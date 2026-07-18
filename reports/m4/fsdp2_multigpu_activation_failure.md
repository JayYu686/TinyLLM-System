# M4.1 双卡 FSDP2、Activation Checkpointing 与 Rank 故障报告

## 1. 结论

M4.1 的 Tiny Model 最小正确性子门禁已取得真实双卡 CUDA/NCCL 证据：物理 GPU 6、7
在严格 Preflight 通过后，使用 BF16、FSDP2 `fully_shard` 和非重入 Activation
Checkpointing 完成 2 个 Optimizer Step。随后独立运行中，Rank 1 在 Step 1 以固定代码
17 强制退出，torchrun 非零结束并保留结构化诊断。

本报告只支持以下结论：

- 两个真实 CUDA Rank 完成了跨卡 NCCL/FSDP2 前向、反向和 Optimizer Step；
- 两个 `TransformerBlock` 均应用 Activation Checkpointing；
- DTensor 本地 Shard 的元素总数等于逻辑参数总数；
- Rank 1 中途退出能够被 torchrun 发现，并在退出前留下只含血缘和故障边界的诊断；
- M4.1 没有 Checkpoint，所以故障 Run 明确标记为不可恢复。

这不是性能 Benchmark，也不证明 Qwen3-8B 能装入两卡或四卡。M4 整体仍为
`IN_PROGRESS`，下一阶段是 M4.2 DCP 分片 Checkpoint/Exact Resume。

## 2. 运行边界

| 项目 | 实际值 |
| -- | -- |
| Git Commit | `7ce1b9621070e5d97ec2359aa5c4a4597ad89e01`；clean |
| 配置 | `configs/fsdp2/tinygpt_debug_nccl_bf16_two_gpu_activation_checkpointing_smoke.yaml` |
| Config SHA256 | `8e9254ac8065830fc780f586cbbafa6dfdcd7ab030a7746e101681c191da3e8d` |
| Python / PyTorch | 3.11.14 / 2.7.1+cu118 |
| CUDA Runtime / NCCL | 11.8 / 2.21.5 |
| Driver | 535.261.03 |
| GPU | 2 × RTX 3090 24GB；物理索引 6、7 |
| 拓扑 | 同 NUMA 1；GPU 6↔7 为 `PIX` |
| 精度 | BF16；不使用 GradScaler；允许 TF32 |
| 模型 | TinyGPT Debug，86,336 个实际参数 |
| World Size / Global Batch | 2 / 4 |
| 序列长度 / Optimizer Step | 32 / 2 |

共享服务器上的其他 GPU 当时正在运行其他任务，本批次没有使用或干预它们。GPU 6、7
分别在每次运行前重新执行 Preflight，不以“之前空闲”替代启动时检查。

## 3. Activation Checkpointing 实现

实现使用 PyTorch 的 `checkpoint_wrapper` 和 `CheckpointImpl.NO_REENTRANT`，按类型选择
TinyGPT 的每个 `TransformerBlock`，完成包装后再应用 FSDP2 `fully_shard`。运行时同时记录：

- 配置是否启用 Activation Checkpointing；
- 被包装模块类型；
- 实际包装数量；
- 包装数量是否等于模型层数。

Schema 在关闭该能力时禁止写入包装类型或非零数量；开启时则要求类型为
`TransformerBlock` 且数量大于零。CPU 单测验证包装后仍可前向、反向并为全部参数产生梯度，
两进程 Gloo 集成测试验证包装与 FSDP2 可组合运行。

## 4. 真实双卡正确性结果

启动 Preflight：

| GPU | 起始显存 | 利用率 | 温度 | 阈值判定 |
| -- | --: | --: | --: | -- |
| 6 | 461 MiB | 0% | 32°C | 通过 |
| 7 | 281 MiB | 6% | 31°C | 通过 |

正式阈值为显存占用不超过 1,024 MiB、利用率不超过 10%、温度不超过 79°C，不提供忙卡
覆盖参数。

运行结果：

| 项目 | 实际结果 | 判定 |
| -- | -- | -- |
| Backend / Device | NCCL / CUDA | 通过 |
| World Size | 2 | 通过 |
| Optimizer Step / Metrics | 2 / 2 | 通过；仅 Rank 0 持久化 |
| Activation Block | `TransformerBlock` × 2 | 通过 |
| 逻辑参数 | 86,336 | 实际实例化结果 |
| Rank 0 / 1 本地 Shard | 43,168 / 43,168 | 通过 |
| Shard 总和 | 86,336 | 与逻辑参数相等 |
| 参数表示 | 两 Rank 均为 DTensor | 通过 |
| Loss Reduce 最大绝对误差 | 0.0 | 通过；固定容差 `1e-12` |
| Gradient Norm 最大 Rank 差异 | 0.0 | 通过；固定容差 `1e-6` |
| 初始/最终完整参数 Hash | 不同 | Optimizer 确实更新参数 |

成功 Run ID：
`20260717T024830Z-tinygpt-debug-fsdp2-nccl-bf16-two-gpu-activation-8e9254ac-2537`。
监督器记录的 7.049 秒仅用于运行血缘，不作为吞吐或扩展效率指标。

## 5. Rank 1 中途退出失败路径

第二次运行重新通过 GPU 6、7 的 Preflight：起始显存分别为 461/281 MiB，利用率为
3%/0%，温度为 33/30°C。所有 Rank 在 Step 1 完成 Optimizer Step 和跨 Rank 指标校验后：

1. Rank 0 原子写入 `forced_rank_exit` 诊断和 `failure_injected` Run 状态；
2. 两个 Rank 同步，确保诊断已持久化；
3. Rank 1 通过 `os._exit(17)` 模拟无清理的进程死亡；
4. torchrun 观察到 Rank 1/退出码 17，并以非零代码 1 结束整个作业。

实际结果：

| 项目 | 实际结果 | 判定 |
| -- | -- | -- |
| 故障 Rank / Step | 1 / 1 | 与注入请求一致 |
| Rank 退出码 | 17 | 被 torchrun 诊断捕获 |
| torchrun 退出码 | 1 | 预期的作业失败 |
| Run 状态 | `failure_injected` | 保留故障边界 |
| 诊断文件 | 1 个 | 通过严格 Schema |
| `correctness.json` | 未生成 | 失败 Run 不冒充成功 |
| Checkpoint 文件 | 0 个 | M4.1 未评估 Checkpoint |
| `resumable` | `false` | 未提前声称恢复能力 |

故障 Run ID：
`20260717T024854Z-tinygpt-debug-fsdp2-nccl-bf16-two-gpu-activation-8e9254ac-ac21`。
这里验证的是故障发现和诊断持久化，不是透明容错或 Exact Resume；恢复能力必须由 M4.2
的 DCP Checkpoint 另行证明。

## 6. 自动验证

提交前执行与 CI 等价的 `make check`：

- 385 项非 GPU 测试通过，2 项 GPU 测试按默认策略取消选择；
- CPU 可测试代码分支覆盖率 85.06%；
- Ruff、Ruff Format、MyPy、JSON Schema Snapshot 全部通过；
- Markdown 链接和公开 Artifact 脱敏检查通过。

此外，双进程 CPU/Gloo 集成测试真实执行 Activation Checkpointing 和 Rank 1 代码 17
退出，避免只用 Mock 证明分布式失败路径。

## 7. 尚未验证与下一步

本批次没有验证：

- DCP Sharded Checkpoint、原子提交、坏 Shard 检测和 Exact Resume；
- 固定 Revision Qwen3-8B 的完整权重加载与四卡显存适配；
- 四卡 50 Step、Step 25 恢复、Peak Memory 或吞吐；
- 单体 Safetensors 部署导出。

下一批应进入 M4.2：先在 Tiny Model/CPU-Gloo 上冻结 DCP Manifest、原子提交和失败矩阵，
再执行真实多卡 CUDA Checkpoint/Resume；通过后才进入 Qwen3-8B 四卡 Memory Probe。

脱敏机器可读证据见
[fsdp2_multigpu_activation_failure.json](raw/fsdp2_multigpu_activation_failure.json)。私有
Artifact Store 保留完整 stdout/stderr、Preflight、Run 目录和环境信息。
