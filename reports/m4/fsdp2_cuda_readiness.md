# M4 隔离依赖与 CUDA 就绪性报告

> 本报告冻结的是 2026-07-16 的依赖与单卡就绪性证据。后续真实双卡通信、Activation
> Checkpointing 和 Rank 故障结果见
> [M4.1 双卡报告](fsdp2_multigpu_activation_failure.md)；下文的“尚未验证”保留为当时边界。

## 1. 结论

M4 的独立依赖环境和单卡 CUDA/NCCL API 路径已经通过真实 Smoke，但 M4.1 仍未完成：
当前没有两张同时满足严格空闲阈值的 GPU，因此没有运行双卡 FSDP2，也没有产生跨卡通信、
Activation Checkpointing、DCP 或 Qwen3-8B 结论。

本批次的可接受结论是：

- `.venv-m4` 与 M0–M3 环境隔离，候选直接依赖已经冻结；
- FSDP2 `fully_shard`、DCP `save/load` 和 Transformers Qwen3 类可导入；
- 无网络、无远程模型文件的 Tiny Qwen CPU 前后向通过；
- 同一环境中的两进程 CPU/Gloo FSDP2 复查通过；
- GPU 9 上的单进程 BF16、CUDA/NCCL、FSDP2 DTensor 和 Optimizer 路径通过；
- 双卡请求在发现 GPU 8 忙碌后被 Preflight 拒绝，未启动训练进程。

因此，下一步必须等待两张真正空闲的同 NUMA GPU 完成双卡 NCCL 正确性，而不是把单卡
NCCL Process Group 描述成多卡通信验证。

## 2. 依赖门禁

| 项目 | 实际结果 | 判定 |
| -- | -- | -- |
| Python | 3.11.14 | 通过 |
| PyTorch | 2.7.1+cu118 | 通过 |
| CUDA Runtime | 11.8 | 通过 |
| NCCL | 2.21.5 | 通过导入/单卡运行 |
| Transformers | 4.57.6 | 通过 |
| Accelerate | 1.12.0 | 通过 |
| Safetensors | 0.6.2 | 通过 |
| Tokenizers | 0.22.2 | 通过 |
| `fully_shard` | 可导入 | 通过 |
| DCP `save/load` | 可导入 | 通过；尚未验证保存/恢复 |
| Tiny Qwen | 82,304 参数，有限 Loss 与有限梯度 | 通过；不是目标模型 |
| 网络/模型下载 | 未发生 | 符合本批次边界 |

依赖证据绑定干净提交 `2a969632b8d132d6fe77ca2f1f4c7ae71ffe9cec`。
`requirements/constraints/m4.txt` 的 SHA256 为
`84de7c42d55bbea874a0b4f0a40a56bf68d30ee29bb4be818faac0f2421a58a7`。
完整 `pip freeze` 保存于私有 Artifact Store，公开证据只保留其 SHA256：
`557b098e4c4898791ba067db5146336dfb80571ad25a2b35c774aa8eceb48737`。

`pip-audit` 没有发现四个已审查例外之外的已知漏洞。例外仅适用于固定 Qwen3、本地 SDPA、
禁止 Remote Code、Safetensors 权重和原生 FSDP2/DCP 路径，详见
[M4 审计例外](../../requirements/m4_security_exceptions.md)。CUDA 11.8 PyTorch Wheel 无法由
PyPI 审计器解析，这是一项审计限制，不代表 PyTorch 没有安全风险。

## 3. CPU/Gloo 隔离环境复查

独立环境在干净提交 `2a96963` 上重新执行两进程 CPU/Gloo Smoke：

| 项目 | 实际结果 |
| -- | -- |
| World Size | 2 |
| Optimizer Step | 2 |
| 逻辑参数 | 86,336 |
| Rank Shard 总和 | 86,336 |
| Loss Reduce 最大绝对误差 | 0.0 |
| Gradient Norm 最大 Rank 差异 | 0.0 |
| 结果 | 通过 |

这证明独立依赖 Profile 没有破坏已合并的 CPU/Gloo FSDP2 路径。

## 4. 单卡 CUDA/NCCL Smoke

运行绑定干净提交 `6c241a09e2a2a08eacfce79f630c1142b9e3a041`，配置为
`configs/fsdp2/tinygpt_debug_nccl_bf16_single_gpu_smoke.yaml`。

Preflight 的真实状态：

| 项目 | 实际结果 |
| -- | -- |
| 物理 GPU | 9 |
| 型号 | NVIDIA GeForce RTX 3090 |
| 起始显存占用 | 1 MiB |
| 起始利用率 | 0% |
| 起始温度 | 31°C |
| Driver | 535.261.03 |

训练正确性结果：

| 项目 | 实际结果 |
| -- | -- |
| Backend / Device | NCCL / CUDA |
| 精度 | BF16；不使用 GradScaler；显式允许 TF32 |
| World Size | 1 |
| Optimizer Step | 2 |
| Global Batch | 2 |
| 逻辑参数 / Shard 总和 | 86,336 / 86,336 |
| DTensor 参数 | 通过 |
| Loss Reduce 最大绝对误差 | 0.0 |
| Gradient Norm 最大 Rank 差异 | 0.0 |
| 初始/最终完整参数 Hash | 不同 |
| Run 持续时间 | 11.383 秒 |
| 结果 | 通过 |

Run ID 为
`20260716T153830Z-tinygpt-debug-fsdp2-nccl-bf16-single-gpu-smoke-1a20d57a-6123`。
这里的持续时间只用于运行血缘，不是吞吐 Benchmark。

## 5. 双卡失败路径

随后请求 GPU 8、9 运行双卡配置。Preflight 观察到 GPU 8 已占用 5,802 MiB、利用率 78%，
超过 `memory_used_mib <= 1024` 和 `utilization_percent <= 10` 的正式阈值，因此监督器拒绝
启动；GPU 9 当时仍为空闲。该失败路径证明 M4 启动器不会因为“显存还有剩余”而使用他人
正在运行的 GPU。

这不是双卡训练失败，也不能计为 NCCL 多卡验证；训练进程从未启动。

## 6. 尚未验证

- 双卡或更多 GPU 的 M4 FSDP2 Collective；
- Activation Checkpointing 和 Rank 中途退出；
- DCP Sharded Checkpoint、完整性校验和 Exact Resume；
- 固定 Qwen3-8B Revision 的许可证、文件哈希、Config、Tokenizer 和权重加载；
- Qwen3-8B 四卡显存适配、Peak Memory、50 Step 或吞吐；
- Safetensors 单体导出。

以上项目均不得从本报告外推为已支持。脱敏机器可读证据见
[fsdp2_cuda_readiness.json](raw/fsdp2_cuda_readiness.json)。

## 7. 下一门禁

1. 等待两张同 NUMA GPU 同时通过严格 Preflight；
2. 运行 `tinygpt_debug_nccl_bf16_two_gpu_smoke.yaml`，验证真实跨卡 NCCL、Shard 覆盖和
   Loss Reduce；
3. 增加 Activation Checkpointing 和 Rank 中途退出失败路径；
4. 进入 M4.2 DCP Sharded Checkpoint/Resume；
5. 前述门禁通过后，才允许获取固定 Revision 的 Qwen3-8B 并做四卡 Memory Probe。

## 8. 自动验证

- `make check`：374 passed，2 个 GPU 测试按默认策略取消选择；
- CPU 可测试核心分支覆盖率：85.19%；
- M4 Dependency Smoke：通过；
- M4 Dependency Audit：通过，4 个例外已单独记录；
- `pip check`：无损坏依赖。
