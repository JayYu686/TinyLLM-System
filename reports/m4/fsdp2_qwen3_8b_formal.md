# M4.3 Qwen3-8B 四卡 FSDP2 正式验收报告

## 1. 结论

M4.3 已通过。固定 Revision 的 Qwen3-8B 在四张 RTX 3090 上真实完成 BF16、FSDP2
`fully_shard`、36 层非重入 Activation Checkpointing、50 个 Optimizer Step、Step 25
DCP 原子 Checkpoint、全新 `torchrun` 进程 Exact Resume、Step 50 最终 Checkpoint，以及
单体 Safetensors 部署导出。

本次正式 Run 的关键事实为：

- 物理 GPU 5、6、7、8 均位于 NUMA 1，三次 Phase 启动前都通过严格空闲 Preflight；
- 一步 Memory Probe 通过，每 Rank 峰值 allocated 约 19.38 GiB，reserved 约 23.15 GiB；
- Fresh Phase 到 Step 25 后提交约 49.15 GB 的完整 DCP Checkpoint，并按计划退出；
- 新进程验证全部分片、配置、数据、Git、环境、World Size 与物理 GPU 身份后，从 Step 26
  继续到 Step 50；
- 指标记录严格覆盖 Step 1–50，没有重复或跳过，Loss 与 Gradient Norm 全部为有限值；
- 最终单体 Safetensors 为 16,381,517,232 Bytes，包含 399 个张量，独立 Hash/Inventory/
  Shape 校验和 Transformers 离线加载均通过。

因此 M4 的四卡正确性、完整分片恢复与导出门禁已经满足。该结论不是吞吐 Benchmark，
不证明模型质量提升，也不外推到 8/10 卡、长上下文、Changed World Size Resume 或 ZeRO-3。

## 2. 固定身份

| 项目 | 实际结果 |
| -- | -- |
| Git Commit | `c8f2b002c3a8b40ebfea4546f5a0599423cf2318` |
| Git 状态 | clean |
| 配置 | `configs/fsdp2/qwen3_8b_four_gpu_formal.yaml` |
| 配置 SHA256 | `59f4821028d05721d646d296488a7547386a6dc415d962bf6e830ce3aab98941` |
| 模型 | `Qwen/Qwen3-8B` |
| 模型 Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| 模型 Artifact SHA256 | `d3dfde74f554d22794ac0591d88c5eac23b864f7f90013b9769b4b3f40ac8d1a` |
| 实际参数量 | 8,190,735,360 |
| 数据版本 | `m2-sft-v1-f82ff32e-m4view-5cef2562` |
| 数据视图 SHA256 | `5cef25622986f75fb882a3aa98decd710e3c3f6e22cd90021fecff22a4e0c2f9` |
| 正式 Run | `20260720T064722Z-qwen3-8b-fsdp2-four-gpu-formal-59f48210-b57f` |
| Python / PyTorch | 3.11.14 / 2.7.1+cu118 |
| CUDA Runtime / Driver | 11.8 / 535.261.03 |
| Transformers / Safetensors | 4.57.6 / 0.6.2 |

完整私有 Artifact Store 保存绝对路径、逐文件 Hash、原始日志与训练权重；公开
[机器可读摘要](raw/fsdp2_qwen3_8b_formal.json)已移除用户名、主机名和绝对路径。

## 3. GPU 与拓扑门禁

正式选择 GPU 5–8，四卡均属于 NUMA 1。Probe 启动前实际值如下：

| GPU | 已用显存 | 利用率 | 温度 | NUMA |
| -- | --: | --: | --: | --: |
| 5 | 1 MiB | 0% | 29°C | 1 |
| 6 | 1 MiB | 0% | 29°C | 1 |
| 7 | 1 MiB | 0% | 30°C | 1 |
| 8 | 1 MiB | 0% | 31°C | 1 |

严格阈值为已用显存不超过 1,024 MiB、利用率不超过 10%、温度不超过 79°C。Fresh 与
Resume Phase 也分别重新执行了同一 Preflight，没有复用 Probe 的旧结论。此前选择忙卡时，
Supervisor 已在 `torchrun` 前拒绝启动；本次没有使用 `--allow-busy`，也没有终止他人进程。

## 4. Memory Probe

Probe 使用与正式 Run 相同的模型、数据视图、World Size、BF16、Sequence Length 512、
Micro Batch 1 和 FSDP2 策略，真实执行一个 Optimizer Step。

| Rank | 物理 GPU | Peak Allocated | Peak Reserved | 结果 |
| --: | --: | --: | --: | -- |
| 0 | 5 | 20,805,166,592 B | 24,859,639,808 B | 通过 |
| 1 | 6 | 20,805,166,592 B | 24,859,639,808 B | 通过 |
| 2 | 7 | 20,805,166,592 B | 24,859,639,808 B | 通过 |
| 3 | 8 | 20,805,166,592 B | 24,859,639,808 B | 通过 |

Peak Reserved 约为 23.15 GiB，而 PyTorch 记录的单卡总显存约为 23.69 GiB，余量很窄。
因此当前结论只适用于固定配置；Sequence Length、Micro Batch、模型或并行策略改变后必须
重新 Probe。

## 5. Step 25 中断与 Step 50 恢复

正式时间线如下：

| Phase | 状态 | Wall Clock | 关键产物 |
| -- | -- | --: | -- |
| Probe | `probe_succeeded` | 97.672 秒 | 一步训练与四 Rank 显存证据 |
| Fresh | `interrupted` | 739.840 秒 | Step 1–25 Metrics、Step 25 DCP、预定退出 |
| Resume | `succeeded` | 1,337.453 秒 | Step 26–50、Step 50 DCP、最终导出 |

上述时长包含模型 Artifact 全量 Hash、模型加载、Checkpoint 写入/复核和导出，不是纯训练
耗时，不能换算为 Tokens/s。当前 Metrics 没有采集逐 Step Duration，所以本报告不发布训练
吞吐结论。

恢复事件明确记录：保留 25 条历史 Metrics、丢弃 0 条、跳过 0 个无效 Checkpoint，并从
Step 25 的下一批数据继续。最终 `metrics.jsonl` 恰好包含连续的 Step 1–50，累计
`102,400` token；Loss 范围为 0.709826–5.846294，Gradient Norm 与 Loss 均无 NaN/Inf。
这些数值只用于 Smoke 正确性，不用于证明收敛或质量提升。

这里的 Exact Resume 指：相同 World Size 下，完整训练状态和数据位置经过校验后恢复，并从
下一 Step 连续推进。M4.2 CPU/Gloo 对照已经证明 Tiny Model 的逐位一致；本次没有运行
未中断四卡数值对照，因此不声称 BF16 结果与未中断 Run 逐位或容差等价。

## 6. Checkpoint 完整性与成本

| Checkpoint | Pin 原因 | 文件数 | DCP 分片 | Rank 状态 | 总大小 |
| -- | -- | --: | --: | --: | --: |
| Step 25 | `interruption` | 12 | 4 | 4 | 49,153,987,502 B |
| Step 50 | `final` | 12 | 4 | 4 | 49,153,987,502 B |

两个 Manifest 都声明模型、Optimizer、Scheduler、GradScaler 不适用状态、Python/NumPy/
PyTorch/CUDA RNG、Stateful Sampler、配置和环境完整覆盖。每个文件均记录大小和 SHA256；
发布流程为临时目录写入、`fsync`、Manifest/Commit Marker、原子 Rename、全量再验证、最后
原子更新 `LATEST`。

单个 Checkpoint 约 45.78 GiB，两个 Pin 点合计约 91.56 GiB。恢复前重复读取完整文件带来
明显 I/O 时间，这是本次真实暴露的工程成本。后续可以缓存已验证摘要以减少同一阶段的重复
Hash，但不能因此取消恢复边界的 fail-closed 校验。

## 7. Safetensors 导出

最终导出与训练 Checkpoint 明确分离，Manifest 的用途为
`deployment_export_not_training_checkpoint`：

| 项目 | 结果 |
| -- | -- |
| 文件大小 | 16,381,517,232 B |
| SHA256 | `f811603cfdb084fa56da15e72f8968b39fd26839b50846471bea13dc92b89933` |
| 张量数量 | 399 |
| 独立 Safetensors 校验 | Hash、Inventory、全部 Shape 通过 |
| 独立 Transformers 加载 | `Qwen3ForCausalLM`、8,190,735,360 参数、BF16，通过 |

导出目录包含模型 Config、Generation Config 和 Tokenizer 文件，但不包含 Optimizer、
Scheduler、RNG 或 Sampler，因此不能用于 Exact Resume，也不能冒充完整训练 Checkpoint。

## 8. 失败路径与完成边界

M4 的组合证据已覆盖：忙卡 Preflight 拒绝、Rank 强制退出、坏 DCP 分片、缺 Commit Marker、
缺 Rank 状态、错误 World Size、配置漂移和数据版本漂移。M4.1 证明多卡 Rank 故障诊断，
M4.2 证明 DCP 原子协议与失败矩阵，M4.3 证明真实 Qwen3-8B 四卡恢复和导出。

M4 至此满足完成门禁。仍未评估且不阻塞 M4 的项目包括：

- 8/10 卡 FSDP2 与 Changed World Size Resume；
- ZeRO-3、CPU Offload、V100 和长上下文；
- 正式吞吐 Benchmark 与 Checkpoint I/O 优化；
- 模型质量提升、Candidate Promotion 和 Production 部署。

下一里程碑为 M5 正式后训练；必须继续沿用 M2 数据/Baseline、M4 分片与恢复契约，不能把
本次 50 Step 系统 Smoke 当作 M5 训练成果。
