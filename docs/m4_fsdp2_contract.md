# M4 FSDP2 分片训练契约

## 1. 目的与边界

M4 验证 TinyLLM-System 能否在共享的 RTX 3090 服务器上，以 PyTorch 原生 FSDP2
完成单卡无法容纳完整训练状态的模型 Smoke Test，并可靠保存和恢复分片 Checkpoint。

M4 的正式目标是：

- 固定 Qwen3-8B 模型 revision、软件环境、数据视图和四张物理 GPU；
- 使用 BF16、Activation Checkpointing 和 FSDP2 `fully_shard` 的完整分片语义；
- 连续完成 50 个 Optimizer Step；
- 在 Step 25 提交 DCP Sharded Checkpoint，模拟退出后 Exact Resume 到 Step 50；
- 保存每个 Rank 的显存、运行身份、Checkpoint 完整性和导出证据；
- 生成单体 Safetensors 部署导出，但不把导出文件视为训练 Checkpoint。

以下内容不属于 M4：ZeRO-3、长期全参数训练、模型质量提升、8 卡发布门禁、Changed
World Size Resume、CPU Offload、推理性能和 Production Promotion。它们不得阻塞 M4。

在真实四卡显存 Probe 通过前，不得声称 Qwen3-8B 配置能够运行，也不得填写吞吐、显存或
恢复耗时。

## 2. 固定身份

| 项目 | M4 固定值或规则 |
| -- | -- |
| 模型 | `Qwen/Qwen3-8B` |
| 模型 revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| 模型许可证 | 下载前重新核验并记录；不得只依赖历史文档 |
| 精度 | RTX 3090 上 BF16；不使用 GradScaler；TF32 必须显式记录 |
| 策略 | PyTorch FSDP2 `fully_shard(..., reshard_after_forward=True)`；语义对应完整分片，不使用旧 FSDP1 `ShardingStrategy` 枚举 |
| Activation Checkpointing | 开启，并记录应用的 Transformer Block 类型 |
| 序列长度 | 512 |
| Micro Batch | 每 Rank 1 |
| Optimizer Step | 50 |
| 恢复点 | Step 25 提交，恢复后运行至 Step 50 |
| 正式 World Size | 4 |
| 可选增强 | 1/2 卡正确性 Smoke；8 卡只在受控资源窗口追加 |

所有正式参数必须进入 YAML 和 resolved config。CLI 只能覆盖 GPU 选择、Artifact Root、
Resume 模式和其他已批准的运行时字段。

## 3. 依赖与环境隔离

M4 不直接修改已经通过 M0–M3 的核心 `.venv`，也不把 M2 Baseline 环境自动视为 FSDP2
兼容环境。M4 使用独立 `.venv-m4`，并在兼容性 Smoke 通过后提交专用 constraints。

依赖冻结顺序：

1. 固定当前验证起点：Python 3.11、PyTorch `2.7.1+cu118`、CUDA Runtime 11.8；
2. 验证 `torch.distributed.fsdp.fully_shard` 和 `torch.distributed.checkpoint` API；
3. 对候选 Transformers、Accelerate 和 Safetensors 版本执行导入、配置加载和 CPU/Gloo
   Tiny Model Smoke；
4. 验证固定 Qwen revision 的 Config、Tokenizer 和 Model 构建路径；
5. 将实际通过的直接依赖固定到 M4 constraints，并保存完整 `pip freeze`；
6. 执行依赖审计；任何例外必须限定范围、说明风险并设置复审日期。

未通过上述 Smoke 的候选版本不得写成“已支持”。M4 环境不得静默改变 M2 数据构建环境的
Tokenizer 版本。

## 4. 数据契约

正式 M4 Smoke 使用已注册的 `m2-sft-v1-f82ff32e` Train Split，不读取原始数据。由于该
数据产品的 Pack 上限为 1024，而 M4 固定序列长度为 512，实现必须创建确定性的训练视图：

- 只读取已验证的注册目录和 Commit Marker；
- 固定 Split、样本顺序、Seed、截断/切片规则和 Padding 规则；
- 保留 Assistant-only Loss Mask 语义；
- 为派生视图记录父 Dataset Version、配置哈希和内容哈希；
- Resume 时校验父版本、视图哈希和 Sampler Cursor；
- 不回写或覆盖 M2 数据产品。

CPU/Gloo Tiny Model 测试可以使用独立的确定性合成 Fixture，但不得用它替代正式 Qwen
四卡 Smoke 的数据血缘。

## 5. GPU 选择与 Preflight

正式运行不硬编码“永远使用某四张卡”，而是在同一 NUMA 节点内选择四张经过协调的 GPU，
并把物理索引、PCI Bus ID、NUMA、拓扑、进程占用和起始空闲显存写入 Run Artifact。

当前主机中 GPU 5–9 位于 NUMA 1，可组成同 NUMA 四卡集合；GPU 0–4 位于 NUMA 0。
实际集合必须在运行时确定。逻辑 Rank 0–3 不得覆盖或隐藏物理 GPU 身份。

每次 GPU 运行前必须：

1. 执行 `tinyllm doctor --distributed --json`；
2. 检查显存、利用率、温度、功耗和外部进程；
3. 确认四张卡属于同一 NUMA 节点且不会影响其他用户；
4. 检查 Artifact Root 和临时 Checkpoint 目录的可用空间；
5. 保存 Git Commit、dirty 状态、PyTorch/CUDA/NCCL、GPU UUID/Bus ID 和完整配置；
6. 运行短 NCCL Collective 或等价的真实 `torch.distributed` 通信 Smoke。

“低利用率”不自动等于“可共享”：显存 Probe 必须拥有足够且稳定的空闲显存。最低空闲显存
阈值由第一次受控 Probe 的真实结果确定，不能预填。

## 6. 实现顺序

### M4.1：接口与最小正确性

- 冻结 FSDP2 YAML Schema、错误码和 resolved config；
- 抽离 Strategy 边界，不在 DDP Worker 中堆叠 FSDP2 分支；
- 使用 CPU/Gloo Tiny Model 验证初始化、参数分片、前后向、Optimizer Step 和 Rank 0 日志；
- 验证参数初始化、Global Batch 和 Loss Reduce；
- 覆盖非法精度、非法 World Size、非有限 Loss 和 Rank 退出。

当前证据：2026-07-16 已在干净提交 `235c625` 上通过两进程 CPU/Gloo Tiny Model
正确性，覆盖显式 CPU DeviceMesh、DTensor Shard、前后向、Optimizer、Loss Reduce、
Rank 0 Artifact 和 World Size 错配拒绝。Activation Checkpointing、Rank 中途退出以及
CUDA/NCCL 仍未评估，因此 M4.1 保持 `IN_PROGRESS`。详见
[M4.1 CPU/Gloo 报告](../reports/m4/fsdp2_cpu_correctness.md)。

### M4.2：DCP Sharded Checkpoint/Resume

- 使用 `torch.distributed.checkpoint` 保存模型和优化器的分片状态；
- 保存 Scheduler、Step/Epoch、Python/NumPy/PyTorch/CUDA RNG、Sampler Cursor、数据视图、
  Git、环境、精度和四卡物理身份；
- 写入临时目录，计算文件大小和 SHA256，完成校验后原子 Rename；
- 最后写 Commit Marker 并原子更新 `LATEST`；
- Exact Resume 只接受相同 World Size、相同物理训练契约和兼容配置；
- 显式拒绝坏分片、缺失 Marker、错误 World Size、数据漂移和配置漂移。

### M4.3：Qwen3-8B 四卡证据

1. 只加载 Config 验证固定 revision 和模型结构；
2. 执行受控四卡 Memory Probe，记录每 Rank 的 allocated/reserved/峰值；
3. Probe 通过后运行 1–2 个 Optimizer Step 的端到端 Smoke；
4. 执行 50 Step 正式运行；
5. Step 25 提交 Checkpoint 并模拟退出；
6. 由新进程校验并 Exact Resume 到 Step 50；
7. 验证单体 Safetensors 导出可被独立加载；
8. 生成中文审查报告和脱敏机器可读证据。

如果四卡 Probe OOM，保留失败 Run 和每 Rank 显存证据，停止正式运行，并通过新 ADR 在
CPU Offload、缩小模型目标或等待更多独占 GPU 之间做选择。不得静默切换到 8/10 卡，
也不得把失败 Probe 描述为成功支持。

## 7. Checkpoint 完整性与恢复判定

M4 Checkpoint 至少包含：

- 分片模型、Optimizer 和必要的 FSDP2/DCP Planner 元数据；
- Scheduler、Step、Epoch、Gradient Accumulation 位置；
- Python、NumPy、PyTorch 和每 Rank CUDA RNG；
- Stateful Sampler Cursor 与数据视图身份；
- 原始/解析配置哈希、模型和 Tokenizer revision；
- Git Commit、依赖环境、World Size、物理 GPU 身份和精度策略；
- 文件清单、大小、SHA256、Schema Version 和 Commit Marker。

M4 的 Exact Resume 成功条件是：恢复进程从 Step 25 的下一训练位置继续，最终到达 Step 50，
状态和指标满足预先冻结的容差。BF16 分布式容差必须由无中断四卡重复基线确定，不能沿用
CPU 逐位一致要求，也不能在看到恢复结果后调整。

## 8. 观测与报告

机器可读 Artifact 至少记录：

- 每 Step Loss、LR、Gradient Norm、Tokens、耗时和非有限值状态；
- 每 Rank 峰值 allocated/reserved 显存；
- Checkpoint 保存、校验、退出、恢复和导出时间线；
- 模型、Tokenizer、数据视图、配置、Git、环境和硬件血缘；
- 所有成功、失败和被 Preflight 拒绝的 Run。

面向用户审查的报告使用简体中文。公开报告必须脱敏，且只引用真实运行产生的结果。

## 9. M4 完成门禁

M4 只有同时满足以下条件才可标记为 `COMPLETE`：

1. 设计契约、Schema 和依赖 profile 已合并；
2. CPU/Gloo Tiny Model 正确性与失败路径通过；
3. 四卡 Qwen3-8B Memory Probe 有真实证据；
4. 四卡 50 Step 正式 Smoke 完成；
5. Step 25 DCP Sharded Checkpoint 经新进程恢复到 Step 50；
6. 坏 Checkpoint、错误 World Size、数据/配置漂移和 Rank 退出均被验证；
7. 单体 Safetensors 导出通过独立加载检查；
8. 中文报告、原始 JSON、文档和 Issues 同步；
9. PR 通过 CI 并合并。

8 卡、ZeRO-3、V100、长期训练和 Changed World Size Resume 均为增强项，不影响 M4 完成。
