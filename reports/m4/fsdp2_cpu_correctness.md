# M4.1 FSDP2 CPU/Gloo 最小正确性报告

执行日期：2026-07-16（Asia/Shanghai）

状态：**PASS（仅限两进程 CPU/Gloo Tiny Model 正确性）**

## 1. 本次验证范围

本报告记录 M4 的第一道实现门禁：在不使用 GPU、Qwen 权重或外部模型依赖的情况下，验证
PyTorch 2.7.1 FSDP2 最小运行路径。测试使用 YAML 配置、两个 torchrun 进程、Gloo 后端、
显式 CPU DeviceMesh 和 TinyGPT Fixture。

本次目标是证明：

- CPU Smoke 不会因为主机存在 CUDA 而隐式占用 `cuda:0`；
- `fully_shard` 后模型参数在每个 Rank 上表现为 DTensor；
- 本地 Shard 完整覆盖一次逻辑模型参数；
- 前向、反向、Gradient Clipping、AdamW 和 Scheduler 可以完成；
- Loss 聚合、完整模型状态重建和 Rank 0 单写 Artifact 正确；
- torchrun World Size 与 YAML 不一致时在创建 Artifact 前失败。

## 2. 可复现身份

| 项目 | 实际结果 |
| -- | -- |
| Git Commit | `235c62533a5e988d8a49b665126d27d65ee62606` |
| Git 状态 | clean |
| 配置 | `configs/fsdp2/tinygpt_debug_gloo_smoke.yaml` |
| 配置 SHA256 | `ba2a44fac9330dab43122322e1911e408ae542ead5fe25d35288277b9eea73f9` |
| Run ID | `20260716T140451Z-tinygpt-debug-fsdp2-gloo-smoke-ba2a44fa-ea44` |
| Python | `3.11.14` |
| PyTorch | `2.7.1+cu118` |
| Backend / Device | Gloo / CPU |
| World Size | 2 |
| Optimizer Step | 2 |
| Global Batch | 4 |

CUDA Runtime 11.8 存在于该 PyTorch 构建中，但本次两个 Rank 的设备均记录为 `cpu`，物理 GPU
索引均为 `null`。这只证明本次运行没有选择 GPU，不是 CUDA/NCCL FSDP2 结果。

## 3. 正确性结果

| 检查项 | 实际结果 | 判定 |
| -- | --: | -- |
| 逻辑模型参数量 | 86,336 | 记录 |
| Rank 0 本地 Shard | 43,168 | PASS |
| Rank 1 本地 Shard | 43,168 | PASS |
| 本地 Shard 总和 | 86,336 | PASS；完整覆盖一次逻辑参数 |
| 参数类型 | 两 Rank 均为 DTensor | PASS |
| Rank 0 持久化 Metrics | 2 条 | PASS；与 Optimizer Step 一致 |
| Loss Reduce 最大绝对误差 | 0.0 | PASS；固定容差 `1e-12` |
| Gradient Norm 最大 Rank 差异 | 0.0 | PASS；固定容差 `1e-6` |
| 初始/最终完整状态 Hash | 不同 | PASS；Optimizer 确实更新模型 |
| Checkpoint | `not_evaluated_m4_1` | 未评估，不计为通过 |

两步观察值如下：

| Step | Loss | Gradient Norm | 累计 Tokens |
| --: | --: | --: | --: |
| 1 | 4.188438892364502 | 2.7215566635131836 | 124 |
| 2 | 4.178703308105469 | 2.799715518951416 | 248 |

两步 Loss 变化只用于证明数值路径可执行，不构成收敛、模型质量或训练性能结论。

## 4. 失败路径

使用相同 YAML 但只启动一个 torchrun 进程时，Worker 返回
`torchrun WORLD_SIZE does not match the FSDP2 config`。torchrun Launcher 最终退出码为 1，
Worker 报告错误类型为 `TrainingError`，且没有创建 Run Artifact。

自动化测试还覆盖：未知/强制类型字段、Gloo/CUDA 错配、CPU/BF16 错配、未经验证的 Gradient
Accumulation、超过四卡的正确性配置、相对输出路径、未分片参数、缺失/非标量/NaN Loss，
以及缺失或非有限 Gradient。

## 5. 自动化门禁

代码提交前的完整 `make check` 结果：

- 363 个非 GPU 测试通过；
- 2 个 GPU 测试按默认 CI 规则取消选择；
- 分支覆盖率 85.19%；
- Ruff、Ruff Format、MyPy Strict、JSON Schema Snapshot、Markdown 链接和公开 Artifact
  检查全部通过。

## 6. 明确限制与下一门禁

本报告不能用于声称以下能力已经完成：

- CUDA/NCCL 或 BF16 FSDP2；
- Activation Checkpointing；
- DCP Sharded Checkpoint 或 Exact Resume；
- Qwen3-8B 加载、四卡显存适配或 50 Step 正式 Smoke；
- 训练吞吐、扩展效率或模型质量。

下一步是建立隔离的 `.venv-m4` 和经过兼容性验证的依赖约束，然后执行 Tiny Model
CUDA/NCCL Smoke。只有这些前置通过后，才允许下载固定 Revision 的 Qwen3-8B 并启动受控
四卡 Memory Probe。

脱敏机器可读证据见 [raw/fsdp2_cpu_correctness.json](raw/fsdp2_cpu_correctness.json)。原始
Run 目录保留在私有 Artifact Store，不进入公共仓库。
