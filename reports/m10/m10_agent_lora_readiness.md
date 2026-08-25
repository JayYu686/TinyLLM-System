# M10.3 Qwen3-8B Agent LoRA 工程就绪报告

## 结论

Qwen3-8B Agent LoRA 的配置、数据与父模型身份、十步显存 Probe、Adapter-only Checkpoint、
Exact Resume、阶段 Gate 和 Evaluation Subject 接口已经形成独立实现。当前状态为
`ENGINEERING_READY`；真实 BF16 显存、速度、Loss 和 Agent Dev 结果均为 `not_evaluated`，
在 GPU 实测前不声明该配置可完成 1M 训练。

## 固定身份

| 项目 | 固定值 |
| -- | -- |
| 父模型 | `qwen3-8b-m9-base-90587dd6` |
| 父模型 Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| 父模型 Artifact | `81fec43ab8b1f03a158e39e50ec23d99cf8701144e8678aea3ca656d12d08de0` |
| M10 数据版本 | `m10-agent-sft-v1-4655d3e3` |
| 数据 Manifest | `6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490` |
| 配置哈希 | `2a47b09ed960150d6d38103e4218734e72d8f10b9d5731392a6c72fa3bf50cd9` |
| Agent Dev 父模型基线 | 45.00%（36/80，同一公开 Tool Name 协议） |

历史 M5 Adapter 不参与初始化。模型从固定 Qwen3-8B Base 加载，新增 Adapter 采用 Rank 16、
Alpha 32、Dropout 0.05，并覆盖 `q/k/v/o/gate/up/down_proj`。

## 训练与恢复契约

- BF16、TF32、Sequence Length 2048、Micro Batch 1、Gradient Accumulation 8；
- 1M、5M、10M Supervised Token 三个阶段使用同一配置和同一 Run；
- Checkpoint 只保存 Adapter、Optimizer、调度位置、RNG、Cursor 和完整血缘；
- 1M/5M/10M 永久 Pin，中间 Epoch 仅作为滚动恢复点；
- Fresh 只能启动到 1M，后续 Resume 必须提供已接受的阶段 Gate；
- 允许在兼容的 RTX 3090 之间恢复，同时保留每次实际物理 GPU 记录；
- Adapter 导出只包含 `adapter_config.json` 与 `adapter_model.safetensors` 的门禁哈希，不复制
  Base 权重。

## 显存与策略回退

正式训练前必须运行十个真实 Optimizer Step Probe。Probe 与训练配置、Git Commit、父模型、
数据和 GPU 型号绑定。只有该固定 BF16 配置产生可复现 CUDA OOM，才允许创建 NF4 QLoRA
新策略身份；GPU 忙碌、其他进程抢占或配置漂移不能作为 QLoRA 回退证据。

当前实测状态：

| 指标 | 状态 |
| -- | -- |
| 10-step Peak Allocated / Reserved | `not_evaluated` |
| 10-step 时长与 Tokens/s | `not_evaluated` |
| 1M Loss、时长与峰值显存 | `not_evaluated` |
| 1M Agent Dev | `not_evaluated` |

## 阶段门禁

1M→5M 要求 Agent Dev 相对 8B Base 至少提升 1pp。5M→10M 还要求 M6 通用聚合回退不超过
2pp。由于 80 条 Dev 的离散步长为 1.25pp，1M 实际至少需要 37/80（46.25%）才能继续。
未达标时保留 Development 证据并停止该路线，不降低阈值。

5M 的通用回归使用专用成对 Summary：父项保持真实的 8B Base 身份，候选项绑定 5M Adapter
Evaluation Subject；二者必须使用相同 M6 v7 配置。该契约不会通过伪造训练血缘把 Base 写成
Candidate。

每个阶段注册独立 `qwen3-8b-m10-agent-lora-{1m|5m|10m}-<hash>` Evaluation Subject，
Gateway 通过固定 Base + Adapter 进行评测。记录包含 Run、Checkpoint、Adapter、Probe、数据、
配置和父模型血缘，并明确 `production_eligible=false`。
