# M10.3 Qwen3-8B Agent LoRA 工程就绪报告

## 结论

Qwen3-8B Agent LoRA 的配置、数据与父模型身份、十步显存 Probe、Adapter-only Checkpoint、
阶段 Gate 和 Evaluation Subject 接口已经形成独立实现。10-step BF16 显存 Probe 与 1M
真实训练均已完成；1M Agent Dev 低于父模型，Continuation Gate 拒绝继续到 5M。完整结果见
[`M10.3 1M 阶段报告`](m10_agent_lora_1m.md)。

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
| 10-step Peak Allocated / Reserved | 20.58 / 22.55 GiB |
| 10-step 时长 | 219.88 秒 |
| 10-step 监督 Token | 9,496 |
| 1M 前/后 100 Step 平均 Loss | 0.7132 / 0.2715 |
| 1M 时长 / Peak Reserved | 22,486.93 秒 / 22.55 GiB |
| 1M Agent Dev | 32.50%（26/80） |

正式 Probe 运行于干净提交 `dbca533f74d5c70a4270e3a0583c756041fd15b1`，结果 SHA256 为
`2426230da1c33650d8769327ae9e438752787e3770187f924f574552828e9c40`。软件环境哈希为
`8e62e483b63c38ddef74495b143ad8b7ee24184784361d26b9063f19917bacef`，硬件兼容哈希为
`5e68c6806e4673ec5a70ba795cb4bce980be86ecf0b8400179b2ab2f77d2f9d2`。原始、无路径结果见
[`m10_agent_lora_memory_probe.json`](raw/m10_agent_lora_memory_probe.json)。

峰值 Reserved 距 24 GiB 物理显存约保留 1.45 GiB，因此正式训练必须使用无其他显存占用的
独占 RTX 3090。该结果证明固定 BF16 LoRA 配置可运行，QLoRA 回退条件未触发。首次尝试在
模型加载前因环境缺少 PEFT 失败，该尝试不计为 CUDA OOM，也没有产生 Probe Artifact；正式
结果使用固定 PEFT 0.19.1 环境完成。

## 阶段门禁

1M→5M 要求 Agent Dev 相对 8B Base 至少提升 1pp。5M→10M 还要求 M6 通用聚合回退不超过
2pp。由于 80 条 Dev 的离散步长为 1.25pp，1M 实际至少需要 37/80（46.25%）才能继续。
未达标时保留 Development 证据并停止该路线，不降低阈值。真实 1M 结果为 26/80，低于父模型
36/80，也低于继续训练所需的 37/80；Gate 以 -12.50pp 正式拒绝，5M/10M 不再执行。

5M 的通用回归使用专用成对 Summary：父项保持真实的 8B Base 身份，候选项绑定 5M Adapter
Evaluation Subject；二者必须使用相同 M6 v7 配置。该契约不会通过伪造训练血缘把 Base 写成
Candidate。

每个阶段注册独立 `qwen3-8b-m10-agent-lora-{1m|5m|10m}-<hash>` Evaluation Subject，
Gateway 通过固定 Base + Adapter 进行评测。记录包含 Run、Checkpoint、Adapter、Probe、数据、
配置和父模型血缘，并明确 `production_eligible=false`。
