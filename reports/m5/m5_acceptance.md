# M5 双模式正式后训练总验收报告

## 1. 验收结论

M5 的代码、真实训练、恢复、失败路径、双模式评测、部署导出和血缘证据已经齐备，可以标记
为完成。项目现已同时具备：

- Qwen3-0.6B 四卡 BF16 Full SFT：50M Token、2M 中断、Exact Resume、五阶段快照与评测；
- Qwen3-8B 单卡 BF16 LoRA：10M Token、5M 中断、Exact Resume、Adapter 与 Model Card；
- 原生 GQA 与显式 Thinking/Non-thinking 模式；
- 数据、模型、配置、Git、环境、硬件、Checkpoint、评测与导出的完整 Hash 血缘；
- 七类训练失败路径和共享服务器 GPU Preflight/温控恢复。

M5 的最优 0.6B 开发点为 10M Full-SFT 快照，8B LoRA 最终点也进入 M6 比较队列。二者当前
均保持 `Development`；M6 才负责独立发布集、通用回归、JSON Valid Rate 与 Candidate 晋级。

## 2. 两条正式路线

| 路线 | 训练 | 恢复 | M5 Dev 结果 | 产物 |
| -- | -- | -- | -- | -- |
| Qwen3-0.6B Full SFT | 四卡 BF16，50M Token | 2,002,739 → 50M Exact Resume | 最优 10M：Thinking 95.0%，Non-thinking 47.5% | 完整 Checkpoint、五阶段 Safetensors |
| Qwen3-8B LoRA | 单卡 BF16，10M Token | 5,000,444 → 10M Exact Resume | 最终：Thinking 99.0%，Non-thinking 72.0% | Adapter Safetensors、Model Card |

0.6B 的 50M 终点为 Thinking 91.5%、Non-thinking 39.0%，低于 10M 快照但仍高于同协议 Base
的 70.5%和 37.0%。完整曲线如实保留，不用末段 Loss 下降替代质量评测结论。

## 3. 完成条件核对

| 完成条件 | 证据 | 状态 |
| -- | -- | -- |
| 双模式设计与严格 Schema | M5 契约、ADR-0005/0006、JSON Schema Snapshot | 通过 |
| 数据 Manifest、拒绝与污染 | M5.1、R3 正式来源、50M 版本化 Mixture | 通过 |
| 训练前 Baseline 与配比消融 | Base、六组消融、R1/R2/R3、Thinking Budget v2 双 Seed | 通过 |
| 0.6B Full SFT | 50M、四卡 DDP、Exact Resume、五阶段评测 | 通过 |
| 8B LoRA | BF16 Probe、10M、Exact Resume、Adapter、Model Card | 通过 |
| 失败路径 | OOM、NaN/Inf、坏 Checkpoint、磁盘、数据漂移、World Size、进程退出 | 通过 |
| 公开报告与血缘 | 中文总验收、英文摘要、脱敏 JSON、私有原始 Artifact | 通过 |

## 4. 真实失败与修复

M5 没有删除异常 Run。保留的关键失败包括：数据来源 Token 预算不足、多个格式修复 Gate
拒绝、GPU 7 驱动失联、Micro Batch 未生效、Checkpoint 血缘字段不完整，以及实际批次 Token
超过阶段目标 532 Token 导致的严格 Schema 拒绝。每个失败都先保存证据，再通过版本化代码或
协议修订处理；最终成功 Campaign 使用 clean Commit `c406e6760c6ea6b5eb19966740af4c494983576d`。

## 5. 证据入口

- [M5 设计契约](../../docs/m5_sft_contract.md)
- [Thinking Budget v2 选优](m5_thinking_budget_v2.md)
- [0.6B Full SFT 正式报告](m5_full_sft_formal.md)
- [8B LoRA 正式报告](m5_lora_formal.md)
- [失败路径报告](m5_failure_paths.md)
- [英文公开摘要](m5_public_summary.en.md)
- [0.6B 脱敏机器摘要](raw/m5_full_sft_formal.json)
- [8B LoRA 脱敏机器摘要](raw/m5_lora_formal.json)

## 6. 下一阶段

M6 将冻结并执行独立的领域与通用评测，对 Base、0.6B 10M 快照、0.6B 50M 终点和 8B LoRA
分别进行可比范围内的回归分析。Promotion Gate 仍要求目标能力提升、Bootstrap 95% CI、通用
任务回退上限、JSON Valid Rate 和完整血缘同时通过。M5 的完成不会自动授予 Candidate 或
Production 状态。
