# M5.2-R3 Teacher 来源策略审查报告

## 1. 结论

基于已经提交的 R2、P0 和 P0-R1 真实证据，下一项来源实验选择两阶段
`solve → compress`，确定性规则 Trace 作为控制组。状态为
`two_stage_contract_authorized`：只授权实现 P1 契约，不代表模型质量提升，也未授权 GPU
Pilot、正式扩展、Mixture 或训练。

## 2. 输入证据

| 实验 | 状态 | 关键结果 |
| -- | -- | -- |
| R2 | `length_ceiling_insufficient` | 1536 投影格式率 98.0% / 96.5%，未恢复 4 / 7 |
| P0 | `fail` | 接受 10/40；Config / Log 5 / 5；英文 / 中文 9 / 1 |
| P0-R1 | `fail` | 接受 12/40；Config / Log 4 / 8；英文 / 中文 10 / 2 |

P0 到 P0-R1 的超 192 Token 候选从 52 降到 46，但触及 Teacher 上限的候选从 11 增加到
14，两个任务族仍未通过门禁。该变化不足以支持继续使用单阶段 Prompt-only 方案。

## 3. 策略决策

| 策略 | 决策 |
| -- | -- |
| `single_stage_prompt_control` | 拒绝 |
| `higher_generation_ceiling_only` | 拒绝 |
| `two_stage_solve_compress` | 选为 P1 |
| `deterministic_rule_trace` | 仅作为控制组 |

P1 将 Qwen3-8B 原生 Thinking 求解和 Non-thinking 受约束压缩分开。Solver 的答案先通过
Exact Verifier，Compressor 生成的短 Rationale 再独立验证答案、Evidence Anchor、长度、
重复度、唯一性和训练序列长度。

## 4. 冻结边界

- P1：40 个新任务，Config / Log 各 20，每类英文 14、中文 6；
- Solver：Thinking、一个候选、最大 896 New Tokens；
- Compressor：Non-thinking Greedy、一个候选、最大 256 New Tokens；
- 接受 Trace：不超过 192 Token，完整训练序列不超过 1024 Token；
- 门禁：每类至少 14 条，其中英文至少 10、中文至少 4；
- 规则 Trace：必须 40/40 通过结构检查，但 `training_source_authorized=false`；
- 不读取 M6 冻结评测；
- `formal_source_expansion_authorized=false`；
- `r3_mixture_authorized=false`；
- `r3_training_authorized=false`。

## 5. 当前完成度

本批次已完成：

- 三份父结果的文件 SHA256 与 Schema 校验；
- 四种来源策略的固定处置；
- P1 模型、Seed、阶段、输出协议、Trace Policy 和 Gate；
- 严格 Pydantic Schema 与 JSON Schema；
- 输入漂移、Schema 漂移和越权解锁的单元测试；
- 路径无关、无私有推理的公开机器审查结果。

下一批实现 P1 Task/Context、Solver/Compressor Artifact、双重 Verifier、规则控制组和 CPU
合成 Smoke。上述接口通过前，`p1_gpu_pilot_authorized` 保持 `false`。

## 6. 证据身份

| 项目 | SHA256 |
| -- | -- |
| 策略配置（Canonical） | `6a59d3a83d9420d7f44bd3432c98b1d296d946d1c99abedaa5048837244fa2d6` |
| R2 Decision 文件 | `04165538efce811240b4d4501b13f74151758af7373704c12d6df882e3044ed6` |
| P0 Result 文件 | `5eff250ef4cde98d044c992a0aaf7e2eb75342faa9c377d265a25945a3d4388b` |
| P0-R1 Result 文件 | `c59ab59fd048620e2b8de6a985a5a1deb877bb786d9b28b813531437b582c0b7` |
| 公开 Review 文件 | `60bd5b763f5bec6f0a881ec426a7acfb991db5c0cf008f45bbecc9abe10f06cd` |

相关入口：

- [完整策略设计](../../docs/m5_r3_teacher_source_strategy.md)
- [冻结配置](../../configs/data/m5_r3_teacher_source_strategy.yaml)
- [机器可读审查](raw/m5_r3_teacher_source_strategy_review.json)
- [P0-R1 实验报告](m5_r3_p0_r1.md)
