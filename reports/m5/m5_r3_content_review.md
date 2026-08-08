# M5.2-R3-P2 内容审查报告

## 1. 当前状态

P2 真实来源 Pilot 已通过自动门禁，33 条接受样本随后完成私有维护者内容审查。维护者确认
Codex 的 33 条逐项草案判定，最终状态为 `APPROVED`：33/33 通过，正式 240 条来源扩展已
授权；R3 Mixture 和训练仍未授权。

这一步是数据质量门禁，不是运行命令审批。Codex 草案只用于减少逐条核对成本，不能写成
人工审查结果。

## 2. 审查输入

| 项目 | 固定值 |
| -- | -- |
| Review | `m5-r3-p2-content-review-v1` |
| P2 Public Result SHA256 | `94b89e3a72af6edb0f8c50772f8b8d807bbc6fa8ccf70b44f610426817fb03f2` |
| P2 Private Raw SHA256 | `2b693da607c880fa456b6701ad8bc2449ed1927d5b07aab33dc662ef2127a7b5` |
| P2 Samples SHA256 | `2d73a4d62b657b98e9da2539d7cf10fdc8a2f3af369e4db4956348b5b79c3ea8` |
| 审查条数 | 33 |
| Config / Log Diagnosis | 17 / 16 |
| 英文 / 中文 | 24 / 9 |
| 私有 Packet SHA256 | `db93d3a01c51b8ab7263e8dd3fbd58185e28b4d62335c50eacafdf5a4ebf7f47` |
| Codex Draft SHA256 | `7860e04788934f04881b4de319cde08133db234e7aa75de9f4bbcdff462071bf` |
| Maintainer Judgments SHA256 | `30b669b2d4ec4aa86208e7dee44962283272a3aec05d17d4b1ef33f927adce7a` |
| 审查结果 | 33/33 通过 |

原始 Prompt、Model-distilled Rationale 和逐条判断只保存在私有 Artifact Store。

## 3. 固定标准

每条样本必须同时满足：

1. 最终标签与直接证据一致；
2. 短推理引用可定位的决定性证据，足以支撑所选标签；
3. 没有无依据事实、其他候选标签或误导性因果陈述。

Codex 初审对 33 条均建议 `[true, true, true]`。维护者已明确确认这 33 条草案判定，
Finalizer 将其转换为独立的维护者 Judgments 并记录私有文件 SHA256。

## 4. 失败闭锁

Review Finalizer 会拒绝：

- Codex 草案被直接当作维护者判断；
- 少于或多于 33 条判断；
- Task ID 重复、缺失或来源不一致；
- P2 Public / Private / Samples SHA256 漂移；
- Criterion 顺序、`passed` 汇总或 Schema 不一致；
- 未显式提供维护者确认；
- 覆盖完整但存在任一内容拒绝时错误授权扩展。

本次 33/33 维护者判断通过，公开 Review Result 已标记
`formal_source_expansion_authorized=true`。Mixture 和训练仍保持 `false`。

## 5. 下一步

执行 240 条正式来源生成和 160 条分层选择。该阶段必须重新记录任务、模型、Seed、生成、
选择与污染身份；不能因为 Pilot 和人工审查通过而绕过正式来源门禁。

相关入口：

- [P2 实验报告](m5_r3_p2.md)
- [公开审查结果](raw/m5_r3_p2_content_review.json)
- [Review Finalizer](../../scripts/finalize_m5_r3_p2_content_review.py)
- [SFT 契约](../../docs/m5_sft_contract.md)
