# M6 独立评测、晋级与发布验收报告

## 结论

M6 已完成。Qwen3-0.6B Full-SFT 模型在独立冻结的 v7 Release Suite 上通过全部 11 项
Candidate Gate，并以 `qwen3-0-6b-m6-d16c2357` 注册为 `Candidate`。M6 的晋级上限保持为
Candidate；`production_eligible=false`，Production 资格由 M7 的真实推理性能门禁决定。

最终结论来自四条相互独立的 300 题领域评测、Base/Candidate 通用任务评测、160 条维护者
人工判断、10,000 次配对 Cluster Bootstrap 和完整血缘校验。公开机器摘要见
[M6 v7 验收数据](raw/m6_v7_acceptance.json)。

## 1. 冻结协议与模型身份

| 项目 | 实际身份 |
| -- | -- |
| Release 协议 | `m6-release-v7` |
| 领域集 | `tinyllm-domain-thinking-boundary-audit-v1-b82cbca1` |
| 领域集内容 SHA256 | `b82cbca1821cadbaf4872636e89c61cef730ebe09413f9c63f34993302b6f955` |
| Release 配置 SHA256 | `a82c4d5f2aa4b2c3be881641c7d97d568901fd6da3241ac43331a469385ab59d` |
| Base | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` |
| Candidate 架构 | 596,049,920 参数、GQA、Full SFT、Thinking/Non-thinking 双模式 |
| Candidate 训练 Run | `20260811T024325Z-m6-domain-contract-r41-seed42-dce956b0-d5b6` |
| Candidate 数据版本 | `m6-domain-generalization-mixture-v2-f2e029e4` |
| Candidate Checkpoint | `checkpoint-tokens-0001000000` |
| Candidate 模型 SHA256 | `63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6` |

v7 在模型进入该 Suite 前冻结，与 v1–v6 的完整 Prompt 交集为零。Base 与 Candidate 使用相同
Prompt、Tokenizer、解码控制、JSON Schema 和评分器。M5 Dev 只用于训练阶段选择；M6 v7
结果没有回流训练或阈值调整。

## 2. 四路领域评测

| 路线 | 正确数 | 分数 | JSON Valid | 格式有效 | 强制闭合 | 可见推理泄漏 |
| -- | --: | --: | --: | --: | --: | --: |
| Base Thinking | 103/300 | 34.33% | 79/80 | 300/300 | 24/300 | 0 |
| Candidate Thinking | 125/300 | 41.67% | 80/80 | 300/300 | 5/300 | 0 |
| Base Non-thinking | 67/300 | 22.33% | 80/80 | 300/300 | 0 | 0 |
| Candidate Non-thinking | 122/300 | 40.67% | 80/80 | 300/300 | 0 | 0 |

Thinking 提升 7.34 个百分点，配对 Cluster Bootstrap 95% CI 为 `[+0.33, +14.29]pp`；
Non-thinking 提升 18.34 个百分点，95% CI 为 `[+12.46, +24.40]pp`。两个区间下界均严格
大于零。

## 3. 通用能力回归

| 指标 | Base | Candidate | 变化 |
| -- | --: | --: | --: |
| ARC-Easy `acc_norm` | 47.26% | 53.28% | +6.02pp |
| HellaSwag `acc_norm` | 42.07% | 45.36% | +3.29pp |
| PIQA `acc_norm` | 66.05% | 64.80% | -1.25pp |
| 三任务等权聚合 | 51.80% | 54.48% | +2.68pp |

通用聚合没有发生回退，通过“最多回退 2pp”的门禁。三项结果来自完整 14,256 样本评测，
聚合规则为等任务 `acc_norm` 均值。

## 4. 人工审查

冻结领域集包含 40 条 Human Rubric 题。Base/Candidate × Thinking/Non-thinking 四路共 160 条
草案均由维护者确认，随后写入不可变 Judgment 与 Commit 文件并重新组装最终评测：

| 路线 | 已审查 | 通过 |
| -- | --: | --: |
| Base Thinking | 40/40 | 2 |
| Base Non-thinking | 40/40 | 0 |
| Candidate Thinking | 40/40 | 1 |
| Candidate Non-thinking | 40/40 | 0 |

人工题按预注册 Rubric 逐条判定；草案生成者与最终维护者确认角色在私有证据中分开记录。公开
报告仅披露统计和内容无关哈希，不包含 Prompt、Reference、模型原始输出或推理正文。

## 5. Candidate Gate

| 门禁 | 阈值 | 实际值 | 结果 |
| -- | --: | --: | -- |
| Thinking 领域增量 | ≥ +3pp | +7.34pp | 通过 |
| Thinking Bootstrap 下界 | > 0 | +0.33pp | 通过 |
| Non-thinking 领域增量 | ≥ +3pp | +18.34pp | 通过 |
| Non-thinking Bootstrap 下界 | > 0 | +12.46pp | 通过 |
| 通用聚合回退 | ≥ -2pp | +2.68pp | 通过 |
| Candidate 双模式 JSON Valid | ≥ 98% | 100% | 通过 |
| Candidate Thinking 格式 | ≥ 99% | 100% | 通过 |
| Candidate Thinking 强制闭合 | ≤ 10% | 1.67% | 通过 |
| Candidate Non-thinking 泄漏 | = 0 | 0 | 通过 |
| 评测完整性 | 完整 | 160/160 人工判断完成 | 通过 |
| 血缘完整性 | 完整且 Clean Git | 完整 | 通过 |

比较记录 SHA256 为
`d16c2357dabb1011f56d6d3b5026b7385e2afb6dd4bd2eddcab40f60c37f5432`。

## 6. Registry 与 Run 查询索引

Candidate 已通过原子写入注册为 `qwen3-0-6b-m6-d16c2357`，注册记录 SHA256 为
`bb9189f4df4ffa8b089fc2892e12d31273875efbe9739eede8ea0f903d95fba9`。重复执行相同晋级可
幂等读取，身份冲突会拒绝覆盖。

M6 同时交付 `tinyllm run rebuild|list|show`。真实重建从私有 Artifact Store 的 57 个
`run.json` 生成 SQLite v1 查询索引，57/57 条成功入库，`PRAGMA integrity_check` 返回 `ok`。
源树 SHA256 为 `42b480f9b73e5eb37180bd1e3d5c02204f724146cc026e57ac327addcb1543ba`。
SQLite 只保存内容无关投影；Run JSON/JSONL 继续作为事实源，索引可随时完整重建。

## 7. 50M 对照与适用边界

M6.4 复用 M6 启动前已完成并冻结的 50M 长程对照：Qwen3-0.6B 在同一 M5 Dev 协议下，10M
快照为 Thinking 95.0%、Non-thinking 47.5%，50M 终点为 91.5%和 39.0%。该证据说明继续训练
出现回退，支持选择短程快照；它不替代 M6 v7 的独立 Base/Candidate 门禁，也未在看到 v7
答案后参与选优。详情见 [M5 Full-SFT 正式报告](../m5/m5_full_sft_formal.md)。

当前公开结论仅覆盖 Qwen3-0.6B Candidate 质量、单机训练与既有真实硬件范围。M7 将测量
vLLM 的 TTFT、TPOT、吞吐、P50/P95、并发稳定性和回滚流程，完成后才能考虑 Production。

## 8. M6 验收清单

- [x] 独立冻结双模式 Release Suite 与污染隔离；
- [x] Base/Candidate 四路 300 题领域评测；
- [x] ARC-Easy、HellaSwag、PIQA 完整评测；
- [x] 160/160 维护者人工判断；
- [x] 10,000 次配对 Cluster Bootstrap；
- [x] 11/11 Candidate Gate；
- [x] Candidate 原子注册，Production 边界保持关闭；
- [x] 可从 Run 目录重建的 SQLite 查询索引；
- [x] 50M 长程回退对照；
- [x] 中文验收、英文公开摘要和 10 分钟中文演示；
- [x] `v0.6.0-rc.1` 版本材料。

M6 状态：`COMPLETE`。下一阶段：M7 推理部署与 Production Gate。
