# M5.2-R1 Thinking 格式可靠性修正报告

## 1. 当前结论

M5.2-R1 的失败归因、修正策略、数据构建、两组真实单卡训练、冻结评测和自动 Gate 均已
完成。本批次状态为 `COMPLETED_GATE_REJECTED`：Non-thinking 回归门禁通过，两个 Seed 的
Thinking 格式率只有 94.5%和 93.5%，未达到 99%门槛。R1 没有解锁 M5.3。

R1 保留 M5.2 的 Base、Reasoning Dev、解码参数、1M Token 训练预算和 99% Thinking 格式
门禁。它只改变训练数据内部的 Thinking 样本构成，用于检验“增加短而完整的格式监督是否能
减少未闭合 Thinking 输出”这一单一假设。

## 2. 真实失败归因

分析器重新读取四个 30%/50% Candidate 的私有 `results.jsonl`，逐条校验 Result SHA256、
Prompt Hash、Response Hash、400 条 Item 身份和 Summary 可复算性，再只输出聚合计数。

| Thinking | Seed | 格式失败 | 达到长度上限且未闭合 | EOS 时未闭合 | Config | Python | Linux |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 30% | 42 | 9 | 8 | 1 | 6 | 2 | 1 |
| 30% | 20260727 | 6 | 6 | 0 | 4 | 2 | 0 |
| 50% | 42 | 8 | 8 | 0 | 5 | 3 | 0 |
| 50% | 20260727 | 15 | 13 | 2 | 11 | 2 | 2 |
| 合计 | — | 38 | 35 | 3 | 26 | 9 | 3 |

38 条失败全部包含一个 `<think>` 开标签但缺少 `</think>`；35 条在 896 个生成 Token 处停止，
另 3 条在 EOS 时仍未闭合。没有观察到缺少开标签、多个 Think 块、嵌套标签、空推理或空最终
答案等其他格式类型。失败中英文 22 条、中文 16 条。

该结果把 M5.2 报告中的“长度限制是首要调查方向”提升为逐条可复算结论。R1 不提高
`thinking_max_new_tokens`，因为改变解码上限会产生新的评测协议，无法与原 Base 和 Candidate
直接比较。

## 3. 预注册修正策略

R1 从训练侧通过 Verifier 的 96 条 Teacher Pilot 样本中构建 `short-complete-balanced-v1`
修复池，规则在训练前固定：

- 每个样本的移位后 Assistant 监督 Token 不超过 512；
- Config、JSON、Linux、Log Diagnosis、Python 各选择 8 条；
- 每类选择 5 条英文和 3 条中文，总计英文 25 条、中文 15 条；
- 每个分层按监督 Token 数、Sample ID 确定性排序，选择最短样本；
- 所有样本都保留完整 `<think>...</think>` 和最终 JSON，不截断推理轨迹；
- Pilot/Dev 污染门禁保持有效，Dev 内容不进入训练数据。

新的 1M Supervised Token 混合固定为：

| 分层 | Token | 占总预算 | 数据作用 |
| -- | --: | --: | -- |
| M2 Non-thinking | 700,000 | 70% | 保持 Non-thinking 能力 |
| 完整 Pilot Thinking | 150,000 | 15% | 保留任务与推理多样性 |
| 短格式修复 Thinking | 150,000 | 15% | 强化完整闭合和短回答先验 |

因此总体 Thinking 比例仍为 30%，修复样本占 Thinking 监督的一半。30%只是 M5.2 中表现最好
的诊断起点，尚未被标记为正式配比。

## 4. 真实数据 Artifact

| 项目 | 实际值 |
| -- | -- |
| Dataset Version | `m5-format-repair-mixture-v1-1396b60b` |
| Content SHA256 | `1396b60b7bb308c476466bcefa9de7c7813a15d9c1e244c5d0a57eb3472826b8` |
| Manifest SHA256 | `2467b5dce0d909b865b73219d2f608bdbc9c6fcc1bb09b93c5ebea8a7b60bd0e` |
| Payload SHA256 | `d14612b5dd06c117940537c5bc1306047e46a8c85ce6e0839b9a76eac1ef3623` |
| Payload Size | 38,251,570 Bytes |
| 总序列数 | 4,150 |
| Non-thinking 序列 | 3,063 |
| 普通 Thinking 序列 | 479 |
| Repair Thinking 序列 | 608 |
| Build Seed | `20260727` |

私有 Artifact 经过 `COMMITTED`、Manifest SHA256、Payload 大小与 SHA256、数组形状、三种
分层标记和精确 Token 计数的重新校验。公开证据不包含 Prompt、Sample ID、Thinking Trace、
用户名、主机名或绝对路径。

## 5. 冻结训练、评测与实际结果

两个训练配置只允许 Seed 不同：

- `configs/sft/m5_format_repair_r1_seed42.yaml`
- `configs/sft/m5_format_repair_r1_seed20260727.yaml`

训练参数保持单卡 BF16、Micro Batch 4、梯度累积 2、Learning Rate `2e-5`、50K Token
Warmup、每 500K Token 完整 Checkpoint 和精确恢复。每个 Run 完成 1M Supervised Tokens。

训练后继续使用 `m5-reasoning-dev-v1-53ddf557` 和原始 Base，门禁顺序保持：

1. 两个 Seed 的 Non-thinking 分数相对 Base 都不得回退超过 2pp；
2. 两个 Seed 的 Thinking 格式有效率都必须至少 99%。

两道门禁同时通过时，R1 才能解锁 M5.3；任一道失败都保留真实结果并进入新的设计审查，
不降低门槛，也不把 R1 标记为正式配比。

两组训练均达到精确 1M Supervised Tokens，并保存最终完整 Checkpoint 和独立模型导出：

| Seed | GPU | Optimizer Step | 初始/最终 Loss | 时长（秒） | Peak Reserved（Bytes） |
| --: | --: | --: | -- | --: | --: |
| 42 | 5 | 519 | 1.42064 / 1.47475 | 880.75 | 17,773,363,200 |
| 20260727 | 6 | 519 | 1.87054 / 0.86651 | 904.29 | 17,773,363,200 |

Loss 是首个和末个优化器组的观测值，不单独作为质量结论。冻结双模式评测结果为：

| Seed | Non-thinking 分数 | Thinking 格式率 | Thinking 分数 | Length-limited | 评测时长（秒） |
| --: | --: | --: | --: | --: | --: |
| 42 | 64.0% | 94.5% | 93.0% | 11 | 699.14 |
| 20260727 | 66.0% | 93.5% | 93.0% | 13 | 671.32 |

Base Non-thinking 分数为 37.0%，因此两个 Seed 都通过“不回退超过 2pp”的门禁。Thinking
格式率均低于 99%，自动 Gate 返回：

```text
status: rejected
gate_reason: thinking_format_gate_failed
exit code: 6
```

R1 的 24 条格式失败全部在 896 Token 上限处留下未闭合 `<think>`。按任务聚合：Config
19 条、Log Diagnosis 4 条、Python 1 条；英文 12 条、中文 12 条。与原 30%消融相比，Seed 42
格式率从 95.5%降至 94.5%，Seed 20260727 从 97.0%降至 93.5%。因此“重复训练侧最短完整
样本即可提高闭合率”的 R1 假设没有得到实验支持。

## 6. CPU 验收结果

`make check` 实际通过：

- Ruff 与 Ruff Format；
- MyPy：220 个 Source File 无类型错误；
- Pytest：521 Passed、2 Deselected；
- CPU 可测核心覆盖率：85.15%；
- JSON Schema Snapshot；
- Markdown Link Check：75 个文件；
- Public Artifact 脱敏检查。

上述结果验证代码、数据契约和失败路径；模型质量结论来自本报告第 5 节的真实 GPU 结果。

## 7. 已覆盖的失败路径

自动测试覆盖：

- 私有 Raw Result SHA256、Prompt Hash、Response Hash 或 Item 身份漂移；
- 四个分析输入缺失、重复、Seed/配比错误或评测协议不一致；
- 修复池任一任务/语言分层不足，或样本超过 512 监督 Token；
- Manifest、Commit Marker、Payload SHA256、数组分层或精确 Token 计数漂移；
- R1 配置将 Thinking 比例改为 0%/50%；
- 两个训练 Run 使用不同数据、错误 Seed、错误配比或未成功完成；
- 任一 Seed 低于 99%格式门禁时拒绝 R1。

## 8. 证据索引

- 失败聚合：[m5_format_failure_analysis.json](raw/m5_format_failure_analysis.json)
- 数据 Manifest：[m5_format_repair_mixture.json](raw/m5_format_repair_mixture.json)
- R1 Gate：[m5_format_repair_gate.json](raw/m5_format_repair_gate.json)
- M5.2 原始选优结论：[m5_ablation_selection.json](raw/m5_ablation_selection.json)
- 设计契约：[M5 SFT 契约](../../docs/m5_sft_contract.md)
- Candidate 原始响应、训练 Run、Checkpoint 和导出：私有 Artifact Store
