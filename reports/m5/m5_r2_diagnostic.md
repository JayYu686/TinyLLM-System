# M5.2-R2 Thinking 长度反事实诊断报告

## 1. 当前状态

M5.2-R2 的设计契约、严格 Schema、D1 离线分析、D2 双 Seed 确定性 GPU 重放、失败路径、
机器判定和文档均已完成。当前状态为 `COMPLETED_DIAGNOSTIC_REJECTED`：长度增加能够恢复
部分闭合和正确答案，但两个 Seed 在 1536 Token 仍分别只有 98.0%和 96.5%格式率，未达到
99%门槛。现有正式评测协议保持不变，M5.3 继续阻塞。

R2 不训练模型，不改变 M5.2/R1 的拒绝结果，也不降低 99% Thinking 格式门禁。它只检验：

1. 原 896 Token 输出能否逐条精确重放；
2. 同 RNG 的 1536 Token 输出前缀是否保持一致；
3. 原失败项在 1024、1280、1536 三个截断点能恢复多少完整格式和正确答案。

## 2. D1 真实离线分析

D1 重新读取两个 R1 私有 `results.jsonl`，校验 Raw SHA256、Prompt Hash、Response Hash、
400 条身份和 Summary 可复算性，再使用固定 Qwen3-0.6B Tokenizer 计算内容脱敏的长度与
重复度指标。公开结果不包含 Response、Prompt、Item ID、Thinking Trace 或绝对路径。

| Seed | 失败数 | 任务族 | 失败 Unique Token Ratio | 失败 Repeated 8-gram Ratio | 对照 Repeated 8-gram Ratio | 最大相同行重复 |
| --: | --: | -- | --: | --: | --: | --: |
| 42 | 11 | Config 9 / Log 1 / Python 1 | 19.47% | 25.82% | 1.28% | 4 |
| 20260727 | 13 | Config 10 / Log 3 | 19.68% | 24.95% | 1.24% | 10 |

失败项全部在 896 Token 处停止且缺少 `</think>`。同任务族的格式有效对照中：

| Seed | 对照数 | Token P50 | Token P90 | Token Max | Unique Token Ratio |
| --: | --: | --: | --: | --: | --: |
| 42 | 109 | 267 | 481 | 852 | 43.94% |
| 20260727 | 67 | 257 | 575 | 845 | 49.25% |

这些结果表明失败项同时具有“触顶”和“重复度显著高于同任务族有效输出”两个特征。该证据
支持继续执行长度反事实诊断，但不能单独证明提高长度上限即可解决问题：更长预算也可能只
延长重复生成。D2 必须同时检查闭合、最终 JSON 和答案正确性。

Repeated 8-gram Ratio 定义为单条输出中重复 8-gram 出现次数占全部 8-gram 窗口的比例，
再对样本取均值；Unique Token Ratio 为不同 Token 数占该输出 Token 数的比例，再对样本
取均值。它们用于比较分布，不设置独立质量门槛。

## 3. D2 冻结协议

每个 Seed 只重放包含原失败项的完整四样本 Batch。每个 Batch 先以 896 Token 重放，再重置
相同 RNG 并以 1536 Token 重放。

运行有效性要求：

- 896 重放的 Response、Token 数、Finish Reason 和全部评分与原结果完全相同；
- 1536 重放的前 896 个生成 Token ID 与 896 重放逐个相同；
- 任一 Batch 不一致时返回 `INVALID_REPLAY`，停止质量归因；
- 禁止运行时补写闭标签或最终 JSON。

两个 Seed 均在提交 `02fed7d5007deaa4a94c40fdb84b786ee892f13b` 上使用 RTX 3090
完成真实重放：

| Seed | GPU | 重放 Batch/Item | 896 精确一致 | 1536 前缀一致 | 时长（秒） | Peak Reserved（Bytes） |
| --: | --: | --: | --: | --: | --: | --: |
| 42 | 7 | 10 / 40 | 40 / 40 | 40 / 40 | 610.47 | 2,613,051,392 |
| 20260727 | 7 | 9 / 36 | 36 / 36 | 36 / 36 | 563.83 | 2,613,051,392 |

全部一致性校验通过，排除本轮环境、Batch 顺序或 RNG 漂移导致原失败结果不可重放的解释。
截断评分结果为：

| 上限 | Seed 42 投影格式率 | Seed 20260727 投影格式率 | 两 Seed 是否达到 99% | 结论 |
| --: | --: | --: | -- | -- |
| 1024 | 95.0% | 95.5% | 否 | 长度不足 |
| 1280 | 97.0% | 96.0% | 否 | 不支持条件协议升级 |
| 1536 | 98.0% | 96.5% | 否 | 长度假设不足 |

对应的 Final-answer 投影分数：

| 上限 | Seed 42 | Seed 20260727 |
| --: | --: | --: |
| 1024 | 93.5% | 94.0% |
| 1280 | 95.0% | 94.0% |
| 1536 | 95.5% | 94.0% |

在 1536 Token 处仍有 11 条未恢复完整格式：Config 8 条、Log Diagnosis 2 条、Python
1 条；英文 4 条、中文 7 条。另有 6 条虽然恢复完整格式和 Final JSON，但最终答案仍错误。
因此，更长的生成预算只能部分缓解闭合问题，无法单独提供可靠的 Thinking 双模式。

## 4. 正式决策

冻结选择器返回：

```text
status: length_ceiling_insufficient
selected_max_new_tokens: null
formal_protocol_changed: false
decision_reason: at_least_one_seed_fails_at_diagnostic_limit
exit code: 6
```

此前“若两个 Seed 在 1280 都达到 99%则允许建立新协议”的条件没有满足。因此：

- 不把正式 Thinking 上限从 896 改为 1280 或 1536；
- 不重跑 Base 和旧 Candidate；
- 保留 M5.2 与 R1 的 Gate 拒绝状态；
- 下一批训练修正仅面向 Config/Log 的高重复与过长推理；
- 新 Teacher Trace 必须简洁、互不重复，并在训练前冻结任务族比例、长度分布和污染检查。

## 5. 已完成验收

- 严格 YAML、私有 Item、公开 Seed Summary 和双 Seed Decision Schema；
- 原评测、训练 Run、模型导出、Tokenizer 和 Git Worktree 血缘检查；
- Batch Offset 与 Seed 计算；
- 896 Response/评分漂移失败路径；
- 1536 前缀漂移失败路径；
- 1024/1280/1536 原 Parser 截断评分；
- 私有 Raw 与公开聚合分离；
- D1 真实离线运行与公开脱敏扫描；
- D2 双 Seed RTX 3090 真实重放；
- 机器判定以退出码 6 拒绝长度假设；
- CPU Fixture、MyPy Strict、Ruff 和 JSON Schema Snapshot。

## 6. 证据索引

- D1 机器结果：[m5_r2_offline_analysis.json](raw/m5_r2_offline_analysis.json)
- D2 机器判定：[m5_r2_length_diagnostic.json](raw/m5_r2_length_diagnostic.json)
- R2 配置：[m5_r2_length_replay.yaml](../../configs/eval/m5_r2_length_replay.yaml)
- 诊断设计：[m5_r2_diagnostic_design.md](../../docs/m5_r2_diagnostic_design.md)
- R1 报告：[m5_format_repair_r1.md](m5_format_repair_r1.md)
- D2 私有逐条结果和两个 Seed Summary：私有 Artifact Store
