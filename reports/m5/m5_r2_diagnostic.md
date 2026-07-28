# M5.2-R2 Thinking 长度反事实诊断报告

## 1. 当前状态

M5.2-R2 的设计契约、严格 Schema、D1 离线分析、D2 确定性重放实现、失败路径和 CPU
Fixture 已完成。D2 真实 GPU 重放尚未执行，因此当前状态为
`IMPLEMENTED_AWAITING_GPU_REPLAY`，尚不能判断 1280 Token 是否足以支持新评测协议。

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

有效重放后，机器结果将在下表记录：

| 上限 | Seed 42 投影格式率 | Seed 20260727 投影格式率 | 两 Seed 是否达到 99% | 结论 |
| --: | --: | --: | -- | -- |
| 1024 | 待运行 | 待运行 | 待运行 | 待运行 |
| 1280 | 待运行 | 待运行 | 待运行 | 待运行 |
| 1536 | 待运行 | 待运行 | 待运行 | 待运行 |

## 4. 条件决策

- 两个 Seed 在 1024 或 1280 的同一最小上限都达到 99%：
  `supports_eval_protocol_revision`。允许建立新 Evaluation Protocol，但尚未改变正式协议。
- 只有 1536 同时达标：`tradeoff_review_required`。先评估生成 Token、评测时长和推理延迟。
- 任一 Seed 在 1536 仍不达标：`length_ceiling_insufficient`。保留 R1 拒绝，后续训练设计
  转向 Config/Log 的新颖简洁 Teacher Trace。
- 896 或前缀无法精确重放：`INVALID_REPLAY`。优先排查软件环境、Batch 顺序和 RNG。

用户已条件确认：若两个 Seed 在 1280 都达到 99%，可以在新版本中把 Thinking 评测上限
改为 1280；生效前必须完整重跑 Base、六个 M5.2 Candidate 和两个 R1 Candidate，并完成
性能成本评估。旧结果继续保留。

## 5. 已完成验收

- 严格 YAML、私有 Item、公开 Seed Summary 和双 Seed Decision Schema；
- 原评测、训练 Run、模型导出、Tokenizer 和 Git Worktree 血缘检查；
- Batch Offset 与 Seed 计算；
- 896 Response/评分漂移失败路径；
- 1536 前缀漂移失败路径；
- 1024/1280/1536 原 Parser 截断评分；
- 私有 Raw 与公开聚合分离；
- D1 真实离线运行与公开脱敏扫描；
- CPU Fixture、MyPy Strict、Ruff 和 JSON Schema Snapshot。

## 6. 证据索引

- D1 机器结果：[m5_r2_offline_analysis.json](raw/m5_r2_offline_analysis.json)
- R2 配置：[m5_r2_length_replay.yaml](../../configs/eval/m5_r2_length_replay.yaml)
- 诊断设计：[m5_r2_diagnostic_design.md](../../docs/m5_r2_diagnostic_design.md)
- R1 报告：[m5_format_repair_r1.md](m5_format_repair_r1.md)
- D2 私有逐条结果：GPU 运行后保存到私有 Artifact Store
- D2 公共双 Seed 结论：GPU 运行后写入 `reports/m5/raw/m5_r2_length_diagnostic.json`
