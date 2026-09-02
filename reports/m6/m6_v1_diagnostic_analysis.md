# M6 v1 Candidate 诊断分析

## 结论

M6 v1 评测结果经过独立复核，评分实现没有发现误判。首个 0.6B Candidate 在领域增量、
Bootstrap 置信区间、双模式 JSON Valid Rate 和 Thinking 强制收束率上均暴露出改进空间。
主要根因是 M5 双模式
训练模板与 Qwen3 官方推理模板不对齐，并叠加了训练来源规模小、模板重复率高的问题。

当前 Candidate 保持 `Development`。现有 M6 v1 证据全部保留，门禁阈值不调整。
公开的内容无关原始汇总见
[m6_v1_diagnostic_summary.json](raw/m6_v1_diagnostic_summary.json)。

## 1. 真实结果

| 指标 | Base | Candidate | 门禁 | 结论 |
| -- | --: | --: | --: | -- |
| Thinking 领域正确 | 80/300（26.67%） | 28/300（9.33%） | 相对提升至少 3pp | 需改进 |
| Non-thinking 领域正确 | 16/300（5.33%） | 18/300（6.00%） | 相对提升至少 3pp | 需改进 |
| Thinking JSON Valid | 52/80（65.00%） | 57/80（71.25%） | 至少 98% | 需改进 |
| Non-thinking JSON Valid | 32/80（40.00%） | 45/80（56.25%） | 至少 98% | 需改进 |
| Thinking 自然闭合 | 292/300（97.33%） | 1/300（0.33%） | 强制收束不超过 10% | 需改进 |
| 通用任务等权 `acc_norm` | 51.80% | 51.15% | 回退不超过 2pp | 通过 |

Candidate 的 Thinking 首段平均只有约 58 Token，中位数 17 Token；大多数失败样本是直接回答
后 EOS，而不是达到 1,536 Token 上限。增加生成长度无法解决该失败。

## 2. 根因证据

固定 Revision Tokenizer 的实际 Generation Prompt 为：

```text
Thinking:
<|im_start|>assistant\n

Non-thinking:
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

旧 M5 Non-thinking 训练序列却是：

```text
<|im_start|>assistant\n答案<|im_end|>
```

Thinking 训练序列是：

```text
<|im_start|>assistant\n<think>\n推理\n</think>\n\n答案<|im_end|>
```

两类样本在完全相同的 Assistant Header 后分别监督“答案首 Token”和 `<think>`，没有独立的
模式上下文。训练数据中 General Thinking 只有 96 个来源、Repair Thinking 只有 40 个来源，
1M Mixture 已存在大量复用，正式 10M Snapshot 又重复约十轮。因此 M5 Dev 的 95% Thinking
分数反映了同模板分布内学习，未迁移到 M6 的七类独立任务。

标签掩码复核没有发现 User/System Token 被训练；问题位于模式上下文，而非 Assistant-only
Loss 边界。M6 控制器和解析器也按冻结协议正确记录了自然闭合与强制收束。

## 3. 修复边界

修复采用新的 `qwen3-chatml-nonthinking-sft-v2`：在 Non-thinking Assistant Header 后加入空
Think 块作为已 Mask 的输入上下文，只监督最终答案。Thinking 继续从 Header 后监督 `<think>`。

第一批修复数据固定为 1M Supervised Tokens：

- 640K：经 v2 模板对齐的 M2 General Non-thinking；
- 60K：M5 R3 领域来源的 Non-thinking Pair，并限制每个来源最多使用 27 次；
- 300K：同一批 M5 R3 来源的 Thinking；
- Thinking 比例仍为 30%，模型 Revision、GQA、优化器和门禁阈值不变；
- 来源均冻结于 M6 运行前，且 `consume_m6_frozen_results=false`。

先执行 CPU 内容身份与标签审计，再进行双 Seed 1M Token 训练。M6 v1 已被用于诊断，最终晋级
必须使用新的独立 M6 v2 内容身份，不能在旧测试集上反复调参后宣称通过。

## 4. 当前状态

- M6 v1：`COMPLETED_GATE_REJECTED`；
- 旧 Candidate：`Development`；
- 模板 v2 与防回归测试：已实现；
- 修复 Mixture：`m5-dual-mode-correction-mixture-v1-4bc342d4`，Manifest SHA256
  `db66ce847fac4bd2966666d125f1bb4e21dd0fd3bb608a1a384806c206f8945c`；对齐后重新
  Packing 为 2,562 条序列，避免将短样本逐条 Pad 到 1,024 所产生的无效计算；
- Token 审计：1,000,000 个监督 Token，10,557 个 Non-thinking 监督起点全部具有被 Mask 的
  v2 模式上下文，2,185 条 Thinking 序列全部从 `<think>` 开始，错误计数均为 0；
- Seed 42 Proxy：已完成；Thinking 93.5%、Non-thinking 79.5%，Non-thinking JSON
  完整率 100%，可见推理泄漏为 0；
- Seed 20260810 Proxy：已完成；Thinking 91.0%、Non-thinking 80.5%，Non-thinking JSON
  完整率 100%，可见推理泄漏为 0；
- M6 v2 正式晋级：尚未执行。

后续正式执行的 Candidate 选择、独立 Suite 和不变门禁已在
[M6 v2 正式执行预注册](m6_v2_execution_plan.md)中固定。
