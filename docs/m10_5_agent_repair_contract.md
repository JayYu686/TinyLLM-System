# M10.5 Agent 能力修复契约

## 1. 目标

M10.5 使用独立 Dataset Revision、评分协议和训练 Run 完成 M10 的能力修订。实验从固定
Qwen3-8B Base 重新初始化，不修改既有 Run、Checkpoint、评测结果或发布标签。

修订完成条件仍由 M9 预注册门禁决定：开发集只负责早停，密封 Release、BFCL、M6 回归和
M7 Serving 只在开发门禁通过后执行，全部阈值沿用预注册契约。

## 2. 首轮诊断与修订目标

首轮数据在结构上合法，但存在三个与运行时不一致的监督信号：

1. 大量最终回答是高重复的通用模板，工具结果中的真实状态、数值、路径和故障码没有充分进入
   监督目标；
2. 失败恢复样本把运行时内部的透明重试写成了两次模型工具调用，导致训练轨迹与 Agent Runtime
   的一次调用、多次执行尝试语义不一致；
3. 等待审批任务要求模型提出写操作并停止，但 v1 评分同时检查不应出现的最终回答文本，且把
   `not_executed` 的审批前工具调用作为普通成功调用处理。

对应修订分别落在数据 v2、评分协议 v2 和独立训练 Campaign 中。历史 v1 证据继续使用
`m9-agent-scoring-v1` 解析，不能被新逻辑静默改写。

## 3. 评分协议 v2

`m10-agent-scoring-v2` 只修改能够由冻结任务契约证明的评分语义：

- 数字参数使用精确十进制语义比较，例如字符串 `1e-5` 与数值 `0.00001` 等价；
- `not_executed` 仅在“需要审批且终态为 waiting_approval”的任务中可接受；
- 已匹配 `requires_final_answer=false` 轨迹时不再要求最终回答内容；
- 父模型和候选模型必须使用同一协议，Gate 拒绝跨协议比较。

8B Base 的既有 80 条 Agent Dev 原始输出经只读重评分后，Task Success 从 45.00% 修正为
47.50%。该操作没有重新生成模型输出，来源摘要、逐项结果和重评分结果均有独立 SHA256。

## 4. DevOps 训练源 v2

修复源仍包含 2,400 条中英文轨迹，但提高最薄弱能力的覆盖，并强制最终回答绑定真实工具事实。

| 类别 | 条数 |
| -- | --: |
| Single Tool | 360 |
| No-tool | 240 |
| Wrong-tool / Irrelevance | 480 |
| Missing Argument / Clarification | 240 |
| Sequential Multi-step | 480 |
| Parallel Independent Tools | 120 |
| Tool Failure Recovery | 360 |
| Grounding / Approval / Security | 120 |

英文/中文保持 70/30。机器质量门禁要求：

- 所有有工具最终回答必须包含工具返回事实和实际 Call Evidence；
- 失败恢复必须是一次模型 Tool Call，Tool Result 记录 `attempts=2`；
- 缺参任务必须输出明确问题；
- 禁止命中登记的通用回答模板；
- 精确重复和跨组近重复必须为零；
- 与 M9 Dev、密封 Release、BFCL Core 和 M6 领域集污染必须为零；
- 80 条分层内容审查获维护者确认前，数据保持 `review_pending`，不得进入正式训练。

## 5. 修复混合与训练策略

修复混合仍为每逻辑 Epoch 精确 1M Supervised Tokens、序列长度 2048、英文/中文 70/30、
Non-thinking/Thinking 94/6。来源比例调整为：

```text
20% ToolACE
10% Hermes Function Calling
40% DevOps v2
20% M6 Domain Replay
10% M2 No-tool Replay
```

训练从 `qwen3-8b-m9-base-90587dd6` 重新开始，保持 BF16 LoRA Rank 16、Alpha 32、Dropout
0.05、Attention/MLP Linear、Sequence Length 2048、Micro Batch 1、Gradient Accumulation 8
和 Gradient Checkpointing。学习率从首轮 `2e-4` 降为 `5e-5`，以降低对 8B Base 已有
Tool Calling 能力的破坏。首个正式检查点仍为 1M Supervised Tokens，确保与首轮成本和证据可比。

## 6. 阶段门禁

开发阶段使用评分协议 v2：

- 固定父模型 Task Success：47.50%；
- 1M 候选至少达到 48.50%，才允许继续；
- Schema、No-tool、工具幻觉、恢复、Grounding 与安全指标必须同时报告；
- 开发集达到续训标准后才消费密封 Release。

最终模型门禁保持不变：Release Task Success 不低于 70%，相对父模型至少提升 5pp 且配对
Cluster Bootstrap 95% CI 下界大于 0，Schema Valid Rate 不低于 98%，No-tool Accuracy
不低于 90%，Tool Hallucination Rate 不高于 2%，Grounding 与 Error Recovery 分别不低于
90% 和 70%，三类安全违规计数必须为零，并通过 BFCL、M6 与 Serving 回归。

## 7. 证据与失败处理

Dataset v2、训练配置、Memory Probe、Run、Checkpoint、Adapter Export、父模型重评分、候选 Dev、
Release、BFCL、M6 与 Serving 结果均使用独立内容哈希。最终模型在全部门禁满足后写入新的
Agent Production 记录，M6/M7 历史身份保持不变。

## 8. Repair v4：唯一轨迹扩容与阶段停止规则

Repair v3 的 1M、3M、4M 和 5M Checkpoint 使用同一 Agent Dev 与评分协议 v3 进行对照。
1M 的 Task Success 为 63.75%、Grounding 为 89.13%；3M、4M、5M 的 Task Success 分别为
52.50%、50.00%、56.25%，Grounding 分别为 47.83%、58.70%、76.09%。因此现有 Run 在
5M 停止，不继续到 10M，也不将降低工具幻觉率单独视为晋级依据。

Repair v4 将 DevOps 训练源扩展到 9,600 条唯一上下文，类别比例保持与修复目标一致，英文/中文
保持 70/30。训练 System Policy 与 M9/M10 Agent Runtime 的工具权限、证据引用、审批、安全边界
和原始 CoT 隐藏规则对齐。机器质量门禁要求至少 8,600 个唯一最终回答、单一最终回答频次不超过
32、精确重复和跨组近重复为零，并继续要求与 Agent Dev、密封 Release、BFCL Core 和 M6 数据
污染为零。

Repair v4 的正式混合必须证明所有来源样本复用计数为零。80 条分层样本已由维护者全部确认，
`approval-v4` 与 `m10-agent-sft-v3-7aa779bf` 冻结混合已经生成；10 个 Stratum 的样本复用计数
均为零。新实验重新从 Qwen3-8B Base 开始，在 1M Token 处先做 Agent Dev 门禁，避免再次用
重复 Epoch 掩盖能力退化。

## 9. 最终状态

最终路线在 160 条密封 Release 上达到 93.12% Task Success，相对父模型提升 18.74pp，
Cluster Bootstrap 95% CI 为 `[+3.47, +36.07]pp`；BFCL 总体 +0.11pp，M6 聚合回归 0pp，
13/13 统一门禁通过。模型已注册为
`qwen3-8b-m10-agent-production-b2d88493`，完整结论见
[`M10 总验收`](../reports/m10/m10_acceptance.md)。
