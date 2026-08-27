# M10.5 评分协议迁移与 Repair v3 就绪报告

## 结论

Qwen3-8B Agent LoRA 的 1M Token 阶段在 `m10-agent-scoring-v3` 下通过继续修复门禁：
父模型 Task Success 为 48.75%，候选为 62.50%，绝对提升 13.75 个百分点。该结论授权继续
修复实验，不代表模型已通过 M10 最终 Release Gate。

旧 v2 训练在 1M Token 时的最终 Loss 为 0.0072，继续使用同一数据和学习率训练存在明显过拟合
风险。因此后续不沿用旧配置机械续训至 5M，而是使用独立的 Repair v3 数据源和更低学习率从
Qwen3-8B Base 重新开始 1M 阶段。

## 同协议对照

| 指标 | Qwen3-8B Base | 1M LoRA | 变化 |
|---|---:|---:|---:|
| Task Success | 48.75% | 62.50% | +13.75pp |
| Tool Selection | 81.25% | 92.50% | +11.25pp |
| Argument Accuracy | 75.00% | 88.75% | +13.75pp |
| Schema Valid | 100.00% | 98.75% | -1.25pp |
| No-tool Accuracy | 66.67% | 93.33% | +26.66pp |
| Multi-step Success | 90.00% | 93.33% | +3.33pp |
| Error Recovery | 100.00% | 100.00% | 0pp |
| Grounding Accuracy | 97.83% | 95.65% | -2.18pp |
| Tool Hallucination | 18.75% | 3.75% | -15.00pp |

候选的路径逃逸、未审批写入和任意命令执行均为 0。门禁事实源见
[`raw/m10_5_lora_1m_scoring_v3_gate.json`](raw/m10_5_lora_1m_scoring_v3_gate.json)。

## 评分迁移原因

`m10-agent-scoring-v3` 修正了早期评分契约中未向模型披露的可选默认参数、过度依赖字面表达的
澄清判定，以及恢复任务中的隐藏标签要求。评分迁移没有修改历史训练 Run、Checkpoint、Adapter
或原始输出；父模型和候选模型均在同一 Git 提交、同一 Dev Suite、同一运行时和同一评分协议下
重新评测。

## 剩余缺口

候选距离 70% Task Success 最低要求还差 6 条成功任务；Tool Hallucination 为 3.75%，尚未达到
不高于 2% 的要求。工具规划本身已经基本正确，主要失败集中在多步任务最终答案遗漏用户明确询问
的服务或组件名称。

## Repair v3 数据

新数据版本为 `m10-devops-training-v3-a5645bc5`，保持 2,400 条和英文 70% / 中文 30%。新增
质量约束包括：

- 480 条顺序双步轨迹必须包含两个有序调用；
- 120 条并行轨迹必须在同一模型决策中包含两个不同调用；
- 360 条恢复轨迹由运行时透明重试，模型仅发出一个逻辑调用；
- 960 条多步与恢复轨迹的最终答案必须保留用户询问的实体；
- 所有需要工具的最终答案必须引用实际 Tool Result；
- Dev、隐藏 Release、BFCL Core 和 M6 Domain 四边界污染扫描必须为 0。

机器检查已经通过；新版本仍等待维护者完成 80 条分层内容审查，在审批前保持
`training_permitted=false`。
