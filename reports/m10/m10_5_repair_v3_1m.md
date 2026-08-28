# M10.5 Repair v3 LoRA 1M 阶段报告

## 结论

Repair v3 已在单张 RTX 3090 上完成 Qwen3-8B BF16 LoRA 的 1,000,000 Supervised Token
训练，并通过 1M→5M Continuation Gate。候选在冻结的 80 条 Agent Dev 上取得 63.75%
Task Success，相对同协议父模型的 48.75% 提升 15 个百分点，超过继续训练所需的 1 个百分点。

该结论只授权 Exact Resume 到 5M 阶段。候选尚未达到 M10 最终 Agent Model Gate，不能标记为
Production Agent Model。

## 训练与血缘

| 项目 | 实际结果 |
|---|---|
| Run ID | `20260827T070839Z-m10-5-agent-repair-v3-lora-qwen3-8b-seed42-d451d02f-3fc2` |
| Git Commit | `5ccf94aae736327357213412bd768684d5930b8d` |
| 数据版本 | `m10-agent-sft-v2-435b9fbc` |
| 数据 Manifest | `e7b7943e3dddee9cad3403e22e26de4c65c92d063efa15b2d4c168d29afe21d2` |
| Optimizer Step | 999 |
| Supervised Token | 1,000,000 |
| 墙钟时间 | 23,736.65 秒（6 小时 35 分 36.65 秒） |
| 初始 / 最终 Loss | 1.3207 / 0.1456 |
| Peak Allocated / Reserved | 20.58 / 22.55 GiB |
| Checkpoint | `checkpoint-tokens-0001000000` |
| Adapter SHA256 | `b8cad495502a2263c0e8c270d2c1c954317f18f08457dcf46841cfb55747ddec` |
| Evaluation Subject | `qwen3-8b-m10-agent-lora-1m-0ad5befe` |

训练未出现 OOM、NaN/Inf 或 Checkpoint 完整性失败。相比 Repair v2 在 1M 时的最终 Loss
0.0072，Repair v3 的低学习率和新版混合把过拟合风险控制在更合理的范围内。

## Agent Dev 结果

父模型和候选均使用 `m10-agent-scoring-v3`、同一 80 条 Dev、Non-thinking 模式和冻结工具协议。

| 指标 | 父模型 | Repair v3 1M | 最终门禁要求 |
|---|---:|---:|---:|
| Task Success | 48.75% | 63.75% | ≥ 70%（Release） |
| Tool Selection | 81.25% | 83.75% | 记录项 |
| Argument Accuracy | 75.00% | 76.25% | 记录项 |
| Schema Valid | 100.00% | 100.00% | ≥ 98% |
| No-tool Accuracy | 66.67% | 100.00% | ≥ 90% |
| Multi-step Success | 90.00% | 56.67% | 记录项 |
| Error Recovery | 100.00% | 100.00% | ≥ 70% |
| Grounding Accuracy | 97.83% | 89.13% | ≥ 90% |
| Tool Hallucination | 18.75% | 10.00% | ≤ 2% |

未审批写操作、路径逃逸和任意命令执行均为 0，Approval Safety 为 100%。

## 继续训练判断

Continuation Gate 的事实结果为：

```text
parent Task Success      48.75%
candidate Task Success   63.75%
absolute improvement    +15.00pp
decision                 accepted
next stage               5M supervised tokens
```

距离 70% 的 Dev 参考线还差 5/80 条成功任务。更严格的最终门禁还要求在隐藏 Release、BFCL、
M6 回归和 Serving Gate 上同时通过，因此 5M 训练完成后必须重新注册不可变评测对象并运行完整对照。

脱敏事实源见
[`m10_5_repair_v3_1m_gate.json`](raw/m10_5_repair_v3_1m_gate.json)。
