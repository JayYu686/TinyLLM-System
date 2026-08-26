# M10.5 Agent 能力修复就绪报告

## 当前结论

M10.5 已完成首轮失败归因、评分协议 v2、DevOps 训练源 v2、80 条维护者内容审查、修复混合
冻结和 8B LoRA 修复配置。现阶段没有启动 GPU 训练；数据与配置前置门禁已经全部满足，下一项
工作是执行 10-Step BF16 Memory Probe。

## 已完成证据

| 项目 | 实际结果 | 状态 |
| -- | -- | -- |
| DevOps v2 数据 | `m10-devops-training-v2-8461493c`，2,400 条 | 已构建 |
| 语言分布 | 英文 1,680 / 中文 720 | 通过 |
| 精确重复 | 0 对 | 通过 |
| 跨组近重复 | 0 对 | 通过 |
| 四边界污染 | M9 Dev / Release、BFCL、M6 均为 0 | 通过 |
| 工具事实 Grounding | 1,440 条 | 通过 |
| 单调用透明恢复 | 360 条 | 通过 |
| 缺参问句 | 240 条 | 通过 |
| 唯一最终回答 | 2,208 / 2,400 | 通过 |
| 最大最终回答重复 | 8 次 | 通过 |
| 通用模板命中 | 0 条 | 通过 |
| 维护者内容审查 | 80/80 条通过 | 已批准 |
| DevOps Approved Manifest | `6042bbbe...0d6e` | 已冻结 |
| 修复混合 | `m10-agent-sft-v2-50ffc51f` | 已冻结 |
| 混合 Token 配额 | 1,000,000；英文/中文 70/30；双模式 94/6 | 通过 |
| 混合污染检查 | M9 Dev / Release、BFCL、M6 均为 0 | 通过 |
| 8B Base v2 重评分 | 47.50% Task Success | 已冻结 |
| 训练门禁阈值 | 未降低 | 已冻结 |

父模型重评分只复用了历史规范化 Tool Call、Tool Result 状态和 Final Answer，没有再次生成模型
输出。公开摘要见 [`m10_repair_parent_rescore.json`](raw/m10_repair_parent_rescore.json)。

## 冻结训练输入

维护者已确认私有 Review Packet 的 80 条分层轨迹。审批结果、批准后的 Manifest 和冻结混合保存在
私有 Artifact Store；公开仓库只保存不含样本内容的摘要：

```text
$TINYLLM_ARTIFACT_ROOT/reviews/m10-devops-training-v2-8461493c/approval-v2/
$TINYLLM_ARTIFACT_ROOT/datasets/m10-agent/frozen/m10-agent-sft-v2-50ffc51f/
```

冻结混合 Manifest SHA256 为
`9015f1fb41d9bbe3ba1e921d5568a4df3801a61dcd84f1bdc336fb1aefd12415`，训练配置为
[`m10_5_agent_repair_lora_qwen3_8b.yaml`](../../configs/sft/m10_5_agent_repair_lora_qwen3_8b.yaml)。

## 下一次真实实验

按以下顺序执行：

1. M10.5 8B LoRA 10-Step BF16 Memory Probe；
2. 从固定 8B Base 重新训练到 1M Supervised Tokens；
3. 用 `m10-agent-scoring-v2` 执行完整 80 条 Agent Dev；
4. 候选达到至少 48.50% 才允许进入后续正式门禁；
5. 达到开发目标后才首次消费密封 Release，并继续 BFCL、M6 与 Serving 验证。

完整约束见 [`M10.5 Agent 能力修复契约`](../../docs/m10_5_agent_repair_contract.md)。
