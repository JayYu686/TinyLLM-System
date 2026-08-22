# M10.1 Agent 训练混合验收报告

## 结论

M10.1 数据冻结与污染检查已通过，最终私有数据版本为
`m10-agent-sft-v1-4655d3e3`。该版本允许进入 M10.2 训练，但不代表任何 Agent Candidate
已经通过 Release、BFCL、M6 或 Serving 门禁。

## 实际产物

| 项目 | 实际结果 |
| -- | --: |
| 固定长度序列 | 8,061 |
| 监督 Token | 1,000,000 |
| 序列长度 | 2,048 |
| 英文 / 中文 | 700,000 / 300,000 |
| Non-thinking / Thinking | 940,000 / 60,000 |
| 私有数组大小 | 148,606,057 bytes |
| 数据内容 SHA256 | `4655d3e35ffb1d46e119b34a6bcdcaaec0e79505b47b07a310058ee8c37b8693` |
| Manifest SHA256 | `6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490` |

来源监督 Token 严格按预注册配比构建：

| 来源 | 监督 Token | 比例 |
| -- | --: | --: |
| ToolACE | 300,000 | 30% |
| Hermes Function Calling | 200,000 | 20% |
| TinyLLM DevOps | 200,000 | 20% |
| M6 Domain Replay | 200,000 | 20% |
| M2 No-tool Replay | 100,000 | 10% |

## 过滤、去重与复用

- ToolACE：10,770 条 Canonical 输入中，3 条完整会话 Exact 重复被移除，81 条超过
  2,048 Token；10,686 条进入可选池。
- Hermes：1,822 条 Canonical 输入中，1,094 条超过 2,048 Token；728 条进入可选池。
- TinyLLM DevOps：2,400 条已批准轨迹全部通过长度检查。
- M6 Replay：3,368 条序列全部通过，并保留原始 Thinking/Non-thinking 标记。
- M2 Replay：4,597 条 Train 样本中，3 条在 Non-thinking 模式对齐后超过既有边界，4,594
  条进入可选池。
- Near Dedup 没有删除记录。Hermes 英文和 DevOps 中文分层使用确定性多轮采样达到精确
  Token 目标；没有合成或改写新样本。

Exact Dedup 比较规范化后的工具定义、完整消息、工具调用、工具结果和最终回答。Near Dedup
要求 Prompt 与 Tool Schema 的 5-gram 相似度同时达到 0.85，公共 MCP Tool Catalog 本身不
构成重复。

## 污染检查

| 评测边界 | Exact | Near | 最高候选相似度 |
| -- | --: | --: | --: |
| M9 Dev（80） | 0 | 0 | 0.00% |
| M9 Release（160，密封） | 0 | 0 | 0.00% |
| BFCL Offline Core（1,840） | 0 | 0 | 30.43% |
| M6 Domain（300） | 0 | 0 | 0.00% |

扫描报告只保存计数、版本和哈希，不公开 Release 正文或匹配片段。M6 Replay 由上游
Manifest 的评测 Prompt 零重叠证据约束；M2 Replay 只读取已注册 Train Split。

## 训练许可边界

本次门禁确认：

- 五个输入版本、Manifest、内容哈希和 DevOps 维护者审批完全绑定；
- Tool Call 与最终 Assistant 回答参与监督，System、User、Tool Result 和 Non-thinking
  空 Thinking 前缀屏蔽 Loss；
- 来源、语言、模式和总监督 Token 均从最终数组复算通过；
- 私有数组、Manifest、去重报告和污染报告由完整 Commit Marker 固定；
- 公开报告不含用户名、主机名、绝对路径、原始消息或密封评测内容。

下一门禁是 M10.2 的 Qwen3-0.6B 1M Supervised Token Full SFT、Checkpoint/Resume 和
Agent Dev/M6 阶段评测。
