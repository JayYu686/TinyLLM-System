# M10 Agent 后训练总验收报告

## 验收结论

M10 已完成 Agent 数据构建、Qwen3-8B LoRA 后训练、Checkpoint/Resume、Dev/Release 双阶段
评测、BFCL 离线核心集、M6 通用能力回归、Serving 血缘复核和最终统一门禁。最终候选
`qwen3-8b-m10-agent-lora-5m-3e8bf1dd` 通过预注册的 13/13 项检查，并晋级为：

```text
qwen3-8b-m10-agent-production-b2d88493
```

私有 Artifact Store 中的 `agent-production` Alias 已通过原子写入指向该不可变记录。M6
Candidate、M7 Production 和 M9 父模型证据均保持不可变；M10 Production 记录只引用这些
历史身份及其 SHA256。

## 正式结果

| 指标 | Qwen3-8B 父模型 | M10 Agent Production | 结果 |
|---|---:|---:|---:|
| Release Task Success（160 条） | 74.38% | **93.12%** | **+18.74pp** |
| Cluster Bootstrap 95% CI | — | — | **[+3.47, +36.07]pp** |
| Schema Valid Rate | 100% | **100%** | 达标 |
| No-tool Accuracy | 66.67% | **100%** | +33.33pp |
| Tool Hallucination Rate | 16.88% | **0%** | -16.88pp |
| Tool Selection / Argument Accuracy | 83.12% / 79.38% | **98.12% / 98.12%** | 提升 |
| Multi-step Success | 78.33% | **100%** | +21.67pp |
| Error Recovery | 100% | **100%** | 保持 |
| Grounding Accuracy | 100% | **100%** | 保持 |
| BFCL v1.3 Offline Core Profile | 39.18%（721/1840） | **39.29%（723/1840）** | +0.11pp |
| M6 通用任务聚合 | 62.64% | **62.64%** | 0pp 回归 |
| 未审批写入 / 路径逃逸 / 任意命令 | 0 / 0 / 0 | **0 / 0 / 0** | 安全边界满足 |

Release Task Success 的差值使用 22 个任务簇、10,000 次重采样和固定 Seed `20260820`
计算。BFCL 最差单类别变化为 -0.50pp，处于预注册的 -2pp 回归界限内。该结果称为
`TinyLLM BFCL v1.3 Offline Core Profile`，不作为官方 BFCL Overall 或排行榜成绩。

## 模型选择与路由

最终模型沿用 Qwen3 的 GQA 架构，父模型固定为
`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`。训练身份绑定：

- 5,000,000 Supervised Token；
- BF16 LoRA，Rank 16、Alpha 32、Dropout 0.05；
- Dataset `m10-agent-sft-v2-435b9fbc`；
- Checkpoint `checkpoint-tokens-0005000000`；
- 配置 SHA256 `d451d02fe30ac00565c37c01f450070c27151cf1f92e579bd8d6a34b929750ec`。

部署采用可哈希的模块路由策略：请求提供完整的七个 TinyLLM DevOps 工具目录时加载 Agent
Adapter；无工具或其他工具目录使用 Base 路径。该策略将专用 Agent 能力限制在审查过的
Tool Allowlist 内，同时保持通用与无工具行为。路由策略 SHA256 为
`b6495e9b4e906338d5c42bb51cbc2e09f1aef66d37e14543a6c0d2b69567f556`。

## 统一门禁

最终门禁逐项检查以下事实源：

1. 160 条密封 Release 的端到端任务成功率；
2. 相对父模型的 Cluster Bootstrap 置信区间；
3. Schema、No-tool、工具幻觉、Grounding、失败恢复和安全边界；
4. 候选与父模型各 1,840 条 BFCL Core 结果；
5. M6 v7 通用能力配对回归；
6. M7 已验收的 Gateway、恢复、回滚与安全平台门禁；
7. 精确模型、Tokenizer、Adapter、路由策略和评测摘要的 SHA256 血缘。

最终 Gate SHA256 为
`b2d88493e308ea93507f069a447920ed956cdfdd1d55ec7b288a33b9b52bfd63`，Production Record
SHA256 为 `eccccd83402254a8626527f647c28a76a765f3a4feaa0ef21023595a2495d78c`。

## 交付能力

- 单张 RTX 3090 上的 Qwen3-8B BF16 LoRA 与完整阶段恢复；
- Assistant/Tool 监督掩码、污染检查和数据血缘；
- OpenAI Tool Calling、MCP、LangGraph 状态机、FTS5 证据检索和显式审批；
- Dev/Release 隔离、BFCL、通用回归与 Bootstrap 统计门禁；
- 不可变 Agent Production Record、原子 Alias、哈希漂移防护与回滚接口；
- 中文公开汇总与私有请求级证据分层。

模型路线与选择依据见 [`M10 Agent 模型路线选择报告`](m10_route_selection.md)。公开聚合证据见
[`m10_final_gate_summary.json`](raw/m10_final_gate_summary.json)。
