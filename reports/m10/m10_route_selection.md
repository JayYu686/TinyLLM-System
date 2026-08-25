# M10 Agent 模型路线选择报告

## 结论

统一训练与 Serving Tool Name 协议后，Qwen3-0.6B Full SFT 的 1M、5M 阶段均低于其
Production 父模型，5M Continuation Gate 再次拒绝。0.6B 路线在 5M 正式早停，10M 不再
执行。Qwen3-8B Base 在相同提交、同一 80 条 Agent Dev 上达到 45.00%，后续进入预注册的
8B Agent LoRA 路线。

## 协议修复

训练轨迹使用公开 MCP Tool Name，例如 `get_run` 和 `search_evidence`；初始 Agent Runtime
向模型发送了带 Server 前缀的私有名称。提交
`b5023ab38d3c9773ea9fabd660921834647c642e` 将 OpenAI Tool Calling 名称统一为公开
`tool_name`，本地仍以 `server_id + tool_name` 执行权限映射。多个 MCP Server 出现同名工具时
直接拒绝加载。

旧评测与新评测均保留。路线选择只使用修复提交上的配对结果。

## 真实 Agent Dev 结果

| 模型 | 角色 | Task Success | Schema Valid | Tool Selection | Grounding | 安全违规 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B Production | 0.6B 父模型 | 21.25% | 75.00% | 36.25% | 41.30% | 0 次未审批写操作 / 12 次路径逃逸 / 0 次任意命令 |
| Qwen3-0.6B Full SFT 1M | 早期诊断 | 7.50% | 60.00% | 22.50% | 15.22% | 0 次未审批写操作 / 13 次路径逃逸 / 0 次任意命令 |
| Qwen3-0.6B Full SFT 5M | Continuation Gate 对象 | 10.00% | 60.00% | 20.00% | 19.57% | 0 次未审批写操作 / 5 次路径逃逸 / 0 次任意命令 |
| Qwen3-8B Base | 8B LoRA 父模型 | **45.00%** | **100.00%** | **82.50%** | **100.00%** | 0 次未审批写操作 / 0 次路径逃逸 / 0 次任意命令 |

1M 到 5M 增加 2 个成功任务，但仍比 0.6B 父模型少 9 个；同时 No-tool、工具幻觉和参数选择
没有达到继续训练条件。5M 的 M6 通用任务回退为 1.78pp，单项通过，但不能抵消 Agent Dev
下降 11.25pp。

8B Base 相比 0.6B 父模型多完成 19 个任务，并在 Schema、Tool Selection 与 Grounding 上形成
清晰优势。它仍需通过 Agent LoRA 解决 No-tool、工具幻觉、多步和失败恢复短板。

## 后续执行边界

- 8B LoRA 父模型固定为 `qwen3-8b-m9-base-90587dd6`，不叠加 M5 历史 Adapter；
- BF16、Rank 16、Alpha 32、Dropout 0.05，覆盖 Attention/MLP Linear；
- 先执行 1M Supervised Token 和 Agent Dev，达到相对父模型至少 +1pp 后再进入 5M；
- 5M 未改善时停止 10M，保留失败证据并进入最终 RC 收尾；
- 最终 Release、BFCL、M6 与 Serving Gate 的阈值保持不变。

聚合事实源见
[`m10_route_selection_protocol_v2.json`](raw/m10_route_selection_protocol_v2.json)。
