# M8 Tool Calling、MCP 与 DevOps Agent 契约

## 1. 目标

M8 在 M7 Production Gateway 之后提供一个能力受限、可恢复、可审计的 DevOps 诊断单
Agent。它复用 OpenAI Chat Completions 的 Tool Calling 协议，通过 MCP 调用本地工具，并将
证据、事件、审批和沙箱写入保存在私有 Artifact Store。

M8 验收只证明运行时协议、工具边界、审批恢复和固定 Qwen3/vLLM Profile 的 Tool Calling
兼容性。Agent 任务成功率、BFCL 成绩及训练后能力由 M9/M10 独立门禁判定。

## 2. 服务接口

Gateway 保留 M7 的 OpenAI-compatible 接口，并增加以下 Agent API：

```text
POST /v1/agent/runs
GET  /v1/agent/runs/{run_id}
GET  /v1/agent/runs/{run_id}/events
POST /v1/agent/runs/{run_id}/approvals/{approval_id}
POST /v1/agent/runs/{run_id}/cancel
```

创建请求固定包含 `schema_version`、`model`、`messages`、`mode`、`mcp_server_ids` 和
`max_steps`。外部调用方只能提交 User/Assistant 文本消息，System Policy、Tool Call 和 Tool
Result 均由服务端生成，避免调用方伪造工具权限或 Observation。

- `max_steps` 为 1–8，总工具调用上限为 12；
- 创建和审批要求 `Idempotency-Key`；
- Bearer Token 仅从环境变量读取，Gateway 默认绑定 `127.0.0.1`；
- SSE 使用单调序号，`Last-Event-ID` 可补发历史事件；
- SSE 断开不取消 Run，取消只能调用显式接口；
- 状态为 `created/running/waiting_approval/succeeded/failed/cancelled/expired`；
- API、事件与日志不公开原始 CoT。

事件类型冻结为：

```text
run.started
model.delta
tool.call.proposed
approval.required
tool.started
tool.completed
message.completed
run.completed
run.failed
```

## 3. Agent 状态图与恢复

```text
接收请求
→ 检查证据检索能力
→ 模型决策
→ Tool Schema 校验
→ 本地权限与循环限制
→ 审批或 MCP 调用
→ Observation 安全校验
→ 下一工具或最终答案
```

LangGraph 使用每个 Run 独立的 SQLite Checkpointer。服务重启后，`created/running` Run 从最近
安全节点恢复；`waiting_approval` Run 保持静止，收到与 Tool Call SHA256 匹配的持久化审批后
恢复。生成中断只保证安全节点级恢复，不表示 Token 级 Exact Resume。

同一工具与参数最多连续出现两次；只读工具最多执行三次（首次加两次 250/500 ms 退避），
沙箱写工具只执行一次。单工具超时 10 秒，Run 默认超时 120 秒。

## 4. MCP 权限模型

只加载管理员 YAML 中注册的 MCP Server。请求不能提供 URL、启动命令或临时工具。MCP Server
返回的 Annotation、描述和检索内容均视为不可信数据，不授予权限；最终权限由本地 Allowlist
决定。

发送到 OpenAI Tool Calling 接口的函数名保持 MCP Tool 的公开 `tool_name`，并在本地映射回
`server_id + tool_name` 权限身份。训练轨迹、Gateway Tool Schema 与模型输出由此使用同一
名称；多个已注册 Server 暴露同名 Tool 时启动失败，不通过添加私有前缀静默改写模型协议。

参考 `tinyllm-devops` stdio Server 暴露：

| 工具 | 权限 | 主要边界 |
| -- | -- | -- |
| `search_evidence` | 只读 | 查询 FTS5/BM25 索引并返回路径、行号和内容哈希 |
| `list_runs` | 只读 | 只返回 Run 安全字段 |
| `get_run` | 只读 | Run ID 唯一匹配，隐藏 Prompt 与 Secret |
| `read_log_excerpt` | 只读 | 仅限 Artifact Store 内允许目录与文本后缀 |
| `query_metrics` | 只读 | 仅限 `metrics.jsonl`/`summary.json`，字段和结果有界 |
| `inspect_config` | 只读 | 仅限允许目录中的 YAML/JSON，递归脱敏 Secret |
| `apply_sandbox_config_patch` | 沙箱写 | 只写 Agent Run 专属配置副本，必须显式审批 |

所有路径逐段拒绝 `..`、绝对路径、NUL、软链接和根目录逃逸。写操作将审批 ID、Tool Call
SHA256、Run ID、Call ID 和参数绑定；重复相同审批返回相同结果，参数漂移或目标冲突会拒绝。
源配置保持不变，目标仅位于 `agent-sandboxes/<run-id>/`。

stdio 是参考实现。Streamable HTTP 只做协议集成验证，要求 HTTPS 与环境变量 Bearer Secret，
并继续受同一 Allowlist 约束。

## 5. 检索与 Grounding

`tinyllm agent index rebuild` 从仓库文档、公开报告、Registry 和脱敏 Run 元数据构建不可变
SQLite FTS5/BM25 索引。每条结果包含文档 ID、相对路径、起止行号、内容 SHA256、相关性分数
和有界摘录。

模型接收的检索结果最多保留三条、每条摘录最多 600 字符。工具结果无法覆盖 System Policy。
执行过工具的最终回答必须包含实际 Call ID 引用；模型遗漏时 Runtime 会附加
`[evidence:<call_id>]`，不会生成不存在的证据标识。

## 6. CLI 与私有 Artifact

M8 交付：

```text
tinyllm agent run
tinyllm agent approve
tinyllm agent cancel
tinyllm agent index rebuild
```

`tinyllm agent eval` 在 M9 评测契约落地时交付。命令支持稳定 `--json`，配置/输入错误使用退出码
2，Agent Runtime、MCP 或工具执行失败使用退出码 8。

私有事实源：

```text
$TINYLLM_ARTIFACT_ROOT/
├── agent-runs/
├── agent-sandboxes/
├── agent-evaluations/
└── agent-indexes/
```

公开报告只保存去标识化摘要。完整消息、工具参数、工具结果、审批记录和原始日志不进入 Git。

## 7. M8 验收门槛

- OpenAI Tool Calling 的 `auto/required/none/named × streaming/non-streaming` 8 格真实验证通过；
- 普通回答和 Tool Call Streaming 均不暴露原始 Tool Markup；
- stdio 与 Streamable HTTP MCP 集成测试通过；
- 写操作经历 `waiting_approval`、进程重启、安全节点恢复和恰好一次沙箱写入；
- 源配置哈希保持不变，重复审批和写入满足幂等；
- 未审批写入、未知工具、非法 Schema、路径/软链接逃逸、循环、超时和私有 CoT 泄漏被拒绝；
- Ruff、Ruff Format、MyPy Strict、Schema Snapshot 和仓库 CPU 核心覆盖率 ≥ 85%；
- 依赖审计没有新增未评审公告；
- 真实证据绑定干净 Git Commit、GPU 身份和不可变 Artifact 哈希。

只有以上条件和中文验收报告全部完成后，M8 才能发布 `v0.8.0-beta.1`。
