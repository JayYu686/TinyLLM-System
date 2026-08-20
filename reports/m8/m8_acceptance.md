# M8 Tool Calling、MCP 与 DevOps Agent 验收报告

## 结论

M8 已完成。固定的 M7 Qwen3-0.6B Production 在单张 RTX 3090 上通过 OpenAI Tool Calling
8 格兼容性矩阵；LangGraph Runtime、MCP Client、参考 DevOps MCP Server、FTS5/BM25 证据
检索、显式审批、服务重启恢复、沙箱写入和 Agent API 已完成真实或集成验证。

M8 验收证明运行时与协议能力，不表示模型已通过 Agent Readiness Gate。Qwen3-0.6B 对提示语言
和表达仍有明显敏感性；BFCL、自建 240 条任务集、Task Success、No-tool Accuracy 和
Hallucination 指标将在 M9 冻结并测量。

## 1. 固定身份

| 项目 | 实际结果 |
| -- | -- |
| 实现 Commit | `cc1c8ba7478d8d6c157996e9b32f38d4ac2ea742` |
| Gateway/Agent 版本 | `0.8.0b1` |
| 服务模型 | M7 Production `qwen3-0-6b-m7-fa678d92` |
| vLLM | `0.8.5.post1+cu118` |
| 正式 GPU | RTX 3090，物理索引 0 |
| Tool Calling Evidence | `m8-tool-calling-e7c2fdc4` |
| Tool Calling Evidence SHA256 | `e277b872723d426f76a3e7c9163d5a16297b8e2c10682903fa5e83b0e13e4cf4` |
| Approval/Recovery Evidence | `m8-agent-contract-dc438db6` |
| Approval/Recovery Evidence SHA256 | `24373ec0669bbee772051e8abdc4179999384af8f6885339570f8813dabcc828` |
| Evidence Index | `m8-evidence-c1c7c591`，153 文档 / 405 Chunk |
| Evidence Index SHA256 | `beb0976f925e89f8cc083a7b0381e87c7fd3faa84c66377d94e6bba699d9dfe5` |

原始 JSON、Agent Run、事件、审批、沙箱、Gateway 日志和依赖审计保存在私有 Artifact Store；
公开报告只保留去标识化身份和汇总。

## 2. Tool Calling 正式矩阵

固定 Qwen3/Hermes Profile 覆盖 `auto/required/none/named` 与 Streaming/Non-streaming
笛卡尔积：

| `tool_choice` | 非流式 | 流式 |
| -- | -- | -- |
| `auto` | 通过，`finish_reason=tool_calls` | 通过，`finish_reason=tool_calls` |
| `required` | 通过，`finish_reason=tool_calls` | 通过，`finish_reason=tool_calls` |
| `none` | 通过，普通文本、无工具 | 通过，普通文本、无工具 |
| 指定 `search_evidence` | 通过，`finish_reason=tool_calls` | 通过，`finish_reason=tool_calls` |

结果为 8/8 通过，所有 Tool Call 名称均结构化为 `search_evidence`，普通回答与 Tool Call 均未
暴露原始 `<tool_call>` 或 JSON Markup。该结果绑定干净 Git Commit 和实际 GPU 身份。

调试阶段保留了多轮失败证据：代理环境误路由、旧 Parser 的 Streaming Finish Reason、
`required + streaming` 增量解析，以及真实写请求产生的非标准 XML Call。最终实现增加回环
Client、Finish Reason 归一化、固定 Qwen/Hermes 增量修复和未知 Markup 拒绝；失败记录没有
删除，也没有计入正式通过数。

## 3. Agent、MCP 与 Grounding

真实模型端到端读工具 Run：

| 项目 | 实际结果 |
| -- | -- |
| Run | `agent-20260820T064524Z-a1313d13-cb1e` |
| 状态 | `succeeded` |
| 模型步骤 | 2 |
| 工具调用 | 1 × `search_evidence` |
| 事件 | 7 条，序号 1–7 单调 |
| 检索 Query | `M7 Production gate` |
| Grounding | 最终答案包含实际 Call ID `call_0674bf6f3e22f6ea62612475` |

Run 通过真实 OpenAI Gateway、LangGraph、stdio MCP 和 FTS5 索引完成。工具返回文档 ID、相对
路径、行号、内容哈希和摘录；最终回答引用实际 Tool Call，不暴露原始 CoT。

同一模型在另一条中文直接指令中选择了 No-tool 并给出“缺少证据”的保守回答，未发生工具幻觉
或伪造引用。这说明 0.6B Base/Production 尚存在提示敏感性，也说明 M8 的协议通过不能替代 M9
任务成功率门禁。本报告保留该观察，不将其计为 Agent 能力通过样例。

## 4. 审批、恢复与沙箱写入

`m8-agent-contract-dc438db6` 使用真实 stdio MCP 验证：

- Run 到达持久化 `waiting_approval`，审批前没有写入；
- 审批绑定 Tool Call SHA256；
- 创建全新的 Runtime/MCP Client 模拟服务重启；
- 从 LangGraph 审批中断安全节点恢复；
- 完成恰好一次 `apply_sandbox_config_patch`；
- 重复审批返回既有决策，重复相同写入返回同一内容哈希；
- 原配置 SHA256 前后均为
  `c97d60e43ada033058ffd38986a708552309be3e92b7449b4d78618b3fedf8b9`；
- 输出仅位于 Agent Run 专属沙箱，文件权限为私有。

该证据的 `git_dirty=false`、`source_unchanged=true`、`restart_resume_succeeded=true`、
`idempotent_approval_succeeded=true`、`idempotent_write_succeeded=true`，总结果为 `passed=true`。

## 5. 自动测试与失败路径

| 检查 | 实际结果 |
| -- | -- |
| 仓库 CPU/Mock 测试 | 993 通过，2 个 GPU 测试按标记排除 |
| CPU 可测试核心覆盖率 | 85.094% |
| M8 定向测试 | 114 通过 |
| Ruff / Ruff Format | 通过 |
| MyPy Strict | 389 个 Source File 通过 |
| JSON Schema Snapshot | 通过 |
| Markdown Link | 119 个文件通过 |
| 公开 Artifact 脱敏 | 通过 |
| stdio MCP | 通过 |
| Streamable HTTP MCP | 通过 |

自动测试覆盖未知工具、非法参数、Server Tool 缺失、MCP 超时与错误输出、只读重试、写操作零
重试、重复 Tool Loop、坏 Observation、私有 CoT 字段、路径遍历、软链接逃逸、未审批写入、
审批哈希漂移、幂等冲突、SSE 断线补发、显式取消、过期和安全节点恢复。

## 6. 依赖与安全

隔离 M8 环境的 `pip-audit` 原始结果包含 8 条记录、6 个唯一公告，全部来自 M7 已审查的
protobuf/Starlette Profile；LangGraph/MCP 没有新增发现。`make audit-agent` 只忽略这 6 个精确
ID，实际结果为“0 个未处理、8 条已审查忽略”。详细适用性和控制见
[M8 Agent 安全实践审查](security_best_practices.md)及
[M8 依赖例外](../../requirements/m8_security_exceptions.md)。

## 7. 验收清单

- [x] 固定 Tool Calling 协议与 8 格真实 GPU 验证；
- [x] MCP Client、stdio Server 与 Streamable HTTP 集成；
- [x] 七个能力受限的参考 DevOps 工具；
- [x] LangGraph 状态图、Step/Tool/循环/超时限制；
- [x] Agent API、Bearer、Idempotency、SSE Replay 与显式取消；
- [x] FTS5/BM25 证据索引、内容哈希和最终答案引用；
- [x] 显式审批、重启恢复和幂等沙箱写入；
- [x] 85% 覆盖率、Schema、链接、脱敏和依赖审计；
- [x] 中文契约、安全审查与总验收报告。

M8 状态：`COMPLETE`。下一阶段为 M9 Agent Readiness Evaluation；在 M9/M10 门禁通过前，
M7 Production 继续作为线上模型，项目不宣称 Agent Production Readiness。
