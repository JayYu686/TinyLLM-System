# M7.0/M7.1 在线推理基础审查报告

## 结论

M7.0/M7.1 的工程实现和真实 RTX 3090 服务验证已经完成，M7 整体状态保持 `IN_PROGRESS`。
已完成 M6 Candidate 的只读解析与模型哈希复核、Production/Deployment Schema、原子 Alias、
FastAPI Gateway、OpenAI Chat Completions 契约、Bearer Auth、健康检查、Prometheus 指标、
受控 Backend Supervisor、依赖安全审计和正式 Benchmark 数据契约。

Qwen3-0.6B Candidate 已在固定 CUDA 11.8 vLLM 环境中通过模型加载、非流式、流式、Thinking /
Non-thinking、崩溃恢复和 Last Known Good 回滚验证。该轮证据产生于未提交工作树，仅用于工程
验证，不能提交 Production Gate。TTFT、TPOT、吞吐和 P95 正式结论仍需由干净提交上的完整
Benchmark 矩阵产生。

## 1. 已验证输入

| 项目 | 实际结果 |
| -- | -- |
| M6 Candidate | `qwen3-0-6b-m6-d16c2357` |
| Candidate Record SHA256 | `bb9189f4df4ffa8b089fc2892e12d31273875efbe9739eede8ea0f903d95fba9` |
| 模型导出 SHA256 | `63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6` |
| Tokenizer Artifact SHA256 | `99d4d297ece6cc43fa551987701e4ded4fa5c860d9448965d212508519bfc382` |
| 模型状态 | Candidate；未生成 Production Alias |

Resolver 从 Candidate 的 `training_run_id` 定位唯一 Run，重新计算导出目录哈希，并从固定的
Qwen Repository/Revision 解析 Tokenizer。以上结果来自本机现有 Artifact Store，只记录内容
身份，不公开私有绝对路径。

## 2. 已实现接口

CLI：

```text
tinyllm serve
tinyllm benchmark inference
tinyllm deploy resolve|show|gate|promote|rollback
```

Gateway：

```text
GET  /v1/models
POST /v1/chat/completions
GET  /health/live
GET  /health/ready
GET  /version
GET  /metrics
```

新增服务与部署错误使用退出码 `7`，M7 Gate 拒绝继续使用退出码 `6`。Tool Calling 字段已经
进入请求 Schema 和透传路径，实际 Qwen/Hermes Parser 验收属于 M8。

## 3. 安全默认值

- Gateway 和 vLLM Backend 仅允许回环地址；
- Swagger/OpenAPI/ReDoc 默认关闭；
- Bearer Token 从环境变量读取，不写入 YAML；
- 认证作为路由依赖统一执行；
- Request ID 与 W3C Trace Context 经过格式校验；
- HTTP Client 禁止环境代理和自动重定向；
- 请求体、并发、速率、消息数和生成长度设置上限；
- 错误响应不披露内部路径或 Traceback；
- Metrics 不使用 Prompt、用户或工具参数作为 Label。
- vLLM Backend 使用独立内部 Bearer、固定路由 Allowlist 和回环绑定；
- 启动前限制模型架构并递归拒绝危险动态配置；
- Backend Crash 时终止完整进程组，等待显存释放后再重启。

## 4. 证据状态

| 验收项 | 状态 |
| -- | -- |
| Pydantic v2 严格 Schema 与 Snapshot | 已完成 |
| Registry/Resolver 哈希漂移失败路径 | 已完成 |
| Candidate 不可变与 Production 原子 Alias | 已完成 Mock 验证 |
| Auth、健康检查、普通/流式请求、工具字段透传 | 已完成 Mock 验证 |
| 错误映射、请求大小、速率限制 | 已完成 Mock 验证 |
| 正式 Direct/Gateway Benchmark 契约 | 已冻结 |
| vLLM CUDA 11.8 依赖安装与 Qwen3 加载 | 已完成真实验证 |
| RTX 3090 非流式/流式/双模式 Smoke | 已完成真实验证 |
| 正式 30 格 × 2 后端 × 3 重复 | `not_evaluated` |
| Backend Crash 与 180 秒恢复 | 调试验证通过：Readiness 41 ms 失效，25.205 s 恢复 |
| Last Known Good 回滚 | 调试验证通过：88 ms，身份保持不变 |
| 依赖安全审计 | 限定部署 Profile 审查通过；44 条观测，18 条 Critical/High 已评估，0 条未缓解 |
| Production Gate / Alias | 未执行 |

## 5. 下一步

1. 合并 M7 实现并在干净提交上重新采集环境、硬件、契约、恢复、回滚和安全证据；
2. 运行正式 Direct/Gateway Benchmark 矩阵；
3. 生成 M7 Production Gate；
4. 只有全部检查通过才写入不可变 Production Record 并原子更新 Alias；
5. 形成最终中文审查报告并发布 `v0.7.0`。

## 6. 依赖解析记录

首次解析 `vLLM 0.8.5.post1+cu118` 时发现其 OpenTelemetry 上限为 `<1.27`，与初稿中独立选择
的 `1.39.1` 不兼容；无上限解析还会选择未进入本项目范围的 Transformers 5.x。最终隔离环境
固定为 PyTorch `2.6.0+cu118`、vLLM `0.8.5.post1+cu118`、Transformers `4.57.6`、
Tokenizers `0.22.1`、Ray `2.56.1` 和 OpenTelemetry `1.26.0`，并已通过 `pip check`、
Qwen3 RTX 3090 Smoke 与限定部署 Profile 安全审查。

安全审查结论只适用于当前单 GPU、回环 Backend、文本 Qwen3、固定本地 Artifact 和受限请求
Schema 的部署 Profile，不表示相关依赖在其他部署方式下不存在风险。
