# M7 在线推理与 Production 晋级契约

## 1. 目标

M7 将 M6 已注册的不可变 Candidate 接入本地 vLLM 服务，提供带认证的
OpenAI-compatible Gateway，并用真实 Direct/Gateway 对照、崩溃恢复和部署回滚证据决定
是否生成新的 Production 记录。

M6 Candidate 记录不发生修改。M7 Production 记录只引用并再次校验 M6 Candidate、模型导出、
Tokenizer、Serving 配置、环境、Benchmark、安全审计和失败恢复证据。

## 2. 运行边界

- 默认监听 `127.0.0.1`，后端必须是回环 HTTP 地址。
- Bearer Token 只从 `TINYLLM_GATEWAY_BEARER_TOKEN` 读取，长度至少 32 个字符。
- TLS、用户系统和分布式限流由外部反向代理提供。
- OpenAPI、Swagger UI 和 ReDoc 默认关闭。
- 服务日志不记录完整 Prompt、工具参数、工具结果或原始 CoT。
- `.venv-serving` 与训练、Baseline、M4/M5 环境隔离。
- 当前 Driver 保持不变；vLLM CUDA 兼容性必须由 RTX 3090 Smoke 证明。

## 3. Model Resolver

支持三类引用：

```text
M6 Candidate 版本
M7 Production 版本
production Alias
```

每次解析都执行：

1. 严格校验 Registry JSON Schema；
2. 从 `training_run_id` 唯一定位 Run；
3. 重算模型导出内容哈希并与 M6 记录比较；
4. 从固定 Repository/Revision 定位 Tokenizer 并计算哈希；
5. Production 再次校验 Alias、Production Record 与 M6 Candidate 的交叉血缘。

哈希漂移、多条 Run 匹配、软链接目录、缺失 Artifact 或非法引用均拒绝启动。

## 4. Gateway API

```text
GET  /v1/models
POST /v1/chat/completions
GET  /health/live
GET  /health/ready
GET  /version
GET  /metrics
```

`/v1/models`、`/v1/chat/completions` 和 `/metrics` 统一使用 Bearer Auth。健康检查和版本接口只
返回路径无关、内容无关的信息。Chat 请求使用 Pydantic v2 严格 Schema，拒绝未知字段，并限制
消息数、消息大小、生成长度、工具数和请求体大小。

Gateway 保留 OpenAI Chat Completions 的消息、Streaming、Usage、Tools、`tool_choice` 和
`stream_options.include_usage` 字段。M7.1 负责透明转发；Tool Parser 的实际能力在 M8 验收。

## 5. 请求语义

- 非流式请求只在后端尚未返回内容且发生 Transport/Timeout/502/503/504 时重试一次。
- Streaming 请求发出首个 Chunk 后不重试。
- 普通 Chat Streaming 的客户端断开会关闭上游流。
- `x-request-id` 只接受安全字符，否则由 Gateway 生成随机 ID。
- 合法 W3C `traceparent` 会透传至后端；非法值不透传。
- 并发由进程内 Semaphore 限制，速率由本地滑动窗口限制。
- 错误使用稳定 OpenAI 风格 Envelope；内部异常和路径不会返回客户端。

## 6. Benchmark

正式矩阵固定为：

```text
并发：1 / 4 / 8 / 16 / 32
输入：128 / 512 / 1024 tokens
输出：128 / 256 tokens
预热：每格 20 请求
测量：每格 100 请求
重复：3 次
后端：Direct vLLM / TinyLLM Gateway
```

原始事实源为 `requests.jsonl`，每条只保存 Request ID、后端、目标长度、实际 Usage、状态码、
TTFT、TPOT、总延迟和错误类别，不保存请求或响应正文。`summary.json` 绑定配置、模型、Tokenizer、
环境、硬件和原始结果 SHA256。

## 7. Production Gate

以下检查必须全部通过：

| 检查 | 阈值 |
| -- | -- |
| API/Auth/Streaming/取消/错误映射 | 全部通过 |
| 正式请求成功率 | ≥ 99.5% |
| 稳定性 | OOM、僵死、未解释 5xx 均为 0 |
| Gateway/Direct 吞吐比 | ≥ 90% |
| 各格 P95 增幅中位数 | ≤ 10% |
| Backend 故障 | 5 秒内 Readiness 失败，180 秒内恢复 |
| 部署失败回滚 | 180 秒内恢复 Last Known Good |
| M6 质量与血缘 | 完整 |
| 安全审计 | 无未缓解 Critical/High |

Gate Schema 会从原始计数重新计算门禁布尔值，手工将检查标为通过会被拒绝。通过后生成不可变
Production Record，并原子更新 `registry/aliases/production.json`。失败时 Candidate 保持原状态。

## 8. 验收状态

M7.0/M7.1 的 Registry、Resolver、Gateway、Schema、Mock 测试、真实 vLLM CUDA Smoke、
Backend Crash、Last Known Good 回滚和安全审计已经完成。已证明固定 Candidate 能在 RTX 3090
和 CUDA 11.8 环境中加载与恢复，但调试轮证据不替代正式发布证据。

正式矩阵、干净提交证据、Production Gate、Production Alias 和发布尚未完成，因此 M7 状态保持
`IN_PROGRESS`，未产生吞吐、TTFT、TPOT 或 P95 的发布结论。
