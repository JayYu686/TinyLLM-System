# M7 在线推理与 Production 晋级验收报告

## 结论

M7 已完成。M6 Candidate `qwen3-0-6b-m6-d16c2357` 在单张 RTX 3090 上通过 vLLM
CUDA 11.8 兼容性、Gateway 契约、正式 Direct/Gateway 性能矩阵、后端崩溃恢复、Last Known
Good 回滚、血缘和限定部署 Profile 安全审查，并晋级为
`qwen3-0-6b-m7-fa678d92` Production。

本报告仅公开去标识化汇总；18,000 条请求级 JSONL、环境、硬件、日志和完整 Gate 保存在私有
Artifact Store。M7 的 Production 表示模型质量与在线服务门禁通过，Agent Tool Use 能力仍由
M9/M10 独立门禁判定。

## 1. 固定部署身份

| 项目 | 实际结果 |
| -- | -- |
| 来源 Candidate | `qwen3-0-6b-m6-d16c2357` |
| Production | `qwen3-0-6b-m7-fa678d92` |
| 模型 | Qwen3-0.6B，596,049,920 参数，GQA，Full SFT |
| 模型 Artifact SHA256 | `63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6` |
| Tokenizer Artifact SHA256 | `99d4d297ece6cc43fa551987701e4ded4fa5c860d9448965d212508519bfc382` |
| Production Gate | `m7-production-gate-b804d235` |
| Production Gate SHA256 | `fa678d9278f8ab7749983ce211d963dd616f518632490e8c5abfd1499c52c739` |

M6 Candidate 记录保持不可变。M7 创建新的 Production 记录引用 Candidate、训练 Run、
Checkpoint、数据版本、评测、模型和 Tokenizer 哈希，随后原子更新 `production` Alias。

## 2. 正式推理 Benchmark

冻结矩阵覆盖输入 `128/512/1024` Token、输出 `128/256` Token、并发
`1/4/8/16/32`、Direct vLLM 与 Gateway 两条路径。每个组合预热 20 个请求，测量 100 个请求，
独立重复 3 次。

| 指标 | 门禁 | 实际结果 | 结论 |
| -- | --: | --: | -- |
| 测量请求 | 完整矩阵 | 18,000 | 通过 |
| 成功率 | ≥ 99.5% | 100.00%（18,000/18,000） | 通过 |
| Gateway / Direct 吞吐 | ≥ 90% | 97.23% | 通过 |
| 各组合 P95 延迟增幅中位数 | ≤ 10% | 4.28% | 通过 |
| OOM | 0 | 0 | 通过 |
| 进程僵死 | 0 | 0 | 通过 |
| 未解释 5xx | 0 | 0 | 通过 |

这些结论只适用于本报告固定的 Qwen3-0.6B、单张 RTX 3090、vLLM
`0.8.5.post1+cu118`、回环 Gateway 和对应请求矩阵。它们不外推到其他模型、GPU、上下文长度
或公网部署。

## 3. 正确性与恢复

| 检查 | 实际结果 |
| -- | -- |
| OpenAI Chat Completions、Models、Usage | 通过 |
| Bearer Auth、错误映射、请求边界 | 通过 |
| Streaming 与客户端断开取消上游 | 通过 |
| Thinking/Non-thinking 双模式且不公开原始 CoT | 通过 |
| vLLM 内部 Backend Guard | 通过 |
| Backend 被终止后 Readiness 失效 | 34 ms |
| Backend 自动恢复并再次完成请求 | 25.188 s |
| 无效配置切换保持 Last Known Good | 82 ms |

## 4. 安全与依赖边界

Gateway 默认绑定回环地址，Bearer Token 仅从环境变量读取；请求大小、JSON 深度、工具 Schema、
并发和速率均有上限。公开日志不保存完整 Prompt、工具参数、工具结果或原始 CoT。旧 CUDA
11.8 vLLM 依赖 Profile 的 44 个 OSV 公告全部保留，18 个 Critical/High 公告逐项完成适用性
审查，未缓解 Critical/High 数量均为 0。

该安全结论只适用于固定文本 Qwen3、回环 Backend、无远程模型代码、哈希校验 Artifact 和受限
请求 Schema。依赖升级后必须重新审计，不能继承本轮结论。

## 5. Production Gate

| Gate | 结果 |
| -- | -- |
| API 契约 | 通过 |
| 正式成功率 | 通过 |
| Runtime 稳定性 | 通过 |
| Gateway 吞吐 | 通过 |
| P95 延迟开销 | 通过 |
| Backend 恢复 | 通过 |
| Deployment 回滚 | 通过 |
| M6 质量与完整血缘 | 通过 |
| 安全审计 | 通过 |

Gate 总结果为 `accepted`，9/9 检查通过，Production Alias 已指向
`qwen3-0-6b-m7-fa678d92`。Registry 目录使用 `0700`，记录与 Alias 使用 `0600`。

## 6. 验收清单

- [x] 隔离 vLLM CUDA 11.8 环境与真实模型加载；
- [x] Model Resolver、Gateway、Supervisor、监控和结构化日志；
- [x] API、鉴权、Streaming、取消、错误映射与 CoT 隐藏；
- [x] 18,000 请求正式 Direct/Gateway Benchmark；
- [x] Backend Crash、Readiness、自动恢复和 LKG 回滚；
- [x] 限定部署 Profile 依赖安全审查；
- [x] 不可变 Production Gate、Production Record 与原子 Alias；
- [x] 中文公开审查报告。

M7 状态：`COMPLETE`。下一阶段：M8 Tool Calling、MCP 与 DevOps 单 Agent。
