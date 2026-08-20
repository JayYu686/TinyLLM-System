# M9 Agent Readiness Evaluation 验收报告

## 结论

M9 已冻结并实现 Agent 训练前评测边界：80 条公开 Dev 与 160 条密封 Release 组成原创
DevOps Agent Suite，外部对照固定为 1,840 条离线任务的
`TinyLLM BFCL v1.3 Offline Core Profile`。M7 Qwen3-0.6B Production、固定 Revision 的
Qwen3-8B Base 和 M5 历史 8B LoRA 均使用相同协议完成 Agent Dev 基线；三者的 BFCL
正式结果均通过完整性闭锁并纳入本报告。M9 的代码、测试、真实 GPU 运行、失败路径、公开
脱敏证据与中文报告现已形成完整验收链路。

M9 的作用是建立 M10 训练前能力起点和不可变门禁口径。下表中的父模型结果不会直接触发
Agent Production 晋级；Release Task Success ≥ 70% 等阈值只用于 M10 训练后 Candidate 与其
父模型的配对比较。

## 评测对象

| 对象 | 不可变模型身份 | 角色 | Agent Dev |
| -- | -- | -- | -- |
| Qwen3-0.6B Production | `qwen3-0-6b-m7-fa678d92` | M7 当前 Production，也是 M10 0.6B 父模型 | 两次正式重复 |
| Qwen3-8B Base | `qwen3-8b-m9-base-90587dd6` | 固定 Revision 的评测对象，也是 M10 8B 父模型 | 一次正式运行 |
| Qwen3-8B 历史 LoRA | `qwen3-8b-m9-historical-lora-b214e902` | M5 历史对照，只用于诊断 | 一次正式运行 |

两个 8B 对象注册在独立 Evaluation Registry，均固定
`production_eligible=false`，不会被 `production` Alias 解析，也不会创建 Candidate 或
Production 记录。历史 LoRA 不作为 M10 初始化点。

## DevOps Agent Dev 基线

套件身份固定为 `tinyllm-devops-agent-dev-v1-f958bcc6`，包含 80 条公开任务。0.6B 的两次
正式运行在 80/80 条 Task Success 上完全一致；为了避免选择性报告，下表同时保留两次结果。

| 指标 | 0.6B Run A | 0.6B Run B | 8B Base | 8B 历史 LoRA |
| -- | --: | --: | --: | --: |
| Tool Selection Accuracy | 41.25% | 40.00% | 56.25% | 65.00% |
| Argument Accuracy | 40.00% | 38.75% | 41.25% | 46.25% |
| Schema Valid Rate | 50.00% | 50.00% | 73.75% | 77.50% |
| No-tool Accuracy | 100.00% | 96.67% | 66.67% | 66.67% |
| Multi-step Success Rate | 0.00% | 0.00% | 6.67% | 16.67% |
| Task Success Rate | **20.00%** | **20.00%** | **36.25%** | **36.25%** |
| Tool Hallucination Rate | 13.75% | 15.00% | 21.25% | 12.50% |
| Error Recovery Rate | 0.00% | 0.00% | 0.00% | 0.00% |
| Grounding Accuracy | 23.91% | 23.91% | 54.35% | 60.87% |
| Approval Safety | 71.43% | 71.43% | 85.71% | 100.00% |
| 平均 Tool Calls / Task | 0.338 | 0.375 | 0.662 | 0.625 |
| 平均 Tokens / Task | 1262.062 | 1292.575 | 1759.138 | 1723.325 |
| P95 端到端延迟 | 4.283 s | 4.046 s | 5.309 s | 16.091 s |
| 未审批写操作 | 0 | 0 | 0 | 0 |
| 路径逃逸尝试 | 3 | 3 | 1 | 0 |
| 任意命令尝试 | 0 | 0 | 0 | 0 |

8B 模型显著提高了 Tool Schema、Grounding 与多步完成能力，但 Base 的 Tool Hallucination
达到 21.25%，三组模型的 Error Recovery 都是 0%。这些短板为 M10 的 Hard Negative、缺参
澄清、失败恢复、证据引用与审批安全样本提供了直接依据。历史 LoRA 在部分 Agent 指标上优于
Base，但其 P95 延迟明显更高，而且没有经过 Agent Release、BFCL 回归和统一 Production Gate，
因此只保留为历史诊断对照。这里的 P95 来自单次串行 Agent Dev 运行，用于资源诊断，不替代
M7 的正式 Serving Benchmark。

## BFCL 离线核心对照

每个对象都完成 1,840/1,840 条任务；三组共 5,520 条结果均通过唯一 ID、完整 Result、
`traceback` 和 `Error during inference` 检查，正式推理失败为 0。

| 类别 | 0.6B Production | 8B Base | 8B 历史 LoRA |
| -- | --: | --: | --: |
| Simple | 35.75% | 56.25% | 53.75% |
| Multiple | 15.50% | 37.50% | 36.00% |
| Parallel | 15.50% | 52.00% | 48.00% |
| Parallel Multiple | 4.50% | 19.00% | 19.50% |
| Irrelevance | 96.67% | 86.67% | 90.42% |
| Multi-turn Base | 0.00% | 21.50% | 8.50% |
| Multi-turn Missing Function | 0.00% | 3.00% | 0.00% |
| Multi-turn Missing Parameter | 0.00% | 11.00% | 5.50% |
| **Core Profile 总分** | **24.24%（446/1840）** | **39.18%（721/1840）** | **36.25%（667/1840）** |

8B Base 相对 0.6B Production 高 14.94pp，主要优势来自单轮工具调用、并行调用和基础多轮；
0.6B 在 Irrelevance 上高 10pp，但三类多轮合计均很弱。历史 LoRA 的总分比 8B Base 低
2.93pp，尤其 Multi-turn Base 低 13pp，因此它没有形成可替代固定 Base 的 Agent 父模型。

这组 BFCL 结果属于固定离线 Core Profile。它证明了训练前能力差距和 M10 数据重点，不代表
官方 BFCL Overall，也不触发任何 Agent Production 晋级。

## 评测边界与失败闭锁

- BFCL 固定 Tag `v1.3`、Commit `ea13468e4423454d0c213704fb87cf7cb3990433`，只覆盖
  Simple、Multiple、Parallel、Parallel Multiple、Irrelevance 与三类 Multi-turn。
- Live、Java、JavaScript、Multi-turn Long Context、Agentic Web Search 和外部 Memory 不在
  本次范围内；结果不得称为 BFCL 官方 Overall 或官方排行榜成绩。
- BFCL 使用独立依赖环境、环回 Gateway 和环境变量 Bearer Token；HTTP Client 不继承宿主
  代理。公开仓库不保存 Token、Prompt、完整 Tool 参数或原始任务输出。
- 早期兼容性诊断暴露了宿主代理、BFCL 非标准 `function.response` 扩展、JSON Schema
  `pattern` 属性误判、4K Context 以及 128 消息上限问题。修复后重新从空目录执行正式 Run；
  旧诊断记录保留在私有 Artifact Store，未混入正式结果。
- 正式协议使用 16K Context、最多 1024 条 Chat Completion 消息和 1 MiB Body 限制；Agent
  API 自身的 8 Step / 12 Tool Call 安全边界没有改变。
- BFCL 独立环境 `pip check` 通过；依赖审计没有发现未豁免的已知漏洞。固定上游依赖的九项
  例外只适用于离线评测进程，不适用于 Gateway 或线上服务。

## M10 门禁冻结状态

M10 正式运行前已经冻结以下关键条件：Release Task Success ≥ 70%，相对父模型提升至少
5pp 且配对 Cluster Bootstrap 95% CI 下界大于 0，Schema Valid ≥ 98%，No-tool ≥ 90%，
Tool Hallucination ≤ 2%，Grounding ≥ 90%，Error Recovery ≥ 70%，三类安全违规均为 0。
BFCL 总分不得低于父模型、任一类别回退不得超过 2pp，M6 能力回退也不得超过 2pp。

160 条 Release 在 M10 正式评测前继续密封，不参与训练、数据筛选或超参数选择。M10 的
0.6B Full SFT 与 8B LoRA 必须分别和自己的父模型比较，不能选择三组基线中的较低分作为
门禁起点。

## 验收清单

| 验收项 | 状态 | 证据 |
| -- | -- | -- |
| 80 Dev / 160 Release 套件冻结 | 通过 | Suite Manifest、内容哈希、公开 Dev 与私有 Release |
| 评测器、Resume、指标与 Gate 契约 | 通过 | 单元、集成、失败路径与 Schema Snapshot |
| 三个不可变评测对象注册 | 通过 | M7 Deployment 与独立 Evaluation Registry |
| 三组 Agent Dev 父模型基线 | 通过 | 4 次真实 RTX 3090 评测及逐题私有事实源 |
| 三组 BFCL Offline Core Profile | 通过 | 5520/5520 条、0 推理失败、原始 Result 与 Score |
| M10 门禁预注册 | 通过 | 固定阈值、配对 Bootstrap、回归与安全门禁 |
| 公开材料脱敏 | 通过 | 仅聚合指标、身份与内容哈希进入仓库 |

最终聚合事实源见 [`raw/m9_baseline_comparison.json`](raw/m9_baseline_comparison.json)。
