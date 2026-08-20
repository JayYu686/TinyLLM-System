# M9 Qwen3-0.6B Production Agent Dev 基线报告

## 结论

M7 Production 模型已完成 M9 公共 Dev 套件的两次正式基线评测。两次运行均使用固定
Suite、模型 Artifact、Deployment、软件环境、GPU 身份、Git Commit、Seed 和串行并发配置，
80 条任务的 Task Success 判定完全一致，均为 **20.0%**。

这组结果建立了 M10 的 0.6B 父模型能力起点。它同时表明当前模型擅长 No-tool 判断，但尚未
形成稳定的 OpenAI Function Call、多步工具使用、失败恢复、证据引用和审批安全能力。该结果
是 M9 基线，不是 Agent Candidate Gate 结果，也不表示 M9 已完成。

## 固定身份

| 项目 | 实际值 |
| -- | -- |
| 模型 | `qwen3-0-6b-m7-fa678d92` |
| Revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Dev Suite | `tinyllm-devops-agent-dev-v1-f958bcc6`，80 条 |
| Git Commit | `ac1e5abd02a7aede02f2838b8447f2e36a8ba26c` |
| Seed | `20260820` |
| 并发 | `1` |
| GPU | 物理 GPU 0，NVIDIA GeForce RTX 3090 |
| Driver | `535.261.03` |

软件环境、硬件、配置、模型、Deployment 和 Suite 均通过 SHA256 绑定。完整逐题结果与运行
事实源保存在私有 Artifact Store，公开聚合事实见
[`raw/agent_dev_production_baseline.json`](raw/agent_dev_production_baseline.json)。

## 双重复结果

| 指标 | Run A | Run B |
| -- | --: | --: |
| Tool Selection Accuracy | 41.25% | 40.00% |
| Argument Accuracy | 40.00% | 38.75% |
| Schema Valid Rate | 50.00% | 50.00% |
| No-tool Accuracy | 100.00% | 96.67% |
| Multi-step Success Rate | 0.00% | 0.00% |
| Task Success Rate | 20.00% | 20.00% |
| Tool Hallucination Rate | 13.75% | 15.00% |
| Error Recovery Rate | 0.00% | 0.00% |
| Grounding Accuracy | 23.91% | 23.91% |
| Approval Safety | 71.43% | 71.43% |
| 平均 Tool Calls / Task | 0.338 | 0.375 |
| 平均 Tokens / Task | 1262.062 | 1292.575 |
| P95 端到端延迟 | 4.283 s | 4.046 s |
| 未审批写操作 | 0 | 0 |
| 路径逃逸尝试 | 3 | 3 |
| 任意命令尝试 | 0 | 0 |

两次运行的 80 条 Task Success 完全一致。6 条任务的辅助评分发生变化，24 条最终文本不完全
一致，因此报告同时保留两次原始结果，不选择其中较好的一次作为唯一成绩。

## 分类诊断

两次运行的 Task Success 分类结果一致：

| 类别 | 通过 / 总数 | 主要现象 |
| -- | --: | -- |
| No-tool | 10 / 10 | 能稳定避免无关工具调用 |
| Wrong-tool / Irrelevance | 6 / 10 | 大部分能拒绝无关工具，少量输出伪工具标签或未完成回答 |
| Missing Argument / Clarification | 0 / 10 | 通常不调用工具，但澄清问题未满足冻结的答案断言 |
| Single Tool | 0 / 13 | 常输出 `<call_id>`、`<search_evidence>` 等非协议标签 |
| Sequential Multi-step | 0 / 15 | 无法稳定完成顺序轨迹，出现错误工具和路径参数 |
| Parallel Independent Tools | 0 / 5 | 未完成两个独立工具的并行轨迹 |
| Tool Failure Recovery | 0 / 10 | 未进入有效工具调用，无法验证重试后的恢复 |
| Grounding / Approval / Security | 0 / 7 | 证据引用和审批状态不足，并出现路径逃逸尝试 |

## 关键解释

Schema Valid Rate 只有 50%，原因是评测器会把伪 XML/标签、未知工具或无法解析的 Tool Call
判定为模型协议失败。空工具轨迹不会再自动得到格式通过，从而避免格式指标虚高。两次运行均
记录 40 条 `AgentModelError`，与当前 0.6B 模型尚未接受 Agent Tool 数据训练的状态一致。

安全侧没有发生未审批写入或任意命令执行，但每次均出现 3 次路径逃逸参数尝试，Approval
Safety 仅为 71.43%。这些结果会作为 M10 数据构建中路径 Hard Negative、审批和 Grounding
样本的直接设计依据。

## 后续

1. 在相同契约下建立固定 Qwen3-8B Base 与 M5 8B LoRA 历史对照。
2. 运行 `TinyLLM BFCL v1.3 Offline Core Profile`，保存 1840 条离线任务结果。
3. 汇总三组父模型基线，冻结 M10 训练前比较表。
4. Release 160 条继续保持密封，M10 正式评测前不用于训练、筛选或调参。
