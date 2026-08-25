# M10 Agent 后训练总验收报告

## 验收结论

M10 已按预注册规则完成数据冻结、两条后训练路线、阶段 Checkpoint/Resume、真实 GPU 运行、
Agent Dev 比较和失败早停。Qwen3-0.6B Full SFT 与 Qwen3-8B Agent LoRA 均未达到各自的
Continuation Gate，因此没有 Agent 模型晋级 Candidate 或 Production。

项目保留 M7 的 `qwen3-0-6b-m7-fa678d92` 作为 Production 模型，M8 Agent Runtime、M9
评测套件和全部 M10 训练证据继续有效。发布状态为 `v1.0.0-rc.1`：系统主链路完整，Agent
后训练模型门禁未通过。该状态不会通过降低阈值、跳过 Release/BFCL 或选择更长但已退化的
Checkpoint 改写。

## 交付范围

| 批次 | 状态 | 验收证据 |
|---|---|---|
| M10.1 数据冻结 | 通过 | 五来源 1M 监督 Token，70/30 语言比例，四边界污染为零 |
| M10.2 0.6B Full SFT | 完成并早停 | 1M→5M Exact Resume；5M Agent Dev 相对父模型 -11.25pp |
| M10.3 8B Agent LoRA | 完成并早停 | 单卡 BF16 1M；Agent Dev 相对父模型 -12.50pp |
| M10.4 统一模型门禁 | 未触发 | 两条路线均未通过开发阶段门禁，不消费密封 Release |

密封的 160 条 Release 未用于失败分析、Prompt 调整或训练数据构建。由于没有通过开发阶段
门禁的 Candidate，BFCL、M6 回归和 M7 Serving 复验没有被无意义地重复执行。

## 路线比较

| 路线 | 父模型 Task Success | 最终评测阶段 | 阶段 Task Success | 变化 | 决策 |
|---|---:|---|---:|---:|---|
| Qwen3-0.6B Full SFT | 21.25% | 5M | 10.00% | -11.25pp | 停止 10M |
| Qwen3-8B Agent LoRA | 45.00% | 1M | 32.50% | -12.50pp | 停止 5M/10M |

0.6B 路线的 M6 通用聚合只回退 1.78pp，但 Agent Dev 未通过；8B 路线在 Tool Selection、
Schema、No-tool 和工具幻觉上改善，却因最终事实回答与无关请求边界退化而降低端到端 Task
Success。两条路线都证明了训练 Loss 不能替代 Agent 能力门禁。

## 已验证工程能力

- 固定数据、模型 Revision、配置、Git、环境、硬件、Checkpoint 和评测血缘；
- 0.6B Full SFT 的单卡 BF16、1M→5M Exact Resume 和完整模型导出；
- 8B BF16 LoRA 在 24 GiB RTX 3090 上的真实 10-step Probe 与 1M 训练；
- Adapter-only 原子 Checkpoint、阶段导出、哈希校验和 vLLM LoRA 加载；
- 训练父模型与候选模型的同协议配对 Agent Dev；
- 开发门禁拒绝后自动阻断后续训练与发布评测；
- M7 Production Alias 保持不变，失败实验只保留 `Evaluation` 身份。

## 后续增强边界

若继续研究 Agent 模型能力，应创建新的 Dataset Revision 和 ADR，重点审查最终答案模板重复、
Tool Result 事实复述、Irrelevance Hard Negative 和 Error Recovery 监督。新版本必须重新执行
污染检查和 1M 阶段 Gate；本次 `m10-agent-sft-v1-4655d3e3` 不再追加训练。

M10 的真实训练与失败分析见
[`0.6B Full SFT 5M 报告`](m10_full_sft_5m.md)、
[`8B Agent LoRA 1M 报告`](m10_agent_lora_1m.md)和
[`路线选择报告`](m10_route_selection.md)。
