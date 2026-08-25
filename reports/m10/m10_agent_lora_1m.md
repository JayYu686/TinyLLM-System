# M10.3 Qwen3-8B Agent LoRA 1M 阶段报告

## 结论

Qwen3-8B Agent LoRA 已在单张 RTX 3090 上完成 1,000,000 Supervised Token 的 BF16
训练、Adapter-only Checkpoint、Safetensors 导出、不可变 Evaluation Subject 注册和 80 条
Agent Dev 评测。训练与血缘链路均通过，但 1M→5M Continuation Gate 拒绝继续：候选模型的
Task Success 为 32.50%，低于同协议 8B Base 的 45.00%，变化为 -12.50pp。

该 Run 在 1M 正式早停，不执行 5M/10M，也不进入 Release、BFCL、M6 或 Serving Gate。
Adapter 保留为 `Evaluation` 证据，不具备 Candidate 或 Production 资格。

## 不可变身份

| 项目 | 实际值 |
|---|---|
| Run ID | `20260825T063020Z-m10-agent-lora-qwen3-8b-seed42-2a47b09e-12fc` |
| 训练提交 | `dbca533f74d5c70a4270e3a0583c756041fd15b1` |
| 配置 SHA256 | `2a47b09ed960150d6d38103e4218734e72d8f10b9d5731392a6c72fa3bf50cd9` |
| 数据版本 | `m10-agent-sft-v1-4655d3e3` |
| 数据 Manifest SHA256 | `6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490` |
| 父评测对象 | `qwen3-8b-m9-base-90587dd6` |
| 1M Checkpoint | `checkpoint-tokens-0001000000` |
| Adapter SHA256 | `be65f1f089e9d1ac6710a4d64ade32392c382cb8b0790accf3c9cf5e3cbaeb61` |
| Evaluation Subject | `qwen3-8b-m10-agent-lora-1m-8c51da63` |
| Evaluation Subject SHA256 | `fb0b57e24492b938ade6766e71926a297157f33b992e5abde9f3ba53eab4ac67` |

Evaluation Subject 对 8B Base、Tokenizer、Adapter、训练 Checkpoint、数据、配置和显存 Probe
执行完整哈希复核，并固定为 `production_eligible=false`。

## 真实训练结果

| 指标 | 实际结果 |
|---|---:|
| Optimizer Step | 1,008 |
| Supervised Token | 1,000,000 |
| 墙钟时间 | 22,486.93 秒（6 小时 14 分 46.93 秒） |
| Trainable / Total Parameter | 43,646,976 / 8,234,382,336 |
| Peak Allocated | 22,101,809,152 B（20.58 GiB） |
| Peak Reserved | 24,209,522,688 B（22.55 GiB） |
| 前 100 Step 平均 Loss | 0.7132 |
| 后 100 Step 平均 Loss | 0.2715 |
| 全阶段平均 Loss | 0.3522 |

训练未出现 OOM、NaN/Inf 或 Checkpoint 完整性失败。`result.json` 的 `initial_loss=0.4339`
和 `final_loss=0.6813` 是两个单 Step 瞬时值；窗口均值显示优化目标持续下降。

正式显存 Probe 和完整训练的 Peak Reserved 均为 22.55 GiB，证明固定 BF16 LoRA 配置可在
独占 RTX 3090 上运行，NF4 QLoRA 回退条件未触发。导出阶段曾因 PEFT 自动检查 Embedding
配置而尝试访问 Hugging Face；网络不可达不影响 Adapter 完整性，后续实现已显式关闭未修改
Embedding 的远程探测。

## Agent Dev 与阶段门禁

父模型和 1M Adapter 使用相同的 80 条 Dev、公开 Tool Name、Non-thinking 模式和 Agent
Runtime。真实结果如下：

| 指标 | 8B Base | 1M LoRA | 变化 |
|---|---:|---:|---:|
| Task Success | 45.00% | 32.50% | -12.50pp |
| Tool Selection | 82.50% | 88.75% | +6.25pp |
| Argument Accuracy | 48.75% | 50.00% | +1.25pp |
| Schema Valid | 100.00% | 100.00% | 0pp |
| No-tool Accuracy | 66.67% | 80.00% | +13.33pp |
| Tool Hallucination | 17.50% | 11.25% | -6.25pp |
| Multi-step Success | 16.67% | 16.67% | 0pp |
| Error Recovery | 0.00% | 0.00% | 0pp |
| Grounding | 100.00% | 100.00% | 0pp |
| Approval Safety | 100.00% | 100.00% | 0pp |
| 未审批写操作 / 路径逃逸 / 任意命令 | 0 / 0 / 0 | 0 / 0 / 0 | 无新增违规 |

1M→5M 要求 Task Success 相对父模型至少提升 1pp。80 条任务的离散步长为 1.25pp，因此
候选至少需要完成 37/80；实际只完成 26/80，Gate 以 `improvement_basis_points=-1250`
给出 `rejected`。

## 失败差分

配对比较中，候选相对父模型净减少 10 个成功任务：12 个任务由成功转失败，2 个任务由失败
转成功。主要变化为：

| 类别 | 父模型成功率 | 1M LoRA 成功率 |
|---|---:|---:|
| Single Tool | 84.62% | 61.54% |
| Wrong-tool / Irrelevance | 90.00% | 30.00% |
| Sequential Multi-step | 33.33% | 26.67% |
| Missing Argument / Clarification | 0.00% | 10.00% |
| No-tool | 100.00% | 100.00% |

失败样本中，模型通常选择了正确工具并生成合法参数，但最终回答退化为“检查已完成”“与评测
事件一致”等模板，没有复述 Tool Result 中的状态、精度或诊断事实；部分无关请求还错误调用了
`search_evidence`。因此 Loss 下降和 Tool Schema 改善没有转化为端到端任务成功。

从现有证据推断，下一版数据应提高 Tool Result→最终事实回答、明确无关请求拒答和失败恢复的
监督权重，并降低通用模板答案的重复度。该推断只进入后续数据版本研究，不修改本次冻结数据、
门禁或历史证据。

脱敏事实源见
[`m10_agent_lora_1m_eval.json`](raw/m10_agent_lora_1m_eval.json) 与
[`m10_agent_lora_1m_gate.json`](raw/m10_agent_lora_1m_gate.json)。
