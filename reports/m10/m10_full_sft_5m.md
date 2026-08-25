# M10.2 Qwen3-0.6B Full SFT 5M 阶段报告

## 结论

Qwen3-0.6B Agent Full SFT 已通过同一 Run 的 1M→5M Exact Resume，5M 阶段 Checkpoint、
Safetensors Export、数据身份、配置身份和父 Production 身份均通过哈希校验。真实
Continuation Gate 已执行并拒绝继续训练：M6 通用能力回退满足阈值，但 Agent Dev Task
Success 相对父模型下降 8.75pp。0.6B 10M 路线保持阻断，不产生新的 Candidate 或 Production
声明。

## 不可变身份

| 项目 | 实际值 |
|---|---|
| Run ID | `20260824T011335Z-m10-agent-full-sft-qwen3-0-6b-seed42-1ac1cad4-7b63` |
| 训练提交 | `bd92349342af53eb94682b3475b3fd41d77d6761` |
| 配置 SHA256 | `1ac1cad439d09e325a4482b357e527051eef4e68401122f823d6e5bb709ee61a` |
| 数据版本 | `m10-agent-sft-v1-4655d3e3` |
| 数据 Manifest SHA256 | `6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490` |
| 父 Production | `qwen3-0-6b-m7-fa678d92` |
| 5M Checkpoint | `checkpoint-tokens-0005000000` |
| Checkpoint Payload SHA256 | `98c0c01d42e016a256e3637107c347f6b0e56fa59deeca934f73b951c1ba8fce` |
| 5M Export SHA256 | `a8de3e2af57cb34870649c6fb67da225401737cc292bc31096b3e669b692ec1e` |
| 评测身份 | `qwen3-0-6b-m10-full-sft-5m-cf9c3994` |

评测身份来自只读的完整 Run 验证。其 Registry 状态固定为 `Evaluation`，并带有
`production_eligible=false`；它只能用于阶段评测，不能通过别名解析为 Production。

## 真实训练结果

| 指标 | 1M 阶段 | 1M→5M Resume 阶段 | 5M 累计状态 |
|---|---:|---:|---:|
| Optimizer Step | 1,008 | 4,032 | 5,040 |
| Supervised Token | 1,000,000 | 4,000,000 | 5,000,000 |
| 墙钟时间 | 4,259.40 秒 | 16,982.59 秒 | 21,241.99 秒 |
| Peak Allocated | 13,777,430,528 B | 13,777,327,104 B | 13,777,430,528 B |
| Peak Reserved | 17,785,946,112 B | 16,003,366,912 B | 17,785,946,112 B |

按 Epoch 统计的平均 Loss 为：

| Epoch | 平均 Loss |
|---:|---:|
| 1 | 0.4630 |
| 2 | 0.3066 |
| 3 | 0.2804 |
| 4 | 0.2662 |
| 5 | 0.2555 |

最后 100 Step 平均 Loss 为 0.2197。训练过程未出现 OOM、NaN/Inf 或 Checkpoint 完整性失败。
`result.json` 中的 `final_loss` 是最后一个 Step 的瞬时值 0.7971，不能替代窗口均值判断趋势。

## Checkpoint 与恢复

- 5M 结果的模式为 `exact_resume`，`resumed_from_tokens=1000000`；
- 1M 与 5M 边界永久 Pin；3M、4M 作为最近两个普通滚动 Checkpoint 保留；
- 5M Checkpoint 的模型、优化器、进度和 RNG 保存在完整训练状态中；
- 5M Safetensors Export 与训练 Checkpoint 分离，Export 只用于 Serving/Evaluation；
- 10M Resume 必须引用同时绑定 Run、配置、5M Export、Agent Dev 和 M6 证据的已接受 Gate。

## Continuation Gate 结果

阶段评测使用两个互相独立的证据：

1. 冻结的 80 条 Agent Dev：5M 模型的 Task Success 相对 M7 Production 父模型至少提升
   1pp；父模型已测基线为 20.00%，所以 80 条离散任务中至少需要达到 17/80，即 21.25%。
2. 冻结的 M6 ARC-Easy、HellaSwag、PIQA 等权聚合：相对父模型 54.48% 的回退不超过 2pp。

真实结果如下：

| 门禁项 | 父模型 | 5M 模型 | 变化 | 判定 |
|---|---:|---:|---:|---|
| Agent Dev Task Success | 20.00% | 11.25% | -8.75pp | 失败 |
| M6 通用任务聚合 | 54.48% | 52.70% | -1.78pp | 通过 |

最终决策为 `rejected`。配对诊断显示净减少的 7 个成功任务全部来自 No-tool 与
Wrong-tool/Irrelevance 边界；5M 模型在这些任务上出现不必要的证据检索和配置修改提议，工具类
任务则没有新增成功。训练 Loss 持续下降，因此该结果属于能力门禁失败，不能由 Loss 趋势替代。
脱敏事实源见 [`m10_full_sft_5m_gate.json`](raw/m10_full_sft_5m_gate.json)。

随后的接口审计发现，训练轨迹监督公开 MCP Tool Name，而当次 Agent Runtime 将带 Server
前缀的私有名称发送给模型，导致部分有效输出被记录为 Unknown Tool。初始拒绝证据保持不可变，
但不能用于评价修复后的统一协议；父模型与 5M 模型必须在同一修复提交上重新配对评测，新的
Gate 也必须形成独立证据，不能覆盖本节结果。

为区分早期有效学习与多轮重复训练造成的策略偏置，已保存的 1M Export 将使用同一冻结 Agent
Dev 单独复评。该诊断不会解锁 10M；只有形成新的、预注册的数据或训练策略后才能启动后续训练。
