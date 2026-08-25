# M10.2 Qwen3-0.6B Full SFT 5M 阶段报告

## 结论

Qwen3-0.6B Agent Full SFT 已通过同一 Run 的 1M→5M Exact Resume，5M 阶段 Checkpoint、
Safetensors Export、数据身份、配置身份和父 Production 身份均通过哈希校验。该结果允许进入
Agent Dev 与 M6 通用能力回归评测；在 Continuation Gate 生成并接受前，不允许继续训练到
10M，也不产生新的 Candidate 或 Production 声明。

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

## 下一门禁

阶段评测使用两个互相独立的证据：

1. 冻结的 80 条 Agent Dev：5M 模型的 Task Success 相对 M7 Production 父模型至少提升
   1pp；父模型已测基线为 20.00%，所以 80 条离散任务中至少需要达到 17/80，即 21.25%。
2. 冻结的 M6 ARC-Easy、HellaSwag、PIQA 等权聚合：相对父模型 54.48% 的回退不超过 2pp。

两个条件同时满足时，系统生成 `accepted` Continuation Gate；任一条件失败时保留 5M
Development 结果并停止 0.6B 10M 路线，不能通过手工改写状态绕过门禁。
