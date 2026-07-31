# M5.2-R3 Mixture v2 构建报告

## 1. 结论

`m5-r3-mixture-v2-b47723e1` 已完成真实离线构建和重新打开校验，状态为
`COMPLETED_TRAINING_AUTHORIZED`。它解决了原协议中 150K Targeted Thinking Token 与
单来源最多四次不可同时满足的问题，并在训练前完成标签重平衡。

本结果只授权两个 1M Token R3 Seed 的短程训练。它不代表模型质量门禁通过，也不读取
M6 冻结评测结果。

## 2. 修订依据

原 160 条选择一轮只有 4,933 个监督 Token，四次总使用最多得到 19,732 Token。Mixture v2
在不改变 1M 总预算和 30% Thinking 比例的前提下：

1. 从已经通过正式来源门禁的 218 条中，按任务族、语言和标签重新确定性选择 160 条；
2. 保留 `reasoning_tokens → repeated_8gram_basis_points → sample_id` 稳定排序；
3. 依据真实 Tokenization 将单来源总使用上限冻结为 30；
4. 构建前固定配置和配额，不读取 R3 训练或 Dev 结果。

真实重平衡来源每轮提供 5,037 个监督 Token。精确 150K 构建后每个来源使用 29–30 次，
没有触及超过 30 次的情况。

## 3. 精确 Token 构成

| 数据分层 | 监督 Token | 来源数 | 训练序列数 |
| -- | --: | --: | --: |
| M2 Non-thinking | 700,000 | 4,597 | 3,050 |
| 原 Pilot General Thinking | 150,000 | 96 | 478 |
| R3 Config/Log Targeted Thinking | 150,000 | 160 | 4,765 |
| 合计 | 1,000,000 | — | 8,293 |

整体 Thinking 比例保持 30%。三个分层各只有最后一个序列使用部分 Loss Mask，因此
`partially_masked_sequences=3`。

## 4. Targeted 来源分布

| 维度 | 分布 |
| -- | -- |
| 任务族 | Config 80；Log Diagnosis 80 |
| 语言 | 英文 112；中文 48 |
| Config 标签 | 21 / 22 / 15 / 22 |
| Log 标签 | 20 / 20 / 20 / 20 |
| 单来源监督 Token/轮 | 5,037 |
| 单来源使用次数 | 最小 29；最大 30 |

Config 的 `unsupported_precision` 只有 15 条通过正式生成，因此不能做到四标签完全各 20
条。v2 使用全部 15 条，并将其余标签控制在 21–22 条；相比最短优先结果
`14 / 29 / 7 / 30`，偏斜显著降低。

## 5. 血缘

| 项目 | 值 |
| -- | -- |
| Config SHA256 | `68fe849f097baa2c60660d2db7a45af55b0338e7aeb91c12d0df38653b3b16a7` |
| Dataset Version | `m5-r3-mixture-v2-b47723e1` |
| Manifest SHA256 | `2a7dac85b2b98909c993234a3c9bd4054eb09a59a5c1a23f862f6a9a1ea0f83f` |
| Content SHA256 | `b47723e112480309403d1700f27b8ae7be7a656099ca51568098b67259b6f2dd` |
| Payload SHA256 | `515ea2b7a991ba860ad605fe85a46cded42955be42e3711c1a8ec0e6b0cad18b` |
| Formal Raw SHA256 | `a71c68f51761f8f041d0961bbe419d2f045efaac15706ad598ad3a2446ac05f9` |
| Targeted Selection SHA256 | `de7b50147ba69cc5890d2bfe8142a8df867d6288406e3872183b7121e6d29313` |

公开机器 Manifest：
[m5_r3_mixture.json](raw/m5_r3_mixture.json)。

## 6. 下一步门禁

使用固定 Seed 42 和 20260727 分别完成 1M Token Qwen3-0.6B BF16 Full SFT，然后在相同
200 条冻结 Reasoning Dev、相同双模式解码配置上评测。两个 Seed 必须同时满足：

- Non-thinking 分数相对 Base 回退不超过 2pp；
- Thinking 格式有效率至少 99%。

未满足时仍可诚实完成本轮系统实验，但模型保持 `Development`，不得把失败结果改写为
Candidate。
