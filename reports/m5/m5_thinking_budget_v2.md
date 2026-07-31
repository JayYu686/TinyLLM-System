# M5 Thinking Budget v2 门禁报告

## 1. 结论

`m5-thinking-budget-v2` 已在同一干净 Commit 上完成 Base 与 R1 两个固定 Seed 的真实 RTX
3090 评测。两个 Candidate 同时通过控制后格式、强制收束率、Thinking 正确率和
Non-thinking 回归四项门禁，`m5_3_authorized=true`，因此 M5.3 正式训练已解锁。

该结论表示 R1 的 30% Thinking 配比适合进入长程训练，不表示 M5 已完成，也不表示模型已经
晋级 Candidate。0.6B Full SFT、8B LoRA、恢复与失败路径仍需完成；M6 将使用独立冻结评测。

## 2. 真实结果

| 模型 | Thinking 格式 | 自然闭合 | 强制收束 | Thinking 正确率 | Non-thinking 正确率 | 时长 |
| -- | --: | --: | --: | --: | --: | --: |
| Base | 100.0% | 93.0% | 7.0% | 70.5% | 37.0% | 922.33s |
| R1 Seed42 | 100.0% | 98.0% | 2.0% | 96.0% | 64.0% | 750.14s |
| R1 Seed20260727 | 100.0% | 96.5% | 3.5% | 96.0% | 66.0% | 767.05s |

两个 Candidate 相对 Base 的 Thinking 正确率均提升 25.5pp；Non-thinking 分别提升
27pp 和 29pp。两个 Seed 都没有 Thinking 长度触顶项，控制后 400 条 Candidate Thinking
结果全部格式有效且最终 JSON 有效。

## 3. 门禁复算

| 门禁 | 阈值 | Seed42 | Seed20260727 | 结果 |
| -- | --: | --: | --: | -- |
| 控制后 Thinking 格式 | ≥99% | 100% | 100% | 通过 |
| 强制收束率 | ≤10% | 2.0% | 3.5% | 通过 |
| Thinking 正确率 | ≥90% | 96.0% | 96.0% | 通过 |
| Non-thinking 回归 | ≥Base−2pp，即 ≥35% | 64.0% | 66.0% | 通过 |

自动选择器验证协议、配置、Commit、模型 Revision、GQA、Suite、训练 Run、Mixture 和 Seed
一致后返回 `status=passed`、`gate_reason=all_protocol_v2_gates_passed`。

## 4. 控制器证据边界

第一阶段使用 1536 Token Thinking Budget。模型未自然产生 `</think>` 时，运行器注入 Qwen
官方示例的固定 early-stopping 文本，再最多生成 128 个最终答案 Token。注入内容不计入模型
生成 Token，也没有被表述为模型自然闭合。

Base 有 14/200 条、两个 Candidate 分别有 4/200 和 7/200 条使用控制器。每条私有结果分别
保存第一阶段输出、注入文本、第二阶段输出、Token 记账和响应哈希。公开摘要只包含脱敏聚合；
逐条响应留在私有 Artifact Store。

## 5. 血缘

- 评测协议：`m5-thinking-budget-v2`
- Dev：`m5-reasoning-dev-v1-53ddf557`
- 模型 Revision：`c1899de289a04d12100db370d81485cdf75e47ca`
- Mixture：`m5-format-repair-mixture-v1-1396b60b`
- Mixture Manifest：
  `2467b5dce0d909b865b73219d2f608bdbc9c6fcc1bb09b93c5ebea8a7b60bd0e`
- 评测 Commit：`6cfd71ed9d4d6483608e17dda5f2110dae7085ea`
- Base Raw SHA256：
  `49f08010eee3fde879c6501212441621dec8dac911d4e2e4710e4472a8d75231`
- Seed42 Raw SHA256：
  `df7114eebd5bea8ad52bea799c219a362fb32ec0988e129b8d989efc005d4abb`
- Seed20260727 Raw SHA256：
  `e0ed9364715251204c88896e1a35e14637679c8760a25445969119ce043ed05b`

机器结果：

- [自动门禁](raw/m5_thinking_budget_v2_gate.json)
- [Base 摘要](raw/m5_thinking_budget_v2_base.json)
- [Seed42 摘要](raw/m5_thinking_budget_v2_seed42.json)
- [Seed20260727 摘要](raw/m5_thinking_budget_v2_seed20260727.json)

## 6. 下一阶段

正式数据冻结采用已选定的 30% Thinking Token 配比。接下来依次完成：

1. 构建 `m5-dual-sft-v1-*` 与确定性重开验证；
2. Qwen3-0.6B 四卡 DDP Full SFT、Checkpoint 和 Exact Resume；
3. Qwen3-8B 单卡 BF16 LoRA Probe、正式训练和 Adapter 导出；
4. 失败路径、中文 M5 验收报告与英文公开摘要。
