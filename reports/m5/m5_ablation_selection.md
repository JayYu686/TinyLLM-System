# M5.2 双模式配比消融与预注册选优报告

## 1. 结论

M5.2 的 Teacher Pilot、三份精确 Token 配比数据、六组短程训练、真实中断恢复、冻结双模式
评测和预注册选优均已执行完成。

本轮选优状态为：

```text
status: no_eligible_arm
selection exit code: 6
selected_thinking_fraction_basis_points: null
selection_reason: no_arm_passed_preregistered_gates
```

0%、30% 和 50% 三个 Thinking Token 配比都通过 Non-thinking 回归门禁，但都未通过两个
Seed 的 Thinking 格式有效率至少 99%这一冻结门禁。因此，本轮没有选出正式配比，M5.3
Qwen3-0.6B 长程 Full SFT 保持阻塞状态。

该结果是门禁正常工作产生的有效失败结论。当前证据不支持降低阈值、忽略失败 Seed，或直接
把表现最好的 30% 配比写成正式选择。

机器可读选优结果见
[m5_ablation_selection.json](raw/m5_ablation_selection.json)。

## 2. 冻结实验身份

| 项目 | 固定值 |
| -- | -- |
| Base Model | `Qwen/Qwen3-0.6B` |
| Model Revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Attention | 原生 GQA |
| Reasoning Dev | `m5-reasoning-dev-v1-53ddf557` |
| Evaluation Config SHA256 | `3735a34e70c06059fbc09f62d02fabc296fd97e79a45d60f3d00dede21108d51` |
| Thinking 配比 | 0% / 30% / 50%，按实际参与 Loss 的 Assistant Token 计算 |
| 每组训练预算 | 1,000,000 Supervised Tokens |
| 训练 Seed | `42` / `20260727` |
| Thinking 解码 | Seed `20260726`，Temperature 0.6，Top-p 0.95，Top-k 20，最多 896 New Tokens |
| Non-thinking 解码 | Greedy，最多 128 New Tokens |
| 每种模式评测规模 | 200 条 |
| Candidate 评测硬件 | 物理 GPU 4，NVIDIA GeForce RTX 3090 |
| Candidate 评测 Git | `be0f439e030b02c4c96be6ecdc3a03eafe50514e`，工作树干净 |

训练与评测均使用冻结 Dev，没有接触 M6 的最终测试指标。Base 与 Candidate 的
`suite_version`、`config_sha256`、模型 Revision 和解码协议一致。Base 在物理 GPU 0 上运行，
Candidate 在物理 GPU 4 上运行；本报告比较质量和格式指标，不比较两张卡的执行时间。

## 3. Teacher Pilot 与数据门禁

修订后的 Label Vocabulary v2 Teacher Pilot 使用固定 Qwen3-8B Revision 和本地离线
BF16 Thinking 推理：

| 项目 | 结果 |
| -- | --: |
| 输入任务 | 100 |
| 接受样本 | 96 |
| 接受率 | 96% |
| 生成尝试 | 110 |
| 英文/中文接受数 | 68 / 28 |
| Config / JSON / Linux / Log / Python | 19 / 19 / 20 / 20 / 18 |
| Dataset Version | `m5-reasoning-pilot-v1-b4db5ac8` |

Pilot 通过至少 80% 接受率和五类覆盖门禁。此前的 Prompt 污染失败与 Placeholder 标签歧义
失败均保留在公开证据中，没有进入训练数据。

三份消融 Mixture 分别包含精确的 0、300,000 和 500,000 Thinking Supervised Tokens；
总预算均为 1,000,000。部分尾序列通过显式 Label Mask 达到精确 Token 预算，复用与 Mask
次数记录在各自 Manifest 中。

## 4. 六组短程训练

| Thinking | Seed | Optimizer Step | 训练时间（秒） | 初始 Loss | 最终 Loss | Peak Allocated | 结果 |
| --: | --: | --: | --: | --: | --: | --: | -- |
| 0% | 42 | 549 | 931.49 | 1.5679 | 2.0151 | 12.83 GiB | 成功；包含真实中断与 Exact Resume |
| 0% | 20260727 | 549 | 920.53 | 2.2948 | 1.1293 | 12.83 GiB | 成功 |
| 30% | 42 | 504 | 861.91 | 1.0955 | 1.1322 | 12.83 GiB | 成功 |
| 30% | 20260727 | 504 | 886.98 | 2.3415 | 1.1654 | 12.83 GiB | 成功 |
| 50% | 42 | 475 | 844.34 | 1.1187 | 0.9452 | 12.83 GiB | 成功 |
| 50% | 20260727 | 475 | 818.39 | 1.6019 | 0.7681 | 12.83 GiB | 成功 |

六组训练都达到精确 1,000,000 Supervised Tokens 并生成独立 Safetensors 导出。0% / Seed 42
先在 21,334 Token 处执行真实中断，随后从完整 Checkpoint 继续到 1,000,000 Token；失败的
CPU/CUDA RNG 映射尝试与最终成功尝试分别保存在 Run 的 `attempts/` 中。

表中的初始/最终 Loss 对应不同数据窗口，且三个 Mixture 的序列组成不同。该数据用于运行诊断，
不作为跨配比模型质量排序依据。

## 5. 冻结双模式评测

Base 结果：

| 模式 | 格式有效率 | Final-answer 分数 | JSON 有效数 | Length-limited |
| -- | --: | --: | --: | --: |
| Thinking | 85.5% | 65.0% | 133 / 200 | 29 |
| Non-thinking | 100% | 37.0% | 123 / 200 | 0 |

六个 Candidate 结果：

| Thinking | Seed | Non-thinking 分数 | Thinking 格式率 | Thinking 分数 | Thinking Length-limited | 评测时间（秒） |
| --: | --: | --: | --: | --: | --: | --: |
| 0% | 42 | 46.5% | 0.0% | 0.0% | 1 | 110.28 |
| 0% | 20260727 | 43.5% | 0.0% | 0.0% | 1 | 90.84 |
| 30% | 42 | 61.0% | 95.5% | 93.5% | 8 | 755.13 |
| 30% | 20260727 | 61.5% | 97.0% | 95.0% | 6 | 791.42 |
| 50% | 42 | 60.5% | 96.0% | 94.5% | 8 | 890.15 |
| 50% | 20260727 | 60.5% | 92.5% | 90.5% | 13 | 819.88 |

所有 Candidate 的 Non-thinking 格式率均为 100%，可见 Thinking 泄漏数均为零。30% 配比在
两个 Seed 上取得最高的平均 Thinking 分数和 Non-thinking 分数，但其格式率仍分别比 99%
门槛低 3.5pp 和 2.0pp。

## 6. 预注册选择结果

选优顺序在训练前已经固定：

1. Non-thinking Dev 相对 Base 回退不超过 2pp；
2. 两个 Seed 的 Thinking 格式有效率都至少为 99%；
3. 最大化两个 Seed 的平均 Thinking Final-answer 分数；
4. 差异不足 1pp 时选择 Thinking 比例更低者。

| Thinking | 平均 Non-thinking 分数 | Non-thinking Gate | 两个 Seed Thinking 格式率 | Format Gate | 平均 Thinking 分数 |
| --: | --: | -- | -- | -- | --: |
| 0% | 45.00% | 通过 | 0.0% / 0.0% | 拒绝 | 0.00% |
| 30% | 61.25% | 通过 | 95.5% / 97.0% | 拒绝 | 94.25% |
| 50% | 60.50% | 通过 | 96.0% / 92.5% | 拒绝 | 92.50% |

由于第二道门禁已经拒绝全部配比，第三和第四条只保留为诊断信息，不能触发正式选择。

## 7. 失败分析

30% 两个 Seed 共出现 15 条 Thinking 格式失败，同时记录 14 条 Length-limited；50% 共出现
23 条格式失败，同时记录 21 条 Length-limited。两组计数高度接近，说明生成长度限制是首要
调查方向，但聚合计数不能证明两者逐条重合。

下一批必须从私有 Raw Results 对失败 Item 做逐条归因，至少区分：

- 达到 896 Token 上限导致 Closing Tag 或 Final JSON 缺失；
- Thinking 内容完成但输出 Envelope 不完整；
- Final JSON Schema 或封闭标签错误；
- 重复生成、模式漂移或其他停止条件异常。

本轮 Dev 已经承担配比选择用途，可以用于工程诊断；M6 冻结测试集继续隔离，不能参与修正。

## 8. 状态与后续门禁

M5.2 的实验执行状态更新为“完成，选优门禁拒绝”。M5 总里程碑继续保持 `IN_PROGRESS`。

后续建立独立的格式可靠性修正批次：

1. 对 30% 和 50% 的 38 条格式失败执行私有逐条归因，公开聚合统计；
2. 在 Train/Pilot 来源上构建版本化 Format-repair 数据，保持 Dev 内容隔离；
3. 以 30% 配比作为诊断起点建立新配置身份，不把它标记为已选配比；
4. 使用两个固定 Seed 重跑短程修正实验；
5. 保持当前 Dev、解码参数和 99%格式门禁；
6. 只有新配比通过全部预注册门禁后，才解锁 M5.3 长程 Full SFT。

任何解码长度、评测 Template 或门槛变化都需要新的评测配置版本，并重新运行 Base 与全部
对照，不能覆盖本轮结果。

## 9. 证据索引

- 设计与冻结参数：[M5 SFT 契约](../../docs/m5_sft_contract.md)
- M5.1 数据与 Teacher 证据：[M5.1 报告](m5_reasoning_data.md)
- Teacher Pilot 摘要：[teacher_pilot_100.json](raw/teacher_pilot_100.json)
- 预注册选优结果：[m5_ablation_selection.json](raw/m5_ablation_selection.json)
- Candidate 原始生成、完整 Run、Checkpoint 和导出：私有
  `$TINYLLM_ARTIFACT_ROOT`
