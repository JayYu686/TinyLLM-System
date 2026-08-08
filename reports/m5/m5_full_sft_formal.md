# M5.3 Qwen3-0.6B 四卡 Full SFT 正式报告

## 1. 结论

Qwen3-0.6B 四卡 BF16 Full SFT 已完成 M5 系统验收。正式 Campaign 在物理 GPU 5、6、8、9
上训练 50M Supervised Token；Fresh 段于 2,002,739 Token 受控中断，全新 `torchrun` 进程
验证模型、优化器、Scheduler、RNG、Sampler Cursor、配置、数据、环境、硬件与 World Size
后 Exact Resume 到 50M。

Run 保存 10M、20M、30M、40M、50M 五个不可变模型快照，并对每个快照真实执行 200 条
Thinking 与 200 条 Non-thinking 冻结 M5 Dev 评测。10M 快照取得最好的联合双模式结果：
Thinking 95.0%，Non-thinking 47.5%；50M 终点为 91.5%和 39.0%。因此 10M 快照进入 M6
优先比较队列，50M 终点作为完整训练、恢复和长程回退证据保留。

## 2. 固定身份

| 项目 | 实际结果 |
| -- | -- |
| 模型 | `Qwen/Qwen3-0.6B` |
| Revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Attention | 原生 GQA |
| 数据 | `m5-dual-sft-v1-b5b9e839`，30% Thinking Token |
| 数据 Manifest SHA256 | `607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6` |
| Run | `20260807T071224Z-m5-formal-qwen3-0-6b-d39dad35-3d15` |
| Campaign | `20260807T071211Z-m5-formal-campaign` |
| Git Commit | `c406e6760c6ea6b5eb19966740af4c494983576d`，clean |
| 配置 SHA256 | `d39dad3534730dfde08d526f24a69344d3be1341a097e610eaec7038041ad676` |
| 软件 | Python 3.11.14、PyTorch 2.7.1+cu118、CUDA 11.8、Transformers 4.57.6 |

公开[机器可读摘要](raw/m5_full_sft_formal.json)已移除用户名、主机名和绝对路径；私有
Artifact Store 保留完整日志、权重、逐条输出和环境快照。

## 3. 四卡训练与恢复

| 阶段 | 结果 | Token | Step | 作业段时长 |
| -- | -- | --: | --: | --: |
| Fresh | 按计划中断 | 2,002,739 | 260 | 479.63 秒 |
| Exact Resume | 成功 | 50,000,000 | 6,485 | 11,614.82 秒 |

每 Rank Peak Allocated 为 16,156,301,312 B，Peak Reserved 为 20,573,061,120 B。训练 Loss
从 1.694187 降至 0.143489，观测范围为 0.032165–2.019683；Loss 与 Gradient Norm 均保持
有限值。`metrics.jsonl` 包含连续 6,485 个 Optimizer Step，SHA256 为
`3b5aed75666dd0f80bc6e453ef6f6efecfcd304768f4542c53207ae0d34d03b7`。

温控守护共执行 25 次暂停与 25 次恢复，采样到的最高温度为 87°C。暂停阈值为 84°C，恢复
阈值为 74°C；超过阈值后进程组暂停，降温后从原状态继续。两个训练作业段均低于 12 小时。

## 4. Checkpoint 与阶段快照

| 用途 | 目标 Token | 实际 Token | Step | 结果 |
| -- | --: | --: | --: | -- |
| 中断恢复 | 2M | 2,002,739 | 260 | Exact Resume 成功 |
| 阶段评测 | 10M | 10,000,532 | 1,297 | Snapshot 与评测完成 |
| 阶段评测 | 20M | 20,001,758 | 2,594 | Snapshot 与评测完成 |
| 阶段评测 | 30M | 30,002,588 | 3,891 | Snapshot 与评测完成 |
| 阶段评测 | 40M | 40,004,805 | 5,188 | Snapshot 与评测完成 |
| 最终点 | 50M | 50,000,000 | 6,485 | Checkpoint、Snapshot、导出完成 |

目标 Token 与实际批次边界分别记录，未把 10,000,532 截断或伪写为 10,000,000。每个训练
Checkpoint 约 3.33 GiB，使用临时目录、Payload SHA256、Commit Marker、原子 Rename 和
`LATEST` 更新；阶段模型快照只用于评测与部署，不能替代完整训练状态。

## 5. 五阶段双模式结果

Base 与五个阶段使用相同 `m5-thinking-budget-v2`、Tokenizer、Prompt 和 200 条 M5 Dev。

| 模型点 | Thinking | Non-thinking | 自然闭合 | 强制收束 |
| -- | --: | --: | --: | --: |
| Base | 70.5% | 37.0% | 93.0% | 7.0% |
| 10M | **95.0%** | **47.5%** | 99.0% | 1.0% |
| 20M | 92.5% | 43.5% | 98.0% | 2.0% |
| 30M | 92.0% | 43.0% | 98.5% | 1.5% |
| 40M | 91.0% | 40.0% | 97.0% | 3.0% |
| 50M | 91.5% | 39.0% | 98.5% | 1.5% |

所有阶段 Thinking/Non-thinking 格式率均为 100%，可见推理泄漏均为 0。10M 相对 Base 的
Thinking 提升 24.5pp、Non-thinking 提升 10.5pp；继续训练后两项指标逐步回落，表明该重复
数据预算下出现过拟合。五次评测在训练完成后由冻结快照依次执行，因此这些结果用于模型
选择和回退分析，不声明训练期间执行了在线 Early Stop。

## 6. 导出与失败证据

50M 部署导出目录树 SHA256 为
`9702ea5115c9d0c8b1545502e70cf8a18a283f1ba0f166a30d3d8fa268dade52`；其中
`model.safetensors` 为 1,192,135,096 Bytes，SHA256 为
`53a9813bde64f7ba4cadbe38bc5084a00010f20ad999c5160898414115fd3e0e`。

正式成功前保留了三类真实失败：早期 Micro Batch 未生效、Checkpoint 血缘字段不完整，以及
阶段目标恰好值与实际批次边界不一致。最后一项在 10,000,532 Token fail-closed，修订 Schema
后从干净 Campaign 重跑通过。另一次 GPU 7 驱动失联使等待器退出，随后等待器改为只查询候选
卡并重试瞬时 `nvidia-smi` 错误。失败目录和日志均未删除。

## 7. 验收边界

本报告完成 0.6B 四卡 Full SFT、Exact Resume、Checkpoint、五阶段曲线、部署导出、温控与
失败恢复证据。结果属于 M5 私有开发集；M6 独立冻结套件尚未消费，模型状态保持
`Development`。10M 快照与 8B LoRA 将在 M6 分别接受领域、通用回归、JSON Valid Rate 和
完整血缘门禁。
