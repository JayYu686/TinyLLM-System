# TinyLLM-System 路线图汇总

> GitHub Issues/Milestones 是执行层；本文件只维护里程碑状态、依赖和完成门禁。
> 代码存在但 Smoke、失败路径、真实报告或 PR 合并缺失时，不得标记完成。

执行入口：[Week 1 #3](https://github.com/JayYu686/TinyLLM-System/issues/3)；
[M1 #4–#7](https://github.com/JayYu686/TinyLLM-System/milestone/2)；
[M2 #8–#11](https://github.com/JayYu686/TinyLLM-System/milestone/3)；
[M3 #12–#15](https://github.com/JayYu686/TinyLLM-System/milestone/4)；
[M4 #16–#18](https://github.com/JayYu686/TinyLLM-System/milestone/5)；
[M5 #19–#22](https://github.com/JayYu686/TinyLLM-System/milestone/6)；
[M6 #23–#26](https://github.com/JayYu686/TinyLLM-System/milestone/7)。

## 当前状态

| 批次 | 状态 | 已完成 | 剩余门禁 |
| -- | -- | -- | -- |
| M0 硬件体检 | `COMPLETE` | Doctor、RTX 3090/BF16、1/2/4/6 卡 NCCL、报告、CI | V100/10 卡为非阻塞项 |
| Week 1 专业化基础 | `COMPLETE` | Apache/README/治理、Typer/Pydantic、公共 Schema、CI、PR #27 | 无；后续修改进入 M1 独立 PR |
| M1 单卡闭环 | `IN_PROGRESS` | TinyGPT-Debug、Toy Data、原生 CPU Trainer、Loss Smoke、数值失败路径 | 完整 Checkpoint、Exact Resume、GPU 精度、真实中断、最终报告 |
| M2–M8 | `NOT_STARTED` | 设计文档 | 对应前置里程碑与 Issue |

## Week 1：专业化基础

- [x] 独立审查并 Squash 合并原 M1 foundation PR。
- [x] 采用 Apache-2.0 并修正包元数据。
- [x] 英文 README、公开脱敏规则、贡献/安全/PR/Issue 模板。
- [x] Typer CLI 和 Pydantic v2 严格 Schema。
- [x] 冻结 Run ID、Artifact Layout 和 Checkpoint Manifest v1。
- [x] 导出并校验 JSON Schema Snapshot。
- [x] 建立 CPU、RTX 3090 CUDA 11.8、未来 V100 Profile。
- [x] 增加 ≥85% CPU 核心覆盖率、Schema、链接、公开 Artifact 和 Docker CI。
- [x] 建立 GitHub M1–M6 Milestones 与可审查 Issues。
- [x] 配置 `main` 保护与必需 CI。
- [x] Week 1 Draft PR 通过 CI、完成审查并 Squash 合并。

## M1：原生单卡 Trainer 与 Exact Resume

依赖：Week 1 合并。

执行批次：

1. Trainer/Optimizer/Scheduler/Metrics 与 CPU Loss 下降。
2. 原子完整 Checkpoint、Retention、Stateful Sampler、Exact/Warm/Transfer Resume。
3. CPU 逐位 Exact Resume 和坏文件/磁盘/配置/World Size 失败矩阵。
4. RTX 3090 BF16 基线、SIGTERM/SIGKILL 恢复、真实 M1 报告与预发布。

完成门禁：全部批次合并，CPU/GPU Smoke 和失败矩阵通过，真实报告发布。

## M2：数据版本化与冻结评测

依赖：固定 revision/许可证重新验证；可与 M1 后半段部分并行。

执行批次：导入/Schema → 规范化/过滤/许可证 → Exact Dedup/分组 Split →
Tokenization/Packing → Manifest/Rejected/统计 → 污染检查 → 冻结 Baseline。

完成门禁：同输入/revision/config/seed 内容哈希一致，无同源跨 Split，评测先于正式训练冻结。

## M3：DDP 与正式扩展

严格依赖：M1、M2。执行批次：分布式正确性 → Checkpoint/Resume/Rank Failure →
受控 1/2/4/8 Strong/Weak Scaling → NUMA 对照 → 报告和 `v0.3.0-beta.1`。

完成门禁：每配置预热 20、测量 100、重复 3 次，原始结果与异常说明完整。

## M4：FSDP2

严格依赖：M1–M3。执行批次：FULL_SHARD/Activation Checkpointing → DCP 分片保存 →
Step 25 退出/恢复到 50 → Peak Memory/导出 → 正式 8 卡报告。

完成门禁：固定 Qwen3-8B revision 真实完成 50 Step 和 Sharded Resume。ZeRO-3 不阻塞。

## M5：正式后训练

依赖：M2 数据/Baseline、M1 训练核心；按路径复用 M4。

- Qwen3-0.6B Full SFT：50M Tokens 下限，按 10M 门禁，100M 上限。
- Qwen3-8B LoRA：10M Tokens 正式预算，最多 30M；NF4 仅为有证据的 OOM 回退。

完成门禁：训练/评测/血缘/Checkpoint/Adapter/Model Card 真实且可复现。

## M6：评测、Candidate Gate 与作品

依赖：M2 冻结评测、M5 候选。

执行批次：评测适配 → 300 条领域集 → 原始输出/CI → Compare → Promotion Audit →
英文报告/图表/简历 Bullet/10 分钟 Demo → `v0.6.0-rc.1`。

完成门禁：只晋级 Candidate；阈值满足且完整血缘可反向追溯。

## Buffer：M7/M8 与研究增强

按顺序：补核心证据 → vLLM/M7 Production Gate → 最小 Planner → ZeRO-3 →
MLflow/V100/GPU Docker → TinyGPT-350M。

MoE、自研 KV Cache/Tensor Parallel/FlashAttention/CUDA Kernel、多节点、Pipeline
Parallel、完整 RLHF、Kubernetes、多租户计费和复杂前端不进入本路线图执行层。
