# TinyLLM-System 实施计划

## 1. 项目目标

TinyLLM-System 面向校招/初级训练系统工程师岗位，核心证明：

- 原生 PyTorch 单卡、DDP 和 FSDP2 训练能力。
- 完整 Checkpoint、Exact Resume 和失败恢复。
- RTX 3090 拓扑感知与真实性能分析。
- 数据、配置、代码、硬件、评测和模型血缘。
- Qwen 后训练、质量回归和 Candidate Promotion Gate。

项目采用“10 周核心 + 2 周缓冲”。第 10 周完成 M1–M6 并形成
`v0.6.0-rc.1`；M3 有真实 DDP 证据后即可开始投递。M7/M8、ZeRO-3、MLflow、
V100 和 TinyGPT-350M 不阻塞核心版本。

详细固定基线见 [ADR-0003](docs/adr/0003-career-release-baselines.md)。

## 2. 不可变工程原则

每个里程碑必须经过：

```text
设计契约 → 最小接口 → 单元测试 → Smoke Test → 失败路径
→ 集成测试 → 真实报告 → 文档与 Issue 同步 → PR 合并
```

代码完成但缺少真实 Smoke、失败路径或报告时，状态保持 `IN_PROGRESS`。
正式实验只从 Schema 校验后的 YAML 启动；所有性能、质量和恢复结果来自真实运行。

## 3. 平台与资源策略

主平台是 10 × RTX 3090 24GB：BF16 优先，可启用 TF32，通常不使用 GradScaler。
正式扩展实验固定 1/2/4/8 卡；共享开发时可显式选择任意空闲卡，例如 GPU 4–9，
但动态分组不替代受控扩展结果。10 卡仅用于边界对照。

8 × V100 32GB 是条件性辅助平台：只允许 FP16 + GradScaler，不允许 BF16/TF32。
没有连接方式和真实 Smoke 前，V100 不进入发布声明。

策略顺序固定为：

```text
单卡正确性 → DDP → FSDP2 → 可选 ZeRO-3
```

DDP 用于完整训练状态可被单卡容纳时的吞吐扩展，不合并显存。FSDP2 用于参数、
梯度、优化器分片和原生分布式 Checkpoint。ZeRO-3 仅在 FSDP2 通过后进行同条件对照。

## 4. 统一接口与事实源

核心 CLI：

```text
tinyllm doctor
tinyllm data prepare|inspect
tinyllm train
tinyllm run list|show|reproduce
tinyllm benchmark train
tinyllm eval
tinyllm compare
tinyllm promote
```

缓冲阶段增加 `tinyllm plan`、`tinyllm serve`、`tinyllm benchmark inference`。
所有命令支持稳定 `--json`。退出码为 0 成功、2 输入/配置错误、3 Preflight 失败、
4 训练失败、5 Checkpoint/Resume 完整性失败、6 评测失败或门禁拒绝。

默认私有 Artifact Store：

```text
/data/yujielun/tinyllm/
├── cache/
├── datasets/
├── models/
├── runs/
└── registry/
```

Run ID 为 `<UTC>-<slug>-<resolved-config-hash8>-<random4>`。每个 Run 保存
`run.json`、`events.jsonl`、原始/解析配置、环境、硬件、指标、Checkpoint、评测和
导出。JSON/JSONL 是事实源；M6 的 SQLite 是可重建索引；MLflow 只可作为投影。

## 5. Milestone 0：专业化基础与硬件体检

状态：`COMPLETE`。Week 1 专业化迁移作为 M1 前置治理批次独立审查。

已完成 M0 输入：根文档、Python 骨架、`tinyllm doctor`、RTX 3090/CUDA/BF16
体检、1/2/4/6 卡 NCCL 正确性、失败路径、报告和远程 CI。

专业化迁移输出：Apache-2.0、英文主 README、公开脱敏政策、Typer/Pydantic v2、
Run/Artifact/Checkpoint Schema、JSON Schema Snapshot、依赖 Profile、覆盖率与审计 CI、
仓库治理模板。

完成条件：独立 PR 通过 CI、审查并 Squash 合并；不以“代码在分支存在”视为完成。

## 6. Milestone 1：确定性单卡训练与 Exact Resume（Weeks 2–3）

状态：`COMPLETE`。真实验收见 [M1 Acceptance](reports/m1/m1_acceptance.md)，发布版本为
`v0.1.0-alpha.1`。

输入：M0 环境、TinyGPT-Debug 模型、Toy Dataset、冻结 Schema 和错误码。

主要工作：

- 原生 Trainer、AdamW、Warmup/Cosine Scheduler。
- Gradient Accumulation、Gradient Clipping、NaN/Inf 保护和结构化 Metrics。
- 原子 Checkpoint、SHA256、Retention、自动 Resume、Stateful Sampler。
- 保存模型、优化器、Scheduler、GradScaler、Step/Epoch、全部 RNG、Sampler Cursor、
  数据/配置/Git/环境/World Size。
- Exact、Warm、Transfer Resume 显式分离。

验证：CPU Loss 下降；CPU Exact Resume 的 Batch/Loss/LR/参数/状态逐位一致；3090 BF16
容差先由无中断重复基线确定；真实 SIGTERM/SIGKILL 恢复；坏 Checkpoint、磁盘不足、
配置漂移和不兼容 World Size 被拒绝。

输出：`tinyllm train`、完整 M1 报告、`v0.1.0-alpha.1`。

完成条件：一条命令执行训练并从中断点可靠恢复；报告与 PR 合并。M1 严格阻塞 M3。

## 7. Milestone 2：数据版本化与评测前置（Week 4）

状态：`COMPLETE`。真实验收见 [M2 Acceptance](reports/m2/m2_acceptance.md)。

输入：ADR-0003 固定数据 revision、统一消息 Schema、Tokenizer revision 和固定 Seed。

主要工作：OASST Ready/Positive/非删除路径过滤；CommitPackFT Python 和许可证 Allowlist；
规范化、过滤、Exact Dedup、按 Tree/Repository 分组切分、Tokenization、Packing、Manifest、
Rejected Samples、许可证统计和训练/评测污染检查。Near Dedup 仅为增强项。

输出：内容寻址数据版本、统计/拒绝报告、固定领域评测集、正式训练前 Baseline Evaluation。

完成条件：相同输入 revision、配置和 Seed 得到相同内容哈希；同源无跨 Split 泄漏；
Manifest 记录输入/输出哈希和许可证。M2 已解除 M3/M4/M5 的数据与 Baseline 前置阻塞；
这些阶段仍须满足各自其余依赖。

## 8. Milestone 3：DDP 与扩展证据（Weeks 5–6）

状态：`IN_PROGRESS`。M3.1 已完成真实 1/2 卡 NCCL/BF16 正确性运行；M3.2 已完成双卡
完整 Checkpoint、Step 6 Exact Resume 和 Step 8 Rank 1 故障恢复。验收见
[M3.1 DDP Correctness](reports/m3/ddp_correctness.md)与
[M3.2 DDP Recovery](reports/m3/ddp_recovery.md)。正式扩展 Benchmark 尚未完成，因此不得
标记 M3 完成或开始 M4。

输入：M1 正确训练/恢复语义、M2 数据版本、TinyGPT-Target-120M 配置。

主要工作：torchrun、DistributedSampler、参数初始化、Global Batch、Loss Reduce、Rank 0
日志、DDP Checkpoint/Resume 和 Rank 退出处理。

Benchmark：正式 1/2/4/8 卡 Strong/Weak Scaling；每组预热 20 Step、测量 100 Step、
独立重复 3 次；保存原始 JSON、中位数/范围、显存、通信、数据等待和 Profiler Trace。
增加 GPU 6–9 同 NUMA与 GPU 4–7 跨 NUMA受控对照，保留并解释异常运行。

输出：可复现扩展报告和 `v0.3.0-beta.1`。

完成条件：DDP 正确性、Global Batch、日志、Resume、异常 Rank 和固定矩阵全部有证据。
M3 阻塞 M4；完成后开始正式投递。

## 9. Milestone 4：FSDP2 分片训练（Week 7）

输入：M1/M2/M3，通过 revision/许可证/依赖 Smoke 的 Qwen3-8B。

主要工作：BF16、Activation Checkpointing、FULL_SHARD、DCP Sharded Checkpoint/Resume、
Peak Memory 和单体 Safetensors 导出。正式验收固定 8 卡；动态 4/6 卡只做正确性 Smoke。

输出：50 Optimizer Step 报告；Step 25 Checkpoint，模拟退出并恢复到 Step 50；分片
Manifest、内存证据和部署导出。

完成条件：状态分片、DCP 恢复和导出全部验证。ZeRO-3 只进入缓冲期，不阻塞 M5/M6。

## 10. Milestone 5：Qwen 正式后训练（Weeks 8–9）

输入：M2 数据和冻结 Baseline，M1 训练核心，M4 分片能力（仅需要的路径）。

Qwen3-0.6B Full SFT：Non-thinking、Assistant-only Loss、BF16、Sequence Length 1024、
Gradient Checkpointing；最低 50M Tokens、最高 100M。每 10M Tokens 检查：验证损失相对
改善至少 0.5%，通用 Dev 回退不超过 2pp，否则停止扩展。每段作业不超过 12 小时。

Qwen3-8B LoRA：BF16 Rank 16/Alpha 32/Dropout 0.05，目标 Attention/MLP Linear；
正式 10M Tokens，最多 30M。只有单卡 Sequence 1024/Micro Batch 1/Gradient
Checkpointing 的 OOM 有证据后才切 NF4 QLoRA。发布 Adapter 和 Model Card，不发布基础权重。

输出：训练/验证曲线、Checkpoint、Adapter、完整血缘、训练前后评测。TinyGPT-350M
和 7B 长期 Full SFT 是挑战项，不是完成条件。

## 11. Milestone 6：评测、晋级与作品交付（Week 10）

输入：冻结 Baseline、M5 候选、版本化评测集和完整血缘。

主要工作：ARC-Easy/HellaSwag/PIQA；300 条领域集；Prompt/Tokenizer/解码配置、原始输出、
评分依据和 Bootstrap 95% CI；Base/Candidate Compare；配置化 Gate 和审计日志。

Candidate Gate：领域至少 +3pp 且 CI 下界 > 0；通用聚合回退 ≤2pp；JSON Valid Rate
≥98%；数据/模型/Checkpoint/环境/评测血缘完整。失败模型保持 Development 并公开回退。

输出：`v0.6.0-rc.1`、英文架构/实验报告、3–5 条只引用真实指标的简历 Bullet、
10 分钟中文 Demo 和公开 Adapter Model Card。

完成条件：只能晋级 Candidate；Production 由 M7 真实推理性能门禁决定。

## 12. Buffer：M7、M8 与增强项（Weeks 11–12）

优先级：

1. 补齐共享 GPU 导致延期的 M1–M6 证据。
2. vLLM 原生 OpenAI API + TinyLLM 血缘/启动包装和推理 Benchmark。
3. 静态内存估算 + 10–20 Step Probe 的最小 `tinyllm plan`。
4. FSDP2 与 ZeRO-3 同条件短对照。
5. 可选 MLflow 投影、GPU Docker 验证和 V100 FP16 Smoke。
6. TinyGPT-350M 挑战。

M7 输出真实 TTFT、延迟、吞吐、显存和失败率门禁后，Candidate 才可晋级 Production。
M8 的估算必须标注为估算，Probe 结果覆盖估算；不实现 ZeRO-2 作为独立核心策略。

## 13. 风险控制

| 风险 | 控制 |
| -- | -- |
| 范围失控 | 核心/缓冲/Future 三层边界；revision 变化走 ADR |
| 3090/10 卡非标准拓扑 | 正式 1/2/4/8；记录 NUMA、温度、频率和背景负载 |
| FSDP2/ZeRO-3 复杂度 | 单卡→DDP→FSDP2；ZeRO-3 不阻塞 |
| Checkpoint 假恢复 | 原子提交、哈希、完整状态、真实中断、坏文件和兼容性拒绝 |
| 数据质量/许可证 | 固定 revision、Allowlist、分组切分、污染检查、拒绝统计 |
| 评测不可信 | 冻结评测、Baseline 先行、CI、失败样例和原始输出保留 |
| 软件版本冲突 | CPU/3090/V100 独立 Profile；每 Run 保存完整环境 |
| 磁盘容量 | 私有 Artifact Store、Preflight、Retention 和 Pin 策略 |
| 消费卡长时稳定性 | 温度/频率监控、≤12h 作业窗口、周期 Checkpoint |
