# M5 Qwen3 双模式正式后训练契约

## 1. 目标与边界

M5 验证同一套版本化数据、原生 PyTorch 训练、Checkpoint/Resume 和评测血缘能否同时支撑：

- Qwen3-0.6B Full SFT；
- Qwen3-8B BF16 LoRA；
- 显式 Thinking 与 Non-thinking 双模式。

固定模型保留原生 GQA。M5 不实现 MLA、RLHF、偏好优化、过程奖励模型、推理服务或自研
KV Cache。训练完成不等于晋级 Candidate；M6 才执行正式 Promotion Gate。

## 2. 不可变身份

| 路线 | Repository | Revision | Attention | 初始策略 |
| -- | -- | -- | -- | -- |
| Full SFT | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | GQA 16 Query / 8 KV | BF16 Full SFT |
| LoRA | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | GQA 32 Query / 8 KV | BF16 LoRA |

两条路线都要求 `trust_remote_code=false`、Sequence Length 1024、Assistant-only Loss 和
显式模式选择。Qwen3-8B LoRA 固定 Rank 16、Alpha 32、Dropout 0.05，覆盖 Attention/MLP
Linear。只有单卡 BF16 LoRA 的受控 Probe 保存 OOM 证据后，才允许建立单独的 NF4 QLoRA
配置身份。

## 3. 双模式数据契约

M2 的 `m2-sft-v1-f82ff32e` 和 `qwen3-chatml-nonthinking-v1` 是只读父数据。M5 新增：

- `m5-reasoning-pilot-v1-*`：配比消融使用的私有 Pilot 数据；
- `m5-dual-sft-v1-*`：由父数据和经过验证的 Thinking 数据组成的不可变正式版本；
- `qwen3-chatml-thinking-v1`：无 Tool 消息的原生 Qwen3 Thinking 子集。

Thinking Assistant 内容固定渲染为：

```text
<|im_start|>assistant
<think>
{reasoning_content}
</think>

{final_answer}<|im_end|>
```

Assistant Header 不监督；`<think>`、可见推理轨迹、`</think>`、最终答案和 `<|im_end|>`
全部监督。System/User 内容全部 Mask。空 Think、空最终答案、多个 Think 块、无法验证和超过
1024 Token 的候选必须拒绝，不能截断后接受。

首版只覆盖 Python、Linux、JSON、YAML/TOML 配置和结构化日志诊断。任务由规则模板生成
标准答案，固定 Qwen3-8B Thinking 模式最多生成两个轨迹候选，接受第一个通过确定性
Verifier 的候选。M5 v1 不执行模型生成的任意 Python 或 Shell 代码。

数据按生成器模板族分组切分；英文/中文 Token 目标为 70%/30%。Teacher Revision、生成参数、
Seed、Verifier、输入/输出哈希、拒绝原因和污染结果必须进入 Manifest。重新调用随机生成器
不要求跨 CUDA 环境逐位一致；一旦原始生成 Artifact 被接收，后续规范化、验证、Tokenization、
Packing 和注册必须可确定性重建。

### 3.1 M5.1 已冻结实现

M5.1 使用 `configs/data/m5_reasoning.yaml` 冻结 Task、Teacher、Verifier、Rejected Record、
Pilot/Dev 污染检查和 Dataset Manifest v1。200 条 Dev 的内容身份为
`m5-reasoning-dev-v1-3eb153c2`，五类各 40 条，英文 140 条、中文 60 条。Pilot、Dev 和 Teacher
Sampling 使用互异 Seed；任意 Exact Prompt 或 Template Family 交叉都会阻断构建。

真实离线 Qwen3-8B Teacher Smoke 已验证加载、Thinking 生成、首个通过候选选择和公开脱敏
证据。该 Smoke 只有一个接受样本，不代表正式 Pilot 规模或模型质量。详见
[M5.1 中文报告](../reports/m5/m5_reasoning_data.md)。

上述 Dev 身份只属于 M5.1 历史证据。M5.2 在任何 Candidate 训练前运行 100 条 Teacher Pilot
时发现，Config、Linux 和 Log 三类 Prompt 使用 `"code"` 作为未定义占位值，Teacher 会合理
地复制占位值或输出语义同义词，导致 100 条中只有 37 条通过 Exact Verifier。该结果不是模型
质量结论，而是评测/数据任务定义存在客观歧义。

M5.2 因此以显式版本 `task_contract_version=label_vocabulary_v2` 修订 Prompt：三类分类任务均
列出封闭标签集合，但不改变期望标签、Verifier、采样参数、规模或门禁。修订后的 Dev 身份为
`m5-reasoning-dev-v1-53ddf557`；旧 Dev、旧 Base 和失败 Pilot 全部保留，但不得与新 Candidate
混合比较。此修订发生在 Candidate 训练之前，不能根据后续 Candidate 结果再次修改。

## 4. 配比与训练门禁

正式数据配比不能凭经验指定。先在不接触 M6 冻结测试指标的情况下，对 Thinking Token
比例 0%、30%、50% 各执行 1M Supervised Tokens、两个固定 Seed 的 0.6B 消融。选择顺序：

1. Non-thinking Dev 回退不超过 2pp；
2. Thinking 格式有效率至少 99%；
3. 最大化 Thinking Final-answer 分数；
4. 差异不足 1pp 时选择 Thinking 比例更低者。

M5 Reasoning Dev 固定为 `m5-reasoning-dev-v1-53ddf557` 共 200 条，五类任务各 40 条，每类
28 条英文、12 条中文，并与 Train
按模板族隔离。它只用于 M5 配比选择，不进入 M6 最终发布指标。

### 4.1 M5.2 冻结执行参数

M5.2 将私有 Pilot 固定扩展为 100 个输入任务：五类各 20 条，每类 14 条英文、6 条中文。
Teacher 仍使用固定 Qwen3-8B Thinking 和最多两个候选；只有接受率至少 80%，且五类均有
通过样本时，Pilot 才通过扩展门禁。被拒绝任务、候选和原因全部保留，不用合成答案填补。
扩容任务使用独立配置 `configs/data/m5_reasoning_label_vocabulary_v2.yaml` 和 Pilot Task Seed
`20260728`，同时保持冻结 Dev Seed 不变。原 M5.1 Seed `20260723` 在 100 条扩容预跑中产生
一条 Python Exact Prompt 碰撞，因此被污染门禁诚实拒绝；随后无碰撞的 Placeholder v1 Pilot
又因上述标签歧义仅接受 37 条。两个失败均保留，不用于训练。混合构建器会重新执行 100 条、
80% 接受率和五类覆盖门禁，不能只依赖公开摘要。

三份消融数据各包含精确 1,000,000 个移位后实际参与 Causal LM Loss 的 Assistant Token。
Thinking 目标分别是 0、300,000 和 500,000 Token，剩余来自只读 M2 Train。配比按 Token
而不是样本数计算；为精确达到预算，只允许在最后一条序列尾部追加 Label Mask，并在
Manifest 中记录复用和部分 Mask 次数。数据构建 Seed 固定为 `20260725`。

训练 Seed 固定为 `42` 和 `20260727`。每个消融臂使用单卡 BF16、Micro Batch 4、梯度累积
2、AdamW、Learning Rate `2e-5`、Weight Decay `0.01`、50K Token Warmup、Token-indexed
Cosine 和 Gradient Clipping 1.0；每 500K Token 保存一次完整训练 Checkpoint，保留最近两个，
中断点和最终点 Pin。Checkpoint 保存模型、Optimizer、RNG、数据顺序、序列游标、配置、
数据身份和 Git 身份；恢复时任一身份漂移都拒绝 Exact Resume。

训练前 Base 和六个训练结果都只在 `m5-reasoning-dev-v1-53ddf557` 上分别运行 Thinking 与
Non-thinking。Thinking 固定 Seed `20260726`、Temperature 0.6、Top-p 0.95、Top-k 20、
最大 896 New Tokens；Non-thinking 使用 Greedy 和最大 128 New Tokens。批大小固定为 4。
该评测不读取 M6 冻结结果。实现配置见
[`configs/eval/m5_reasoning_dev.yaml`](../configs/eval/m5_reasoning_dev.yaml)。

### 4.2 M5.2 实际选优结果

六组 1M Supervised Token 训练和冻结双模式评测均已真实完成。0%、30%、50% 三个配比都
通过 Non-thinking 回归门禁；两个 Seed 的 Thinking 格式有效率分别为：

- 0%：0.0% / 0.0%；
- 30%：95.5% / 97.0%；
- 50%：96.0% / 92.5%。

三个配比都未达到每个 Seed 至少 99%的冻结门槛。预注册选择器返回
`status=no_eligible_arm` 和退出码 6，没有产生正式 Thinking 配比。M5.2 实验执行已收口为
门禁拒绝，M5.3 长程 Full SFT 在新的格式可靠性修正批次通过同一门槛前保持阻塞。

本轮完整结果见
[M5.2 中文报告](../reports/m5/m5_ablation_selection.md)和
[机器可读选优结果](../reports/m5/raw/m5_ablation_selection.json)。本节记录实际结果，
不改变训练前冻结的选择顺序、解码参数或门槛。

### 4.3 M5.2-R1 格式可靠性修正

对 30%/50% 四个 Candidate 的私有原始响应完成逐条复算后，38 条 Thinking 格式失败全部
属于开标签存在但闭标签缺失；其中 35 条达到 896 Token 生成上限，3 条在 EOS 时未闭合。
公开聚合见
[失败分析](../reports/m5/raw/m5_format_failure_analysis.json)。

R1 固定从训练侧已验证 Pilot 中选择每类 8 条短而完整的样本：每类英文 5 条、中文 3 条，
移位后 Assistant 监督不超过 512 Token。总训练预算仍为 1M Token 和 30% Thinking，其中
700K 来自 M2 Non-thinking、150K 来自完整 Pilot Thinking、150K 来自短格式修复池。数据
版本为 `m5-format-repair-mixture-v1-1396b60b`，Manifest SHA256 为
`2467b5dce0d909b865b73219d2f608bdbc9c6fcc1bb09b93c5ebea8a7b60bd0e`。

两个训练 Seed、优化参数、Base、Dev、解码设置和门禁全部沿用 M5.2。两个 Seed 的
Non-thinking 回退都不超过 2pp 且 Thinking 格式率都至少 99%时，R1 才能解锁 M5.3。
两组真实训练和冻结评测已经完成：Non-thinking 分数为 64.0%/66.0%，Thinking 格式率为
94.5%/93.5%，Thinking 分数均为 93.0%。自动 Gate 以
`gate_reason=thinking_format_gate_failed` 和退出码 6 拒绝 R1。24 条格式失败全部达到
896 Token 上限且缺少闭标签；短完整样本复用策略未改善格式率，M5.3 继续保持阻塞。详见
[M5.2-R1 中文报告](../reports/m5/m5_format_repair_r1.md)和
[机器可读 Gate](../reports/m5/raw/m5_format_repair_gate.json)。

### 4.4 M5.2-R2 长度反事实诊断

R2 在设计阶段只诊断 896 Token 上限，不训练新模型。它使用原 R1 模型、失败 Item 所属完整
Batch、任务顺序和 RNG，先以 896 重放并要求 Response SHA256 与原结果一致，再以 1536
重放并要求前 896 个 Token ID 完全相同。随后在 1024、1280、1536 三个截断点复算格式和
Final-answer 指标。

只有两个 Seed 在同一个上限都投影达到至少 99%时，才支持建立新的 Evaluation Protocol；
新协议仍需完整重跑 Base、六个 M5.2 Candidate 和两个 R1 Candidate。若 1536 仍不达标，
后续训练修正转向 Config/Log 的新颖简洁 Teacher Trace。若重放不一致，R2 结果无效并优先
处理环境和 RNG 可复现性。完整协议见
[M5.2-R2 诊断设计](m5_r2_diagnostic_design.md)。

D1 离线分析已完成：两个 Seed 的 24 条失败全部触及 896 Token 上限，失败输出的平均重复
8-gram 比例为 25.82%/24.95%，同任务族格式有效对照为 1.28%/1.24%。该结果同时支持长度
触顶和重复生成风险，尚不能替代 D2 GPU 重放结论。若 D2 证明两个 Seed 在 1280 都达到
99%，已条件允许建立上限为 1280 的新协议；生效前必须完成全部 Base/Candidate 重跑和性能
成本评估。详见[M5.2-R2 中文报告](../reports/m5/m5_r2_diagnostic.md)。

D2 已在两个 Seed 上完成真实 RTX 3090 重放。896 输出和 1536 前缀分别以 40/40、36/36
全部精确一致；但两个 Seed 在 1536 的投影格式率只有 98.0%和 96.5%。冻结选择器返回
`length_ceiling_insufficient`、`formal_protocol_changed=false` 和退出码 6。正式上限不改为
1280 或 1536，后续修正必须面向 Config/Log 的高重复和过长推理，不能继续把增加解码预算
作为主要方案。

### 4.5 M5.2-R3 Config/Log 定向修复

R3 保持 R1 的 1M Token、30% Thinking、训练参数、双 Seed、冻结 Dev 和 99%格式门禁，只将
150K Repair Thinking 从五类通用短样本替换为新的 Config/Log 简洁 Trace。现有 Pilot 的真实
CPU 审计显示，19 条 Config 和 20 条 Log Trace 中分别只有 2 条和 4 条满足 192 Token、
低重复和唯一性规则，远低于每类 80 条的来源门禁，因此禁止直接复用。

40 任务 R3-P0 已完成固定任务生成器、污染检查、严格 Schema、CPU 合成契约 Smoke、失败
路径和真实 Qwen3-8B Teacher Pilot。真实实验只接受 10/40 条：Config 5 条、Log 5 条；
英文 9 条、中文 1 条。52 个候选因推理超过 192 Token 被拒，11 个候选触及 384 Token
生成上限。两个任务族均未达到 14 条及 10/4 语言门禁，因此正式 240 条扩展保持阻断。

下一批只允许建立新的 Prompt 控制诊断，保持 Teacher、采样、192 Token Trace 上限、384
Token 生成上限和污染规则不变；新 Prompt 必须使用独立版本与身份。正式评测上限继续保持
896，M5.3 在 R3 双 Seed Gate 通过前继续阻塞。
完整协议见[R3 定向修复设计](m5_r3_targeted_repair_design.md)，真实审计见
[R3 中文报告](../reports/m5/m5_r3_targeted_repair.md)，P0 实验状态见
[R3-P0 中文报告](../reports/m5/m5_r3_p0.md)。

P0-R1 冻结为 `m5-r3-p0-r1-v1`：新 Task/Template/Case Reference、Task Seed `20260801`
和 Generation Base Seed `20260802` 与父 P0 分离，父 P0 公开结果哈希写入配置并在运行前
校验。Prompt 唯一变化是要求“先给结论、引用一处直接证据、不讨论其他标签”；Teacher、
采样分布、候选数、192/384 Token、任务分布、Verifier、污染检查和 14/10/4 Gate 均保持
不变。

真实 GPU Pilot 接受 12/40 条：Config 4 条（英文 3、中文 1），Log 8 条（英文 7、中文 1）。
46 个候选超过 192 Token，14 个候选触及 384 Token 生成上限。两个任务族均未通过来源
门禁，P0-R1 状态为 `COMPLETED_GATE_REJECTED`；正式 240 条扩展和 R3 训练继续阻断。
本结果结束同类 Prompt-only 修正，下一步必须重新审查 Teacher 来源策略。完整证据见
[P0-R1 中文报告](../reports/m5/m5_r3_p0_r1.md)。

Teacher 来源策略审查已选择两阶段 `solve → compress` 作为
`m5-r3-p1-two-stage-v1`，确定性规则 Trace 仅作控制组。该审查只授权实现 P1
Task/Artifact/Verifier 与 CPU 契约；在这些接口通过前，GPU Pilot 保持阻断。即使 P1
GPU Pilot 通过，也必须先完成人工内容审查，才能另行决定 240 条来源扩展；Mixture 和训练
仍需后续独立门禁。详见
[Teacher 来源策略设计](m5_r3_teacher_source_strategy.md)。

P1 两阶段运行接口、CPU 合成契约和真实单卡 Qwen3-8B Pilot 已完成。真实运行接受
11/40 条：Config 5 条、Log Diagnosis 6 条；29 条拒绝由缺少证据锚点、讨论其他标签、
Solver 长度上限和 Compressor JSON 失败组成。四路污染和规则控制通过，但两个任务族均
未达到 14/10/4 门禁，因此正式扩展、Mixture 和训练继续阻断。P1 证据见
[P1 中文报告](../reports/m5/m5_r3_p1.md)。

0.6B 正式路径先做单卡 BF16 Smoke，再用四张通过 Preflight 的 RTX 3090 执行 DDP。最低
50M Tokens、最高 100M，每 10M 执行继续训练门禁；每 2M 保存滚动 Checkpoint。8B 路线先做
单卡 Memory Probe，再训练最低 10M、最高 30M Tokens；每 1M 保存滚动 Checkpoint、每 2M
执行 Dev 评测。每段作业不超过 12 小时。

## 5. 配置与恢复

新配置使用 `config_kind=qwen_sft` 和独立 `schema_version`，不改变 M1 Schema。至少记录模型
Revision、`attention_architecture=gqa`、Full/LoRA/QLoRA 身份、数据与混合 Manifest、模式、
Token 预算、精度、World Size、Checkpoint 策略和评测版本。CLI 只允许运行时覆盖物理 GPU、
输出根目录和 Resume 模式。

0.6B DDP Exact Resume 必须保持 World Size、模型、数据版本、配比和优化配置兼容。8B LoRA
Checkpoint 必须包含 Adapter、Optimizer、Scheduler、RNG、Sampler Cursor、基座 Revision、
数据版本和配置哈希。部署导出与训练 Checkpoint 分离；8B 只导出 Adapter Safetensors。

## 6. M5 完成条件

M5 只有在以下证据全部合并后才能标记完成：

1. 双模式设计、Schema、数据 Manifest、拒绝统计和污染报告；
2. 训练前双模式 Baseline 与配比消融；
3. 0.6B Full SFT 的真实训练、恢复、曲线和 Checkpoint；
4. 8B LoRA 的真实 Probe、训练、恢复、Adapter 和 Model Card；
5. OOM、NaN/Inf、坏 Checkpoint、磁盘不足、数据漂移、错误 World Size 和进程退出失败路径；
6. 中文主验收报告、英文公开摘要和完整血缘。

结果没有质量提升时可以作为诚实的 M5 系统实验完成，但模型保持 `Development`。只有 M6
满足 Thinking 提升、Non-thinking/通用回归、JSON Valid Rate 和血缘门禁后才能晋级。
