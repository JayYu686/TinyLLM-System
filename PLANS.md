# TinyLLM-System 初步完整实施计划

---

## 0. 项目目标

构建一套运行在 10×RTX 3090 主服务器和 8×V100 辅助服务器上的硬件感知 LLM 系统，覆盖：

```text
数据版本化
→ 训练计划生成
→ 单卡与分布式训练
→ Checkpoint 与恢复
→ 自动评测
→ 模型晋级
→ 推理服务
→ 性能压测
```

项目以真实可复现结果为核心，不以堆叠框架数量为目标。

---

# Milestone 0：文档优先初始化与硬件体检

## 目标

建立项目边界、文档体系、硬件 Profile 和最小 Python 工程骨架。

## 主要工作

### 文档

- README.md
- AGENTS.md
- PLANS.md
- TASKS.md
- 产品范围
- 系统架构
- 数据契约
- 训练设计
- 评测规范
- 实验血缘
- 推理设计
- Benchmark 计划
- 简历对齐
- ADR

### 工程骨架

```text
src/tinyllm/
configs/
tests/
scripts/
reports/
```

### 硬件体检

收集：

- GPU 型号、数量、显存。
- Compute Capability。
- CUDA Driver 和 Runtime。
- PyTorch、NCCL 版本。
- GPU 拓扑。
- NUMA 拓扑。
- P2P 可用性。
- NCCL All-Reduce 带宽。
- 本地磁盘读写速度。
- 数据盘容量。

## 输出

- `reports/hardware/rtx3090_inventory.md`
- `reports/hardware/v100_inventory.md`（取得辅助服务器访问方式后补充，不阻塞 3090 主平台的 M0 验收）
- `reports/hardware/nccl_topology.md`
- 硬件 Profile 配置草案。

## 验收

- 新开发者能仅通过文档理解项目目标。
- `tinyllm doctor` 的接口和输出格式确定。
- 能根据实时空闲状态选择显式 GPU 组，并完成至少一个跨 NUMA 多卡 NCCL Smoke。
- 3090 服务器的推荐 8 卡训练组被明确为正式扩展实验基线；其 8 卡实测移至 M3。
- 不包含任何未经运行的性能结果。

---

# Milestone 1：确定性单卡 Debug 训练闭环

## 目标

实现最小可验证训练链路。

## 模型

TinyGPT-Debug：

- Decoder-only。
- RMSNorm。
- RoPE。
- Causal Attention。
- SwiGLU。
- Weight Tying。
- 参数量约 1M–5M。

## 数据

- Toy Vocabulary。
- 固定随机 Token。
- 固定 Seed。
- 可重复采样。

## 功能

- 配置加载与校验。
- 单卡前向和反向。
- BF16 与 FP16 Profile。
- Gradient Accumulation。
- Gradient Clipping。
- Checkpoint 保存。
- Checkpoint 恢复。
- 结构化日志。
- CPU Smoke Test。
- GPU Smoke Test。

## 必须验证

1. Loss 能下降。
2. 相同 Seed 下前若干 Step 结果可重复。
3. 中断后恢复不重复 Step。
4. 恢复前后学习率连续。
5. 3090 使用 BF16。
6. V100 使用 FP16 + GradScaler。

## 输出

- `reports/m1/debug_training_report.md`
- 单元测试。
- 恢复测试。
- 最小 CLI：`tinyllm train`。

## 验收

一条命令完成 100–500 Step 训练，并在模拟中断后恢复。

---

# Milestone 2：真实数据管线与 Dataset Registry

## 目标

让训练数据具备明确格式、版本和血缘。

## 内部数据格式

```json
{
  "id": "sample-000001",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "source": "dataset_name",
  "split": "train",
  "metadata": {}
}
```

## 数据流水线

1. 数据导入。
2. Schema 校验。
3. Unicode 与空白规范化。
4. 长度过滤。
5. 规则过滤。
6. 精确去重。
7. 近似去重。
8. 数据隔离。
9. Tokenization。
10. Sequence Packing。
11. Manifest 生成。
12. 统计报告。

## 输出

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `dataset_manifest.json`
- `statistics.json`
- `rejected_samples.jsonl`

## 验收

- 任意 Run 都只能引用已注册的数据版本。
- 数据内容变化会导致哈希变化。
- 训练、验证、测试在去重后切分。
- 同一版本重复构建得到相同 Manifest。

---

# Milestone 3：DDP 多卡扩展

## 目标

实现单卡可容纳模型的吞吐扩展。

## 工作内容

- `torchrun` 启动。
- DistributedSampler。
- Rank-aware Logging。
- Global Batch 计算。
- 梯度同步。
- 分布式指标聚合。
- 多卡 Checkpoint。
- 失败退出处理。
- 1/2/4/8 卡运行。

## 对比原则

保持以下条件一致：

- 模型。
- 数据版本。
- Global Batch Size。
- Seed。
- 训练 Step。
- 序列长度。

## 指标

- Tokens/s。
- Step Time。
- Peak Memory。
- Scaling Efficiency。
- NCCL 通信占比。
- 数据加载等待占比。

## 验收

- DDP 与单卡 Loss 曲线在合理误差内一致。
- 1/2/4/8 卡报告可自动生成。
- 8 卡异常退出后可恢复。
- 不把 DDP 作为解决单卡模型显存不足的手段。

---

# Milestone 4：FSDP2 优先的分片训练与 ZeRO-3 对照

## 目标

训练单卡无法容纳的模型，并形成策略选择依据。

## FSDP2

实现：

- Transformer Block 自动包装。
- FULL_SHARD。
- Mixed Precision Policy。
- Activation Checkpointing。
- Sharded State Dict。
- Full State Dict 导出。
- Distributed Checkpoint。
- Resume。

FSDP2 是 MVP 的默认分片实现，必须先通过正确性、Checkpoint 和 Resume 门禁。

## ZeRO-3

实现：

- 参数、梯度、优化器分片。
- 配置模板。
- 可选 CPU Offload。
- Checkpoint 恢复。
- 与 FSDP2 的对比接口。

ZeRO-3 在 FSDP2 Smoke Test 通过后进入，不与 FSDP2 同时首发。若时间或兼容性限制，ZeRO-3 对照不得阻塞小模型 Full SFT、7B LoRA 和后续质量闭环。

## Smoke Test

推荐模型：

- 3B 级 Full SFT。
- 7B 级 Full SFT Smoke Test。
- 序列长度先从 512/1024 开始。

## 验收

- 至少生成 DDP 与 FSDP2 的显存与吞吐对比；ZeRO-3 完成后追加同条件对照。
- 7B 级模型完成 20–100 Step 分片训练。
- Sharded Checkpoint 可恢复。
- 可导出单体部署权重或明确记录无法导出的限制。

---

# Milestone 5：正式模型实验

## 目标

形成真正可写入简历的训练结果。

## 实验 A：TinyGPT-350M 从零预训练

目标：

- 验证预训练数据管线。
- 验证长期训练稳定性。
- 验证 DDP/FSDP2。
- 产出 Loss、吞吐、显存和生成样例。

注意：

- 不以高通用能力为目标。
- 训练 Token 数依据资源再确定。
- 先跑短程稳定性实验，再决定正式预算。

## 实验 B：0.5B–1.5B Full SFT

目标：

- 建立开源模型完整后训练闭环。
- 比较训练前后任务能力。
- 生成真实能力和回退报告。

## 实验 C：7B–8B LoRA SFT

目标：

- 验证多卡消费级 GPU 的实用微调。
- 比较 LoRA Rank、序列长度和吞吐。
- 产出可部署 Adapter 或合并权重。

## 实验 D：7B–8B Full SFT

定位：

- 高级挑战。
- 非 MVP 阻塞项。
- 先完成短程 Smoke Test，再评估正式训练价值。

## 验收

至少完成：

- TinyGPT-350M 一次正式训练。
- 一个小模型 Full SFT。
- 一个 7B 级 LoRA SFT。
- 一个 7B 级分片训练 Smoke Test。

---

# Milestone 6：自动评测与模型晋级

## 目标

将“训练完成”与“模型可用”分离。

## 评测层次

### 通用评测

选择少量稳定任务：

- ARC Easy。
- HellaSwag。
- PIQA。
- BoolQ。
- WinoGrande。
- 中文基础任务子集。

### 自建评测

建议 300–500 条：

- Python 解释。
- Linux 命令理解。
- JSON 生成。
- 配置修改。
- 错误日志分析。
- 简短代码补全。
- 无依据问题拒答。

### 系统评测

- JSON Valid Rate。
- Latency。
- TTFT。
- Tokens/s。
- Peak Memory。
- Failure Rate。

## Promotion Gate

阶段：

```text
development → candidate → production → archived
```

门禁检查：

- 必须指标。
- 允许回退上限。
- 性能回归上限。
- 评测完整性。
- 模型和数据血缘完整性。

## 验收

- 训练完成自动触发评测。
- 模型对比报告自动生成。
- 未通过门禁的模型无法晋级。
- 晋级行为具有审计记录。

---

# Milestone 7：推理服务与性能压测

## 目标

将晋级模型部署为可监控服务。

## 后端

- TransformersBackend。
- VLLMBackend。
- MockBackend。

## API

- OpenAI-compatible Chat Completions。
- Streaming。
- Health。
- Model Info。
- Metrics。
- Request ID。
- Error Mapping。

## 单卡测试

- RTX 3090。
- V100。
- FP16/BF16。
- 不同输入输出长度。
- 不同 Batch 和并发。

## 多卡测试

- Tensor Parallel。
- Data Parallel。
- TP×DP 组合。
- 负载均衡。
- 实例故障摘除。

## 指标

- P50/P95。
- TTFT。
- Inter-token Latency。
- Tokens/s。
- Peak Memory。
- 并发吞吐。
- Failure Rate。

## 验收

- candidate 或 production 模型可一条命令部署。
- 推理请求可追溯到模型版本。
- 自动生成标准 Benchmark 报告。
- 3090 和 V100 的结果分开记录。

---

# Milestone 8：硬件感知训练规划器

## 目标

根据模型、序列长度和硬件自动推荐训练策略。

## 命令

```bash
tinyllm plan --config configs/sft/qwen_7b_full.yaml
```

## 输出

- 硬件信息。
- 模型参数估算。
- DDP 显存估算。
- ZeRO-2 估算。
- FSDP2/ZeRO-3 估算。
- 推荐 World Size。
- 推荐精度。
- 推荐 Micro Batch。
- 推荐 Gradient Accumulation。
- 推荐 Activation Checkpointing。
- 风险提示。

## 流程

```text
静态估算
  → 候选策略
  → 10–20 Step Probe
  → 真实峰值显存
  → 自动调整
  → 最终执行计划
```

## 验收

- 对不可行配置明确拒绝。
- 所有估算标注为估算。
- Probe 结果覆盖静态估算。
- 推荐策略能够落地运行。

---

# 项目阶段优先级

## 依赖说明

- M0 完成后，M1 与 M2 可以并行。
- M3 严格依赖 M1 的单卡正确性与恢复语义。
- M4 严格依赖 M1、M2 和 M3；FSDP2 必须先于 ZeRO-3。
- M6 的评测 Schema、版本化与最小 Baseline 适配器应在 M5 正式训练前准备；M6 的模型比较和晋级验收依赖 M5 产出的模型。
- M6 先完成质量门禁，M7 的真实推理结果再补充性能门禁。
- M8 的完整自动规划器依赖 M3、M4 和 M7 的真实数据；M0–M4 只提供必要的静态 preflight，不提前实现自动 Probe 和 OOM 回退。

## 第一优先级

M0 → M1 → M2 → M3

目标：先建立正确、可恢复、可复现的训练系统。

## 第二优先级

M4 → M5 → M6

目标：形成真实模型结果和自动评测闭环。

## 第三优先级

M7 → M8

目标：补齐部署、压测和硬件感知差异化能力。

---

# 风险控制

| 风险 | 应对 |
|---|---|
| 3090 多卡通信慢 | 先测试拓扑，默认使用 8 卡组 |
| 10 卡非标准并行规模 | 8 卡训练，2 卡评测和备用 |
| 7B Full SFT 成本过高 | 只要求 Smoke Test，正式训练可选 |
| 框架版本冲突 | 固定版本矩阵与硬件 Profile |
| Checkpoint 过大 | Sharded Checkpoint、轮换和压缩策略 |
| 数据质量不足 | 先建立小而干净的自建评测集 |
| 项目范围失控 | Future Work 隔离，里程碑门禁 |
| Benchmark 不可信 | 所有指标由脚本自动生成并保留环境信息 |
