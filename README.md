# TinyLLM-System

> 面向消费级多 GPU 集群的硬件感知大语言模型训练、评测与部署平台。

TinyLLM-System 的目标不是重复实现一个完整的大模型框架，也不是把若干 Hugging Face 脚本简单拼接起来，而是在 **10×RTX 3090 24GB** 主服务器和 **8×V100 32GB** 辅助服务器上，建立一套可复现、可恢复、可评测、可晋级、可部署的大语言模型实验系统。

项目重点解决以下问题：

1. 如何让任意一次训练都能追溯到唯一的数据、配置、代码和硬件环境。
2. 如何根据 GPU 型号、数量、显存和通信拓扑，选择 DDP、FSDP2 或 ZeRO-3。
3. 如何在多卡训练中实现可靠的 Checkpoint、异常恢复和结果复现。
4. 如何自动比较基础模型与候选模型，并阻止能力回退的模型进入部署阶段。
5. 如何将训练完成的模型转换成可监控、可压测、可回滚的推理服务。

---

## 1. 项目定位

### 1.1 一句话介绍

TinyLLM-System 是一套面向消费级多 GPU 集群的 LLM 生命周期平台，覆盖：

```text
数据准备
  → 训练计划生成
  → 单卡/多卡训练
  → Checkpoint 与恢复
  → 自动评测
  → 模型晋级
  → 推理部署
  → 性能基准测试
```

### 1.2 核心特色

- **硬件感知**：自动识别 RTX 3090、V100 的精度能力、显存和通信拓扑。
- **策略可解释**：给出 DDP、FSDP2、ZeRO-3 的可行性估算和推荐理由。
- **实验可复现**：记录数据版本、配置、Git Commit、依赖版本、随机种子和硬件信息。
- **训练可恢复**：支持单卡和分布式 Checkpoint、自动恢复与损坏检测。
- **评测可门禁**：候选模型只有通过质量与性能门禁后才能晋级。
- **部署可追踪**：线上模型能够追溯到具体 Run、Checkpoint、数据和评测报告。
- **结果不造假**：README 只记录真实跑出的指标，不提前填写虚构结果。

---

## 2. 硬件资源

### 2.1 主服务器

```text
10 × NVIDIA RTX 3090 24GB
总物理显存：240GB
默认精度：BF16
默认训练组：GPU 0–7
评测/服务组：GPU 8
开发/备用组：GPU 9
```

主要用途：

- Qwen 等开源模型 SFT、LoRA，以及后期可选的 DPO。
- DDP、FSDP2、ZeRO-3 分布式训练。
- 多卡推理、Tensor Parallel 与 Data Parallel。
- 新版 PyTorch、Transformers、TRL 和 vLLM 适配。
- 1/2/4/8 卡扩展效率测试。

### 2.2 辅助服务器

```text
8 × NVIDIA V100 32GB
默认精度：FP16
训练可用卡数：按实际权限配置
```

主要用途：

- Volta 架构兼容性验证。
- FP16 与 BF16 路线对照。
- 单卡 32GB 显存实验。
- V100 多卡拓扑和性能对比。
- 部分对显存更敏感、但对新内核依赖较低的任务。

---

## 3. 项目主线

项目分为两条互补路线。

### 3.1 路线 A：TinyGPT 从零预训练

作用是验证训练系统本身，而不是与成熟开源大模型竞争。

建议配置：

| 模型 | 规模 | 用途 |
|---|---:|---|
| TinyGPT-Debug | 1M–5M | CPU/GPU Smoke Test、CI |
| TinyGPT-120M | 约 120M | 单卡训练基线 |
| TinyGPT-350M | 约 350M | 正式预训练主模型 |
| TinyGPT-1B | 可选 | 后期挑战目标 |

主要验证：

- Decoder-only Transformer 正确性。
- FP16/BF16 混合精度。
- DDP 与 FSDP2 的数值一致性。
- Checkpoint 和恢复。
- 数据版本化与实验血缘。
- 1/2/4/8 卡扩展效率。

### 3.2 路线 B：开源模型后训练

这是项目的求职价值主线。

建议分层：

| 模型规模 | 训练方式 | 项目用途 |
|---|---|---|
| 0.5B–1.5B | Full SFT | 快速建立完整闭环 |
| 3B–4B | Full SFT / LoRA | 正式能力实验 |
| 7B–8B | LoRA / QLoRA | 日常领域微调 |
| 7B–8B | FSDP2 / ZeRO-3 Full SFT | 高级系统挑战 |
| 14B | 多卡 LoRA / 推理 | 后期扩展 |

MVP 必须完成：

1. 一个 0.5B–1.5B 模型的 Full SFT。
2. 一个 7B–8B 模型的 LoRA SFT。
3. 一个 7B–8B 模型的 FSDP2 或 ZeRO-3 多卡 Smoke Test。
4. Base 与 Fine-tuned 模型的自动评测对比。
5. 候选模型的部署与压测。

---

## 4. 系统架构

```text
┌─────────────────────────────────────────────┐
│               CLI / REST API                │
│ doctor / plan / train / eval / promote      │
│ reproduce / serve / benchmark               │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│             Experiment Orchestrator         │
│ 配置校验 / 资源规划 / 状态机 / 自动恢复       │
└────────────┬───────────────┬────────────────┘
             │               │
   ┌─────────▼────────┐ ┌────▼────────────────┐
   │ Dataset Registry│ │ Training Runtime     │
   │ 清洗/去重/切分   │ │ Single/DDP/FSDP/ZeRO│
   │ Tokenize/Packing│ │ AMP/Checkpoint       │
   └─────────┬────────┘ └────┬────────────────┘
             │               │
             └───────┬───────┘
                     ▼
           ┌────────────────────┐
           │ Artifact & Run Store│
           │ 配置/日志/模型/数据 │
           │ Checkpoint/Manifest │
           └─────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │ Evaluation Service │
          │ 通用任务/自建评测   │
          │ 回归/格式/性能测试   │
          └──────────┬──────────┘
                     │
             ┌───────▼────────┐
             │ Promotion Gate │
             │ dev/candidate  │
             │ production     │
             └───────┬────────┘
                     │
          ┌──────────▼──────────┐
          │ Inference Gateway   │
          │ Transformers/vLLM  │
          │ OpenAI-compatible  │
          └─────────────────────┘
```

---

## 5. 统一命令设计

### 5.1 M0 本地安装

```bash
make bootstrap-gpu
source .venv/bin/activate
tinyllm --help
tinyllm doctor
tinyllm doctor --distributed --json
```

M0 使用项目隔离 `.venv`，不会修改已有 Conda base。GPU 环境固定为 PyTorch 2.7.1 + CUDA 11.8 wheel；版本依据和实测结果见硬件报告。

```bash
# 环境和拓扑检查
tinyllm doctor
tinyllm doctor --distributed

# 数据准备与注册
tinyllm data prepare --config configs/data/instruction_v1.yaml
tinyllm data inspect --version instruction-v1.0.0

# 生成训练计划
tinyllm plan --config configs/sft/qwen_7b_full.yaml

# 启动训练
tinyllm train --config configs/pretrain/tinygpt_350m.yaml
tinyllm train --config configs/sft/qwen_1_5b_full.yaml

# 自动评测
tinyllm eval --run-id <run-id> --suite configs/eval/default.yaml

# 基线比较
tinyllm compare --baseline <run-id-a> --candidate <run-id-b>

# 模型晋级
tinyllm promote --run-id <run-id> --stage candidate
tinyllm promote --run-id <run-id> --stage production

# 复现实验
tinyllm reproduce --run-id <run-id>

# 启动推理
tinyllm serve --model production --backend transformers

# 性能测试
tinyllm benchmark --profile configs/benchmark/rtx3090_single.yaml
```

---

## 6. MVP 范围

### 必做

- 配置驱动的训练入口。
- Dataset Manifest 和数据哈希。
- TinyGPT-Debug 与 TinyGPT-120M。
- 3090 单卡 BF16 训练。
- DDP 1/2/4/8 卡训练。
- 分布式 Checkpoint 和恢复。
- Qwen 小模型 Full SFT。
- Qwen 7B 级 LoRA SFT。
- FSDP2 或 ZeRO-3 Smoke Test。
- 通用评测和自建评测。
- Promotion Gate。
- OpenAI-compatible 推理 API。
- P50/P95、TTFT、Tokens/s 和显存测试。
- Run ID 级别实验血缘。

### 暂不做

- 自研 CUDA Kernel。
- 自研 FlashAttention。
- 完整实现 vLLM。
- MoE 正式训练。
- 多节点训练。
- Pipeline Parallel。
- 自研 Tensor Parallel。
- 完整 RLHF。
- 大型 Kubernetes 平台。
- 多租户计费。
- 复杂前端管理系统。

---

## 7. 里程碑

| 阶段 | 目标 | 关键产出 |
|---|---|---|
| M0 | 文档与硬件体检 | 文档骨架、GPU Profile、NCCL 报告 |
| M1 | 单卡 Debug 闭环 | TinyGPT-Debug、Checkpoint、Resume |
| M2 | 数据版本化 | Manifest、去重、切分、Packing |
| M3 | DDP 扩展 | 1/2/4/8 卡报告 |
| M4 | FSDP2/ZeRO-3 | 7B 分片训练 Smoke Test |
| M5 | 正式训练 | TinyGPT-350M、1.5B Full SFT、7B LoRA |
| M6 | 自动评测与晋级 | Eval Report、Promotion Gate |
| M7 | 推理与压测 | OpenAI API、多卡推理、性能报告 |
| M8 | 自动策略规划 | `tinyllm plan`、显存探测、策略推荐 |

详细内容见 [PLANS.md](PLANS.md)。

---

## 8. 真实验收标准

项目不以“支持了多少框架”作为完成标准，而以可验证结果为准：

- 任意 Run 都能追溯到数据、配置、代码、环境和硬件。
- 模拟中断后可从有效 Checkpoint 恢复。
- DDP 的 Global Batch 和单卡基线保持一致。
- 1/2/4/8 卡均生成真实吞吐和扩展效率报告。
- 训练完成后自动触发评测。
- 未通过门禁的模型无法晋级。
- 部署模型可以追溯到训练 Run。
- 推理服务输出 P50/P95、TTFT、Tokens/s、显存和错误率。
- 所有展示指标均来自可复现脚本，不手工编造。

---

## 9. 文档导航

- [AGENTS.md](AGENTS.md)：AI Agent 和开发者协作规则。
- [PLANS.md](PLANS.md)：完整里程碑和验收标准。
- [TASKS.md](TASKS.md)：可执行任务清单。
- [docs/product_scope.md](docs/product_scope.md)：产品边界与目标用户。
- [docs/architecture.md](docs/architecture.md)：系统架构。
- [docs/hardware_strategy.md](docs/hardware_strategy.md)：3090/V100 资源规划。
- [docs/dataset_contract.md](docs/dataset_contract.md)：数据格式和版本契约。
- [docs/training_design.md](docs/training_design.md)：训练和分布式设计。
- [docs/evaluation_spec.md](docs/evaluation_spec.md)：评测和门禁。
- [docs/experiment_lineage.md](docs/experiment_lineage.md)：实验血缘。
- [docs/inference_design.md](docs/inference_design.md)：推理服务设计。
- [docs/benchmark_plan.md](docs/benchmark_plan.md)：性能测试计划。
- [docs/resume_alignment.md](docs/resume_alignment.md)：简历能力对齐。
- [docs/doctor_contract.md](docs/doctor_contract.md)：M0 环境体检命令与输出契约。
- [docs/nccl_test_plan.md](docs/nccl_test_plan.md)：M0 NCCL 测试矩阵与结果记录规则。
- [reports/hardware/rtx3090_inventory.md](reports/hardware/rtx3090_inventory.md)：主服务器真实硬件与 BF16 报告。
- [reports/hardware/nccl_topology.md](reports/hardware/nccl_topology.md)：1/2/4/6 卡 NCCL 与拓扑报告。
- [reports/m0/m0_acceptance.md](reports/m0/m0_acceptance.md)：M0 验收状态与剩余阻塞项。

---

## 10. 当前状态

```text
项目阶段：M0 已完成；下一阶段进入 M1 单卡 Debug 闭环与 M2 数据版本化
代码状态：最小 Python 骨架和 tinyllm doctor 已实现
硬件体检：CUDA/BF16、1/2/4/6 卡 NCCL 已实测；开发 Smoke 按实时空闲卡选择
Benchmark：仅记录 M0 真实 NCCL 原始结果；训练和推理 Benchmark 仍为 TBD
默认主平台：10×RTX 3090 24GB
辅助平台：8×V100 32GB
标准训练组：GPU 0–7（M3 正式 1/2/4/8 卡基线）；共享状态下动态选择空闲卡
```
