# 训练系统设计

## 1. 训练配方

所有正式训练必须由 YAML 配置启动。

```yaml
run:
  name: qwen_1_5b_sft_v1
  seed: 42

model:
  name_or_path: model-path
  revision: immutable-model-commit
  gradient_checkpointing: true

tokenizer:
  name_or_path: tokenizer-path
  revision: immutable-tokenizer-commit
  chat_template_revision: v1

data:
  dataset_version: instruction-v1.0.0
  max_length: 2048
  packing: true

training:
  mode: full_sft
  epochs: 3
  learning_rate: 2.0e-5
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  max_grad_norm: 1.0

distributed:
  strategy: fsdp2
  world_size: 4

precision:
  dtype: bf16

checkpoint:
  save_steps: 200
  keep_last: 2
  resume: auto

evaluation:
  eval_steps: 200
  suites:
    - default
```

## 2. Global Batch

```text
global_batch =
micro_batch
× gradient_accumulation
× data_parallel_world_size
```

训练启动前必须打印并校验。

## 3. 策略接口

```text
TrainingStrategy
├── SingleStrategy
├── DDPStrategy
├── FSDP2Strategy
└── DeepSpeedStrategy
```

统一能力：

- setup。
- wrap_model。
- backward。
- optimizer_step。
- save_checkpoint。
- load_checkpoint。
- reduce_metrics。
- teardown。

## 4. Checkpoint 内容

- 模型。
- 优化器。
- 调度器。
- GradScaler。
- Step/Epoch。
- RNG。
- Sampler。
- 配置快照。
- 数据版本。
- Git Commit。
- 软件环境。
- World Size。
- Checksum。

单卡/DDP 保存完整 PyTorch 训练状态；FSDP2 使用
`torch.distributed.checkpoint` 分片。写入临时目录后必须校验文件清单和 SHA256，
原子 Rename，最后原子更新 `LATEST`。普通滚动点保留最近两个；中断点、最佳点和
最终点永久 Pin。Safetensors 只用于部署导出，不得冒充完整训练 Checkpoint。

## 5. 恢复语义

分为：

- Exact Resume：尽可能恢复到同一训练位置。
- Warm Resume：只加载模型权重。
- Transfer Resume：加载部分权重用于新任务。

用户必须显式知道当前恢复模式。

## 6. 异常处理

记录：

- OOM。
- NaN/Inf。
- NCCL Timeout。
- 数据错误。
- Checkpoint 损坏。
- 磁盘空间不足。
- 进程退出。

OOM 不能无限自动重试，最多执行有限次 Micro Batch 回退。

## 7. 训练回调

第一阶段支持：

- LoggingCallback。
- CheckpointCallback。
- EvaluationCallback。
- EarlyStopCallback。
- MemoryCallback。
- ThroughputCallback。
- NaNGuardCallback。

## 8. 模型路线

### TinyGPT

- 用于验证系统。
- 不追求 SOTA。
- 核心顺序为 Debug 和 Target-120M；350M 只在 M1–M6 完成后进入挑战队列。

### 开源模型

- 小模型 Full SFT。
- 7B LoRA。
- 8B FSDP2 Full SFT Smoke Test。
- DPO 后置。

固定模型 revision、训练预算和回退条件见
[ADR-0003](adr/0003-career-release-baselines.md)。
M4 的四卡 FSDP2、DCP 恢复、数据视图和失败门禁见
[M4 FSDP2 分片训练契约](m4_fsdp2_contract.md)。
