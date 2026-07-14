# M1 单卡确定性训练契约

## 1. 目标与边界

M1 用 TinyGPT-Debug 建立一条可测试、可恢复的单卡训练链路，证明训练系统的正确性，不用于宣称模型质量或吞吐优势。

M1 包含：

- YAML 配置校验。
- Decoder-only TinyGPT-Debug。
- 确定性 Toy Dataset。
- CPU 与单 GPU 前向、反向和短程训练。
- Gradient Accumulation、Gradient Clipping 和 LR Scheduler。
- 完整 Checkpoint、损坏检测和 Exact Resume。
- 结构化日志和真实 Smoke 报告。

M1 不包含 DDP、FSDP2、ZeRO-3、真实数据训练、开源模型微调或推理服务。

## 2. TinyGPT-Debug 冻结配置

默认配置位于 `configs/pretrain/tinygpt_debug.yaml`：

| 字段 | 值 |
|---|---:|
| Vocabulary | 256 |
| Hidden Size | 192 |
| Layers | 4 |
| Attention Heads | 6 |
| Head Dimension | 32 |
| SwiGLU Intermediate | 512 |
| Max Sequence Length | 128 |
| Dropout | 0 |
| Weight Tying | 启用 |

模型使用 RMSNorm、RoPE、Causal Self-Attention、SwiGLU 和 Causal LM Loss。Attention 使用 PyTorch 原生 `scaled_dot_product_attention`；M1 不实现自研 FlashAttention、KV Cache 或 CUDA Kernel。

默认参数量必须落在 1M–5M。测试可使用更小配置缩短 CPU 时间，但不得把测试配置冒充默认模型。

## 3. 配置规则

- 正式入口只接受 YAML。
- 未知字段、缺失字段和类型错误必须失败，不静默采用猜测值。
- `model.vocab_size` 必须等于 `data.vocab_size`。
- `data.sequence_length` 不得超过模型最大长度。
- Hidden Size 必须能被 Attention Heads 整除，Head Dimension 必须为偶数。
- `global_batch = micro_batch_size × gradient_accumulation_steps`；M1 的 World Size 固定为 1。
- 配置加载后保存解析完成的快照；Checkpoint 还需保存原始配置哈希。

## 4. 确定性等级

M1 区分：

1. 数据确定性：相同 Seed 生成相同 Token。
2. 初始化确定性：相同 Seed 生成相同初始参数。
3. Step 确定性：相同环境与配置下，前若干 Optimizer Step 的 Loss 和 LR 一致。
4. Resume 确定性：中断恢复后的下一个 Batch、Global Step、LR 和参数更新与未中断对照一致。

跨 PyTorch、CUDA 或 GPU 架构的逐位一致不属于 M1 承诺；环境差异必须进入报告。

## 5. Step 与梯度语义

- `micro_step`：每次前向和反向。
- `global_step`：每次成功的 Optimizer Step；日志、保存和恢复均以它为主键。
- Loss 在反向前除以 `gradient_accumulation_steps`。
- Gradient Clipping 在累积完成、Optimizer Step 之前执行。
- Scheduler 只在 Optimizer Step 成功后前进。
- 非有限 Loss 或 Gradient 必须失败并记录首个异常 Step，不得静默跳过。

M1 固定使用 AdamW（Betas 0.9/0.999、Epsilon 1e-8）。参数维度大于等于 2 的权重
进入 Weight Decay 组；Norm 等一维参数进入 No-Decay 组。学习率按 Optimizer Step
计算：前 `warmup_steps` 线性升温，剩余 Step 余弦下降；最后一次参数更新仍使用正 LR，
训练完成后才归零。Step 指标记录本次参数更新实际使用的 LR，而不是 Scheduler 为下一
Step 设置的 LR。

结构化 Optimizer-Step 指标至少包含：Schema Version、Global/Micro Step、Epoch、
累积窗口平均 Loss、实际 LR、裁剪前 Gradient Norm、本次是否裁剪、累计预测 Token 数。
Loss、LR 和 Gradient Norm 必须拒绝 NaN/Inf。M1.1 只提供内存 Sink；JSONL 持久化在
Run Store 接入后实现，不能由测试日志冒充事实源。

稳定训练失败码：

- `TRAIN_OUTPUT_INVALID`：模型没有返回标量 Loss。
- `NON_FINITE_LOSS`：反向前发现非有限 Loss。
- `NON_FINITE_GRADIENT`：Optimizer Step 前发现非有限 Gradient/Norm。
- `EMPTY_DATALOADER`：数据无法形成一个完整 Micro Batch。
- `UNSUPPORTED_PRECISION`：当前实现或硬件不支持请求的精度。

## 6. 精度规则

| 环境 | Dtype | GradScaler | TF32 |
|---|---|---|---|
| CPU Smoke | FP32 | 否 | 否 |
| RTX 3090 | BF16 | 否 | 可配置 |
| V100 | FP16 | 是 | 不允许 |

M1 首批只实现和测试 FP32 模型基础。BF16、FP16 与 GradScaler 随训练器批次实现，未运行前不得勾选或宣称通过。

## 7. Checkpoint 与恢复契约

Checkpoint 至少保存模型、优化器、调度器、GradScaler、Global Step、Micro Step、Epoch、Python/NumPy/PyTorch/CUDA RNG、采样器、完整配置、数据标识、Git Commit、环境与完整性元数据。

保存流程必须先写临时目录、完成校验后原子发布。恢复前校验结构、哈希、配置、数据、精度和设备约束。任何字段缺失都不能宣称 Exact Resume。

M1 单卡 Checkpoint 目录固定包含：

```text
checkpoint-step-00000025/
├── training_state.pt
├── config.resolved.json
├── environment.json
├── manifest.json
└── COMMITTED
```

`COMMITTED` 保存 Manifest SHA256；Manifest 保存其余文件的大小和 SHA256。先在同一
父目录写临时目录并 `fsync`，校验后原子 Rename，再原子更新 `LATEST`。任何失败都必须
清理临时目录，不能发布部分 Checkpoint。普通点滚动保留最近 `keep_last` 个；
interruption、best、final 点使用明确 Pin Reason，不参与清理。

Toy Data 使用 Stateful Sequential Sampler。其 `num_samples`、`epoch` 和下一个样本
`cursor` 必须进入 Checkpoint；恢复时样本数量或 Cursor 不合法必须拒绝。M1.2 只验证
保存、完整性和下一 Batch Cursor，M1.3 才验证完整训练恢复语义。

## 8. 验收顺序

```text
配置与模型单元测试
→ CPU 前向/反向
→ CPU Loss 下降
→ Checkpoint 保存/读取
→ 模拟损坏失败路径
→ 中断与 Exact Resume 对照
→ 空闲 RTX 3090 BF16 Smoke
→ 真实 M1 报告
```

只有完成以上流程并将结果合入主分支，M1 才能标记为 Complete。
