# Benchmark 计划

## 1. 原则

- 只报告实测。
- 测试条件完整。
- 预热后测量。
- 多次运行。
- 记录均值和分位数。
- 不混用不同硬件结果。

## 2. 训练 Benchmark

### 变量

- GPU：3090/V100。
- World Size：1/2/4/8。
- Strategy：DDP/FSDP2/ZeRO-3。
- Precision：BF16/FP16。
- Sequence Length：512/1024/2048/4096。
- Micro Batch。
- Activation Checkpointing。

### 指标

- Tokens/s。
- Samples/s。
- Step Time。
- Peak Memory。
- Scaling Efficiency。
- Communication Time。
- Data Loading Time。
- Checkpoint Time。
- Resume Time。

## 3. 推理 Benchmark

### 输入组合

- 128 输入 / 128 输出。
- 512 输入 / 256 输出。
- 2048 输入 / 512 输出。
- 4096 输入 / 512 输出。

### 并发

- 1。
- 2。
- 4。
- 8。
- 16。
- 32。

### 指标

- TTFT。
- P50/P95。
- Inter-token Latency。
- Tokens/s。
- Request/s。
- Peak Memory。
- Failure Rate。

## 4. 报告模板

每份报告必须包含：

- 日期。
- Git Commit。
- 模型。
- 数据。
- 硬件。
- 拓扑。
- 软件版本。
- 配置。
- 原始数据文件。
- 汇总表。
- 异常说明。
- 结论。

## 5. 禁止比较

不直接比较：

- 不同模型规模却不归一化。
- 不同输入输出长度。
- 不同 Batch。
- 不同量化方式。
- 不同软件版本。
- 不同拓扑却不说明。

## 6. 初始报告

- 3090 单卡训练。
- 3090 1/2/4/8 卡 DDP。
- 3090 FSDP2 与 ZeRO-3。
- V100 单卡 FP16。
- 3090/V100 单卡推理。
- 7B LoRA。
- 7B Full SFT Smoke Test。
