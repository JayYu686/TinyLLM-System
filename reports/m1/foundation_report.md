# M1 Foundation 阶段报告

## 1. 状态

```text
M1 总状态：IN_PROGRESS
本批状态：PASS
范围：设计契约、配置 Schema、TinyGPT-Debug、Toy Dataset、Seed、CPU 前后向单元测试
```

本报告不代表 M1 完成。训练器、Checkpoint、Exact Resume、Loss 下降、RTX 3090 BF16 和 V100 FP16 尚未验收。

## 2. 已实现接口

- `TinyGPTConfig`：类型、范围、Head 划分和未知字段校验。
- `TinyGPT`：RMSNorm、RoPE、Causal Self-Attention、SwiGLU、Transformer Block、Weight Tying 和 Causal LM Loss。
- `ToyTokenDataset`：按 Seed 生成可重复的模运算 Token 序列。
- `M1TrainingConfig`：严格 YAML Schema、Global Batch 和跨字段校验。
- `seed_everything`：Python、NumPy、PyTorch 和可见 CUDA Seed 入口。

## 3. 默认模型实测静态值

| 项目 | 实际结果 |
|---|---:|
| 参数量 | 1,820,352 |
| 可训练参数量 | 1,820,352 |
| Head Dimension | 32 |
| M1 默认 Global Batch | 8 |

参数量由当前代码实例化后调用 `parameter_count()` 获得，不是估算值。它满足 M1 约 1M–5M 的 Debug 模型范围。

## 4. 测试结果

最终本地命令：

```text
make check
```

实际结果：

- Ruff：通过。
- Ruff Format：通过。
- MyPy Strict：29 个 Source File 无问题。
- Pytest：32 Passed，1 GPU Test Deselected。

新增测试覆盖：

- 默认模型参数规模。
- 前向、Causal LM Loss 和反向。
- Weight Tying 共享存储。
- 未来 Token 不影响过去 Logits。
- RMSNorm 数值参考和反向。
- Toy Dataset 相同 Seed 一致、不同 Seed 有差异。
- Python/NumPy/PyTorch Seed 重置。
- YAML 示例加载、未知字段和跨字段冲突失败路径。

## 5. 本批发现并修复的问题

首次反向测试发现 RMSNorm 在 FP32 输入上使用原地乘法，导致 Autograd 报告变量版本被修改。实现已改为非原地计算，并加入 RMSNorm 输入梯度回归测试。

## 6. 下一批

1. 定义单卡 Trainer、Optimizer 和 Scheduler 接口。
2. 实现 Gradient Accumulation、Gradient Clipping 与非有限值失败路径。
3. 增加 CPU 短程 Loss 下降 Smoke。
4. 设计并实现完整 Checkpoint 的原子保存、校验和 Exact Resume。
5. 以上通过后再运行空闲 RTX 3090 BF16 Smoke。
