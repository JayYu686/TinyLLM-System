# 硬件策略

## 1. 总体决策

RTX 3090 服务器作为主平台，V100 服务器作为辅助和兼容性平台。

## 2. RTX 3090 服务器

```text
10 × RTX 3090 24GB
```

默认资源划分：

| GPU | 用途 |
|---|---|
| 0–7 | 标准训练组 |
| 8 | 自动评测 / 推理 |
| 9 | 开发 / 备用 / Embedding |

这是独占资源充足时的推荐逻辑划分，不是共享服务器上的硬编码调度。开发、M0 体检和功能 Smoke Test 每次先用 `tinyllm doctor` 或 `nvidia-smi` 检查实时状态，再显式选择当时空闲、温度正常的 GPU；必须记录实际 GPU 索引，不能抢占未知进程。

共享服务器的正式扩展效率实验固定使用 1/2/4 卡和可复现的嵌套分组，避免把不同 World
Size、拓扑或后台负载的结果直接比较。动态 3/5/6/7 卡组只证明正确性和资源适应能力，
不替代正式扩展基线。8 卡和跨 NUMA对照仅在获得不影响其他用户的受控窗口时追加，
不阻塞核心发布，详见 [ADR-0004](adr/0004-shared-server-4gpu-acceptance.md)。

原因：

- 1/2/4 卡能形成标准的二次幂扩展曲线，并在共享资源上稳定复现。
- 多数 Tensor Parallel 配置更适合 2 的幂。
- 保留独立评测卡，避免训练结束后等待资源。
- 保留备用卡，提高实验连续性。

全部 10 卡只用于：

- 极限 FSDP/ZeRO-3。
- 长上下文。
- 显存边界实验。
- World Size 对照。

## 3. V100 服务器

默认用途：

- FP16 兼容性。
- 单卡 32GB 实验。
- 3090/V100 对比。
- Volta 路线验证。
- 特定旧版推理框架测试。

## 4. 精度策略

| 硬件 | 默认精度 | GradScaler | TF32 |
|---|---|---|---|
| RTX 3090 | BF16 | 否 | 可启用 |
| V100 | FP16 | 是 | 不支持 |

## 5. 并行策略选择

### DDP

使用条件：

- 单卡能容纳完整模型状态。
- 主要目标是提高吞吐。

### FSDP2 FULL_SHARD

使用条件：

- 单卡无法容纳参数、梯度和优化器状态。
- 希望使用原生 PyTorch。
- 需要 Sharded Checkpoint。

### ZeRO-3

使用条件：

- 需要成熟的 ZeRO 配置。
- 需要 CPU Offload。
- 需要与 DeepSpeed 生态兼容。

## 6. 必做硬件检查

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id,power.limit --format=csv
numactl --hardware
```

NCCL 测试：

- All-Reduce。
- All-Gather。
- Reduce-Scatter。
- M0 按空闲资源完成多卡正确性 Smoke。
- M3 固定进行 1/2/4 卡扩展对比。
- 8 卡与受控跨 NUMA对照是可选增强证据。
- 10 卡仅作为可选边界对照。

## 7. 风险

- 3090 可能跨 NUMA。
- 10 卡拓扑可能不对称。
- 消费级卡长时间高负载可能受散热或功耗限制。
- 不同 GPU 可能有降频差异。
- 3090 不使用 ECC。
- V100 与新版框架兼容性下降。

系统必须记录实际拓扑和功耗状态，不只记录 GPU 型号。
