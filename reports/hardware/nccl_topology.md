# RTX 3090 NCCL 与拓扑报告

## 1. 测试条件

| 字段 | 实际值 |
|---|---|
| 日期 | 2026-07-13 |
| nccl-tests | NVIDIA `v2.13.4`，Commit `d313d20` |
| 官方归档 SHA256 | `00852d7b74548e171da86c459f3e6e03715fa7414a0f47eaab1110550a8db2f7` |
| 编译 CUDA_HOME | `/usr/local/cuda-11.8` |
| 运行时 `libcudart` | `/usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudart.so.11.0` |
| 运行时 NCCL | `/lib/x86_64-linux-gnu/libnccl.so.2`，包版本 2.13.4+cuda11.7 |
| 消息范围 | 8 B → 512 MiB，Factor 2 |
| Warmup / Iterations | 5 / 20 |
| Correctness Check | `-c 1` |

源码使用 NVIDIA 官方 tag 归档。服务器直连 GitHub clone 返回 HTTP 403，因此记录 tag、Commit 和归档 SHA256 代替 `.git` 元数据。

## 2. 物理拓扑

- GPU 0–4：NUMA 0。
- GPU 5–9：NUMA 1。
- PIX Pair：GPU 1–2、GPU 3–4、GPU 6–7、GPU 8–9。
- 同一 NUMA 内其他连接主要为 PXB。
- 跨 NUMA GPU 连接为 SYS。
- 所有 NVLink 均为 inactive。
- P2P Read 查询对所有 GPU Pair 返回 `CNS`（Chipset Not Supported）。

## 3. 实测结果

以下 `Avg bus bandwidth` 是 nccl-tests 对 8 B–512 MiB 全消息范围的单次平均值，仅用于记录本次环境，不用于训练吞吐声明。

| GPU 组 | 拓扑目的 | Collective | 退出码 | Correctness Error | Avg busbw |
|---|---|---|---:|---:|---:|
| 6 | 单卡工具 Smoke | All-Reduce | 0 | 0 | 0 |
| 6 | 单卡工具 Smoke | All-Gather | 0 | 0 | 0 |
| 6 | 单卡工具 Smoke | Reduce-Scatter | 0 | 0 | 0 |
| 8,9 | 2 卡 PIX | All-Reduce | 0 | 0 | 3.78226 |
| 8,9 | 2 卡 PIX | All-Gather | 0 | 0 | 3.08950 |
| 8,9 | 2 卡 PIX | Reduce-Scatter | 0 | 0 | 3.11437 |
| 6,7,8,9 | 4 卡同 NUMA | All-Reduce | 0 | 0 | 2.38485 |
| 6,7,8,9 | 4 卡同 NUMA | All-Gather | 0 | 0 | 2.11790 |
| 6,7,8,9 | 4 卡同 NUMA | Reduce-Scatter | 0 | 0 | 2.11994 |
| 4,5,6,7 | 4 卡跨 NUMA | All-Reduce | 0 | 0 | 2.56792 |
| 4,5,6,7 | 4 卡跨 NUMA | All-Gather | 0 | 0 | 2.44799 |
| 4,5,6,7 | 4 卡跨 NUMA | Reduce-Scatter | 0 | 0 | 2.37768 |
| 4,5,6,7,8,9 | 动态空闲 6 卡 | All-Reduce | 0 | 0 | 1.73375 |
| 4,5,6,7,8,9 | 动态空闲 6 卡 | All-Gather | 0 | 0 | 1.71595 |
| 4,5,6,7,8,9 | 动态空闲 6 卡 | Reduce-Scatter | 0 | 0 | 1.51900 |

本轮只运行一次矩阵，不足以得出“跨 NUMA 更快”或“某组最优”等性能结论。正式结论需要固定频率/温度、重复运行并比较同一消息大小。

## 4. 安全失败路径

在单卡 GPU 6 Smoke 结束后，脚本检测到 GPU 6 的瞬时利用率仍为 67%，自动将紧随其后的 2/4 卡任务标为：

```text
status: not_run
reason: busy_gpus
```

冷却并重新确认利用率为 0 后，2 卡和 4 卡任务才分别执行。这验证了脚本不会默认抢占忙卡。

## 5. 动态空闲卡验证与未执行项

2026-07-13 再次检查时，GPU 4–9 均为 1 MiB 显存占用、0% 利用率，脚本在未启用 `--allow-busy` 的情况下完成 6 卡三种 Collective。该结果验证共享服务器上可以按实时状态选择显式 GPU 组，但单次 6 卡结果不用于计算正式 Scaling Efficiency。

| 分组 | 状态 | 原因 |
|---|---|---|
| GPU 0–7 标准 8 卡组 | `deferred_m3` | GPU 0–3 被已有进程持续占用；固定 8 卡扩展测试属于 M3 |
| GPU 1,2,3,4,6,7,8,9 平衡候选组 | `not_run` | GPU 1–3 被已有进程持续占用 |
| GPU 0–9 | `not_run` | 10 卡不是默认规模，且 GPU 0–3 忙碌 |

当前结果确认了 1/2/4/6 卡 Collective 正确性和动态空闲卡选择。GPU 0–7 是否适合作为固定 8 卡性能基线，仍需 M3 在受控资源窗口验证。

## 6. 原始证据

- `reports/hardware/raw/nccl-v2.13.4/`
- `reports/hardware/raw/nccl-v2.13.4-pair/`
- `reports/hardware/raw/nccl-v2.13.4-quad/`
- `reports/hardware/raw/nccl-v2.13.4-cross-numa/`
- `reports/hardware/raw/nccl-v2.13.4-six-gpu/`
- `reports/hardware/raw/rtx3090_doctor.json`
