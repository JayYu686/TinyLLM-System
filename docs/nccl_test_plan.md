# M0 NCCL 测试计划

## 1. 目标

用真实 Collective 测试验证 RTX 3090 服务器的 GPU 分组、动态空闲卡选择和跨 NUMA 通信特征。M0 关注工具正确性与安全调度；ADR-0002 的固定 8 卡性能验证属于 M3 正式扩展实验。

本测试不用于宣称训练吞吐，也不等价于 DDP/FSDP2/ZeRO-3 Benchmark。

## 2. 前置条件

- 记录 `nccl-tests` Git Commit，不使用浮动源码。
- 本次 M0 目标版本为 `v2.13.4`（Commit `d313d20`），与服务器系统 NCCL 2.13.4 对齐；若使用官方 tag 归档，同时记录归档 SHA256。
- 记录 Driver、CUDA Toolkit、NCCL 库和主机拓扑。
- 运行前检查目标 GPU 的利用率和显存占用。
- 目标 GPU 忙碌、温度异常或存在未知进程时拒绝启动。
- 不终止或抢占其他用户进程。

## 3. Collective

- All-Reduce。
- All-Gather。
- Reduce-Scatter。

统一参数：

```text
起始消息：8 B
结束消息：512 MiB
增长因子：2
Warmup：5
Iterations：20
Correctness Check：启用
```

任何参数变化必须单独形成 Run，不得合并比较。

## 4. GPU 分组

| 分组 | GPU | 目的 |
|---|---|---|
| 1 卡基线 | 6 | 验证工具与原始日志格式 |
| 2 卡同 PIX | 6,7 | 最接近的 PCIe 对照 |
| 4 卡同 NUMA | 6,7,8,9 | 单 NUMA/多 PCIe Bridge 对照 |
| 4 卡跨 NUMA | 4,5,6,7 | 验证 `SYS` 路径成本 |
| 动态空闲 6 卡 | 4,5,6,7,8,9 | 验证共享状态下的显式空闲卡选择 |
| 标准 8 卡 | 0,1,2,3,4,5,6,7 | M3 验证 ADR-0002 当前默认组 |
| 平衡 8 卡候选 | 1,2,3,4,6,7,8,9 | 4+4 NUMA 候选，仅用于拓扑对照 |
| 10 卡边界 | 0–9 | 非标准规模，只在全部 GPU 空闲时执行 |

正式扩展实验仍优先 1/2/4/8 卡。10 卡结果不得作为默认配置依据。

## 5. 输出

每个 Run 保存：

- 完整命令。
- `CUDA_VISIBLE_DEVICES`。
- 开始和结束时间。
- 退出码。
- 原始 stdout/stderr。
- `nccl-tests` Commit。
- Driver、CUDA、NCCL 版本。
- 运行前 GPU 利用率、显存和温度。

报告指标来自 `nccl-tests` 原始输出：

- 算法带宽。
- Bus Bandwidth。
- Correctness Error。
- 失败或超时信息。

未执行组合必须写为 `not_run` 并说明原因，不得留空或估算。

## 6. M0 判定

- 至少完成空闲资源上的 1/2/4 卡三种 Collective Smoke。
- 至少完成一个实时选择的跨 NUMA 多卡组，并保存选择时的利用率、显存和温度。
- M0 不要求等待固定 GPU 0–7 空闲，也不把 6 卡结果当作扩展效率基线。
- M3 在受控资源窗口完成固定 1/2/4/8 卡实验；若标准组和候选组差异显著，再单独提交 ADR 修订。
