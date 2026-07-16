# M3.3–M3.4 DDP Benchmark 与扩展验收契约

## 1. 目标与边界

本契约只回答原生 PyTorch DDP 在主服务器上的训练性能与扩展行为，不重新验证 M3.1
的基础正确性，也不替代 M3.2 的 Checkpoint/Exact Resume 证据。M3 Benchmark 使用
`TinyGPT-Target-120M` 目标配置，实例化后的实际参数量必须由程序记录，不能从名称推断。

以下内容不属于本契约：FSDP2、ZeRO-3、跨节点、弹性成员管理、模型质量和推理性能。

## 2. 唯一正式配置

正式入口读取 `configs/benchmark/m3_tinygpt_120m_ddp.yaml`。配置固定：

- 模型：`hidden=768`、`layers=12`、`heads=12`、`intermediate=2304`、
  `vocab=32768`、`sequence_length=1024`、Weight Tying；
- 精度：RTX 3090 BF16、允许 TF32、不使用 GradScaler；
- 预热：20 个 Optimizer Step；
- 测量：100 个 Optimizer Step；
- 重复：每个 Profile 独立运行 3 次；
- Micro Batch：每 Rank 1；
- Strong Scaling：Global Batch 固定为 8，1/2/4/8 卡分别使用 8/4/2/1 次
  Gradient Accumulation；
- Weak Scaling：每 Rank Batch 固定为 1，Global Batch 随 World Size 变为 1/2/4/8；
- Profiler：每个配置的第一次重复采集前 5 个测量 Step，所有 Rank 都保存 Trace。

GPU 编号、输出位置、Profile、World Size 和重复编号属于允许的运行时字段；模型、数据、
精度、优化器、预热与测量窗口不得通过 CLI 静默覆盖。

## 3. 时间与吞吐口径

每个 Rank 使用 CUDA Event 测量完整 Optimizer Step。一次 Step 包含 H2D、前向、
反向、DDP Gradient Synchronization、Gradient Clipping、AdamW 和清零；不包含下一批
数据从 DataLoader 取出的 CPU 等待时间。多 Rank 有效 Step Time 取同一步所有 Rank 的
最大值，避免用快 Rank 掩盖 Straggler。

吞吐按下式计算：

```text
Tokens/s = 测量窗口内全局有效预测 Token 数 / 有效 Step Time 总和
Samples/s = 测量窗口内全局样本数 / 有效 Step Time 总和
```

有效预测 Token 数按 `batch × (sequence_length - 1)` 计算。报告同时保留 100 个 Step
和每个 Rank 的原始时间数组，不用单个均值替代原始证据。

## 4. 通信、数据等待、显存与 Profiler

- `data_wait_ms` 直接测量 DataLoader 迭代器取批时间；报告总量、中位数、P95 以及其相对
  有效 Step Time 的比例。它不包含 H2D。
- Peak Memory 使用预热结束后重置的 `torch.cuda.max_memory_allocated()`，报告 Rank 最大值。
- DDP 通信使用 PyTorch Profiler 中名称包含 NCCL/AllReduce/ReduceScatter/AllGather 的
  Device Event 实测时间；报告 Profile 窗口、匹配的 Event Key 和各 Rank 数值。
- Profiler 只覆盖配置中声明的测量窗口，其通信时间不能冒充全部 100 Step 的累计时间。
- Trace、完整 stdout/stderr、遥测和 Run 目录保存在私有 Artifact Store；公开 JSON 保存
  脱敏汇总与 SHA256。

若当前 PyTorch/Driver 无法产生可识别的通信 Event，结果必须标记为 `unavailable`，不能
填 0 或估算值。World Size 1 没有跨 Rank 通信，其状态明确记为 `not_applicable`。

## 5. 遥测与拓扑

监督器在每次启动前执行 fail-closed GPU Preflight，并在子进程存活期间定期采集：

- 显存占用、GPU 利用率、温度；
- SM Clock、功耗；
- 物理 GPU 编号和 Driver；
- `nvidia-smi topo -m` 的完整快照。

显存占用超过 1024 MiB、利用率超过 10% 或温度超过 79°C 的选中 GPU 必须拒绝；正式
入口不提供 Busy Override。失败、超时、预检拒绝和异常运行必须留在独立 Evidence 目录，
不得覆盖或删除。

## 6. 正式矩阵

正式 1/2/4/8 卡 Strong/Weak Scaling 使用相同代码 Commit、YAML、模型、数据生成算法和
重复 Seed。每个 Profile/World Size 都必须有三次成功原始运行。Scaling Efficiency：

```text
Strong Efficiency(N) = Throughput(N) / (N × Throughput(1))
Weak Efficiency(N)   = Per-GPU Throughput(N) / Per-GPU Throughput(1)
```

每个单元格用三次 Tokens/s 的中位数作为比较值，同时报告最小值和最大值。失败运行不进入
中位数，但必须保留并单独解释；不能补写、复制或插值缺失重复。

NUMA 对照固定为四卡 Weak Scaling：

- 同 NUMA：GPU 6–9；
- 跨 NUMA：GPU 4–7。

两组使用同一配置和三次独立重复。若实际拓扑与上述假设不一致，必须停止并修订 ADR/契约，
不能继续沿用标签。

## 7. 完成门禁

M3 只有在以下条件全部满足后才能标记 `COMPLETE`：

1. Benchmark 配置、Schema、聚合逻辑和 Trace 契约有单元测试；
2. CPU/Gloo 小配置通过真实多进程集成测试；
3. Busy GPU、错误 World Size、配置漂移、已有 Evidence 目录、超时和无效 Worker 输出均被拒绝；
4. 正式 1/2/4/8 Strong/Weak 每格完成三次真实运行；
5. 两个四卡 NUMA 组各完成三次真实运行；
6. 原始私有证据可被严格 Schema 重新加载，公开证据不包含用户名、主机名或绝对路径；
7. 中文报告解释环境、拓扑、温度、频率、后台负载、异常和结论边界；
8. 全量质量门禁、CI、PR 合并和 Issue #14/#15 关闭。
