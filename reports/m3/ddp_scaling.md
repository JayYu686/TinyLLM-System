# M3.3–M3.4 DDP 扩展与性能报告

## 1. 审核结论

在 [ADR-0004](../../docs/adr/0004-shared-server-4gpu-acceptance.md) 定义的共享服务器
资源边界下，M3 正式 `1/2/4 卡 × Strong/Weak` 矩阵通过证据验收：6 个正式单元均完成
3 次独立真实运行，共 18 次成功运行。另有 3 次同 NUMA 四卡补充运行和 2 次由 GPU
Preflight 正确拒绝的失败尝试，失败证据均保留。

本报告不声称完成 8 卡扩展，也不声称得到跨 NUMA 性能结论。机器汇总明确记录
`eight_gpu_status=not_collected` 和 `numa_comparison_status=partial`。项目所有者已于
2026-07-16 确认本中文报告通过；PR #55 的质量、依赖和 Docker CI 均通过。本报告随
PR #55 合入 `main` 后，M3 状态正式成为 `COMPLETE`。

## 2. 证据身份与实验环境

| 项目 | 实际值 |
| -- | -- |
| 实验时间 | 2026-07-16 06:27:31–06:45:53 UTC |
| Benchmark 源代码 Commit | `a373b4623e22ab14360100793e9af396cfc88d98` |
| Git 状态 | 所有正式 Run 均为 clean |
| 配置 | `configs/benchmark/m3_tinygpt_120m_ddp.yaml` |
| 解析配置 SHA256 | `5dc8fb4d8534c63c890b13a169d37f430dbed8bdd293dbb6c2bf8ba228e919f3` |
| 模型 | `TinyGPT-Target-120M`，实际 `117,197,568` 参数 |
| 数据 | 固定 Seed 生成的 Toy Token 数据；本实验测系统吞吐，不测模型质量 |
| Python | CPython 3.11.14 |
| PyTorch / CUDA Runtime | PyTorch 2.7.1+cu118 / CUDA 11.8 |
| NVIDIA Driver | 535.261.03；Driver 报告 CUDA 12.2 |
| GPU | NVIDIA GeForce RTX 3090 24 GiB，Compute Capability 8.6 |
| 精度 | BF16，TF32 可用，不使用 GradScaler |
| 后端 | 原生 PyTorch DDP + NCCL |
| 正式 GPU 集合 | 单卡 `8`；双卡 `8,9`；四卡 `6,7,8,9` |
| 拓扑 | GPU 6–9 均属于 NUMA 1；组内为 PIX/PXB；未检测到 NVLink |
| 私有事实源 | `<private-artifact-root>/benchmarks/m3/<benchmark-commit>/` |
| 公开机器汇总 | [ddp_scaling_summary.json](ddp_scaling_summary.json) |
| 公开汇总 SHA256 | `fe14a89cadd25b59029148859f84d91b83d378b88031adde05d33eab99be0379` |

启动前 Doctor 发现 GPU 0–4 正在执行其他任务，因此这些卡没有被选择。所有正式运行均
执行 fail-closed Preflight：显存占用不超过 1024 MiB、利用率不超过 10%、温度不超过
79°C。运行期活跃样本的温度范围为 38–65°C，SM Clock 为 1905–1965 MHz，功耗为
106.13–301.66 W；没有发现超过温度门限的运行。

## 3. 固定测量协议

- 模型结构：Hidden 768、12 层、12 Heads、Intermediate 2304、Vocabulary 32768、
  Sequence Length 1024、Weight Tying。
- 每个单元预热 20 个 Optimizer Step，随后测量 100 Step，独立重复 3 次。
- Micro Batch 固定为每 Rank 1。
- Strong Scaling 固定 Global Batch 8；1/2/4 卡的 Gradient Accumulation 分别为 8/4/2。
- Weak Scaling 固定每 Rank Batch 1；Global Batch 分别为 1/2/4。
- 每个单元第一次重复对前 5 个测量 Step 采集 PyTorch Profiler Trace。
- 多 Rank Step Time 取同一步所有 Rank 的最大值；表格使用三次 Tokens/s 中位数，
  同时保留三次原值和最小/最大范围。

## 4. Strong Scaling 实测

| GPU | GPU 索引 | 三次 Tokens/s | 中位数 [最小, 最大] | 相对加速 | 扩展效率 | Step 中位数 | Peak Memory |
| --: | -- | -- | -- | --: | --: | --: | --: |
| 1 | 8 | 22,986.22 / 25,852.17 / 25,696.71 | 25,696.71 [22,986.22, 25,852.17] | 1.000× | 100.00% | 316.49 ms | 3.369 GiB |
| 2 | 8,9 | 34,090.18 / 38,463.31 / 37,939.28 | 37,939.28 [34,090.18, 38,463.31] | 1.476× | 73.82% | 215.01 ms | 3.369 GiB |
| 4 | 6,7,8,9 | 35,748.28 / 36,957.96 / 37,487.73 | 36,957.96 [35,748.28, 37,487.73] | 1.438× | 35.96% | 219.72 ms | 3.376 GiB |

双卡相对单卡吞吐提升 47.64%，但四卡没有继续提升，且比双卡中位数低 2.59%。这说明在
117M 参数、Sequence Length 1024 和 PCIe 组网条件下，继续增加 Rank 带来的同步开销已经
抵消了每 Rank 计算量下降。M3 没有预设最低效率门槛，因此该结果通过“真实、可复现、
口径正确”的证据门禁，但不能包装为良好的四卡线性扩展。

## 5. Weak Scaling 实测

| GPU | GPU 索引 | 三次 Tokens/s | 中位数 [最小, 最大] | 总吞吐相对单卡 | 每卡扩展效率 | Step 中位数 | Peak Memory |
| --: | -- | -- | -- | --: | --: | --: | --: |
| 1 | 8 | 17,601.86 / 20,242.18 / 19,734.72 | 19,734.72 [17,601.86, 20,242.18] | 1.000× | 100.00% | 51.10 ms | 2.925 GiB |
| 2 | 8,9 | 20,400.21 / 21,201.38 / 21,226.91 | 21,201.38 [20,400.21, 21,226.91] | 1.074× | 53.72% | 95.95 ms | 2.925 GiB |
| 4 | 6,7,8,9 | 22,864.75 / 22,987.49 / 23,235.08 | 22,987.49 [22,864.75, 23,235.08] | 1.165× | 29.12% | 175.99 ms | 2.925 GiB |

Weak Scaling 增加了全局工作量，但四卡总吞吐只比单卡提高 16.48%，每卡效率降至
29.12%。DataLoader 等待占 Step Time 的中位数分别为 0.91%、0.33% 和 0.21%，因此本次
低扩展效率不能主要归因于数据读取；更符合小模型单 Rank 计算量不足、DDP 梯度同步和
PCIe 通信占比上升的表现。

## 6. Profiler 与通信证据

| Profile | GPU | Profiler 状态 | 前 5 个测量 Step、各 Rank 匹配通信 Device Time |
| -- | --: | -- | -- |
| Strong | 1 | `not_applicable` | 单 Rank 无跨 Rank Collective |
| Strong | 2 | `measured` | 969.01–1,341.87 ms |
| Strong | 4 | `measured` | 2,196.81–2,466.52 ms |
| Weak | 1 | `not_applicable` | 单 Rank 无跨 Rank Collective |
| Weak | 2 | `measured` | 982.70–1,013.51 ms |
| Weak | 4 | `measured` | 2,277.21–2,497.92 ms |

Profiler 在多卡 Run 中实际识别到 `c10d::allreduce_`、`nccl:all_reduce` 和 NCCL
AllReduce Kernel。上表是 Profiler 窗口内匹配事件的原始 Device Time 范围，仅用于证明
通信发生并分析相对趋势；它不代表完整 100 Step 的通信累计时间，也不用于估算未运行的
8 卡结果。

## 7. NUMA 补充证据与边界

GPU 6–9 同 NUMA 四卡 Weak 补充组完成三次运行：22,856.61 / 23,223.30 /
22,652.73 Tokens/s，中位数 22,856.61。它与正式四卡 Weak 的同物理 GPU 组中位数仅相差
-0.57%，可以作为额外重复性检查。

GPU 4–7 跨 NUMA 组未运行，因为 GPU 4 长期被其他任务占用且不满足正式 Preflight。
因此，本报告不能判断同 NUMA 相对跨 NUMA 快多少，也不能把同组重复差异解释为拓扑收益。

## 8. 失败与异常留存

| 失败单元 | 原因 | 处理 |
| -- | -- | -- |
| Strong / 4 卡 / Repeat 1 首次尝试 | GPU 7 启动前利用率 55%，超过 10% 门限 | 监督器拒绝启动；原目录保留；资源恢复后写入新的 `attempt2` 目录 |
| Weak / 4 卡 / Repeat 2 首次尝试 | GPU 7 启动前利用率 47%，超过 10% 门限 | 监督器拒绝启动；原目录保留；资源恢复后写入新的 `attempt2` 目录 |

两次失败均未进入中位数，也没有被删除、覆盖或改写。单卡重复间范围相对更大，说明共享
服务器仍存在背景噪声；报告因此使用三次中位数和完整范围，而不是挑选最好结果。

## 9. 完成条件核对

| 门禁 | 状态 | 证据 |
| -- | -- | -- |
| 严格 YAML、Schema 与聚合器 | 通过 | 配置哈希、JSON Schema Snapshot、机器汇总 v1.1 |
| CPU/Gloo 多进程集成 | 通过 | 默认非 GPU 测试套件 |
| 配置漂移、错误 World Size、Busy GPU、超时、坏输出拒绝 | 通过 | 单元/集成测试与两次真实 Preflight 拒绝 |
| 1/2/4 Strong/Weak 各三次 | 通过 | 18 个成功 Run ID 与逐次标量结果 |
| Profiler、显存、Data Wait、遥测 | 通过 | 私有 Trace/原始数组与公开脱敏汇总 |
| 8 卡扩展 | 未采集、非阻塞 | ADR-0004；公开状态为 `not_collected` |
| 跨 NUMA 对照 | 部分采集、非阻塞 | 同 NUMA 已完成；公开状态为 `partial` |
| 中文报告与公开脱敏 | 通过 | 本报告和机器汇总 |
| 所有者内容审查 | 通过 | 2026-07-16 明确确认 |
| CI | 通过 | PR #55 的 quality、baseline-dependencies、docker |
| PR 合并 | 通过后生效 | 本报告随 PR #55 合入 `main` 即完成原子发布 |

## 10. 可公开结论

可以公开说明：项目在 RTX 3090 上完成原生 PyTorch DDP 的 1/2/4 卡正确性、完整恢复和
受控扩展测量；TinyGPT 117.2M Strong Scaling 双卡达到 1.476×、四卡达到 1.438×，并用
Profiler 证实 NCCL AllReduce 是实际通信路径。

不能公开说明：四卡接近线性扩展、已经验证八卡、已经完成跨 NUMA 性能比较，或这些结果
可以直接代表 Qwen/FSDP2。下一阶段 M4 必须重新执行四卡显存 Probe 和 FSDP2 专用测量。
