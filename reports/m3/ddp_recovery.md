# M3.2 DDP Checkpoint 与恢复报告

## 验收结论

M3.2 在本 PR 合并后通过验收。真实的双卡 RTX 3090 BF16 运行证明了以下能力：原子发布
完整 DDP Checkpoint、同 World Size 的 Exact Resume、协调中断后的规范指标连续性，以及
非零 Rank 强制退出后的可靠恢复。

M3 整体仍为 `IN_PROGRESS`。本报告不验收训练吞吐、扩展效率、Profiler 结果、变更
World Size 后的恢复、FSDP2、ZeRO-3、弹性成员管理或节点故障恢复。Issue #14 和 #15
仍是 M3 完成前的强制门禁。

## 复现方式

先通过 `tinyllm doctor --distributed --json` 选择两张处于同一 NUMA 节点的空闲 GPU，
随后执行：

```bash
.venv/bin/python scripts/run_m3_ddp_recovery_smoke.py \
  --config configs/pretrain/tinygpt_debug_ddp_recovery_2gpu_bf16_smoke.yaml \
  --output-root "$TINYLLM_ARTIFACT_ROOT/runs" \
  --evidence-dir "$PRIVATE_EVIDENCE_ROOT/m3-ddp-recovery" \
  --gpu-indices 5,6
```

监督器首先执行两次无中断基线，并在恢复实验开始前冻结容差。随后分别执行两条失败路径：

1. 在 Step 6 进行协调中断，再通过同一 Exact Resume 接口恢复；
2. 在 Step 8 Checkpoint 提交后强制 Rank 1 退出，再通过同一接口恢复。

已脱敏且不包含本机路径的机器可读事实保存在
[`raw/ddp_recovery.json`](raw/ddp_recovery.json)。完整 Doctor 输出、启动前检查、torchrun
标准输出与错误输出、Run 目录、模型与优化器状态、故障诊断均保留在私有 Artifact Store。
公开证据仅记录私有 Summary、容差文件、Doctor 报告，以及已验收 Checkpoint Manifest 和
提交标记的 SHA256。

## 环境与血缘

| 项目 | 实际结果 |
| -- | -- |
| 源代码 Commit | `03e1cf3b4da11470ff8fa7c74f9a5713105ef823` |
| 执行时 Git Dirty | `false` |
| Python | 3.11.14 |
| PyTorch | 2.7.1+cu118 |
| PyTorch CUDA Runtime | 11.8 |
| NVIDIA Driver | 535.261.03 |
| NCCL | 2.21.5 |
| GPU | 2 × RTX 3090 24 GiB，物理编号 5、6 |
| 拓扑 | 同一 NUMA 节点，`PXB` 路径 |
| 精度 | BF16，启用 TF32，不使用 GradScaler |
| 模型 | TinyGPT-Debug，1,820,352 个参数 |

Doctor 总状态为 `warn`，原因是其他 GPU 正在使用、P2P Read 报告 `CNS`、NVLink 未激活，
且当前环境没有独立的 nccl-tests 可执行文件。每个验收阶段都独立重复了更严格的启动前
检查。两张被选 GPU 的显存占用均为 1 MiB、利用率均为 0%；各阶段记录的启动前温度范围
为 29°C–46°C。

上述警告和拓扑信息限定了结论边界：这些结果是真实的单机 DDP 正确性与故障恢复证据，
不是通信性能 Benchmark。

## 实测结果

| 检查项 | 两次基线 | 协调中断 | Rank 故障 |
| -- | --: | --: | --: |
| World Size | 2 | 2 | 2 |
| Global Batch | 8 | 8 | 8 |
| 最终 Optimizer Step | 12 | 12 | 12 |
| 中断/故障边界 | — | Step 6 | Step 8 |
| 强制退出方式 | — | Worker 协调退出 | Rank 1，退出码 17 |
| Exact Resume 来源 | — | Step 6 | Step 8 |
| 规范指标记录数 | 每次 12 | 12 | 12 |
| 缺失或重复 Step | 0 | 0 | 0 |
| 相对基线的最大 Loss 差异 | 0 | 0 | 0 |
| 最终参数 SHA256 与基线相同 | 是 | 是 | 是 |
| Checkpoint 完整性 | 通过 | 通过 | 通过 |
| 最终状态 | 通过 | 通过 | 通过 |

两次无中断基线的逐 Step 最大 Loss 差异为 `0.0`。因此，预先声明的规则
`max(1e-6, 2 × 基线差异)` 在任何恢复对照开始前将容差冻结为 `1e-6`。两条恢复路径
相对基线的 Loss 序列差异均为 `0.0`；所有验收运行的最终模型参数 SHA256 均为：

```text
d0be4763d8682db8eaa634d01e5c7794d09a98341270a53377a2527d3e3c4c16
```

监督器记录的总时长为 48.924 秒，其中包含六次 torchrun 启动和故障编排。该时长只作为
操作过程证据保留，不作为训练吞吐指标。

## Checkpoint 与故障证据

每个验收 Checkpoint 都包含一份共享的完整训练状态，以及连续的 `rank-00000.pt` 和
`rank-00001.pt`。Manifest 覆盖模型、AdamW、Scheduler、Trainer 进度、解析后配置、
运行环境，以及每个 Rank 独立的 Sampler 和 RNG 状态。发布流程使用临时目录、逐文件哈希、
最终 `COMMITTED` 标记、原子 Rename 和原子 `LATEST` 更新。

协调中断 Run 在重启前后保持同一个 Run ID。加载 Step 6 后只新增 Step 7–12。Rank 故障
Run 在 Step 8 提交 Checkpoint，记录 Rank 1 和退出码 17，然后由 torchrun 终止剩余进程组。
重启时选择有效的 Step 8 Checkpoint，只新增 Step 9–12。两个最终 `metrics.jsonl` 均严格
包含 Step 1–12，没有缺失或重复。

自动化测试还覆盖并拒绝以下失败情况：错误 World Size、Rank 状态缺失或损坏、Sampler
Cursor 非法、环境或配置漂移、Checkpoint 目标重复、向非初始 Trainer 恢复，以及优化器
状态应用失败。CPU/Gloo 集成测试通过真实的双进程 torchrun 执行了两条故障路径。

## M3 剩余门禁

M3.2 只关闭 Issue #13。剩余工作为：

1. Issue #14：Benchmark Harness，包括预热、测量窗口、三次独立重复、显存、数据等待、
   通信和 Profiler 证据；
2. Issue #15：受控的 1/2/4 卡 Strong/Weak Scaling。根据 ADR-0004，8 卡和跨 NUMA
   对照是非阻塞增强项，未实测时不得给出相应结论。

M3.2 的恢复编排时长、Checkpoint 写入时间、Tokens/s 和 Scaling 均不得被提前包装成
Benchmark 结果。
