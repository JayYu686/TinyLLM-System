# RTX 3090 主服务器硬件报告

## 1. 报告元数据

| 字段 | 实际值 |
|---|---|
| 采集时间 | 2026-07-13T08:32:39.484842Z |
| 主机名 | `<redacted-host>` |
| 项目根目录 | `<project-root>` |
| doctor Schema | `1.0` |
| doctor 总状态 | `warn` |
| 原始报告 | `reports/hardware/raw/rtx3090_doctor.json` |

`warn` 表示硬件与 CUDA/BF16 Probe 可用，但存在忙卡、温度、Git 血缘和拓扑限制；不是训练可立即启动的声明。

## 2. 操作系统与 CPU

| 项目 | 实际值 |
|---|---|
| OS | Ubuntu 20.04.6 LTS |
| Kernel | `5.4.0-216-generic` |
| CPU | 2 × Intel Xeon Gold 6326，16 Core/Socket，64 Logical CPU |
| NUMA | 2 Nodes |
| NUMA 0 CPU | `0-15,32-47` |
| NUMA 1 CPU | `16-31,48-63` |
| 内存 | 251 GiB Total；采集时约 205 GiB Available |
| Swap | 67 GiB |

`numactl` 当前未安装；doctor 使用 `lscpu` 和 `nvidia-smi topo -m` 完成 NUMA 交叉验证。

## 3. GPU

| GPU | 型号 | 显存 | PCI Bus | NUMA | Compute Capability |
|---:|---|---:|---|---:|---:|
| 0 | RTX 3090 | 24576 MiB | `4F:00.0` | 0 | 8.6 |
| 1 | RTX 3090 | 24576 MiB | `52:00.0` | 0 | 8.6 |
| 2 | RTX 3090 | 24576 MiB | `53:00.0` | 0 | 8.6 |
| 3 | RTX 3090 | 24576 MiB | `56:00.0` | 0 | 8.6 |
| 4 | RTX 3090 | 24576 MiB | `57:00.0` | 0 | 8.6 |
| 5 | RTX 3090 | 24576 MiB | `CE:00.0` | 1 | 8.6 |
| 6 | RTX 3090 | 24576 MiB | `D1:00.0` | 1 | 8.6 |
| 7 | RTX 3090 | 24576 MiB | `D2:00.0` | 1 | 8.6 |
| 8 | RTX 3090 | 24576 MiB | `D5:00.0` | 1 | 8.6 |
| 9 | RTX 3090 | 24576 MiB | `D6:00.0` | 1 | 8.6 |

采集时 GPU 0–3 被已有进程持续占用约 14.2 GiB/卡，利用率约 90–100%。GPU 2 的瞬时温度达到 80°C 以上。M0 没有终止、抢占或修改这些进程。

## 4. CUDA 与 Python 环境

| 项目 | 实际值 |
|---|---|
| Python | 3.11.14，项目 `.venv` |
| PyTorch | `2.7.1+cu118` |
| PyTorch CUDA Runtime | 11.8 |
| PyTorch NCCL | 2.21.5 |
| Driver | 535.261.03 |
| Driver 报告 CUDA | 12.2 |
| `/usr/local/cuda` | CUDA Toolkit 12.1 |
| `nvcc` 是否在 PATH | 否 |
| 系统 NCCL | 2.13.4，CUDA 11.7 包 |

环境隔离结论：

- 原 Conda base 中的 PyTorch 2.9.1 安装不完整，`torch.__init__.py` 缺失，不可使用。
- M0 创建独立 `.venv`，没有修改 Conda base。
- 项目 PyTorch 使用官方 CUDA 11.8 wheel；独立 nccl-tests 使用系统 NCCL 2.13.4，两者不能混写成同一 NCCL 结果。

## 5. CUDA/BF16 Smoke

在空闲 GPU 9 上实测：

| 检查 | 结果 |
|---|---|
| `torch.cuda.is_available()` | `True` |
| 可见 GPU 型号 | NVIDIA GeForce RTX 3090 |
| Compute Capability | `(8, 6)` |
| `torch.cuda.is_bf16_supported()` | `True` |
| BF16 16×16 Matmul | 通过 |
| 结果均值 | 16.0 |
| Pytest GPU Marker | 1 Passed |

该结果只证明 M0 环境下单卡 CUDA/BF16 基础运算可用，不证明任何模型配置可训练。

## 6. 存储

| 路径 | 总容量 | 可用容量 | 结论 |
|---|---:|---:|---|
| 项目所在根文件系统 | 3.28 TiB | 约 738 GiB | 可开发，不建议长期保存大 Checkpoint |
| `/data` | 14.55 TiB | 约 8.05 TiB | 推荐作为数据、模型和 Run Store，等待最终路径约定 |

## 7. 当前结论

- 10 张 RTX 3090 型号、显存和 Compute Capability 与项目文档一致。
- BF16 已通过 PyTorch 实测，不再是静态推断。
- GPU 拓扑为 5+5 双 NUMA，不存在 Active NVLink。
- `nvidia-smi topo -p2p r` 返回 `CNS`，只能通过 NCCL 实测判断 Collective 行为。
- GPU 0–7 是 5+3 跨 NUMA 组合；ADR-0002 将其作为 M3 正式 8 卡基线，性能适用性仍需受控实测。
- 当前 GPU 0–3 忙碌且至少一张卡温度较高，因此 M0 没有等待或抢占，而是显式选择空闲 GPU 4–9 完成 6 卡 NCCL Smoke。
- 开发和正确性 Smoke 采用动态空闲卡策略；正式扩展效率仍固定使用 1/2/4/8 卡。
