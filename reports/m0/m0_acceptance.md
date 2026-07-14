# Milestone 0 验收记录

## 1. 当前状态

```text
状态：COMPLETE
技术验收：通过
工程验收：通过
已完成：设计、最小工程、doctor、测试、CUDA/BF16、1/2/4/6 卡 NCCL、硬件报告、初始提交和远程 CI
```

M0 已满足 AGENTS.md 要求的代码、测试、Smoke、失败路径、真实结果、文档和后续依赖条件。

## 2. 验收表

| 条目 | 状态 | 证据 |
|---|---|---|
| 根文档与设计一致性检查 | 通过 | 关键范围冲突已修正 |
| `pyproject.toml` 与 `src/tinyllm/` | 通过 | 包可编辑安装 |
| `tinyllm --help` | 通过 | 从 `/tmp` 调用成功 |
| `tinyllm doctor` 人类输出 | 通过 | 本地 Smoke |
| `tinyllm doctor --json` | 通过 | Schema 1.0，可解析 |
| doctor 失败路径 | 通过 | 坏 PyTorch、无 GPU、坏 CSV、坏路径、忙卡 |
| Ruff | 通过 | `ruff check .` |
| Ruff Format | 通过 | `ruff format --check .` |
| MyPy Strict | 通过 | `mypy` |
| Pytest CPU | 通过 | 18 Passed，1 GPU Test Deselected |
| Pytest GPU | 通过 | GPU 9，BF16 Smoke 1 Passed |
| 基础 CI | 通过 | GitHub Actions Run [29297900401](https://github.com/JayYu686/TinyLLM-System/actions/runs/29297900401)，Quality Job 26s |
| 3090 Inventory | 通过 | `reports/hardware/rtx3090_inventory.md` |
| NUMA/Topology | 通过 | 5+5 NUMA，PIX/PXB/SYS 已记录 |
| NCCL 1/2/4 卡 | 通过 | 三种 Collective，Correctness Error 0 |
| NCCL 动态 6 卡 | 通过 | GPU 4–9，三种 Collective，Correctness Error 0 |
| NCCL 标准 8 卡 | 移至 M3 | 固定 1/2/4/8 卡扩展实验需要受控资源窗口 |
| Git 初始化 | 通过 | Branch `main` |
| 首个 Git Commit | 通过 | `ae33809`，已推送至 `origin/main` |
| V100 报告 | 条件性未执行 | 本轮无 V100 服务器连接方式，不阻塞 3090 主平台实现 |

## 3. 已验证失败路径

- 项目外调用时输出目录不存在：退出码 2，返回 `CLI_OUTPUT_ERROR`。
- PyTorch 缺失或损坏：doctor 总状态 `fail`，继续采集其他硬件信息。
- `nvidia-smi` 缺失：必需检查失败，不静默跳过。
- GPU CSV 字段变化：解析失败并提供 remediation。
- GPU 忙碌：NCCL Runner 标记 `not_run`，不自动抢占。
- NCCL 子进程超过 300 秒：退出码 124，并保留超时日志。

## 4. 后续非阻塞动作

- 取得 V100 服务器访问方式后补充辅助平台报告。
- 标准 GPU 0–7 的 8 卡 NCCL 和可选候选组对比在 M3 的受控资源窗口执行。
- 10 卡 NCCL 仅作为可选边界对照。
