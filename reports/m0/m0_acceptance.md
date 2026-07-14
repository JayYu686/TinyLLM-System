# Milestone 0 验收记录

## 1. 当前状态

```text
状态：PENDING_REMOTE_CI
技术验收：通过
已完成：设计、最小工程、doctor、测试、CUDA/BF16、1/2/4/6 卡 NCCL、硬件报告、Git author 和初始提交准备
工程流程未完成：初始提交推送后的首次远程 CI
```

根据 AGENTS.md 的里程碑规则，在首次远程 CI 验证和最终状态同步前不得将状态标记为 M0 Complete。

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
| 基础 CI | 等待远程运行 | `.github/workflows/ci.yml`；远程仓库已连接 |
| 3090 Inventory | 通过 | `reports/hardware/rtx3090_inventory.md` |
| NUMA/Topology | 通过 | 5+5 NUMA，PIX/PXB/SYS 已记录 |
| NCCL 1/2/4 卡 | 通过 | 三种 Collective，Correctness Error 0 |
| NCCL 动态 6 卡 | 通过 | GPU 4–9，三种 Collective，Correctness Error 0 |
| NCCL 标准 8 卡 | 移至 M3 | 固定 1/2/4/8 卡扩展实验需要受控资源窗口 |
| Git 初始化 | 通过 | Branch `main` |
| 首个 Git Commit | 已准备 | Git author 已配置；本验收记录随初始提交发布 |
| V100 报告 | 条件性未执行 | 本轮无 V100 服务器连接方式，不阻塞 3090 主平台实现 |

## 3. 已验证失败路径

- 项目外调用时输出目录不存在：退出码 2，返回 `CLI_OUTPUT_ERROR`。
- PyTorch 缺失或损坏：doctor 总状态 `fail`，继续采集其他硬件信息。
- `nvidia-smi` 缺失：必需检查失败，不静默跳过。
- GPU CSV 字段变化：解析失败并提供 remediation。
- GPU 忙碌：NCCL Runner 标记 `not_run`，不自动抢占。
- NCCL 子进程超过 300 秒：退出码 124，并保留超时日志。

## 4. 完成 M0 的剩余动作

1. 推送初始提交并等待基础 CI 完成。
2. 保存远程 CI 结果。
3. 更新本文件、TASKS.md 和 README 状态为 Complete。

标准 GPU 0–7 的 8 卡 NCCL 和可选候选组对比移至 M3，不再阻塞 M0。
