# M4.2 FSDP2 DCP 分片 Checkpoint 与 Exact Resume 报告

## 1. 结论

M4.2 已通过。TinyLLM-System 现在能够使用 PyTorch Distributed Checkpoint（DCP）保存
FSDP2 分片模型与优化器状态，并把 Scheduler、训练进度、各 Rank RNG、Stateful Sampler、
配置、数据版本、Git 和环境身份纳入同一原子提交契约。

两进程 CPU/Gloo 的正式可复现实验得到以下结果：

- 无中断基线从 Step 1 连续运行至 Step 6；
- 第二个 Run 在 Step 2 提交 DCP 后协调退出；
- 新 torchrun 进程从 Step 2 恢复并继续至 Step 6；
- 恢复 Run 的指标 Step 严格为 `1, 2, 3, 4, 5, 6`，没有重复或跳过；
- 无中断与恢复 Run 的每步 Loss、LR、Gradient Norm 和 Tokens 逐条一致；
- 两者最终完整模型 SHA256 均为
  `fff26c6809c359695fa6536d89b65119cc83fa23062918260e70e305d1dea256`。

该结论只证明当前 Tiny Model CPU/Gloo 契约的逐位 Exact Resume；Qwen3-8B BF16 四卡容差
仍必须由 M4.3 的真实重复基线确定。

## 2. 正式运行身份

| 项目 | 实际结果 |
| -- | -- |
| Git Commit | `cf0f96c8b4f2cb71f335c4de9814f6cc40cf5e9f` |
| Git 状态 | clean |
| 配置 Hash | `372c60457b76f04b98bd87937b8c6db2620aad90e66e2887c0a23b6717a4f747` |
| Backend / World Size | Gloo / 2 |
| 无中断 Run | `20260718T082029Z-tinygpt-debug-fsdp2-gloo-dcp-recovery-372c6045-6084` |
| 恢复 Run | `20260718T082048Z-tinygpt-debug-fsdp2-gloo-dcp-recovery-372c6045-adf8` |
| 中断点 | Step 2 |
| 最终点 | Step 6 |
| 最终参数 Hash | 两个 Run 完全一致 |
| 指标 | 两个 Run 逐条一致 |

私有 Artifact Store 保存完整绝对路径与原始 Checkpoint；公开证据已移除用户名、主机名和
绝对路径，见 [机器可读摘要](raw/fsdp2_dcp_recovery.json)。

## 3. Checkpoint 内容与提交顺序

Step 2 Checkpoint 包含：

- DCP `.metadata`；
- 两个 `.distcp` 分片，大小分别为 603,481 和 662,139 Bytes；
- `rank-00000.pt` 与 `rank-00001.pt`；
- `runtime_state.pt`；
- `config.resolved.json` 与 `environment.json`；
- `manifest.json` 与最后提交的 `COMMITTED`。

Manifest 对每个文件记录路径、角色、大小和 SHA256，并声明模型、优化器、Scheduler、
GradScaler 不适用标记、Python/NumPy/PyTorch/CUDA RNG、Sampler、配置和环境均已覆盖。

发布顺序固定为：所有 Rank 在同一临时目录完成 DCP 写入和本地状态持久化，Rank 0 核对
Rank 连续性与 DCP 元数据，计算逐文件 Hash，写 Manifest 和 Commit Marker，原子 Rename
目录，重新完整校验，最后原子更新 `LATEST`。任何中间失败都不能成为有效恢复点。

## 4. Exact Resume 语义

恢复前先校验：

- Checkpoint 文件清单、大小与 SHA256；
- DCP Metadata 可解析；
- World Size 与 Rank 连续性；
- Run ID、配置 Hash、数据版本、Git Commit 与环境；
- Scheduler、TrainerState、Sampler Cursor 和各 Rank RNG Schema。

通过后才由所有 Rank 集体加载 DCP 模型/优化器状态，再恢复 Scheduler、Sampler、训练进度
和本地 RNG。恢复 Run 保留 Step 1–2 指标，并从下一批数据开始写 Step 3。

## 5. 失败路径

自动集成测试验证下列情况均 fail closed，并映射到 Checkpoint 退出类别：

| 失败条件 | 结果 |
| -- | -- |
| `.distcp` 追加损坏字节 | SHA256/大小校验拒绝 |
| 缺少 `COMMITTED` | 不完整 Checkpoint 拒绝 |
| 缺少一个 Rank 状态文件 | 文件清单/Rank 连续性拒绝 |
| 请求错误 World Size | Exact Resume 兼容性拒绝 |
| 修改训练配置 | Run 配置血缘拒绝 |
| 修改数据版本 | Checkpoint 数据血缘拒绝 |

部署 Safetensors 不参与上述恢复选择，也不能冒充完整训练 Checkpoint。

## 6. 自动验证

- `make check`：388 passed，2 个 GPU 测试按默认策略取消选择；
- CPU 可测试核心分支覆盖率：85.13%；
- Ruff、Ruff Format、MyPy Strict：通过；
- JSON Schema Snapshot、Markdown 链接和公开 Artifact 规则：通过；
- DCP 中断/恢复集成测试包含独立无中断基线与新进程恢复对照。

## 7. M4 边界

M4.2 完成不等于 M4 完成。M4.3 仍需真实四卡完成：固定 Revision Qwen3-8B Memory
Probe、50 Step、Step 25 到 Step 50 恢复、每 Rank Peak Memory 和独立 Safetensors 导出。
在这些证据产生前，项目不得宣称 Qwen3-8B 四卡训练已通过。
