# M5 正式训练失败路径验收

## 1. 结论

M5 契约要求的七类失败路径已经全部通过安全 CPU 故障注入验收：CUDA OOM、NaN/Inf、
Checkpoint 损坏、磁盘空间不足、数据身份漂移、World Size 错误和训练子进程异常退出。七个
场景都由正式训练路径复用的校验函数拒绝，汇总状态为 `passed`。

本报告只证明系统能够识别并拒绝这些错误，不把故障注入描述为真实 GPU OOM、真实磁盘写满
或模型质量结果。Qwen3-8B BF16 LoRA 的正式 Probe 和 10M Token 训练均已成功，因此没有触发
QLoRA 回退。

## 2. 验收矩阵

| 失败路径 | 注入方式 | 预期行为 | 实际结果 |
| --- | --- | --- | --- |
| CUDA OOM | 构造 `torch.OutOfMemoryError` | 归一化为稳定的 M5 OOM 错误 | 按预期拒绝 |
| NaN/Inf | 向训练指标校验传入 `NaN` | 在 Optimizer 更新前中止 | 按预期拒绝 |
| 坏 Checkpoint | 修改 Payload 并保留原摘要 | SHA256 完整性校验失败 | 按预期拒绝 |
| 磁盘不足 | 注入低于最低余量的容量信息 | 在占用 GPU 前拒绝启动 | 按预期拒绝 |
| 数据漂移 | 注入不同的数据版本和 Manifest 摘要 | 拒绝静默切换数据 | 按预期拒绝 |
| 错误 World Size | 以 2 对照正式要求的 4 | 在模型构造前拒绝启动 | 按预期拒绝 |
| 进程退出 | 启动固定退出码为 17 的安全子进程 | 不接受部分产物 | 按预期拒绝 |

## 3. 生产路径绑定

- 0.6B Full SFT 启动前要求输出文件系统至少保留 64 GiB；
- 8B LoRA 启动前要求输出文件系统至少保留 16 GiB；
- Full SFT 和 LoRA 共用数据版本、Manifest、有限值和子进程状态校验；
- Full SFT 的四卡 Worker 在 CUDA 初始化前验证 `WORLD_SIZE=4`；
- 已有 Full SFT 与 LoRA Checkpoint Store 继续执行 Manifest、Commit Marker 和 Payload
  SHA256 校验；
- 训练子进程返回非零退出码时，Campaign 不会生成成功结果。

## 4. 证据边界

机器可读结果见 [m5_failure_paths.json](raw/m5_failure_paths.json)。其中
`model_generated=false`、`quality_metric=false` 和
`injection_kind=safe_cpu_fault_injection` 明确限制了结论范围。真实 GPU 训练、恢复、显存与
评测结果由各自 Run、Campaign 和最终 M5 验收报告单独给出。
