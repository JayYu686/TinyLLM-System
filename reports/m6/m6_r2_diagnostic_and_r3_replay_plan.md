# M6 R2 失败诊断与 R3 回放修复预注册

## R2 实际结果

R2 Seed 42 完成 1,000,000 Supervised Tokens，训练过程无 NaN/Inf，Checkpoint、导出和血缘均完整。
开发代理评测得到：

- Thinking：51/200（25.50%），格式有效 199/200，长度截断 1/200；
- Non-thinking：30/200（15.00%），格式有效 200/200；
- 作为对照，上一版双模式修复 Candidate 分别为 187/200（93.50%）和 159/200（79.50%）。

R2 明显改善了推理长度，但能力回退远超可接受范围，因此不得进入冻结的 M6 v3 正式评测，也不得
作为 Candidate 导入。

## 根因

R2 从固定 Qwen3-0.6B 基座重新训练。它保留了 400K 通用 Non-thinking Token，但用新创作任务
替换了上一版 60K 领域 Non-thinking 与 300K Thinking 纠错监督。模型因此学会短格式和新任务，
却遗忘了上一版已经建立的领域任务映射。该现象属于数据分布替换导致的灾难性遗忘，而非训练进程、
CUDA、Checkpoint 或评测脚本故障。

## R3 预注册

R3 使用 Continual-learning Replay，从两个已校验私有 Artifact 按监督 Token 重采样：

- 上一版成功纠错监督：400K Non-thinking + 150K Thinking；
- R2 新门禁修复监督：300K Non-thinking + 150K Thinking；
- 合计：700K Non-thinking + 300K Thinking，共 1,000,000 Supervised Tokens；
- 训练仍从固定 Qwen3-0.6B 基座开始，优化器、学习率和训练预算保持不变；
- 主 Candidate 固定 Seed 42，Seed 20260812 只验证稳定性；
- 数据版本：`m6-gate-replay-mixture-v1-6c169970`；
- Manifest SHA256：`c5ceb1e5597a8e253d7c370484f9aa06d22b0a26dbfe597043d9302d8e580fa9`。

R3 必须先恢复开发代理集的双模式能力，并保持低截断率，才能进入一次性的 v3 正式门禁。冻结的
v3 Prompt、Reference 与逐项结果均未被数据构建器读取。
