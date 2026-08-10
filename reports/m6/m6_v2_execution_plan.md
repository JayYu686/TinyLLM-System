# M6 v2 正式执行预注册

## 状态

本文件在第二 Seed Proxy 结果产生前冻结 M6 v2 的 Candidate 选择与执行边界。门禁阈值沿用
M6 v1，不因 v1 拒绝或 Proxy 指标调整。

## Candidate 选择

正式 Candidate 固定为首个预注册 Run：

- Run：`20260810T095257Z-m6-dual-mode-fix-seed42-ffd49bd9-aeb6`；
- Seed：42；
- 训练：1,000,000 Supervised Tokens，321 Step；
- 数据：`m5-dual-mode-correction-mixture-v1-4bc342d4`；
- 模型导出 SHA256：
  `fd5f6b3d781c2b9a70e36f50a4efa3e205f70d9131b7e7069f58a7fc46e4a78c`；
- Attention：GQA；双模式模板使用 ADR-0007 的对齐策略。

Seed 20260810 Run 只用于检查修复方向的稳定性，不能根据它的 Proxy 分数替换 Seed 42，避免
在开发集上做事后选优。

## 发布集与执行

- Protocol：`m6-release-v2`；
- Suite：`tinyllm-domain-holdout-v1-c0c948cc`；
- 300 条 Prompt 均在 Seed 42 Proxy 完成前冻结；与 v1 精确 Prompt 交集为 0；
- Base 与 Candidate 都重新执行 Thinking/Non-thinking 领域集；
- Candidate 重新执行 ARC-Easy、HellaSwag 和 PIQA；Base 通用结果可复用未变化任务上的完整
  M2 原始证据；
- 四个领域 Pass 各完成 40 条人工 Rubric，自动评分项目不人工改分；
- 最终只由 `tinyllm compare` 的固定 11 项检查决定 Candidate 是否晋级。

## 固定门禁

- Thinking 与 Non-thinking 领域分数分别相对 Base 提升至少 3pp；
- 两种模式的 Paired Cluster Bootstrap 95% CI 下界分别大于 0；
- 通用等权 `acc_norm` 回退不超过 2pp；
- 两种模式 JSON Valid Rate 均至少 98%；
- Thinking Format 至少 99%，强制收束不超过 10%；
- Non-thinking 可见推理泄漏为 0；
- 人工审查与训练/评测血缘完整。
