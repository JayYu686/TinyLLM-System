# M6 R2 修复与 v3 正式执行预注册

## v2 诊断结论

M6 v2 证明双模式模板冲突已经修复，但首个修复 Candidate 仍有两个阻塞：

- Thinking 强制收束 43/300（14.33%），超过不高于 10% 的门禁；
- 新拒答题上经常把缺失证据误写成根因，未稳定学会“证据不足则拒绝归因并索要材料”。

同时，Candidate Non-thinking 的客观题从 Base 的 18/260 提升到 50/260，JSON Valid 从
31/80 提升到 78/80；通用任务等权 `acc_norm` 从 51.80% 提升到 56.27%。这些结果说明模板、
双模式隔离和通用能力保持已经有效，R2 只处理剩余两项失败机制。

除上述两项外，Candidate Thinking JSON Valid 为 72/80（90.00%），Candidate Non-thinking
JSON Valid 为 78/80（97.50%）；两者均未达到 98% 的门槛。因此 R2 中的领域 Non-thinking
与简洁 Thinking 样本也显式强化严格 JSON 输出，不能把接近阈值视为通过。

## R2 数据边界

- 数据：`m6-gate-repair-mixture-v1-be2aa7fa`；
- 总预算：1,000,000 Supervised Tokens；
- 400K 通用 Non-thinking、300K 领域 Non-thinking、300K 简洁 Thinking；
- 680 对独立创作双模式任务，其中 300 对为证据拒答；
- 英文/中文比例 70/30；
- Thinking 单样本最大监督长度 49 Token；
- 与 M6 v1/v2/v3 精确 Prompt 交集均为 0；
- 生成器不读取任何 M6 模型输出、Reference Answer 或逐项得分。

首选 Candidate 固定为 Seed 42 的完整 1M Token Run。Seed 20260811 只验证修复稳定性，不能
根据分数替换 Seed 42。

## v3 正式门禁

- Protocol：`m6-release-v3`；
- Suite：`tinyllm-domain-holdout-v1-2b167ce6`；
- Base 与 Seed 42 Candidate 均执行 Thinking/Non-thinking 领域评测；
- Candidate 执行完整 ARC-Easy、HellaSwag、PIQA 通用回归；
- 四个领域 Pass 分别完成 40 条人工 Rubric；
- 继续使用原 11 项门禁、10,000 次 Paired Cluster Bootstrap 和既有阈值；
- v2 不作为最终晋级证据，所有 v2 失败 Artifact 保留。
