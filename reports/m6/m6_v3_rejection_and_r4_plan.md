# M6 v3 拒绝原因与 R4 收敛方案

## 1. 结论

R3 Candidate 无法通过 v3 门禁。该结论由自动评分即可确定，无需为了形成拒绝结论而补写
40 条人工判断。门禁保持原值，v3 保留为失败证据与开发集；最终晋级只接受训练前冻结的 v4
密封终审结果。

## 2. v3 实测结果

| 模型 / 模式 | 自动正确（260） | JSON Valid（80） | 格式有效（300） | 强制收束（300） |
| -- | --: | --: | --: | --: |
| Base Thinking | 87 | 66 | 300 | 21 |
| R3 Thinking | 73 | 69 | 298 | 2 |
| Base Non-thinking | 45 | 49 | 300 | 0 |
| R3 Non-thinking | 75 | 67 | 300 | 0 |

R3 的 Thinking 强制收束率已从代理阶段的问题降至 `0.67%`，但同时出现三项确定性失败：

1. Thinking 自动正确数比 Base 少 14 条，无法达到两种模式均提升至少 3pp；
2. Thinking JSON Valid Rate 为 `86.25%`，低于 `98%`；
3. Non-thinking JSON Valid Rate 为 `83.75%`，低于 `98%`。

因此 v3 状态应为 `Development / Rejected`。通用评测与人工评分不会改变以上三项失败事实，
本轮不消耗额外 GPU 时间补跑无效门禁步骤。

## 3. 根因

逐类别审计显示 R3 并未学习任务族，只记住了狭窄修复样例：

- R3 数据中的 JSON 主要覆盖单一字段修改，v3 包含对象包装、数组、集合与多种结构变换；
- Python 监督主要覆盖一个 `sum/range` 模式，v3 包含十类 Python 语义；
- Linux 只有少量命令模板，日志诊断没有覆盖冻结的四选一输出契约；
- R3 从 Base 权重开始做 1M Token 修复，没有继承 M5 10M 正式 SFT 已获得的广泛能力；
- 小规模修复数据重复率高，造成精确映射提升和跨模板泛化下降并存。

这解释了此前“每次修复只有局部小幅变化”的现象：优化对象是失败样例而非能力分布。

## 4. R4 方案

R4 改为一次性处理结构性根因：

1. 从 M5 10M 正式 Checkpoint 暖启动，保留原有通用与领域能力；
2. 构建 900 条独立训练任务，完整覆盖 Config、JSON、Linux、Logs、Python、Refusal、
   Short Code 七类任务；
3. 每条任务同时生成 Thinking 与 Non-thinking 监督，保持 30% / 70% Token 比例；
4. 1M Token 中保留 250k 通用 Non-thinking 回放，450k 领域 Non-thinking，300k 领域
   Thinking；
5. 使用较低学习率 `1e-5`，降低对 10M 正式能力的破坏；
6. 训练集与所有 v1–v4 完整 Prompt 的交集为 0；
7. v3 只做开发诊断，v4 只执行一次最终门禁。

R4 数据身份：

```text
mixture_version: m6-domain-generalization-mixture-v1-6c2f59e6
content_sha256: 6c2f59e6463679bf3407783345d6bc74bd0e33cf06d133eb2edac8efe9e1947f
manifest_sha256: 40c7a85edb392b165e2a05f50dbe998cc62ffe96115af27896bf8d5d15401eb9
evaluation_prompt_overlap_count: 0
```

v4 终审身份：

```text
suite_version: tinyllm-domain-final-audit-v1-bac25144
content_sha256: bac25144d53d186693514f6a421e3894a820bddb039c75ca29c2484190b7913a
```

## 5. 验收顺序

R4 必须依次完成：暖启动血缘检查、1M Token 训练、v3 开发验证、v4 Base/Candidate 双模式
评测、40 条人工判断、完整通用评测、Bootstrap 对比与 Candidate Promotion。任何失败结果均
保留原始 Artifact，不调整门禁阈值，也不把 v4 重新用作训练反馈。

## 6. R4 v3 开发验证与 R4.1

R4 在 clean `b257a33` 上完成 1M Token 训练，v3 仅作为退役开发集执行自动项：

| 模式 | Base 自动正确 | R4 自动正确 | R4 JSON Valid | R4 格式有效 | R4 强制收束 |
| -- | --: | --: | --: | --: | --: |
| Thinking | 87/260 | 135/260 | 72/80 | 296/300 | 2/300 |
| Non-thinking | 45/260 | 147/260 | 75/80 | 300/300 | 0/300 |

七类任务泛化已大幅改善，强制收束也通过阈值；剩余失败集中为数组/标量结果遗漏最外层 JSON
对象，以及证据拒答把被怀疑组件复述为根因。另有 4 条 Thinking 在自然关闭后继续生成第二个
`</think>`，属于控制器停止条件缺失，而非长度预算问题。

R4.1 保留 R4 全部 900 条任务，新增 240 条完整 JSON 对象契约和 360 条证据拒答契约；仍从
M5 10M 正式快照暖启动，仍为 1M Token、70/30 双模式、v1–v4 Prompt 零重合。Thinking
控制器在首个 `</think>` 停止首段，再生成 Final Answer。R4.1 数据身份：

```text
mixture_version: m6-domain-generalization-mixture-v2-f2e029e4
content_sha256: f2e029e430ccf68753beeb09a9ba875b441246284c546c77fbb96422e0e0503d
manifest_sha256: 288b0c88c91c49b466e9aeee07f9087a69c0f6618f19462621730390831289aa
authored_source_tasks: 1500
evaluation_prompt_overlap_count: 0
```

R4 及其 v3 开发输出保留为诊断证据；只有 R4.1 可以进入 v4 密封终审。

## 7. v4 输出控制边界

R4.1 的 v3 Non-thinking 开发结果仍有 7 条 JSON 输出只缺语法外壳，继续增加同类 SFT 数据
未带来改善。v4 因此冻结两项对 Base/Candidate 完全相同的推理控制：

- JSON Syntax-only Repair 只允许把可解析的裸值放入 Prompt 已声明的唯一顶层键、给成员片段
  补外层花括号，或补一个缺失的右花括号；禁止读取 Reference，禁止改写任何已解码叶子值；
- Evidence-grounding System Policy 要求在题目明确缺少证据时不复述可疑组件为根因，明确说明
  证据不足，并请求题目列出的全部证据。

每条 JSON 修复同时保存原始答案、原始哈希、修复动作和最终答案。最终报告必须同时给出
Raw 与 Post-control 数量；v4 Base 和 Candidate 使用相同配置与代码路径。该控制器是部署输出
契约的一部分，不改变领域提升、Bootstrap、通用回退或人工 Rubric 阈值。
