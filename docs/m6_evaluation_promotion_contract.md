# M6 独立评测与 Candidate 晋级契约

## 1. 目标与边界

M6 使用独立冻结发布集验证 M5 训练产物的目标能力、通用能力、结构化输出和完整血缘，并将
通过全部门禁的模型从 `Development` 晋级为 `Candidate`。M6 不授予 `Production`；真实
推理延迟、吞吐、显存与失败率由 M7 门禁负责。

第一条正式晋级路线固定为 Qwen3-0.6B：

- Base：`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`；
- Candidate：M5 四卡 Full SFT 的 10M 最优开发快照
  `checkpoint-tokens-0010000532`；
- 50M 终点只作为预注册过拟合对照，不参与首个 Candidate 的阈值选择；
- Qwen3-8B LoRA 必须先补齐同 Revision、同协议的 8B Base，才能进入独立比较。

M5 Dev 结果不能代替 M6 指标，也不能用于修改本契约阈值。

## 2. 冻结输入

领域套件保持 `tinyllm-domain-v1-83bdd8ef` 的 300 条内容身份：210 条英文、90 条中文，覆盖
Python、Linux、JSON、配置修改、日志诊断、简短代码和无依据拒答。通用任务保持：

| 任务 | Split | 样本数 | 门禁指标 |
| -- | -- | --: | -- |
| ARC-Easy | test | 2,376 | `acc_norm` |
| HellaSwag | validation | 10,042 | `acc_norm` |
| PIQA | validation | 1,838 | `acc_norm` |

通用聚合在看到 Candidate 结果前固定为三项 `acc_norm` 的等权平均。原始样本、Prompt、输出和
人工判断保存在私有 Artifact Store；公共仓库只保存聚合指标、失败 Item ID 和不可逆 Hash。

## 3. 双模式执行

Base 与 Candidate 都必须完成相同 300 条领域任务的 Thinking 和 Non-thinking 两种模式：

- Thinking 使用 Qwen3 官方采样参数与 Thinking Budget 控制器，分别记录自然闭合和强制收束；
- Non-thinking 使用 Greedy Decoding，显式关闭 Thinking；
- 两种模式分别评分、比较和报告，禁止把较高模式覆盖较低模式；
- 40 条无依据拒答在每个模型、每种模式下都必须完成维护者人工判断；
- 通用任务只使用 Non-thinking Chat Template，保持与 M2 Base 可比。

领域总分以 300 条中的通过数为分母；客观评分和 Human Rubric 权重均为一条一票。JSON Valid
Rate 以 80 条 `json_object` 项为分母，并要求两种模式分别达标。

生成协议也属于 Release 配置身份：Thinking 固定 1,536 Token 思考预算、512 Token 最终答案
预算、Temperature 0.6、Top-p 0.95、Top-k 20 和 Seed `20260809`；Non-thinking 固定 Greedy
与 512 Token 输出预算。两种 Generation Template、通用任务 Chat Template、Task Adapter、
数据 Revision、样本数和 lm-eval 版本均进入同一配置 Hash，任一差异都会拒绝比较。

## 4. 配对 Bootstrap

领域增量使用 Base/Candidate 同 Item 配对差值执行 10,000 次 Cluster Bootstrap。90 组中英
翻译对按“类别 + Pair ID”构成一个 Cluster，120 条英文独立项各自构成一个 Cluster，共 210
个重采样单元。每次有放回抽取 210 个 Cluster，并保留 Cluster 内全部 Item，避免把翻译对
误当成相互独立的证据。

随机 Seed 固定为 `20260809`；Thinking 与 Non-thinking 使用相邻、确定性 Seed。95% 区间
使用 `nearest-rank-v1` 百分位方法。报告同时保存 Point Delta、下界、上界和配置 Hash。

## 5. Candidate AND Gate

以下检查必须同时通过：

1. Thinking 领域总分相对同模型 Base 提升至少 3pp；
2. Thinking 配对 Bootstrap 95% CI 下界大于 0；
3. Non-thinking 领域总分相对同模型 Base 提升至少 3pp；
4. Non-thinking 配对 Bootstrap 95% CI 下界大于 0；
5. ARC-Easy、HellaSwag、PIQA 等权 `acc_norm` 聚合回退不超过 2pp；
6. Thinking 与 Non-thinking 的 JSON Valid Rate 都至少为 98%；
7. Thinking 控制后格式率至少为 99%，强制收束率不超过 10%；
8. Non-thinking 可见推理泄漏率为 0；
9. 两种模式的 40 条 Human Rubric 全部完成审查；
10. 模型、数据、训练 Run、Checkpoint、配置、Git、软件、硬件、评测和导出血缘完整。

任一条件失败时，结果和失败样例继续保留，模型状态维持 `Development`。门禁代码返回退出码
6；配置错误返回 2，环境或 Artifact 错误返回 3。

## 6. CLI 与 Artifact

首批稳定接口为：

```text
tinyllm eval m6-assemble --role base|candidate ... --output ... --json
tinyllm compare --config ... --baseline ... --candidate ... --output ... --json
tinyllm promote --comparison ... --registry-root ... --json
```

`m6-assemble` 在生成最终 `evaluation.json` 前重新校验导入记录、双模式原始结果、维护者判断、
软件/硬件快照和通用评测结果。v1 Base 可复用协议完全相同的 M2 Non-thinking 与通用证据；
v2/v3 Holdout Base 只复用模型身份和协议未变化的通用证据，Thinking 与 Non-thinking 领域结果
均须在对应冻结题集上重新执行。Candidate 必须提供本轮 Thinking、Non-thinking 和完整通用
评测。最终文件只保存内容无关的逐项得分和组合 Hash，不复制私有 Prompt、模型输出或
Thinking 内容。

`compare` 只接受相同模型 Revision、Tokenizer、Prompt、Suite 和 M6 配置身份的完整结果。
`promote` 只接受 `accepted` 比较结果，在私有 Registry 中原子写入不可变 Candidate 记录；相同
Comparison 可幂等读取，冲突身份拒绝覆盖。

正式评测 Run 至少保存：

```text
run.json
config.resolved.json
environment.json
hardware.json
domain/thinking/results.jsonl
domain/nonthinking/results.jsonl
domain/*/human_review/
general/raw/
evaluation.json
comparison.json
```

公开报告必须说明 M5 Dev 与 M6 Release 的隔离、人工评审状态、控制器干预率、失败项、置信
区间和适用边界。

## 7. 执行批次

1. M6.0：冻结本契约、Schema、Cluster Bootstrap、比较与 Candidate Registry 接口；
2. M6.1：导入并校验 0.6B Base 的 Non-thinking 历史证据，补跑 Base Thinking；
3. M6.2：对 10M Candidate 执行双模式领域和完整通用评测，完成 Human Rubric；
4. M6.3：生成比较、执行 Promotion Gate、重建 SQLite 查询索引和发布中文报告；
5. M6.4：运行 50M 过拟合对照；资源允许时再建立 8B Base/LoRA 可比批次；
6. M6.5：发布 `v0.6.0-rc.1`、演示脚本和真实指标版本说明。

M6.1 的历史复用仅适用于与 Release 配置逐字段等价的 M2 正式证据。导入器必须重新校验 Run
状态、配置 Hash、模型文件、300 条输出、40 条人工判断、通用任务原始结果树、软件环境和硬件
快照；只复制聚合数字或缺少原始文件时直接拒绝。Thinking 没有兼容历史证据，必须通过
`tinyllm eval m6-domain --mode thinking` 在 clean `main` 上重新生成，并由
`tinyllm eval m6-domain-review` 完成 40 条维护者判断。

## 8. v1 执行结果与修复边界

首个 0.6B 10M Candidate 已完成 v1 双模式领域、人工审查和通用评测。门禁有效拒绝：Thinking
领域分数从 Base 的 26.67% 降至 9.33%，Non-thinking 只提升 0.67pp；两种模式 JSON Valid
Rate 分别为 71.25%和 56.25%，Thinking 强制收束率为 99.67%。通用任务等权 `acc_norm` 仅
回退 0.65pp，通过该单项门禁。

代码审计确认 M5 Non-thinking SFT 缺少 Qwen3 Hard Switch 使用的空 Think 上下文，导致双模式
在 Assistant Header 后形成竞争目标。该问题按
[ADR-0007](adr/0007-qwen3-dual-mode-sft-template-alignment.md) 修复；完整证据见
[M6 v1 门禁拒绝分析](../reports/m6/m6_gate_rejection_analysis.md)。v1 结果保持不可变，旧
Candidate 保持 `Development`。

修复模型不能在已经用于诊断的 v1 发布集上反复选优。后续晋级使用新的 M6 v2 内容与配置
身份，继续沿用第 5 节的量化阈值、双模式分别报告、完整人工审查、通用回归和血缘要求。

M6 v2 已在修复模型 Proxy 结果产生前冻结为
`tinyllm-domain-holdout-v1-c0c948cc`，完整内容 SHA256 为
`c0c948cc5282cfaa15baae689ddf0bf51c0d59ece6e01554df480bc16a6d3842`。v2 保持 300 条、
英文 210/中文 90、七类任务、80 条 JSON、40 条人工 Rubric 和 90 个双语 Cluster；与 v1
的精确 Prompt 交集为 0。生成与重建方法见 [v2 评测集说明](../evals/domain/v2/README.md)。

## 9. v2 诊断与 v3 独立复判

M6 v2 证明 Non-thinking 模板对齐和通用能力保持已生效，同时识别出 Thinking 强制收束率超限
以及证据拒答泛化不足。R2 使用独立创作的短推理、证据拒答双模式数据处理这两个失败机制；
训练数据不读取 M6 v3 的答案、模型输出或逐项得分。

R2 的开发代理评测显示长度问题已改善，但 Thinking/Non-thinking 正确率分别回退到 25.50%和
15.00%。根因是 R2 从基座重新训练时替换了上一版成功的领域纠错监督，形成灾难性遗忘。R2
因此不进入 v3。R3 预注册为 55% 上一版纠错回放与 45% R2 修复监督的精确 Token 混合，保持
70/30 双模式比例；完整诊断和数据身份见
[R2 失败与 R3 回放计划](../reports/m6/m6_r2_failure_and_r3_replay_plan.md)。

最终复判使用训练前冻结的 `tinyllm-domain-holdout-v1-2b167ce6`，完整内容 SHA256 为
`2b167ce67a3761558bf2c556131d86eb572dc5d36e533a668a539a78eb86d6e2`。v3 与 v1/v2 的
精确 Prompt 交集均为 0，继续沿用第 5 节全部门禁阈值。生成方法见
[v3 评测集说明](../evals/domain/v3/README.md)，执行预注册见
[M6 R2/v3 计划](../reports/m6/m6_r2_v3_execution_plan.md)。R3 只有先通过开发代理检查，才可使用
该冻结 v3 执行一次正式复判。

## 10. v3 拒绝与 v4 密封终审

v3 自动评分确认 R3 的 Thinking 能力相对 Base 回退，且两种模式 JSON Valid Rate 分别只有
86.25% 和 83.75%，因此无需等待人工评分即可确定拒绝。根因是修复集覆盖面过窄并从 Base
重新训练，模型学到局部模板而没有形成七类任务的泛化能力。完整数据见
[v3 拒绝与 R4 计划](../reports/m6/m6_v3_rejection_and_r4_plan.md)。

R4 从已冻结的 M5 10M 正式快照暖启动，使用不读取任何评测答案或模型输出的 900 条七类任务
做 1M Token 低学习率训练。最终门禁使用 R4 训练前冻结的
`tinyllm-domain-final-audit-v1-bac25144`，内容 SHA256 为
`bac25144d53d186693514f6a421e3894a820bddb039c75ca29c2484190b7913a`。v4 与 v1–v3 的
完整 Prompt 交集为 0，并继续沿用第 5 节的全部阈值；v4 一旦用于结果诊断，不再回流训练。

R4 的 v3 开发验证将剩余问题收敛到 JSON 顶层对象契约、证据拒答和 Thinking 首个关闭标签
停止条件。R4.1 在不读取 v4 内容的前提下扩展到 1500 个训练任务，并从同一 M5 10M 正式
快照重新训练，避免顺序微调造成血缘不清或灾难性遗忘。R4.1 通过 v3 开发门禁后才允许首次
执行 v4。

v4 的推理协议固定启用可审计输出控制。JSON 控制器只修复语法外壳且保留原始输出，不允许
读取 Reference 或修改叶子值；证据拒答使用固定双语 System Policy。Base 与 Candidate 必须
使用同一控制配置，比较报告同时保留修复计数。控制器不能替代 3pp + Bootstrap CI、人工
Rubric、通用回退和完整血缘等门禁。

Thinking 首段在首个 `</think>` 自然停止后，控制器必须注入 Qwen3 模板定义的 `\n\n`
Final-Answer 分隔符再续写。分隔符的 ID、SHA256、原文与 Token 数均进入配置或私有 Transcript；
强制闭合路径继续记录完整注入文本。该规则只修复 Chat Template 边界，不读取 Reference、
不生成答案内容，也不改变采样参数和评分阈值。

Thinking 与 Final Answer 使用分阶段解码：思考首段保留 Qwen3 的 Temperature 0.6、Top-p 0.95、
Top-k 20 采样；关闭标签后的 Final Answer 固定使用 Greedy，避免重启随机采样造成模板边界
重复。Final Answer Batch Size 固定为 4，并使用左填充；Batch Shape 属于可复现配置身份，不得
在同一比较中改为逐条续写。续写上下文固定由 Prompt、解码后的可见思考首段和控制文本整体
重新 Tokenize，禁止把 `stop_strings` 为已完成 Row 补出的隐藏 Pad Token 直接拼入上下文。
JSON Syntax-only v3 继承 v2 的代码块去除、裸标识符键补引号、单键箭头/裸词外壳和必需顶层键
提升，并允许在缺失一个右花括号后串联一次必需键提升。串联动作只读取 Prompt 中公开的必需键，
补齐对象外壳并移动完整子树，不读取 Reference，也不改写任何叶子值。原始答案、原始哈希、动作
和规范化结果均须保留；重复 Thinking 标签、格式错误和语义错误继续计为失败。

v4 正式执行确认 Thinking 边界控制有效，但 Candidate 的 Thinking/Non-thinking JSON Valid
分别只有 `74/80` 和 `77/80`，低于 `79/80` 门禁，因此在人工评分和通用评测前机械拒绝。
其余自动指标为：Thinking `127/260`、格式 `297/300`、强制闭合 `8/300`、泄漏 `0/300`；
Non-thinking `123/260`、格式 `300/300`、泄漏 `0/300`。Base 对应客观正确为 `93/260`
和 `60/260`。v4 结果保持不可变，不允许基于失败正文修改后重跑同一套。

在读取 v4 JSON 失败正文前，项目冻结 v5 `tinyllm-domain-json-audit-v1-3e5fffd7`，内容
SHA256 为 `3e5fffd7d408a6d2d237f4da7f5e3ecfb72523bd5f9e42b6e74f24e9199b1bfe`。
v5 与 v1–v4 的完整 Prompt 交集为 0，模型权重保持不变，只验证版本化 JSON 约束解码是否跨
独立内容泛化。v4 失败输出只能用于推理解码器诊断，不能进入训练数据或 v5 内容。

v5 的 80 条 JSON-object 任务固定使用 `xgrammar-json-shape-v1` 与 XGrammar 0.2.4。每题
Schema 只保留冻结评分契约中的字段名、对象/数组层级和 JSON 类型，删除所有 Reference 叶子值，
且不限制数组长度。Grammar 只约束 Token 级输出结构，不改变模型权重、Prompt、采样种子、
评分答案或门禁阈值。Thinking 只约束 `</think>` 后的 Final Answer，私有思考首段继续使用
冻结采样策略；Non-thinking 直接约束最终输出。

Base 与 Candidate 必须使用完全相同的逐题 Schema。配置记录解码器 ID、版本和 Schema Policy；
私有 Transcript 记录逐题 Schema SHA256，Summary 必须证明 80 条 JSON 全部经过约束。依赖
缺失、版本漂移、不同 Scorer 混批或 JSON 任务绕过约束时，评测失败关闭。v5 仍只允许一次
完整正式审计；v4 的 9 条失败重放仅用于选择解码机制，不得作为 v5 成绩。

v5 Candidate Non-thinking 达到 `80/80` JSON，但一条正确标量答案后继续生成 `</think>`，
造成可见泄漏 `1/300`，因此 v5 在首路后即机械拒绝，其余三路不再执行。v5 原始证据不可修改。

v6 在实现停止策略前冻结为 `tinyllm-domain-output-boundary-audit-v1-c34f63a8`，内容 SHA256
为 `c34f63a87c05910f421db19c71eede7368328028f81bbf08870070bb2fba6002`；与 v1–v5 完整
Prompt 交集为 0。v6 固定增加 `truncate-before-first-thinking-tag-v1`：Non-thinking 在首次
生成 `<think>` 或 `</think>` 时停止，只发布标签前文本。私有 Transcript 必须保留原始文本、
停止原因和截断动作；空前缀或错误前缀仍按原评分器失败。该策略不用于 Thinking，不修改模型
权重、答案正文、JSON Schema 或门禁阈值。

v6 Candidate Non-thinking 达到 JSON `80/80`、格式 `300/300`、泄漏 `0/300`；Candidate
Thinking 达到 JSON `80/80`、强制闭合 `9/300`、泄漏 `0/300`，但格式为 `296/300`，低于
`297/300` 门槛，因此 v6 在 Candidate 机械检查后拒绝，未执行 Base、人工评分和通用评测。
4 条格式失败均为 Final Answer 续写先生成非空答案，随后再次生成 `</think>` 与重复答案。

在实现 Thinking Final Answer 停止策略前，项目冻结 v7
`tinyllm-domain-thinking-boundary-audit-v1-b82cbca1`，内容 SHA256 为
`b82cbca1821cadbaf4872636e89c61cef730ebe09413f9c63f34993302b6f955`；与 v1–v6 完整
Prompt 交集为 0。v7 增加 `truncate-before-next-thinking-tag-v1`：只在 Thinking 的 Final
Answer 续写阶段注册 `<think>` 与 `</think>` 停止串，并只评分首次 Thinking 关闭标签之后、
下一 Thinking 标签之前的非空答案。私有 Transcript 保留包含停止标签的原始响应、停止原因和
逐条截断证据，Summary 记录截断条数。空前缀、错误答案和任何首次关闭前的失败仍按原规则计分。
该策略不改变模型权重、Prompt、Reference、JSON Schema、采样策略、评分器或门禁阈值。

## 11. v7 最终验收

v7 完成 Base/Candidate × Thinking/Non-thinking 四路 300 题评测、三项完整通用任务和 160 条
维护者人工判断。Candidate Thinking 从 34.33% 提升至 41.67%，Cluster Bootstrap 95% CI 为
`[+0.33, +14.29]pp`；Non-thinking 从 22.33% 提升至 40.67%，95% CI 为
`[+12.46, +24.40]pp`。通用等任务 `acc_norm` 从 51.80% 提升至 54.48%。Candidate 双模式
JSON Valid 均为 100%，Thinking 格式为 100%、强制闭合为 1.67%，Non-thinking 可见推理
泄漏为 0。

全部 11 项 Candidate Gate 通过。模型以 `qwen3-0-6b-m6-d16c2357` 原子注册为 Candidate，
同时从 57 个历史 Run Manifest 真实重建 SQLite v1 查询索引。M5 已冻结的 10M/50M 长程曲线
作为 M6.4 过拟合对照复用，不替代 v7 独立评测。完整结果见
[M6 验收报告](../reports/m6/m6_acceptance.md)。M6 状态更新为 `COMPLETE`，Production 门禁继续
由 M7 负责。
