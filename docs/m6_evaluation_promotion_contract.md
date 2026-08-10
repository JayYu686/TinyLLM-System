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
软件/硬件快照和通用评测结果；Base 复用已验证的 M2 Non-thinking 与通用证据，Candidate 必须
提供本轮 Thinking、Non-thinking 和完整通用评测。最终文件只保存内容无关的逐项得分和组合
Hash，不复制私有 Prompt、模型输出或 Thinking 内容。

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

最终复判使用训练前冻结的 `tinyllm-domain-holdout-v1-2b167ce6`，完整内容 SHA256 为
`2b167ce67a3761558bf2c556131d86eb572dc5d36e533a668a539a78eb86d6e2`。v3 与 v1/v2 的
精确 Prompt 交集均为 0，继续沿用第 5 节全部门禁阈值。生成方法见
[v3 评测集说明](../evals/domain/v3/README.md)，执行预注册见
[M6 R2/v3 计划](../reports/m6/m6_r2_v3_execution_plan.md)。
