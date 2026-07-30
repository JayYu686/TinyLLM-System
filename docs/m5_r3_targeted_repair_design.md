# M5.2-R3 Config/Log 定向简洁推理修复设计

## 1. 目标

M5.2-R3 用新的、可验证且低重复的 Config/Log Teacher Trace，检验训练侧的任务定向数据能否
改善原生 Thinking 闭合可靠性。R3 保持模型、GQA、1M Supervised Token 预算、训练参数、
冻结 Dev 和 99%格式门禁不变，只替换 R1 中 150K Repair Thinking Token 的来源策略。

R3 不通过增加解码上限、运行时补标签或降低门禁来掩盖失败。正式 Thinking 上限继续使用
896，M5.3 在 R3 双 Seed Gate 通过前保持阻塞。

## 2. 设计依据

R2 已排除“896 输出无法重放”的可复现性问题，也证明仅增加长度不足：

| Seed | 1536 投影格式率 | 1536 未恢复格式 |
| --: | --: | --: |
| 42 | 98.0% | 4 |
| 20260727 | 96.5% | 7 |

1536 仍未恢复的 11 条输出中，Config 8 条、Log Diagnosis 2 条、Python 1 条；失败输出平均
Repeated 8-gram Ratio 为 25.82%/24.95%，同任务族有效对照只有 1.28%/1.24%。因此 R3
聚焦 Config/Log 的推理长度和重复，而不重做五类通用短样本池。

现有 96 条 Teacher Pilot 的真实审计进一步显示：

| 任务族 | 已有 Trace | 推理 Token P50/P90/Max | 满足 R3 Trace Policy | 英文/中文 |
| -- | --: | -- | --: | -- |
| Config | 19 | 323 / 764 / 800 | 2 | 2 / 0 |
| Log Diagnosis | 20 | 242 / 272 / 392 | 4 | 3 / 1 |

现有 Pilot 只有 6 条满足 R3 标准，无法支撑至少 160 条的定向来源门禁。R3 必须建立新的数据
版本，禁止继续高频复用 R1 的 40 条短样本。

## 3. 不可变输入与范围

R3 保持：

- `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`；
- `attention_architecture=gqa`；
- `m5-reasoning-dev-v1-53ddf557`；
- Thinking 最大 896 New Tokens，Non-thinking 最大 128 New Tokens；
- 两个训练 Seed：`42`、`20260727`；
- 单卡 BF16、Micro Batch 4、梯度累积 2、Learning Rate `2e-5`；
- 1M Supervised Token 和原 Checkpoint/Exact Resume 契约；
- Non-thinking 回退不超过 2pp、两个 Seed Thinking 格式率均至少 99%的 Gate。

R3 不读取 M6 冻结评测，不改变 M5.2/R1/R2 的历史结果，也不进入 M5.3 长程训练。

## 4. Trace 接受策略

每条 R3 Teacher Trace 必须同时满足：

1. 使用固定 Qwen3-0.6B Tokenizer 计算的可见推理不超过 192 Token；
2. Final JSON 通过 `m5-json-exact-v1`；
3. 生成以完整 `</think>` 和最终答案自然结束，禁止截断后接受；
4. Repeated 8-gram Ratio 不超过 5%；
5. 同一 Trace 内不存在重复的非空规范化行；
6. 规范化 Trace SHA256 在整个 R3 来源中唯一；
7. 完整 ChatML 序列不超过 1024 Token；
8. 不执行 Teacher 生成的 Python 或 Shell 内容。

5%的重复阈值高于 R2 有效对照的约 1.2%，同时显著低于失败输出的约 25%，用于拒绝明显循环，
不作为模型质量分数。Trace 不允许人工改写、自动摘要、补闭标签或字符串后处理。

## 5. 新任务与 Teacher Pilot

### 5.1 任务多样性

新任务只覆盖 Config 和 Log Diagnosis，并使用独立 Task Seed、ID Namespace 和 Template
Family。每类仍保留四个冻结标签，但每个标签至少提供六种独立证据模板；只改变 Case ID 的
Prompt 不计为新的证据变体。

任务集必须同时与冻结 Dev 和历史 Pilot 检查：

- Exact Prompt；
- 去除 Case ID 后的规范化 Prompt；
- Template Family；
- 期望答案与证据模板组合重复。

任一污染命中都会阻断 Teacher 生成。

### 5.2 R3-P0 可行性 Pilot

正式扩容前先生成 40 个任务：

- Config 20 条、Log Diagnosis 20 条；
- 每类英文 14 条、中文 6 条；
- Teacher 固定为
  `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`；
- 原生 Thinking、BF16、GQA、`trust_remote_code=false`；
- Temperature 0.6、Top-p 0.95、Top-k 20、Repetition Penalty 1.0；
- 每个任务最多两个候选，最大 384 New Tokens；
- 使用新的 Task Seed 和 Sampling Seed。

P0 每类至少接受 14 条，其中英文至少 10 条、中文至少 4 条，并且所有接受样本通过第 4 节
Trace Policy，才允许正式扩容。P0 失败时不得降低 Trace 标准或提高生成上限；必须保留失败
统计并重新审查任务模板或 Teacher 控制策略。

### 5.3 正式来源门禁

正式输入为每类 120 个任务，其中英文 84 条、中文 36 条。最终确定性选择：

| 任务族 | 英文 | 中文 | 合计 |
| -- | --: | --: | --: |
| Config | 56 | 24 | 80 |
| Log Diagnosis | 56 | 24 | 80 |
| 合计 | 112 | 48 | 160 |

每个分层按推理 Token 数、重复率、Sample ID 稳定排序。训练混合中同一来源最多出现四次；
无法满足数量或复用上限时，数据构建失败，不用旧 Pilot 补齐。

## 6. R3 训练混合

总预算与 R1 相同：

| 分层 | Supervised Token | 变化 |
| -- | --: | -- |
| M2 Non-thinking | 700,000 | 不变 |
| 原完整 Pilot Thinking | 150,000 | 不变 |
| R3 Config/Log Targeted Thinking | 150,000 | 替换 R1 通用短修复池 |

这样保持总体 Thinking 比例 30%，只检验“多样、简洁、任务定向的 Trace”相对 R1
“少量、五类均衡、反复复用的短 Trace”是否改善格式可靠性。Manifest 必须保存三个分层的
来源身份、精确 Token、复用次数、部分 Mask 次数、语言/任务分布和内容哈希。

## 7. 训练与评测决策

R3 继续执行两个 1M Token Seed，并使用原冻结双模式 Dev：

1. 两个 Seed 的 Non-thinking 分数相对 Base 回退均不超过 2pp；
2. 两个 Seed 的 Thinking 格式率均至少 99%；
3. 报告 Thinking Final-answer 分数、Length-limited 数和任务族回退；
4. 只有 1 和 2 同时满足才允许解锁 M5.3；
5. 任一失败都保留 `Development` 状态，不继续增加 Repair Token 或解码上限。

R3 不是 M6 Candidate Gate。即使通过，也只确定 M5.3 可采用的训练数据策略。

## 8. 失败路径

实现必须覆盖：

- Pilot、R2 Decision、Reasoning Config 或 Tokenizer 哈希漂移；
- Config/Log 任一任务、语言分层不足；
- 推理超过 192 Token、重复 8-gram 超限、重复行或 Trace 重复；
- Teacher 输出不闭合、Final JSON 错误、序列超过 1024；
- Dev/历史 Pilot 污染；
- Mixture 中单个来源复用超过四次；
- 精确 1M Token 或 700K/150K/150K 分层计数漂移；
- 两个 Seed、冻结评测协议或训练身份不一致；
- 任一 Gate 失败时错误地解锁 M5.3。

## 9. 执行顺序

```text
现有 Pilot CPU 审计
→ 新任务生成器与污染检查
→ CPU Fixture 和失败路径
→ R3-P0 Teacher Pilot
→ P0 人工内容审查
→ 正式 240 任务 Teacher 生成
→ 160 条来源选择与 Manifest
→ R3 Mixture 构建
→ 两个 Seed 训练
→ 冻结双模式评测
→ 自动 Gate
```

当前已完成 R3-P0：固定 40 条任务及独立身份、Dev/历史 Pilot 污染检查、严格 Schema、
CPU 合成契约 Smoke、失败路径和真实 Qwen3-8B Teacher Pilot。P0 只接受 10/40 条，Config
与 Log Diagnosis 各 5 条，两个任务族均未通过门禁；正式 240 条扩展保持阻断。
真实运行使用隔离的两阶段环境：Teacher 环境只生成候选，冻结 `tokenizers=0.21.4` 的
Policy 环境负责 Token 计数、Trace 选择和 Gate；禁止为了复用单一环境而绕过版本校验。

下一步先冻结 P0-R1 Prompt 控制诊断：使用新的 Task/Template 版本，把“简洁推理”改为
“先给结论，再用一个直接证据说明，不枚举备选项”的双语结构约束。Teacher、采样、候选数、
192 Token Trace 上限、384 Token 生成上限和污染规则全部保持不变。该诊断通过前不实现
240 条扩展，不启动 R3 训练。

P0-R1 设计现已冻结并实现：

- Pilot Version：`m5-r3-p0-r1-v1`；
- Task/Generation Seed：`20260801` / `20260802`；
- Template：`pilot.<family>.r3-targeted-p0r1.v1`；
- 父 P0 公开结果以 SHA256 绑定并在加载模型前校验；
- 确定性任务集 SHA256：
  `4cc14273c8351b94c3221c3b7c0e934afb026169534f9a0cc2d8d862b46d0688`；
- Dev 与历史 Pilot 的 Exact、Normalized 和 Template 污染计数全部为 0；
- CPU 合成契约 Smoke 通过，但不构成 Teacher 质量证据。

下一步是一次 40 条真实 Qwen3-8B GPU Pilot。只有 Config 和 Log 都满足 14 条总接受、
10 条英文和 4 条中文门禁，才进入人工内容审查与 240 条来源扩展。详见
[P0-R1 中文准备报告](../reports/m5/m5_r3_p0_r1.md)。
