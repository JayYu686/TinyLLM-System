# M5.2-R3 Teacher 来源策略

## 1. 决策

R3 下一项有界实验采用两阶段 `solve → compress`：

1. Qwen3-8B 原生 Thinking 负责求解，输出先通过冻结的 Exact Final-answer Verifier；
2. 同一 Qwen3-8B 的 Non-thinking 模式把已验证解答压缩为短、证据可核验的 Distilled
   Rationale；
3. 确定性规则 Trace 与 P1 使用同一任务，作为结构控制组，但不具备训练来源资格。

新实验身份为 `m5-r3-p1-two-stage-v1`。策略审查只授权实现 P1 契约，当前没有授权 GPU
Pilot、240 条正式来源扩展、R3 Mixture 或训练。

## 2. 真实证据

| 证据 | 实际结果 | 对策略的约束 |
| -- | -- | -- |
| R2 长度重放 | 1536 Token 投影格式率 98.0% / 96.5%，仍有 4 / 7 条未闭合 | 排除仅提高长度上限 |
| R3-P0 | 接受 10/40；Config / Log 各 5；52 个候选超过 192 Token | 单阶段原生 Thinking 不满足来源门禁 |
| R3-P0-R1 | 接受 12/40；Config 4、Log 8；46 个候选超过 192 Token | Prompt 控制改善有限且不稳定 |

P0-R1 的答案、JSON、标签和污染拒绝均为 0，失败主要集中于生成长度。下一项实验因此拆分
“找到正确答案”和“生成可训练的短 Rationale”，不继续调整同类 Prompt。

## 3. 策略对照

| 策略 | 处置 | 原因 |
| -- | -- | -- |
| 单阶段 Prompt 控制 | 不采用 | P0 与 P0-R1 的达标样本数均不足 |
| 仅提高生成长度 | 拒绝 | R2 在 1536 Token 仍未达到两个 Seed 均 99% |
| 两阶段求解与压缩 | 选为 P1 | 将答案正确性与 Trace 长度控制分开验证 |
| 确定性规则 Trace | 控制组 | 可测量格式上界，但模板化内容不能直接代表 Teacher 质量 |

规则 Trace 不能因结构检查通过而进入训练。若未来希望将它用于混合数据，必须建立新的数据
身份、单独的训练消融和质量门禁。

## 4. P1 两阶段协议

### 4.1 任务范围

- Config 20 条、Log Diagnosis 20 条；
- 每类英文 14 条、中文 6 条；
- Task Seed `20260803`；
- 使用新 Task ID、Case Reference 和 Template Family；
- 每个标签至少六种独立证据变体；
- 与冻结 Dev、历史 Pilot、P0 和 P0-R1 执行 Exact、Normalized、Template 和证据组合
  污染检查。

P1 是新的来源可行性实验，不把新任务结果与 P0/P0-R1 当作严格同任务 A/B 指标。

### 4.2 Solver

Solver 固定为
`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`，GQA、BF16、原生
Thinking、`trust_remote_code=false`：

- 一个候选；
- Temperature 0.6、Top-p 0.95、Top-k 20；
- 最大 896 New Tokens；
- Generation Base Seed `20260804`。

Solver 的完整原始输出保存在私有 Artifact Store。只有自然闭合、Final JSON 与冻结期望答案
完全一致的结果才进入 Compressor；达到长度上限、答案错误或解析失败均拒绝，不补标签、
不截断后接受。

896 是 Teacher 求解阶段的上限，不修改冻结 Dev 的 896 Token 正式评测协议，也不能被解释
为 R2 已支持单纯扩展评测长度。

### 4.3 Compressor

Compressor 使用同一模型 Revision 的 Non-thinking 模式、Greedy 解码、一个候选、最大 256
New Tokens 和 Base Seed `20260805`。输入包含：

- 原始任务与证据；
- Solver 的完整私有推理；
- 已通过 Exact Verifier 的最终答案；
- 冻结输出协议。

输出必须是 `m5-r3-compressed-rationale-json-v1` JSON Envelope，只包含压缩后的
`reasoning` 和 `final_answer`。压缩结果必须重新验证，不能因为 Solver 正确而自动接受。

被接受内容称为 Model-distilled Rationale。报告不将它描述为模型真实的内部推理过程，也不
声称压缩结果与 Solver 的每一步推理语义等价。

### 4.4 Trace Policy

压缩结果必须同时满足：

1. Final JSON 与冻结答案完全一致；
2. 使用 Qwen3-0.6B Tokenizer 计算不超过 192 Reasoning Token；
3. 包含任务中预注册的直接 Evidence Anchor；
4. 不提及当前标签之外的其他候选标签；
5. Repeated 8-gram 不超过 5%；
6. 不存在重复规范化行；
7. 全局规范化 Trace SHA256 唯一；
8. 渲染后的 Qwen3 Thinking 训练序列不超过 1024 Token；
9. 不包含嵌套 Think 标签，不执行任何生成代码。

Pipeline 不允许通过截断、补写 `</think>`、规则替换答案或人工改写来修复候选。

## 5. 控制组

确定性规则 Trace 从任务的冻结标签和 Evidence Anchor 构造一个短说明，用于验证：

- 40 个任务能否全部生成结构合法、证据可定位且不重复的目标序列；
- P1 的失败是否来自模型压缩能力，而非任务、Tokenizer 或渲染接口；
- 模板化 Trace 与模型生成 Trace 的长度和重复度差异。

控制组结果标记为 `control_only` 和 `training_source_authorized=false`。它不计入 P1
14/10/4 Teacher 门禁。

## 6. P1 门禁

P1 沿用 P0 的门禁：

| 任务族 | 总接受 | 英文 | 中文 |
| -- | --: | --: | --: |
| Config | ≥14/20 | ≥10 | ≥4 |
| Log Diagnosis | ≥14/20 | ≥10 | ≥4 |

两个任务族都通过、污染为 0、Git 和环境身份完整后，才允许人工内容审查。人工审查通过后
才能另行授权 240 条来源扩展。P1 通过仍不直接授权 R3 Mixture 或训练。

## 7. 失败路径

实现必须拒绝：

- R2、P0、P0-R1、配置、模型或 Tokenizer 哈希漂移；
- Solver 未闭合、答案不一致、达到上限或运行失败；
- Compressor 输出非严格 JSON、字段缺失、多余字段或 Final-answer 漂移；
- 压缩 Trace 超过 192 Token、遗漏 Evidence Anchor、讨论其他标签或重复；
- Solver 与 Compressor Task ID、Prompt Hash 或答案身份不一致；
- Dev、历史 Pilot、P0、P0-R1 污染；
- 私有 Solver 推理进入公开结果；
- 控制组被错误标记为正式 Teacher 或训练来源；
- 任一门禁结果非通过时错误解锁扩展、Mixture 或训练。

## 8. 实现顺序

```text
严格 Schema 与父证据校验
→ P1 Task/Context 与污染检查
→ Solver/Compressor 私有 Artifact 契约
→ Compressor Parser 与双重 Verifier
→ 确定性规则 Trace 控制组
→ CPU 合成 Smoke 与失败路径
→ 中文准备报告
→ 单卡 Qwen3-8B P1 GPU Pilot
→ 人工内容审查
→ 扩展决策
```

机器可读策略审查见
[m5_r3_teacher_source_strategy_review.json](../reports/m5/raw/m5_r3_teacher_source_strategy_review.json)。

## 9. 实现状态

P1 严格 Schema、40 个新 Task/Context、Dev/历史 Pilot/P0/P0-R1 污染检查、Solver 与
Compressor 私有 Artifact、严格 JSON Envelope、双重答案验证、Evidence Anchor、规则
Trace 控制组、CPU 合成 Smoke 和失败路径已经实现。

CPU 合成链路 40/40 通过，规则控制组 40/40 且 Trace 全部唯一；缺少 Evidence Anchor、
Solver Seed 漂移和父任务污染均被拒绝。真实单卡 Qwen3-8B P1 随后接受 11/40 条：
Config 5 条、Log Diagnosis 6 条。主要拒绝原因为缺少原文证据锚点 10 条、讨论其他标签
10 条、Solver 达到上限 6 条和 Compressor 非严格 JSON 3 条。两个任务族均未达到
14/10/4 门禁，正式扩展、Mixture 和训练继续阻断。详见
[P1 中文实验报告](../reports/m5/m5_r3_p1.md)。

P2 使用 `m5-r3-p2-fallback-isolated-v1` 新身份：P1 的 40 个 Solver 作为 SHA256
绑定的父证据，只对 6 个拒绝项追加一次 Thinking 候选；全部有效答案随后进入不含原始
Solver 推理和其他标签的 Non-thinking Compressor。真实单卡 Qwen3-8B Pilot 接受
33/40 条，Config 17 条、Log Diagnosis 16 条，两个任务族均通过 14/10/4 门禁，四路
污染为 0。该结果授权人工内容审查和正式来源扩展；Mixture 与训练继续保持 `false`。
详见 [P2 中文实验报告](../reports/m5/m5_r3_p2.md)。
