# M5.2-R2 Thinking 长度与闭合诊断设计

## 1. 目标

M5.2-R2 是只读模型诊断批次，用来区分三个可能原因：

1. `H1_LENGTH_CEILING`：模型能够完成推理和最终答案，但 896 个生成 Token 的上限过低；
2. `H2_FAMILY_VERBOSITY`：Config/Log 等任务族诱发过长推理，需要任务定向的数据修正；
3. `H3_GENERATION_INSTABILITY`：相同模型、Prompt、Batch 和 Seed 无法重放原输出，现有评测
   的可复现性不足。

R2 不训练新模型，不修改 M5.2/R1 结果，也不降低 99% Thinking 格式门禁。诊断结果只决定
下一步应建立新评测协议，还是设计新的训练数据与优化策略。

## 2. 已知证据

R1 两个 Seed 共出现 24 条 Thinking 格式失败，全部在 896 Token 上限处保留一个未闭合
`<think>`：

| Seed | 格式失败 | Config | Log Diagnosis | Python | 英文 | 中文 |
| --: | --: | --: | --: | --: | --: | --: |
| 42 | 11 | 9 | 1 | 1 | 6 | 5 |
| 20260727 | 13 | 10 | 3 | 0 | 6 | 7 |
| 合计 | 24 | 19 | 4 | 1 | 12 | 12 |

Config 的长度分布明显不同于其他任务族：

| Seed | Config P50 | Config P90 | 成功样本最大长度 | Config 失败 |
| --: | --: | --: | --: | --: |
| 42 | 362.5 | 896 | 852 | 9 / 40 |
| 20260727 | 424.5 | 896 | 845 | 10 / 40 |

JSON 和 Linux 在两个 Seed 中均没有格式失败。R1 使用短完整样本后，Thinking 格式率相对原
30%消融分别下降 1.0pp 和 3.5pp。因此，继续复用当前 40 条短样本缺少实验依据；长度上限和
Config 任务族是 R2 首先需要验证的变量。

## 3. 不可变输入

### 3.1 模型与训练 Run

| Seed | Training Run | 原评测 GPU |
| --: | -- | --: |
| 42 | `20260727T075422Z-m5-format-repair-r1-seed42-7c825907-1c02` | 5 |
| 20260727 | `20260727T075432Z-m5-format-repair-r1-seed20260727-59c6d0e9-3af2` | 6 |

两个 Run 必须保持：

- `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`；
- `attention_architecture=gqa`；
- `m5-format-repair-mixture-v1-1396b60b`；
- 原始模型导出 SHA256；
- 原始训练和评测 Summary。

### 3.2 原评测身份

```text
Suite:
m5-reasoning-dev-v1-53ddf557

Evaluation Config SHA256:
3735a34e70c06059fbc09f62d02fabc296fd97e79a45d60f3d00dede21108d51

Thinking Batch Size:
4

Thinking Base Seed:
20260726

Original Max New Tokens:
896
```

原始私有 Result SHA256：

```text
Seed 42:
87e478b92c3992fa4f1196d05e32686291f2f2a4b559777e559fbc80988bd50d

Seed 20260727:
1e1fd1d43d170ddad4752114222c0655e1c814cebc23f8f3208575548c6b8cd7
```

R2 不读取 M6 冻结评测，不改变 Prompt、Tokenizer、Temperature、Top-p、Top-k、Repetition
Penalty、Batch Size、任务顺序或 Seed。

## 4. 两阶段诊断

### 4.1 D1：私有离线结构分析

D1 只读取已经存在的 R1 `results.jsonl`，不加载 GPU 模型。它必须：

1. 重新校验 Summary 与 Raw Result SHA256；
2. 重新生成 200 条 Dev，并校验 Item ID、Prompt SHA256 和任务顺序；
3. 复算 Thinking/Non-thinking Summary；
4. 对失败项输出任务族、语言、Finish Reason 和 Token 长度聚合；
5. 计算内容脱敏的连续重复指标：
   - Token Unique Ratio；
   - Repeated 8-gram Ratio；
   - 相同 Line Hash 的最大重复次数；
6. 同时报告失败项和同任务族成功项的分布，不用单个阈值提前判定“生成失控”。

公开结果只保留计数、分位数、比例和哈希。Response、Item ID、Prompt 和 Thinking Trace 只
保存在私有 Artifact Store。

### 4.2 D2：896→1536 反事实重放

D2 只重放包含原始失败 Item 的 Thinking Batch，避免重新评测全部 400 个输出。每个 Batch
执行两次：

1. 使用原上限 896 重放；
2. 重置完全相同的 RNG 后，使用诊断上限 1536 重放。

原失败 Item 所属的整个四样本 Batch 必须保持原任务顺序。每个 Batch 的 Seed 继续使用：

```text
20260726 + original_batch_offset
```

重放必须满足两道一致性校验：

- 896 重放的 Response SHA256、Generated Tokens、Finish Reason 和评分必须与原 Result 一致；
- 1536 重放的前 896 个生成 Token ID 必须与同批 896 重放逐个相同。

任一校验失败时，本次诊断状态为 `INVALID_REPLAY`，不得讨论长度上限或 R2 数据策略。

对 1536 重放结果分别截取前 1024、1280、1536 个 Token，使用原始 Parser 和 Scorer 复算：

- Thinking 格式有效数；
- Final JSON 有效数；
- Final-answer 正确数；
- 仍未闭合数；
- 达到各上限的数量；
- Closing Tag 首次出现位置分布。

诊断不允许通过字符串补写 `</think>` 或最终 JSON。运行时补标签无法恢复缺失的最终答案，
也不能作为原生 Thinking 能力证据。

## 5. 机器可读接口

后续实现应增加：

```text
configs/eval/m5_r2_length_replay.yaml
scripts/analyze_m5_r2_failures.py
scripts/run_m5_r2_length_replay.py
scripts/select_m5_r2_diagnostic.py
```

建议的私有目录：

```text
$TINYLLM_ARTIFACT_ROOT/evaluations/m5/r2-length-replay/
├── seed42/
│   ├── raw_results.jsonl
│   └── summary.json
└── seed20260727/
    ├── raw_results.jsonl
    └── summary.json
```

公开机器结果：

```text
reports/m5/raw/m5_r2_length_diagnostic.json
```

所有 Schema 使用 `schema_version`、`extra="forbid"` 和可导出的 JSON Schema。正式接口退出码
保持：

- `0`：诊断成功并生成结论；
- `2`：配置或输入错误；
- `3`：GPU/环境 Preflight 失败；
- `6`：重放不一致或长度假设不足。

## 6. 决策规则

对每个候选上限 `1024 / 1280 / 1536`，计算：

```text
Projected Format Valid
= 原 896 上限有效数
+ 原失败项中在该上限内恢复完整格式的数量
```

Projected Format Basis Points 必须继续以 200 条完整 Dev 为分母。两个 Seed 使用同一个最小
上限进行判断。

### 6.1 支持长度上限假设

若两个 Seed 在同一上限都达到至少 99%：

- 记录 `LENGTH_CEILING_SUPPORTED`；
- 报告满足条件的最小上限；
- 不直接替换现有评测；
- 建立新的 Evaluation Protocol Version；
- 使用新版本重新运行 Base、六个 M5.2 Candidate 和两个 R1 Candidate；
- 新旧结果并列保留，禁止覆盖。

如果只有 1536 达标，状态改为 `TRADEOFF_REVIEW_REQUIRED`。正式采用前必须评估生成 Token、
评测时长和后续推理延迟；1536 不自动成为新上限。

### 6.2 长度上限不足

若任一 Seed 在 1536 仍低于 99%：

- 记录 `LENGTH_CEILING_INSUFFICIENT`；
- 保持现有评测协议与 R1 拒绝结论；
- 停止通用短样本复用；
- 后续训练设计只面向 Config/Log，增加新的、互不重复的简洁 Teacher Trace；
- 在训练前冻结任务族比例、推理长度分布和污染检查。

### 6.3 重放不一致

若 896 重放或前缀一致性失败：

- 记录 `INVALID_REPLAY`；
- 检查 PyTorch、Transformers、Tokenizer、GPU、Batch 顺序和 RNG；
- 在重放可复现之前，不执行新的质量实验。

## 7. 验收条件

R2 诊断实现只有同时满足以下条件才可运行 GPU：

1. 严格配置和结果 Schema；
2. 原始 Summary、Raw SHA256、模型导出和训练 Run 血缘校验；
3. Batch/Seed 计算的单元测试；
4. 896 重放不一致失败路径；
5. 1536 前缀漂移失败路径；
6. 1024/1280/1536 截断评分测试；
7. 私有 Raw 与公开聚合脱敏检查；
8. CPU Fixture Smoke；
9. 中文报告模板；
10. 干净 Git Worktree 和 GPU Preflight。

## 8. 范围限制

R2 诊断阶段不包含：

- 新的 SFT、LoRA 或 QLoRA；
- 修改 99%门槛；
- 修改正式 Dev 内容；
- 运行时补写 Closing Tag；
- Grammar-constrained Decoding；
- 使用 M6 冻结结果调参；
- 将诊断结果标记为 Candidate 或 Production。

## 9. 条件确认

用户已条件确认：若 1280 Token 能让两个 Seed 都达到 99%，允许在新的 Evaluation
Protocol Version 中把正式 Thinking 最大生成长度从 896 提高到 1280。该确认不直接修改
现有协议；新版本生效前仍必须完成 Base、六个 M5.2 Candidate、两个 R1 Candidate 的完整
重跑，并报告生成 Token、评测时长和后续推理性能成本。旧结果不得覆盖。
