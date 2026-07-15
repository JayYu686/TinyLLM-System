# 评测规范

## 1. 目标

评测用于回答：

- 模型是否提升目标任务。
- 模型是否回退通用能力。
- 模型输出是否满足格式。
- 模型部署性能是否可接受。
- 模型是否可以晋级。

## 2. 评测层次

### 通用能力

- ARC Easy。
- HellaSwag。
- PIQA。

### 自建任务

固定 300 条，英文/中文目标 70%/30%：

- Python。
- Linux。
- JSON。
- 配置修改。
- 日志分析。
- 简短代码。
- 无依据拒答。

### 系统性能

- TTFT。
- P50/P95。
- Tokens/s。
- Peak Memory。
- Failure Rate。
- JSON Valid Rate。

## 3. 评测记录

每次评测必须记录：

- 模型版本。
- Checkpoint。
- Tokenizer。
- Prompt Template。
- 评测集版本。
- 解码参数。
- 硬件。
- 软件版本。
- Seed。
- 原始输出。
- 汇总结果。

## 4. 模型比较

报告必须同时包含：

- 提升项。
- 回退项。
- 不显著变化项。
- 失败样例。
- 性能变化。
- 不确定性说明。

## 5. Promotion Gate 示例

```yaml
required:
  json_valid_rate:
    min: 0.98
  domain_accuracy:
    min_delta: 0.03

regression_limits:
  general_eval:
    max_drop: 0.02
  p95_latency:
    max_increase: 0.15
```

## 6. 晋级条件

候选模型必须满足：

- 评测完整。
- 数据血缘完整。
- 无严重通用能力回退。
- 格式有效率达标。
- 性能回归在阈值内。
- Checkpoint 可加载。
- 推理 Smoke Test 通过。

M6 Candidate 的量化门禁为：领域总分相对 Baseline 至少提升 3pp 且 Bootstrap
95% CI 下界大于 0；ARC-Easy/HellaSwag/PIQA 聚合回退不超过 2pp；JSON Valid Rate
至少 98%；血缘完整。M7 前不以未实测的推理性能项阻止或通过 Candidate。

门禁分阶段启用：

- M6 先启用评测完整性、数据/模型血缘、能力回退和格式有效率门禁。
- M7 在真实推理服务与 Benchmark 可用后，启用延迟、吞吐、显存和失败率门禁。
- 尚未具备实测基础设施的性能项必须标记为 `not_evaluated`，不得以估算值通过门禁。

## 7. 禁止事项

- 只展示最好的一次结果。
- 删除失败样例。
- 使用测试集调参。
- 省略解码参数。
- 混用不同硬件结果。
- 将估算值当实测值。

## 8. M2.4 冻结评测集契约

M2.4 按三个独立批次完成：先冻结评测身份与污染检测契约，再发布 300 条领域评测集，最后
在任何正式后训练之前运行 Base Model Baseline。代码、评测内容和真实模型运行必须分别
审查，不能以合成 Smoke 代替 300 条评测集或 Baseline。

### 8.1 评测项

每个领域评测项必须包含稳定 ID、语言、类别、Prompt 消息、Canonical Reference、评分器和
许可/来源声明。Prompt 只允许一个可选 System 消息和一个 User 消息；Reference 作为唯一
Assistant 消息参与完整样本污染检查。文本必须已经是 NFC、LF 换行、无首尾空白且不含非法
控制字符，Loader 不得静默修正公开评测内容。

评分器固定为带版本的严格联合类型：

- `exact_match`：一个或多个允许答案和显式大小写策略；
- `multiple_choice`：冻结选项顺序和正确索引；
- `json_object`：Canonical JSON 参考值；
- `required_terms`：必含/禁含项与大小写策略；
- `human_rubric`：明确判定条件、阈值和必须保留的人工依据。

评测集内容身份按 Item ID 排序，对每条 Canonical JSON 使用 8 字节大端长度前缀流式
SHA256；配置身份独立计算，最终内容 SHA256 同时绑定 Items、Tokenizer、Template、解码
配置和预期语言/类别计数。构建时间、主机、用户名和绝对路径不参与内容身份。

### 8.2 固定领域分布

正式 `tinyllm-domain-v1` 必须恰好包含 300 条：210 条英文、90 条中文。类别计数固定为：

| 类别 | 总数 | 英文 | 中文 |
| -- | --: | --: | --: |
| Python | 50 | 35 | 15 |
| Linux | 45 | 32 | 13 |
| JSON | 40 | 28 | 12 |
| 配置修改 | 40 | 28 | 12 |
| 日志诊断 | 45 | 31 | 14 |
| 简短代码 | 40 | 28 | 12 |
| 无依据拒答 | 40 | 28 | 12 |

90 条中文项必须分别与同类别的一条英文项形成相同任务和难度的翻译配对，并使用
`bilingual-pair-NNN` Tag 记录；其余 120 条英文项使用 `english-only`。配对项用于显式分析
语言差异，统计置信区间时不得把翻译对误当成完全独立的证据。

这些数量属于内容契约；修改数量、Item、Reference、评分器、Prompt 或解码配置必须生成新的
评测版本，不能静默更新现有版本。

## 9. M2.4 Exact 污染检测

污染检测只读取通过 `COMMITTED`、完整文件清单和 SHA256 校验的 Dataset Registry 版本。
M2 核心只对 Train Split 建索引，不把 Validation/Test 当成训练数据。每个 Train Sample 从
Pack 的 `sample_token_counts` 边界重建，并计算两个不可逆指纹：

1. `full_sequence`：完整 ChatML `input_ids`；
2. `prompt_prefix`：从序列开头到第一个非 `-100` Label 之前的 `input_ids`，包含
   System/User 消息和 Assistant Header，但不包含答案内容。

指纹编码固定为 `token-sequence-sha256-v1`：8 字节大端 Token 数量，随后为每个非负 Token
ID 的 4 字节大端整数。评测项必须使用与注册数据完全相同的固定 Qwen3 Tokenizer 和
`qwen3-chatml-nonthinking-v1` Template 生成相同两类指纹。Tokenizer、Template 或最大长度
不一致时直接拒绝，不能比较不兼容的哈希。

报告只保存评测 Item ID、匹配类型以及训练 Sample ID 的 SHA256，不输出训练文本、原始
Sample ID、Token IDs 或绝对路径。一个 Item 命中任一指纹即标记为污染；`tinyllm eval
contamination` 使用退出码 6。输入/配置错误使用 2，Registry/缓存/环境失败使用 3。

Near-Dedup 是后续增强项。只完成 Exact 指纹扫描时必须记录 `near_dedup=not_evaluated`，不得
声称已排除语义改写污染。

## 10. Baseline 顺序门禁

正式顺序固定为：

```text
评测 Schema/污染契约
→ 300 条领域集内容审查
→ 对注册 Train Split 执行污染检查并排除命中项
→ 冻结评测版本、Prompt、Tokenizer 和解码配置
→ 运行 Base Model Baseline 并保留原始输出
→ 才允许正式后训练
```

污染为零不等于评测质量合格；300 条内容仍必须经过语言、类别、评分客观性、许可和人工
质量审查。Baseline 未真实运行时必须保持 `not_evaluated`。

## 11. M2.4c 训练前 Baseline 契约

M2.4c 固定使用 `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` 的
Post-trained 权重、BF16、Non-thinking ChatML、Greedy Decoding 和单张 RTX 3090。正式运行
必须在已合并实现的 clean `main` 上从经过 SHA256 校验的私有缓存离线加载模型；不得让
Transformers 静默回退到远端默认分支或执行 Remote Code。

Baseline 使用独立 `.venv-baseline`：Transformers 4.57.6 需要 Tokenizers 0.22.2，而 M2
数据构建仍固定 Tokenizers 0.21.4。两者不能共用解释器后再声称数据可复现。安装与审计命令
固定为 `make bootstrap-baseline` 和 `make audit-baseline`；审计例外及其失效条件记录在
`requirements/baseline_security_exceptions.md`。

领域集生成使用 `qwen3-chatml-nonthinking-generation-v1`：对冻结的 System/User 消息使用
与 M2 数据一致的 ChatML 消息格式，并追加 Assistant Generation Header 与 Qwen3 官方
`enable_thinking=false` 所需的空 `<think></think>` 块。它与污染检测中
包含 Reference 的 `qwen3-chatml-nonthinking-v1` 是两个显式身份，不能混用。300 条输出全部
保存在私有 Run；公共仓库只保存内容哈希、分项汇总、脱敏失败 Item ID 和人工判断依据。

客观评分规则固定为：Exact Match 按 Item 的大小写/首尾空白策略；JSON 先解析并独立记录
Valid Rate，再比较对象；Multiple Choice 比较冻结选项文本；Required Terms 按显式必含/
禁含项。40 条 `human_rubric` 必须由维护者逐项保存三个布尔判定和非空 Rationale，未完成人工
判断时整个 Baseline 状态只能是 `awaiting_human_review`。

通用任务通过 `lm-eval==0.4.12` 的本地 YAML Adapter 运行，均为 0-shot、固定 Batch、应用
Qwen Chat Template 且 `enable_thinking=false`，同时保存 `--log_samples` 原始记录。任务与
数据版本固定为：

| Task | Dataset revision | Split | Samples | Public metric |
| -- | -- | -- | --: | -- |
| ARC-Easy | `allenai/ai2_arc@210d026faf9955653af8916fad021475a3f00453` | test | 2376 | `acc`, `acc_norm` |
| HellaSwag | `Rowan/hellaswag@218ec52e09a7e7462a5400043bb9a69a41d06b76` | validation | 10042 | `acc`, `acc_norm` |
| PIQA | `baber/piqa@142f6d7367fd9877f0fb3b5734ea6a545f54cdd1` | validation | 1838 | `acc`, `acc_norm` |

HellaSwag/PIQA 的 Hub Mirror 未声明许可证，原始样本、Prompt 和模型输出因此只保存在私有
Artifact Store；这不影响只公开聚合指标，但必须作为报告限制。正式运行前先用独立 Smoke
配置对每类至少两条样本验证模型加载、输出、评分、OOM/缓存缺失失败路径和原始日志写入。
运行前必须单独执行 `lm-eval validate`；不使用 v0.4.12 PyPI Wheel 中会因未打包源码测试
目录而失败的 `--check_integrity` 运行时开关。

GPU 由 CLI 的 `--gpu-index` 显式覆盖。模型加载前必须重新采集物理卡显存占用、利用率和
温度；超过 1024 MiB、利用率超过 10% 或温度达到 80 C 时拒绝运行。通过的 M2.4c 兼容性
Smoke 记录在 `reports/m2/baseline_smoke.md`，其 2 条/任务限制不构成质量 Baseline。

正式 Domain 生成完成后，Run 必须保持 `awaiting_human_review`，直到 40 条拒答 Item 都有
一条私有 JSONL Judgment。每条 Judgment 固定保存三个按 Item Criteria 顺序排列的布尔值、
三项全通过得到的 `passed`、非空 Rationale 和 `reviewer_role=maintainer`。Judgment 必须按
Run 中 Item 顺序一次性完整提交，禁止部分提交或覆盖已提交结果：

```bash
.venv-baseline/bin/tinyllm eval baseline-review \
  --run-id "$BASELINE_RUN_ID" \
  --judgments "$PRIVATE_JUDGMENTS_JSONL" \
  --artifact-root "$TINYLLM_ARTIFACT_ROOT" \
  --json
```

提交器会重新验证 Run Config、Domain 原始响应哈希、待审汇总和 Judgment ID，并先原子发布
`human_review/` 目录，再将 Run 从 `evaluating` 更新为 `succeeded`。公共报告可以发布汇总和
脱敏失败 Item ID；原始响应与完整 Judgment 仍保留在私有 Artifact Store。若进程在评审目录
发布后、Run 状态更新前中断，相同 Judgment 提交会校验已发布 Commit 并幂等完成外层状态；
不同内容不能覆盖既有 Commit。
