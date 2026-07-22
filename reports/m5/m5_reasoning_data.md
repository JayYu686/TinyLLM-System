# M5.1 Reasoning 数据契约与 Teacher Smoke 审查报告

## 1. 结论

M5.1 在本批定义的门禁内完成：Reasoning Task、Teacher Generation、Verifier、Rejected
Record、污染检查和 Dataset Manifest 均已成为严格、版本化的公共 Schema；200 条独立 M5
Reasoning Dev 已确定性构建；公开 CPU 合成链路和真实 Qwen3-8B 离线 Thinking Teacher Smoke
均有可复核证据。

M5 整体仍为 `IN_PROGRESS`。本报告不声称模型质量提升、推理能力提升、吞吐、扩展效率或
Candidate 晋级，也不把 CPU 合成 Fixture 当作模型输出或正式训练数据。

## 2. 冻结接口

正式配置为 `configs/data/m5_reasoning.yaml`，固定：

- 父数据 `m2-sft-v1-f82ff32e`；
- Thinking Template `qwen3-chatml-thinking-v1`；
- 最大序列长度 1024；
- Qwen3-8B Revision `b968826d9c46dd6066d109eabc6255188de91218`；
- 原生 GQA、BF16、`trust_remote_code=false`、`local_files_only=true`；
- Sampling：Temperature 0.6、Top-p 0.95、Top-k 20、最多两个候选；
- Verifier `m5-json-exact-v1`，只比较规范 JSON Object，禁止执行模型生成的 Python/Shell。

Pilot Task、Dev Task 和 Teacher Sampling 使用三个互异 Seed。未知字段、错误模型 Revision、
非 GQA 身份、空 Think、多个 Think、非法 JSON、错误答案、超长序列和污染失败都会被拒绝。

## 3. 200 条 Reasoning Dev

确定性构建结果：

| 项目 | 实际结果 |
| -- | -- |
| Task Set Version | `m5-reasoning-dev-v1-3eb153c2` |
| Tasks SHA256 | `3eb153c2defeedd59e4b3c4ba33c13b1f28af7af5c4d1b61cd89cf870eff2549` |
| 总数 | 200 |
| 任务类别 | Python、Linux、JSON、配置、日志诊断各 40 |
| 语言 | 英文 140、中文 60；每类英文 28、中文 12 |
| 用途 | 仅用于 M5 配比选择，不产生 M6 最终求职指标 |

Dev Template Family 全部使用 `dev.*` 命名空间；Pilot 使用 `pilot.*`。构建时不仅检查命名空间，
还执行 Exact Prompt Hash 和 Template Family 双重污染检查。

## 4. CPU 合成链路

`reports/m5/raw/reasoning_data_smoke.json` 使用 50 条公开合成 Pilot Task 和 100 条合成候选，
用于验证确定性接口，不调用模型：

| 项目 | 实际结果 |
| -- | -- |
| Evidence Kind | `synthetic_cpu_contract_smoke` |
| Model Generated | `false` |
| Quality Metric | `false` |
| 合成 Pilot Version | `m5-reasoning-pilot-v1-ecc4f02f` |
| 接受样本 | 50 |
| 非法 JSON 候选拒绝 | 10 |
| Candidate Exhaustion | 0 |
| Exact Prompt Match | 0 |
| Template Family Overlap | 0 |

该版本只能作为 Contract Smoke 身份，禁止用于训练或模型质量描述。

## 5. 真实 Qwen3-8B Teacher Smoke

### 5.1 首次失败保留

第一次运行使用 512 个最大生成 Token。两个候选均达到上限，且都没有生成闭合 `</think>`：

| 项目 | 实际结果 |
| -- | -- |
| 状态 | `fail` |
| 输入 Token | 63 |
| 生成 Token | 512 / 512 |
| 拒绝 | `teacher_length_limit=2`、`no_candidate_passed=1` |
| Git Commit | `ddfb9f49ac1ca25e42cd97516020fd5203a9cb43` |

失败摘要保存在 `reports/m5/raw/teacher_offline_smoke_512_failure.json`，没有删除或改写为成功。
处理方式不是放宽 Verifier，而是在 1024 总长度约束内将生成上限调整为 896，并要求可见推理
控制在 192 Token 内。

### 5.2 最终通过结果

最终运行在物理 GPU 9 上完成，运行前该卡显存占用 1 MiB、利用率 0%。

| 项目 | 实际结果 |
| -- | -- |
| 状态 | `pass` |
| Git Commit | `5289e6e003360d06c962689d64f6c6606c75d311`，Dirty=`false` |
| PyTorch / Transformers | `2.7.1+cu118` / `4.57.6` |
| 输入 Token | 72 |
| 两个候选生成 Token | 513 / 543 |
| 接受样本 | 1；首个候选通过 JSON Exact Verifier |
| 未使用候选 | 1；首个候选通过后不进入 Dataset |
| Dataset Version | `m5-reasoning-pilot-v1-f551031f` |
| Pilot/Dev 污染门禁 | `pass`；Exact Prompt=0、Template Family=0 |
| 运行耗时 | 33.897 秒 |
| Peak Allocated | 16,500,440,064 Bytes |
| Peak Reserved | 16,645,095,424 Bytes |
| 私有原始 Artifact SHA256 | `f1364f994b0673362a011ce6e360e6a323505a08d84b9ebef947bea6664bbb3a` |

耗时和显存只是单次正确性 Smoke 观测，不是 Benchmark。原始 Prompt、Thinking Trace 和最终答案
只保存在私有 Artifact Store；公开仓库仅保存路径无关的哈希、版本、计数和软硬件摘要。

## 6. 失败路径覆盖

自动测试覆盖：

- Teacher 离线快照缺文件、错误 Revision 和错误 GQA 身份；
- Teacher 调用失败、达到长度上限和 Candidate Exhaustion；
- 缺少、多个、嵌套或空白 Think；
- 空最终答案、非法 JSON 和答案不匹配；
- 总序列超过 1024；
- Generation ID、Prompt Hash、候选索引和 Manifest 计数漂移；
- Pilot/Dev Exact Prompt 或 Template Family 污染；
- Dirty Git Run 试图标记为通过；
- Rejected Record 泄露原始内容。

## 7. M5.1 边界与下一步

M5.1 证明的是数据契约、确定性 Dev、失败审计和真实 Teacher 可用性。真实 Teacher Smoke 只有
一个接受样本，不构成正式 Pilot 规模、配比结论或训练数据质量证明。

下一步 M5.2 必须先按同一契约扩展私有 Pilot，并保留逐条 Verifier/拒绝/污染证据；随后运行
训练前 Thinking/Non-thinking Baseline，以及 0%/30%/50% Thinking Token 的 1M Token、双 Seed
消融。消融完成前不得冻结正式混合比例或启动 M5.3 长程训练。
