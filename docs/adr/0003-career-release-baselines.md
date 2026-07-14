# ADR-0003：求职版本核心范围与固定实验基线

## 状态

Accepted

## 背景

早期计划同时列出 TinyGPT-350M、ZeRO-3、完整推理、V100 和自动规划器，容易让
MVP 被研究挑战阻塞。求职版本需要优先证明训练系统正确性、恢复能力、分布式工程、
后训练和质量门禁，并且每项结论必须有真实证据。

## 决策

### 发布边界

- 10 周核心范围为 M1–M6，目标预发布 `v0.6.0-rc.1`。
- M3 形成首个可投递版本，后续只增加真实结果。
- M6 最高只能晋级 Candidate；Production 必须等待 M7 真实推理性能门禁。
- M7、M8、ZeRO-3、MLflow、V100、TinyGPT-350M 属于缓冲或增强项。

### 数据基线

- `OpenAssistant/oasst1@fdf72ae0827c1cda404aff25b6603abec9e3399b`
- `bigcode/commitpackft@fc56fe33c030c6daa414c2b112c932b8eed085e6`

目标为约 60%/40% Token 比例和英文 70%/中文 30%。OASST 按 Conversation Tree、
CommitPackFT 按 Repository 分组切分。Exact Dedup 和训练/评测污染检查属于核心；
Near Dedup 属于增强项。数据导入时必须重新验证 revision 可取得性、Dataset Card、
样本来源许可证字段和 Allowlist，不得仅依据本文宣称合规。

### 模型基线

- DDP：`TinyGPT-Target-120M`，`hidden=768`、`layers=12`、`heads=12`、
  `intermediate=2304`、`vocab=32768`、`sequence_length=1024`、Weight Tying。
  实例化前不以名称冒充实际参数量。
- FSDP2：`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`，
  BF16、Activation Checkpointing、FULL_SHARD、Sequence Length 512、50 个
  Optimizer Step；Step 25 保存分片 Checkpoint，退出后恢复到 Step 50。
- Full SFT：`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`，
  Post-trained、Non-thinking、Assistant-only Loss、BF16、Sequence Length 1024、
  Gradient Checkpointing；正式下限 50M Tokens，上限 100M，按 10M Tokens 门禁推进。
- LoRA：同一 Qwen3-8B revision，BF16、Rank 16、Alpha 32、Dropout 0.05、Attention/
  MLP Linear；只有记录规定配置的 OOM 后才能回退 NF4 QLoRA。

模型导入时必须重新验证 revision、许可证、Transformers 兼容版本和真实 Smoke。
任何 revision 变化都需要新 ADR，不允许浮动到最新版本。

### 评测与晋级

- 通用任务固定 ARC-Easy、HellaSwag、PIQA。
- 冻结领域集 300 条，覆盖 Python、Linux、JSON/配置、日志诊断、无依据拒答，
  英文/中文目标 70%/30%。
- Candidate 要求领域总分相对 Baseline 至少提升 3pp，Bootstrap 95% CI 下界大于 0，
  通用聚合回退不超过 2pp，JSON Valid Rate 至少 98%，且血缘完整。
- 未通过模型保留 Development，报告必须保留回退和失败样例。

## 影响

核心版本更聚焦于可复现训练系统和真实求职证据。缓冲项仍保留研究价值，但不得改变
M1–M6 的完成条件或成为预发布阻塞项。未来修改上述模型、数据、评测或发布边界时，
必须提交替代 ADR 并建立新的数据/模型版本。
