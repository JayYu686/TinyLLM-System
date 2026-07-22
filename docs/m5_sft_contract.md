# M5 Qwen3 双模式正式后训练契约

## 1. 目标与边界

M5 验证同一套版本化数据、原生 PyTorch 训练、Checkpoint/Resume 和评测血缘能否同时支撑：

- Qwen3-0.6B Full SFT；
- Qwen3-8B BF16 LoRA；
- 显式 Thinking 与 Non-thinking 双模式。

固定模型保留原生 GQA。M5 不实现 MLA、RLHF、偏好优化、过程奖励模型、推理服务或自研
KV Cache。训练完成不等于晋级 Candidate；M6 才执行正式 Promotion Gate。

## 2. 不可变身份

| 路线 | Repository | Revision | Attention | 初始策略 |
| -- | -- | -- | -- | -- |
| Full SFT | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | GQA 16 Query / 8 KV | BF16 Full SFT |
| LoRA | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | GQA 32 Query / 8 KV | BF16 LoRA |

两条路线都要求 `trust_remote_code=false`、Sequence Length 1024、Assistant-only Loss 和
显式模式选择。Qwen3-8B LoRA 固定 Rank 16、Alpha 32、Dropout 0.05，覆盖 Attention/MLP
Linear。只有单卡 BF16 LoRA 的受控 Probe 保存 OOM 证据后，才允许建立单独的 NF4 QLoRA
配置身份。

## 3. 双模式数据契约

M2 的 `m2-sft-v1-f82ff32e` 和 `qwen3-chatml-nonthinking-v1` 是只读父数据。M5 新增：

- `m5-reasoning-pilot-v1-*`：配比消融使用的私有 Pilot 数据；
- `m5-dual-sft-v1-*`：由父数据和经过验证的 Thinking 数据组成的不可变正式版本；
- `qwen3-chatml-thinking-v1`：无 Tool 消息的原生 Qwen3 Thinking 子集。

Thinking Assistant 内容固定渲染为：

```text
<|im_start|>assistant
<think>
{reasoning_content}
</think>

{final_answer}<|im_end|>
```

Assistant Header 不监督；`<think>`、可见推理轨迹、`</think>`、最终答案和 `<|im_end|>`
全部监督。System/User 内容全部 Mask。空 Think、空最终答案、多个 Think 块、无法验证和超过
1024 Token 的候选必须拒绝，不能截断后接受。

首版只覆盖 Python、Linux、JSON、YAML/TOML 配置和结构化日志诊断。任务由规则模板生成
标准答案，固定 Qwen3-8B Thinking 模式最多生成两个轨迹候选，接受第一个通过确定性
Verifier 的候选。M5 v1 不执行模型生成的任意 Python 或 Shell 代码。

数据按生成器模板族分组切分；英文/中文 Token 目标为 70%/30%。Teacher Revision、生成参数、
Seed、Verifier、输入/输出哈希、拒绝原因和污染结果必须进入 Manifest。重新调用随机生成器
不要求跨 CUDA 环境逐位一致；一旦原始生成 Artifact 被接收，后续规范化、验证、Tokenization、
Packing 和注册必须可确定性重建。

## 4. 配比与训练门禁

正式数据配比不能凭经验指定。先在不接触 M6 冻结测试指标的情况下，对 Thinking Token
比例 0%、30%、50% 各执行 1M Supervised Tokens、两个固定 Seed 的 0.6B 消融。选择顺序：

1. Non-thinking Dev 回退不超过 2pp；
2. Thinking 格式有效率至少 99%；
3. 最大化 Thinking Final-answer 分数；
4. 差异不足 1pp 时选择 Thinking 比例更低者。

M5 Reasoning Dev 固定 200 条，五类任务各 40 条，每类 28 条英文、12 条中文，并与 Train
按模板族隔离。它只用于 M5 选择，不生成最终求职指标。

0.6B 正式路径先做单卡 BF16 Smoke，再用四张通过 Preflight 的 RTX 3090 执行 DDP。最低
50M Tokens、最高 100M，每 10M 执行继续训练门禁；每 2M 保存滚动 Checkpoint。8B 路线先做
单卡 Memory Probe，再训练最低 10M、最高 30M Tokens；每 1M 保存滚动 Checkpoint、每 2M
执行 Dev 评测。每段作业不超过 12 小时。

## 5. 配置与恢复

新配置使用 `config_kind=qwen_sft` 和独立 `schema_version`，不改变 M1 Schema。至少记录模型
Revision、`attention_architecture=gqa`、Full/LoRA/QLoRA 身份、数据与混合 Manifest、模式、
Token 预算、精度、World Size、Checkpoint 策略和评测版本。CLI 只允许运行时覆盖物理 GPU、
输出根目录和 Resume 模式。

0.6B DDP Exact Resume 必须保持 World Size、模型、数据版本、配比和优化配置兼容。8B LoRA
Checkpoint 必须包含 Adapter、Optimizer、Scheduler、RNG、Sampler Cursor、基座 Revision、
数据版本和配置哈希。部署导出与训练 Checkpoint 分离；8B 只导出 Adapter Safetensors。

## 6. M5 完成条件

M5 只有在以下证据全部合并后才能标记完成：

1. 双模式设计、Schema、数据 Manifest、拒绝统计和污染报告；
2. 训练前双模式 Baseline 与配比消融；
3. 0.6B Full SFT 的真实训练、恢复、曲线和 Checkpoint；
4. 8B LoRA 的真实 Probe、训练、恢复、Adapter 和 Model Card；
5. OOM、NaN/Inf、坏 Checkpoint、磁盘不足、数据漂移、错误 World Size 和进程退出失败路径；
6. 中文主验收报告、英文公开摘要和完整血缘。

结果没有质量提升时可以作为诚实的 M5 系统实验完成，但模型保持 `Development`。只有 M6
满足 Thinking 提升、Non-thinking/通用回归、JSON Valid Rate 和血缘门禁后才能晋级。
