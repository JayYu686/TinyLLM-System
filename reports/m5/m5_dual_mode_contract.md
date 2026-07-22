# M5.0 Qwen3 GQA 与双模式契约审查报告

## 1. 结论

M5.0 第一批契约实现通过本地验收，可以进入代码审查。本批只完成设计、严格配置、Thinking
ChatML 渲染、Assistant-only Label 边界和公共 JSON Schema；M5 整体仍为 `IN_PROGRESS`。

本批没有启动 GPU 作业、生成 Teacher 推理轨迹、构建 Reasoning 数据、训练模型或运行质量
评测，因此不产生 Loss、准确率、吞吐、显存或 Candidate 晋级结论。

## 2. 已冻结决策

- Qwen3-0.6B 与 Qwen3-8B 保留固定 Revision 的原生 GQA，不转换 MLA。
- 0.6B Full SFT 与 8B LoRA 都使用显式 Thinking/Non-thinking 双模式。
- M2 的 `m2-sft-v1-f82ff32e`、`qwen3-chatml-nonthinking-v1` 和训练前 Baseline 保持不可变。
- M5 新增 `qwen3-chatml-thinking-v1`；可见推理轨迹不是模型内部推理真实性声明。
- 0.6B 正式路线固定为四卡 DDP；8B LoRA 固定为单卡，NF4 QLoRA 必须引用真实 BF16 OOM
  Evidence Run。
- M5 只做监督式双模式 SFT，不加入 RLHF、DPO、GRPO、过程奖励模型或自研 Attention/KV Cache。

## 3. 机器可校验接口

新增 `M5SFTConfig` 和 `m5-sft-config-v1.schema.json`，明确校验：

- 固定 Repository、Revision、Apache-2.0、`model_type=qwen3`、`attention_architecture=gqa`；
- `full_sft` 只能绑定 Qwen3-0.6B；`lora/qlora` 只能绑定 Qwen3-8B；
- LoRA Rank 16、Alpha 32、Dropout 0.05、Attention/MLP Linear Scope；
- 双模式数据版本、父 M2 数据版本、Template SHA256 和 0%/30%/50% 消融候选；
- Smoke、1M Token 单卡消融、50M–100M 四卡 Full SFT 和 10M–30M 单卡 LoRA 的不同边界；
- BF16、Sequence Length 1024、最长 12 小时作业、Token 评测/Checkpoint 周期和 M5 Dev 身份；
- 未知字段拒绝和 QLoRA OOM 证据要求。

M1 配置 Schema 与加载路径没有改变。M5 Runtime 尚未接入 `tinyllm train`，因此本批不会把只有
Schema 的配置宣传为可训练配置。

## 4. Thinking Template 与 Label 边界

Thinking Assistant 固定渲染为：

```text
<|im_start|>assistant
<think>
{reasoning_content}
</think>

{final_answer}<|im_end|>
```

单元测试确认 System/User/Header/尾随换行保持 Mask，`<think>`、推理轨迹、`</think>`、最终
答案和 `<|im_end|>` 全部监督。以下输入会硬失败：

- 没有 Assistant Response；
- Assistant 数量与 Reasoning 数量不一致；
- 空白 Reasoning；
- Reasoning 或最终答案包含嵌套 Think 标签；
- Token Offset、词表和特殊 Token 身份不符合既有 Tokenizer 契约。

## 5. 验证记录

执行日期：2026-07-22（Asia/Shanghai）。

| 检查 | 实际结果 |
| -- | -- |
| M5/Tokenizer 目标单测 | 53 passed |
| 全仓 CPU/集成测试 | 442 passed，2 deselected（GPU Marker） |
| 分支覆盖率 | 85.39%，门槛 85% |
| Ruff Lint | 通过 |
| Ruff Format Check | 通过，192 files |
| MyPy Strict | 通过，192 source files |
| JSON Schema Snapshot | 通过 |
| Markdown Link Check | 通过，72 files |
| Public Artifact Policy | 通过 |
| 文档 Manifest | 声明 39，实际 39 |

完整验证入口为：

```bash
make check
```

## 6. 当前限制与下一批

M5.0 不是训练正确性或模型质量证据。合并本批后，M5.1 按以下顺序推进：

1. 定义 Reasoning Task、Teacher Generation、Verifier、Rejected Record 和 Manifest Schema；
2. 生成只包含公开合成 Fixture 的 CPU Smoke 数据；
3. 建立 200 条独立 M5 Reasoning Dev 的确定性生成与模板族切分；
4. 验证固定 Qwen3-8B Thinking Teacher 的离线加载、采样参数和失败路径；
5. 通过审查后才生成私有 Pilot 数据，不提前启动正式训练。
