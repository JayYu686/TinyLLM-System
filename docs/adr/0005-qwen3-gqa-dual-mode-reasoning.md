# ADR-0005：M5 保留 Qwen3 GQA 并采用原生双模式推理

## 状态

Accepted；自 M5 起生效。

## 背景

M5 原计划只训练 Qwen3 的 Non-thinking 路径。项目随后提出两个增强目标：避免传统 MHA，
以及让候选模型具有可控的 Thinking/CoT 能力。固定 Revision 的 Qwen3-0.6B 和 Qwen3-8B
已经分别使用 16/8 和 32/8 个 Query/KV Head，属于 GQA，而不是每个 Query Head 都保存独立
KV 的传统 MHA。它们的 Tokenizer Template 也原生支持 `enable_thinking` 和
`reasoning_content`。

将已经后训练的 Qwen3 权重直接改造成 MLA 并不是等价替换：投影结构、RoPE 布局、
Checkpoint 身份和推理缓存都将改变，需要新的转换或继续预训练研究，并会破坏 M2/M4 已冻结
的模型血缘。MLA 的主要系统收益还依赖推理阶段的 KV Cache 测量，不能在 M5 训练阶段凭架构
名称宣称成立。

## 决策

1. M5 保留固定 Qwen3 Revision 的原生 GQA，不实现或转换 MLA。
2. Qwen3-0.6B Full SFT 与 Qwen3-8B LoRA 都必须支持显式 `thinking` 和
   `non_thinking` 两种模式。
3. M2 的 `qwen3-chatml-nonthinking-v1`、注册数据和训练前 Baseline 保持不可变；M5 使用
   新的 `qwen3-chatml-thinking-v1`、Reasoning 数据版本和双模式配置。
4. Thinking 样本监督 `<think>`、可见推理轨迹、最终答案和 Assistant 结束 Token；
   Non-thinking 样本继续使用 M2 已冻结的 Assistant-only Mask。
5. 正式训练、评测和部署配置必须显式声明模式，不能依赖 Qwen3 的隐式默认值。Prompt 中
   `/think` 和 `/no_think` 仅可用于交互演示，不能代替实验配置。
6. 报告使用“可见推理轨迹”描述 CoT，不把生成文本宣称为模型真实或忠实的内部推理过程。
7. M5 只实现监督式双模式 SFT。RLHF、DPO、GRPO、过程奖励模型、自研 KV Cache 和自研
   Attention 不进入本里程碑。

## 后果

- M2/M4 的历史证据和固定模型 Revision 继续有效。
- M5 必须先增加 Reasoning 数据、模板、双模式 Baseline 和独立 Dev Set，之后才能启动正式
  训练；原两周估计不再作为承诺。
- Thinking 使用采样生成，必须保存解码参数和 Seed；现有 Non-thinking Greedy Baseline 不得
  被静默替换。
- Qwen3-8B LoRA 仍需真实单卡显存 Probe。若 BF16 LoRA OOM，只能在保存失败证据后使用
  独立身份的 NF4 QLoRA 回退。
- 模型未通过 M6 双模式质量与回归门禁前只能保持 `Development` 状态。

## 重新评估条件

只有在 M6 核心交付完成，且项目能够对原生 MLA 模型执行同条件训练与 M7 KV Cache/推理
Benchmark 时，才考虑把 MLA 作为独立研究对照；它不得回溯修改本 ADR 下的 Qwen3 主线。
