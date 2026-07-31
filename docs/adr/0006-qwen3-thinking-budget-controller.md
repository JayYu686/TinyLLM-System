# ADR-0006：采用 Qwen3 官方 Thinking Budget 控制器

## 状态

Accepted；自 M5.2-R4 起生效。

## 背景

M5.2、R1 和 R3 均使用 Qwen3 官方推荐的 Thinking 采样参数，但把单次生成硬限制为
896 Token，并要求模型自然生成完整 `</think>` 和最终答案。真实结果显示：

- R1 两个 Seed 的自然格式率为 94.5% 和 93.5%；
- R2 将失败 Batch 精确重放到 1536 Token 后，投影格式率仍只有 98.0% 和 96.5%；
- R3 用新的 Config/Log 简洁 Trace 训练后，Seed42 的格式率下降到 92.5%。

R3 Seed42 已足以使“双 Seed 均达到 99%”的 AND Gate 失败，因此第二 Seed 在 672,024
Supervised Tokens 时停止；其 500,721 Token Checkpoint 保留。

Qwen3 官方文档为有限 Thinking Budget 提供了两阶段方案：第一阶段达到预算且尚未闭合时，
追加固定的 early-stopping 引导与 `</think>`，第二阶段继续生成最终回答；官方建议有效
Thinking Budget 高于 1024 Token。继续用少量重复数据拟合自然闭标签既没有解决机制问题，
也会增加过拟合和结果选择风险。

参考实现：

- [Qwen3 Thinking Budget](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md)
- [Qwen3-0.6B 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)
- [LLaMA-Factory Qwen3 支持](https://github.com/hiyouga/LlamaFactory)
- [ms-swift Qwen3 模板与 Loss Scale](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)

## 决策

1. 保留 M5 Dev、模型 Revision、Thinking/Non-thinking 模式、采样参数、Seed 和 99% 格式
   门槛；历史协议和失败结果保持只读。
2. 新协议身份为 `m5-thinking-budget-v2`。第一阶段 Thinking Budget 固定为 1536 Token，
   第二阶段最终答案最多生成 128 Token。
3. 第一阶段未自然生成 `</think>` 时，注入 Qwen 官方示例的固定 early-stopping 文本，再
   继续生成最终答案。
4. 每条私有结果必须分别保存第一阶段模型输出、控制器注入文本、第二阶段模型输出、自然
   闭合状态、强制收束状态、Token 数和哈希。
5. 公开摘要同时报告：
   - 控制后格式有效率；
   - 自然闭合率；
   - 强制收束率；
   - Thinking 最终答案正确率；
   - Non-thinking 分数；
   - Token 与运行时间成本。
6. Candidate 必须同时满足：
   - 控制后格式有效率至少 99%；
   - 强制收束率不高于 10%；
   - Thinking 最终答案分数至少 90%；
   - Non-thinking 相对同协议 Base 回退不超过 2pp。
7. 注入的 `</think>` 属于推理控制器行为，不得表述为模型自然生成。M6 继续使用独立冻结
   评测和 Promotion Gate。

## 后果

- M5 从“反复训练模型记住闭标签”转为“模型能力 + 明确推理控制器”的系统设计。
- 格式可靠性与自然闭合能力成为两个独立指标，报告可以解释质量、延迟和控制器依赖。
- R1 两个完整训练 Run 作为协议 v2 的首个 Candidate；R3 保留为失败消融，不继续消耗第二
  Seed 的剩余训练预算。
- 协议 v2 通过前，M5.3 仍保持阻塞；通过后才能冻结正式数据并进入 Full SFT/LoRA。

## 重新评估条件

若后续 Qwen Revision 原生支持可验证的 Thinking Budget API，或自然闭合率稳定达到 99%，
可通过新 ADR 减少控制器介入；不得回写本 ADR 下的真实结果。
