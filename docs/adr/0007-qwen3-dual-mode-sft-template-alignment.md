# ADR-0007：修正 Qwen3 双模式 SFT 模板对齐

## 状态

Accepted；M6 v1 门禁拒绝后的修复批次生效。

## 背景

M6 v1 独立发布评测发现，0.6B 10M Full-SFT Candidate 在 M5 Dev 上有 99% 的自然 Thinking
闭合率，但在 300 条发布集上只有 1 条自然生成 `</think>`，其余 299 条均由控制器强制收束。
同一 Base 模型有 292/300 条自然闭合，因此问题不属于生成上限不足或评测器漏判。

代码与固定 Tokenizer Revision 的逐 Token 对照随后确认：

- Qwen3 `enable_thinking=true` 的 Generation Prompt 结束于 Assistant Header，模型从该位置生成
  `<think>`；
- Qwen3 `enable_thinking=false` 会在 Assistant Header 后预填空的
  `<think>\n\n</think>\n\n`，模型只生成最终答案；
- 历史 `qwen3-chatml-nonthinking-v1` 在 M5 训练中没有预填空 Think 块，而是从同一个
  Assistant Header 位置直接监督最终答案；
- Thinking 样本则从同一位置监督 `<think>`。

因此旧数据把“模式选择”变成了两个互相竞争的首 Token 目标。30% Thinking 配比只能让模型
在重复出现的 M5 Prompt 模板上学习主题相关的隐式模式选择，无法在独立 Prompt 上稳定执行
显式 Thinking 请求。

## 决策

1. `qwen3-chatml-nonthinking-v1` 保持不可变，只用于验证 M2/M5 历史 Artifact。
2. 新增 `qwen3-chatml-nonthinking-sft-v2`：空 Think 块属于输入上下文并全部 Mask，只监督最终
   答案和 Assistant End Token。
3. Thinking 继续使用 `qwen3-chatml-thinking-v1`，从 Assistant Header 后监督完整 Think 块、
   最终答案和 Assistant End Token。
4. 修复数据只使用 M6 运行前已经冻结、污染检查为零的 M2 Train 与 M5 R3 来源；同一 R3 任务
   同时构造 Thinking 和 Non-thinking Pair，不读取 M6 Prompt、Reference 或模型输出。
5. M6 v1 的 Base、Candidate、人工 Judgment 和拒绝结论保持不可变，不重新解释或覆盖。
6. M6 v1 已用于失败诊断，不能继续充当最终发布集。修复模型先通过独立 Proxy，随后在新的
   内容身份上执行 M6 v2；门禁阈值不得降低。

## 后果

- 新 Candidate 必须拥有新的数据版本、训练配置、Run、Checkpoint、模型 Hash 和评测身份。
- 历史 10M Candidate 继续保持 `Development`，不能因模板已修复而追溯晋级。
- 修复实验先执行 1M Token 双 Seed；只有模式闭合、JSON、领域正确率和通用回归同时满足预注册
  Proxy 门禁，才允许进入较长训练或正式 M6 v2。
- 该修复不会改变 GQA、模型 Revision、M6 量化阈值或 Candidate/Production 的边界。

## 参考

- Qwen3 固定 Revision 的本地 `tokenizer_config.json` 与 `chat_template.jinja`；
- [Qwen3-0.6B 模型说明](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md)。
