# M10 Agent 后训练契约

## 1. 目标

M10 在 M7 Production 模型和 M9 训练前基线之上构建 Agent Candidate。训练重点来自真实基线
短板：工具选择、参数生成、多步调用、失败恢复、证据引用、审批安全和 No-tool 判断。0.6B Full
SFT 与 8B LoRA 使用同一数据版本、同一评测协议和各自独立的父模型门禁。

M10 分为四个可独立审查的批次：

```text
M10.1 数据冻结与污染检查
→ M10.2 Qwen3-0.6B Full SFT
→ M10.3 Qwen3-8B LoRA
→ M10.4 Release、BFCL、M6 与 Serving 统一门禁
```

M9 的 160 条 Release 在 M10 正式评测前保持密封，只用于最终污染扫描和门禁评测。Release
正文不会进入训练样本生成、数据筛选、Prompt 调整或超参数选择。

## 2. 固定来源与身份

外部来源使用完整 Git Revision 和文件 SHA256 固定。原始数据只进入私有 Artifact Store，公开
仓库保存契约、内容无关画像、哈希与合成测试夹具。

| 来源 | 固定身份 | 监督 Token 比例 | 当前状态 |
| -- | -- | --: | -- |
| ToolACE | `lockon/ToolACE@6bda777...`，`data.json` | 30% | Canonical `5ff7e195` 已提交 |
| Hermes Function Calling | `NousResearch/hermes-function-calling-v1@dae3e1d...`，`func-calling.json` | 20% | Canonical `fb8b61ba` 已提交 |
| TinyLLM DevOps 轨迹 | `tinyllm/devops-agent-training` | 20% | 2,400 条已冻结；80 条分层审查通过 |
| M6 领域能力 Replay | `m6-domain-generalization-mixture-v2-f2e029e4` | 20% | 已注册 |
| M2 No-tool Replay | `m2-sft-v1-f82ff32e` | 10% | 已注册 |

比例按过滤、去重和 Tokenization 后实际参与 Loss 的监督 Token 计算。样本条数只用于数据质量
报告，不用于近似替代 Token 配比。最终语言目标同样按监督 Token 计算：英文 70%，中文 30%。

Hermes 核心只选用完整的 `func-calling.json`。`func-calling-singleturn.json` 与完整文件存在 793
个重复 ID，因此不进入同一训练版本；JSON Mode、Glaive 等其他子集也不在本版本中静默混入。

## 3. 规范化会话

内部会话使用 OpenAI 风格的工具定义与以下角色：

```text
system → user → assistant(tool_calls) → tool → assistant(final)
```

多轮和并行调用可重复出现 `user/assistant/tool` 段。每条规范化记录至少包含：

- 稳定样本 ID、来源 ID、固定 Revision 和原始记录 SHA256；
- 规范化工具定义及其内容哈希；
- 有序消息、工具调用 ID、工具名和结构化参数；
- 语言、任务类型、License、分组 ID 和拒绝/接收状态；
- 每条消息的监督掩码和内容哈希。

监督只覆盖 Assistant 的工具调用和最终回答。System、User 与 Tool Result 用作上下文并从 Loss
中屏蔽。Agent 工具数据以 Non-thinking 为主，M6 Thinking Replay 用于维持原生
Thinking/Non-thinking 双模式。M10 不引入无法自动验证的合成 CoT 教师轨迹。

## 4. 来源适配规则

### 4.1 Hermes

- `tools` 字符串必须解析成 OpenAI `type=function` 工具数组。
- `<tool_call>...</tool_call>` 内部必须是包含 `name` 和对象型 `arguments` 的 JSON。
- 接收 `system/human/gpt`、`system/human/gpt/tool` 和
  `system/human/gpt/tool/gpt` 三类轨迹。
- 工具结果早于 Assistant 工具调用的轨迹以 `invalid_role_path` 拒绝。
- 工具名规范化后发生冲突时，整条记录进入人工/规则修复队列，不静默重命名。

### 4.2 ToolACE

- 从固定 System Envelope 中解析 JSON 工具数组。
- 参数 Schema 的 `type=dict` 规范化为 `type=object`；顶层 `required=null` 被移除并计数。
- `[Tool Name(arg=value)]` 使用自定义、引号与括号感知的解析器转换；参数值只接受
  `json.loads` 或 `ast.literal_eval` 可安全解析的字面量。
- 工具名转换为稳定 OpenAI-safe 名称，同时保存原名；规范化冲突时拒绝整条记录。
- Tool 消息必须由 Assistant 工具调用触发，非法角色转换以稳定原因码拒绝。

任何格式扩展都需要新增 Golden Test、更新契约版本并重新构建数据版本。

## 5. 自建 DevOps 训练轨迹

自建 20% 轨迹覆盖以下能力，并保持英文 70%、中文 30%：

- Single Tool、No-tool 与相似工具 Hard Negative；
- 缺参澄清、顺序多步和并行独立工具；
- 超时、Tool Execution Error 与可恢复失败；
- 证据引用、只读/写操作边界、审批和路径安全。

训练轨迹使用与 M8 参考 MCP Server 相同的公开 Tool Schema，但环境状态、参数、结果和最终
断言独立创作。M9 Dev 可用于暴露能力类别短板；其具体题目和答案不会被复制或改写为训练
样本。M9 Release 保持完全密封。

## 6. 去重、切分与污染闭锁

处理顺序固定为：

```text
source verify
→ parse
→ canonicalize
→ license/filter
→ cross-source exact dedup
→ prompt/tool-schema 5-gram MinHash near dedup
→ eval contamination scan
→ tokenize
→ overlength reject
→ supervised-token balance
→ fixed-length sequence materialization
→ register
```

Exact Dedup 使用规范化 Prompt、Tool Schema、Tool Calls 和最终回答的内容哈希。Near Dedup
以 Prompt 的 5-gram MinHash 命中为必要条件，阈值固定为 0.85；Tool Schema 用于确认工具协议
身份和组合相似度，但七个样本共享同一公开 MCP Tool Catalog 本身不构成重复或污染。去重先于
筛选，具有相同来源会话或生成模板的记录使用同一 Group ID。M10 使用独立冻结的 M9
Dev/Release 作为开发与发布评测边界，因此来源数据只构建 Train 混合，不从训练源再次派生
数据相关的 Validation Split。

污染检查覆盖 M9 Dev、密封 M9 Release、BFCL Offline Core 和 M6 领域评测。Exact 或 Near
命中都会阻止正式数据注册。针对 Release 的扫描只向公开侧输出计数、算法版本和输入/输出
哈希，不输出任务正文、匹配片段或可逆标识。

## 7. 数据状态与训练前门禁

[`configs/data/m10_agent.yaml`](../configs/data/m10_agent.yaml) 保持不可变的来源预注册状态
`preregistered`、`training_permitted=false`；最终训练身份由
[`configs/data/m10_agent_frozen.yaml`](../configs/data/m10_agent_frozen.yaml) 与新的 Dataset
Manifest 共同表达。只有以下条件全部满足后，才能创建 Frozen Config 和 Dataset Manifest：

1. 两个外部固定 Artifact 的文件大小、SHA256、许可证据与结构画像通过；
2. DevOps 训练轨迹完成内容审查、许可声明、确定性重建和哈希冻结；
3. M6 与 M2 Replay 的 Manifest、内容哈希和 Tokenizer 验证通过；
4. 许可过滤、Exact/Near Dedup 和四类污染检查全部通过；
5. 实际监督 Token 比例和 70/30 语言比例达到契约；
6. Canonical JSONL、Rejected JSONL、Shard、Manifest 与 Commit Marker 原子写入并校验；
7. 同一输入、配置、Seed 和代码版本重复构建得到相同内容哈希。

M10.1 已完成上述七项门禁。最终版本 `m10-agent-sft-v1-4655d3e3` 包含 8,061 条
2,048 长度序列和精确 1,000,000 个监督 Token；来源比例为 30/20/20/20/10，语言比例为
70/30，Thinking 比例为 6%。M9 Dev、密封 Release、BFCL Core 和 M6 Domain 的 Exact/Near
污染均为零。完整计数和哈希见
[`M10.1 Agent 训练混合验收报告`](../reports/m10/m10_frozen_mixture.md)。

## 8. 训练与评测阶段

0.6B 父模型固定为 M7 Production，执行 1M、5M、10M Supervised Token 三阶段 Full SFT。
5M 到 10M 只有在 Agent Dev 提升至少 1pp 且 M6 回退不超过 2pp 时继续。

8B 父模型固定为 Qwen3-8B Base，执行相同阶段的 BF16 LoRA。历史 M5 Domain Adapter 仅保留
诊断身份，不作为初始化点。BF16 LoRA 在固定最小配置真实 OOM 后，才创建独立策略身份切换
到 NF4 QLoRA。

最终选择严格使用 M10 预注册 Gate：Release、父模型配对 Bootstrap、Schema、No-tool、工具
幻觉、Grounding、失败恢复、安全、BFCL、M6 和 M7 Serving 证据必须同时通过。门禁失败时
保留 M7 Production，并将 M10 Candidate 保持在 Development 状态。
