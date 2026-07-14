# 数据契约

## 1. 内部样本格式

```json
{
  "schema_version": "1.0",
  "id": "sample-000001",
  "messages": [
    {"role": "system", "content": "可选"},
    {"role": "user", "content": "问题"},
    {"role": "assistant", "content": "回答"}
  ],
  "source": "dataset_name",
  "split": "train",
  "metadata": {
    "language": "zh",
    "category": "programming",
    "license": "unknown"
  }
}
```

## 2. 必填字段

- schema_version
- id
- messages
- source
- split

## 3. 校验规则

- `messages` 至少包含一组 user/assistant。
- role 只能为 system/user/assistant。
- content 不能为空。
- id 在同一数据版本内唯一。
- split 只能为 train/validation/test。
- 不接受无法解析的控制字符。
- 超长样本必须记录过滤原因。

## 4. 数据流水线

```text
import
→ validate
→ normalize
→ filter
→ deduplicate
→ split
→ tokenize
→ pack
→ register
```

## 5. Manifest

```json
{
  "dataset_name": "instruction",
  "version": "1.0.0",
  "schema_version": "1.0",
  "raw_hash": "...",
  "processed_hash": "...",
  "pipeline_version": "...",
  "tokenizer": "...",
  "tokenizer_revision": "...",
  "max_length": 2048,
  "packing": true,
  "num_raw_samples": 0,
  "num_valid_samples": 0,
  "num_rejected_samples": 0,
  "created_at": "...",
  "git_commit": "..."
}
```

Manifest 分为确定性内容和构建元数据：

- `raw_hash`、`processed_hash`、处理配置、Tokenizer Revision 和 Split Hash 参与内容身份计算。
- `created_at` 等易变字段不参与内容哈希。
- 相同输入、代码版本和处理配置必须得到相同内容哈希；构建时间可以不同。

## 6. 数据切分

必须先完成跨来源去重，再进行切分。

禁止：

- 同一问答的改写版本跨训练和测试。
- 使用测试集进行 Prompt 调整。
- 将评测集拼入 SFT 数据。
- 训练时读取未注册目录。

## 7. 版本规则

- 内容变化：提升 Patch 或 Minor。
- Schema 变化：提升 Major。
- Tokenizer 变化：创建新数据构建版本。
- Packing 参数变化：创建新处理版本。

## 8. 隐私和许可

Manifest 必须记录：

- 来源。
- 许可状态。
- 是否包含个人信息。
- 是否允许再分发。
- 清洗策略。

不明确许可的数据不得随项目仓库发布。

## 9. 求职版本固定来源

M2 使用 ADR-0003 固定的 OASST1 和 CommitPackFT revision。OASST 按 Conversation
Tree、CommitPackFT 按 Repository 分组切分，防止同源泄漏。核心必须完成 Exact Dedup
和训练集/评测集污染检查；Near Dedup 仅作为增强，不阻塞 M2。导入时必须验证 Dataset
Card 和样本来源许可证，不能把 ADR 中的选择当作自动许可。

### 9.1 固定导入快照

| 来源 | Dataset ID | Revision | Dataset Card 许可 |
| -- | -- | -- | -- |
| OASST1 | `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | Apache-2.0 |
| CommitPackFT | `bigcode/commitpackft` | `fc56fe33c030c6daa414c2b112c932b8eed085e6` | MIT |

Revision 不允许静默漂移。导入 Manifest 必须记录 Dataset ID、完整 Revision、输入内容
SHA256 和处理配置哈希。Dataset Card 许可只描述数据集发布物，不能覆盖 CommitPackFT
样本中 `license` 字段所表示的底层仓库许可。

### 9.2 OASST1 导入语义

- 只把以 `assistant` 消息结束的路径转换为训练样本；同一 Tree 的不同回复共享分组 ID。
- 路径上的每条消息都必须属于 `ready_for_export` Tree、`review_result=true` 且
  `deleted=false`。
- 路径必须从 `prompter` 开始，并严格按 `prompter`/`assistant` 交替；缺失父消息、循环、
  空内容或非法角色必须拒绝。
- M2 求职数据版本只接收 `en` 和 `zh`；最终 70%/30% 语言比例在分组切分和 Token
  预算阶段实现，导入器不通过复制样本来强行配比。
- OASST1 样本许可证固定记录为 `apache-2.0`。

### 9.3 CommitPackFT 导入语义与许可策略

- 只接收 `lang=Python`、指令非空、目标文件内容非空且 Repository 标识存在的样本。
- 多 Repository 字段按逗号拆分、去重并排序；后续 Split 必须把共享任一 Repository
  的样本视为同一连通分组，不能只选择第一个 Repository。
- M2 核心白名单只包含明确宽松的 SPDX 许可：`mit`、`apache-2.0`、
  `bsd-2-clause`、`bsd-3-clause`、`isc`、`cc0-1.0`、`unlicense`。
- `unknown`、`agpl-3.0`、`lgpl-2.1`、`mpl-2.0`、`epl-1.0` 和
  `artistic-2.0` 暂不进入训练数据。扩大白名单必须经过许可审查、契约变更和新数据版本。
- 公开仓库只保存合成的小型测试夹具，不保存原始数据内容。

### 9.4 导入阶段输出

导入阶段产生 `ImportedSample`、`RejectedRecord` 和 `DataImportManifest`。接收样本保存组成
该样本的原始记录哈希列表；拒绝记录只保存来源记录 ID、原始记录哈希和稳定原因码，
不复制可能受许可或隐私限制的完整内容。
OASST 的 Prompter 行用于组装上下文，不单独计为候选训练样本；Manifest 因此分别记录
`source_rows` 和 `candidate_samples`，不能用“原始行数 = 接收数 + 拒绝数”的错误假设。

导入产物仍不是可训练数据。只有完成规范化、Exact Dedup、分组切分、Tokenization、
Packing 和注册，并产生最终 Dataset Manifest 后，Trainer 才能读取该版本。

## 10. M2.2 规范化契约

规范化必须保守且可复现，不能把语义不同的文本折叠为同一条数据：

1. Unicode 使用 NFC，不使用可能改变代码符号的 NFKC。
2. `CRLF` 和单独的 `CR` 统一为 `LF`。
3. 移除消息开头的单个 Unicode BOM，并裁剪消息首尾空白；不折叠消息内部空格、Tab、
   空行或代码缩进。
4. 除 `LF` 和 Tab 外的 Unicode `Cc` 控制字符直接拒绝，不静默删除。
5. 默认单消息最多 131,072 字符、单样本最多 262,144 字符；超限原因必须计入拒绝统计。

这些规则的任何变化都会改变处理配置哈希并要求新数据版本。长度限制只控制导入产物；
Tokenizer 长度和 Packing 规则在 M2.3 独立生效。

## 11. Exact Dedup 契约

- Exact 内容身份是规范化后按顺序排列的 `role + content` 消息列表 SHA256，不包含来源、
  Split 或随机字段，因此可以跨 OASST1 和 CommitPackFT 去重。
- 保留项按显式来源优先级和 Sample ID 决定；默认优先保留人工对话来源 OASST1。
- 每个被删除的重复样本都产生 `exact_duplicate` 拒绝记录，并指向保留 Sample ID。
- 保留样本必须合并所有重复来源的 Sample ID 和命名空间化 Group Key。否则同一内容可能
  通过另一个 Repository/Tree 连接关系进入不同 Split。
- Near Dedup 不属于 M2.2 核心，不得用模糊阈值替代可验证的 Exact Dedup。

## 12. 分组切分契约

默认 Split 使用整数 Basis Points：Train 9,800、Validation 100、Test 100，固定 Seed 42。
比例和 Seed 都进入配置哈希。切分单位不是单个样本，而是 Group Key 的连通分量：

- OASST Group Key 为 `oasst1:<conversation-tree-id>`。
- CommitPackFT Group Key 为 `commitpackft:<repository-id>`。
- 一个样本关联多个 Repository 时，这些 Repository 必须 Union 到同一连通分量。
- Exact Duplicate 合并的跨来源 Group Key 也必须 Union。
- 连通分量 ID 由排序后的完整 Group Key 集合计算；使用 Seed + Component ID 的 SHA256
  映射到 0–9,999，再按整数边界分配 Split。

Manifest 必须记录输入内容哈希、处理配置哈希、输出内容哈希、每个 Split 的独立哈希、
拒绝原因、重复数、Split 计数和连通分量数。相同输入集合、配置和 Seed 必须得到完全相同
的内容哈希，输入迭代顺序不得影响结果。小型 Smoke 数据不保证每个 Split 都非空；正式
数据报告必须检查实际比例偏差，并按 Group/Token 两种口径解释。

集合内容哈希按 Sample ID 排序后，对每条 Canonical JSON 使用 8 字节大端长度前缀进行
流式 SHA256；空集合也有确定哈希。该编码避免构造整份数据的内存副本，并消除简单文本
拼接的边界歧义。

## 13. M2.3a Tokenizer 与 Chat Template 契约

M2、M5 使用与 `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`
相同 revision 的 Tokenizer，不允许浮动到 `main`：

| 文件 | 字节数 | SHA256 |
| -- | --: | -- |
| `tokenizer.json` | 11,422,654 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | 9,732 | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |

Tokenizer 后端固定为 `tokenizers==0.21.4` 兼容范围，词表（含 Added Tokens）为 151,669。
`<|endoftext|>`/Pad ID 为 151643，`<|im_start|>` 为 151644，`<|im_end|>`/EOS 为
151645。加载时必须先检查文件大小和 SHA256，再检查词表及特殊 Token ID；不需要下载模型
权重。

M5 的 SFT 模式固定为 `qwen3-chatml-nonthinking-v1`：

```text
<|im_start|>{role}\n{content}<|im_end|>\n
```

它是官方 Qwen3 ChatML 在 system/user/assistant 消息上的 Non-thinking 子集，不添加
`<think>`/`</think>`，不添加 Generation Prompt，也不支持本阶段范围外的 Tool 消息。
Template 规范的 Canonical JSON SHA256 为
`d41161e0416a1047b0f31cce1497e610a4050fbe4d3fb7bda19cc56a1523cb33`。

Assistant-only Loss 的 Label Mask 规则：

- system/user Header、内容和结束符全部为 `-100`。
- assistant Header 为 `-100`。
- assistant 内容和紧随其后的 `<|im_end|>` 使用真实 Token ID 作为 Label。
- assistant 后的换行不监督。
- Offset 必须与完整渲染文本严格对齐；Offset 越界、逆序、数量不一致或 Token ID 越界
  属于 Tokenizer 契约失败，必须终止构建，不能按普通坏样本跳过。
- 超过 1,024 Token 的样本记录为 `sequence_too_long`；没有任何监督 Token 的样本记录为
  `no_supervised_tokens`。两者都不保存原始文本。

本阶段只产生 `TokenizedSample`，仍不代表最终注册数据。确定性混合、Packing、最终 Manifest
和 Registry 分别由 M2.3b/M2.3c 完成。
