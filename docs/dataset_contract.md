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
