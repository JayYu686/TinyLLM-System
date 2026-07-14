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
