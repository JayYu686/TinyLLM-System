# M10 外部 Agent 数据 Canonical Import 报告

## 结论

ToolACE 与 Hermes Function Calling 两个固定 Artifact 已完成真实全量 Canonical Import。输入
13,193 行，接收 12,592 行，隔离 601 行；两份私有制品均通过逐文件 SHA256、Commit Marker
和原子目录提交。公开聚合不包含 Prompt、工具参数、工具结果、路径或可逆样本标识。

内容无关原始结果见
[`raw/m10_external_canonical_import.json`](raw/m10_external_canonical_import.json)，文件 SHA256 为
`fec31e75a9c739a50412db44a5e5a1d0ce53eecefa88c5a8611df17e9f4daed8`。

## 正式版本

| 来源 | Canonical Version | 输入 | 接收 | 隔离 |
| -- | -- | --: | --: | --: |
| ToolACE | `m10-toolace-canonical-v1-5ff7e195` | 11,300 | 10,770 | 530 |
| Hermes Function Calling | `m10-hermes-canonical-v1-fb8b61ba` | 1,893 | 1,822 | 71 |
| 合计 | — | 13,193 | 12,592 | 601 |

ToolACE 内容 SHA256 为
`5ff7e19533e330722975605a03d0e8bedb2c0295dace04a6e93d7486b9c8dcca`；Hermes 内容 SHA256 为
`fb8b61ba289fe192afe5c815fb5b58a57ac4d75337b8fea49409e915df2d7be1`。

## Canonical 规则

两种源格式被转换为统一的严格消息契约：

- 工具定义统一为名称、描述和 `type=object` 的输入 JSON Schema；
- ToolACE 的 `type=dict` 规范化为 `object`，源文件顶层 `required=null` 不进入函数 Schema；
- 工具名执行确定性 OpenAI-safe 规范化，规范化后冲突的整行数据进入隔离区；
- Assistant Tool Call 转为结构化名称、参数和稳定 Call ID；Tool Result 绑定实际 Call ID；
- Assistant Tool Call 与最终回答参与监督，System、User 和 Tool Result 屏蔽 Loss；
- Hermes 的标准 JSON 和安全 Python Literal 历史格式均只做解析，不执行表达式；
- 可见 `<think>`、未知工具、非法角色顺序和无法配对的 Tool Result 失败闭锁；
- 每条消息、工具、调用、样本、输入行和最终 Manifest 都保存独立内容哈希。

ToolACE 共形成 13,272 条监督消息、24,042 条屏蔽消息和 18,139 次工具调用；Hermes 共形成
2,749 条监督消息、4,734 条屏蔽消息和 2,920 次工具调用。

## 隔离统计

| 来源 | 原因 | 行数 |
| -- | -- | --: |
| ToolACE | 非法行结构 | 5 |
| ToolACE | 非法或冲突工具 Schema | 524 |
| ToolACE | Tool Call 无法安全解析 | 1 |
| Hermes | 非法角色顺序 | 10 |
| Hermes | Tool Call 存在但工具 Schema 缺失 | 61 |

隔离记录只保存在私有 Artifact Store，包含源行序号、源行内容哈希和稳定原因码，不公开源内容。

## 语言分布风险

按用户消息中的 CJK 字符执行确定性语言识别后，ToolACE 为英文 10,763 条、中文 7 条；Hermes
为英文 1,822 条、中文 0 条。外部源明显偏向英文，不能直接满足最终 70%/30% 监督 Token
目标。后续混合必须在固定 Qwen3 Tokenizer 上测量实际参与 Loss 的 Token，再从已批准的
TinyLLM DevOps 与 M6 Replay 中完成中文配平；不得按样本条数近似比例。

## Artifact 与失败路径

每个来源的私有目录包含：

```text
items.jsonl
rejected.jsonl
manifest.json
COMMITTED.json
```

写入先在同一父目录建立 Staging，再写入所有内容和文件哈希，最后原子 Rename。相同版本只有
在所有文件逐字节一致时才能复用；版本目录存在但任一文件漂移时直接失败。输入文件名、大小或
固定 SHA256 不符时不会开始转换。

## 当前边界与下一步

Canonical Import 只完成两个外部来源的结构统一与隔离。M10 配置继续保持
`preregistered`、`training_permitted=false`。下一步需要：

1. 将已批准的 2,400 条 DevOps 轨迹映射到统一 Canonical Contract；
2. 验证并导入 M6 Domain Replay 与 M2 No-tool Replay；
3. 在五个来源之间执行 Exact Dedup、Prompt/Tool Schema MinHash Near Dedup；
4. 扫描 M9 Dev、密封 Release、BFCL Core 与 M6 Domain 污染边界；
5. 使用固定 Tokenizer 按监督 Token 构建 30/20/20/20/10 来源比例与 70/30 语言比例；
6. 原子签发最终 Frozen Dataset Manifest 后，才允许进入 GPU 训练。
