# M10 Agent 数据基础审查报告

## 结论

M10 第一批数据基础已完成：训练混合、语言比例、监督掩码、外部 Revision、内部 Replay、
Exact/Near Dedup 和四类污染边界均由严格 Schema 固定。ToolACE 与 Hermes 固定 Artifact 已在
CPU 上完成真实哈希验证和内容无关画像；公开聚合与私有正式结果逐字节一致。

当前数据配置保持 `preregistered`、`training_permitted=false`。这是预期的失败闭锁状态：自建
DevOps 训练轨迹已构建为 `m10-devops-training-v1-2ac97fcd`，真实去重与四边界污染扫描通过；
80 条分层内容审查已由维护者全部确认，自建来源获准进入完整混合构建；M10 正式训练仍未启动。详见
[`m10_devops_training_foundation.md`](m10_devops_training_foundation.md)。

## 固定混合

| 来源 | 监督 Token 比例 | 状态 |
| -- | --: | -- |
| ToolACE | 30% | Artifact 与许可证据已固定 |
| Hermes Function Calling | 20% | Artifact 与许可证据已固定 |
| TinyLLM DevOps 轨迹 | 20% | 2,400 条已冻结；内容审查通过 |
| M6 领域能力 Replay | 20% | Dataset Version、内容与 Manifest 哈希已固定 |
| M2 No-tool Replay | 10% | Dataset Version、内容与 Manifest 哈希已固定 |

语言目标为英文 70%、中文 30%，两者都按实际监督 Token 统计。Loss 只覆盖 Assistant Tool
Call 和最终回答，System、User、Tool Result 全部屏蔽。工具数据以 Non-thinking 为主，通过
M6 Replay 保持 Thinking/Non-thinking 双模式。

## 外部数据真实画像

数据配置 Canonical SHA256：
`7a6015e50c96bec5709c924bb2c2c054f4180c83f3c3c733cb42d56bca3f1e0e`。

| 指标 | ToolACE | Hermes | 合计 |
| -- | --: | --: | --: |
| 原始行 | 11,300 | 1,893 | 13,193 |
| 结构可接收 | 10,770 | 1,822 | 12,592 |
| 隔离 | 530 | 71 | 601 |
| 工具定义 | 33,663 | 3,582 | 37,245 |
| Tool-call 候选行 | 9,354 | 1,883 | 11,237 |
| No-tool 候选行 | 1,946 | 10 | 1,956 |
| 成功解析的 Tool Call | 18,169 | 2,920 | 21,089 |

ToolACE 隔离项包括 5 条非法行结构、524 条工具 Schema 问题和 1 条独立 Tool Call 解析问题。
524 条中，518 条属于另一版 System Envelope，6 条在 OpenAI-safe 工具名规范化后发生冲突。
标准 Envelope 中的 33,663 个 `type=dict` 和顶层 `required=null` 都已被识别并计入确定性
规范化规则。

Hermes 隔离项包括 10 条工具结果早于 Assistant 调用的非法角色顺序，以及 61 条包含 Tool
Call 但独立 `tools` 数组为空的记录。解析器同时覆盖标准 JSON Tool Call 和 793 条以安全
Python 字面量表达的旧式 Information Extraction Call；参数值只经过 JSON 或
`ast.literal_eval`，不会执行表达式。

完整公开聚合见
[`raw/m10_external_source_profile.json`](raw/m10_external_source_profile.json)，文件 SHA256 为
`f36e1da659078d07573314b27cb9f9c43ed37ff58a924dec0625cadb130b3f08`。该文件不包含 Prompt、
Tool 参数、Tool Result、绝对路径、用户名或主机名。

## 契约与失败路径

- 配置和持久化报告使用 Pydantic v2 严格模式、`extra="forbid"` 与版本字段。
- 外部 Artifact 的文件名、大小和 SHA256 任一漂移都会中止画像。
- Hermes 工具名必须存在于当前记录的固定工具数组；ToolACE 参数只接受安全字面量。
- 工具名规范化冲突、非法角色转换和不受支持的 Envelope 使用稳定原因码隔离。
- M9 Release 只允许执行私有、内容无关污染扫描；正文不会进入数据构建或调参。
- 5 个来源全部达到 `ready` 且配置提升为 `frozen` 前，Schema 拒绝
  `training_permitted=true`。

## 当前验收状态

| 验收项 | 状态 |
| -- | -- |
| M10 数据契约与固定混合 | 通过 |
| 严格 Schema 与 JSON Schema Snapshot | 通过 |
| ToolACE/Hermes 固定 Artifact 哈希 | 通过 |
| 安全 Tool Call 解析和合成失败路径 | 通过 |
| 真实外部源内容无关画像 | 通过 |
| DevOps 自建训练轨迹 | 2,400 条已构建；80/80 内容审查通过 |
| 跨来源 Exact/Near Dedup | 待数据齐备后执行 |
| M9 Dev/Release、BFCL、M6 污染检查 | 自建来源通过；完整混合待执行 |
| Frozen Dataset Manifest | 待完成 |

下一批工作是实现 Canonical Importer、跨来源去重、Replay 接入、污染闭锁、Token 配平和注册。
当前无需占用 GPU。
