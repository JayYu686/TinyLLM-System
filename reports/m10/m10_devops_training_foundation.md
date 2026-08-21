# M10 DevOps Agent 训练轨迹构建报告

## 结论

M10 自建 DevOps 训练源已经完成确定性构建、Schema 校验、分组去重和四边界污染扫描。
正式版本为 `m10-devops-training-v1-2ac97fcd`，完整 2,400 条 Canonical JSONL 与 80 条
分层内容审查包均保存在私有 Artifact Store。维护者已于 `2026-08-21T01:00:18Z` 确认
80/80 条抽样轨迹，来源状态为 `approved`，允许进入完整 M10 混合构建。该批准不授权在最终
混合、Token 配平和污染门禁完成前启动训练。

公开的内容无关构建摘要见
[`raw/m10_devops_training_build.json`](raw/m10_devops_training_build.json)，内容审查审批摘要见
[`raw/m10_devops_content_review.json`](raw/m10_devops_content_review.json)。

## 数据构成

| 类别 | 样本数 |
| -- | --: |
| Single Tool | 360 |
| No-tool | 360 |
| Wrong-tool / Irrelevance | 360 |
| Missing Argument / Clarification | 360 |
| Sequential Multi-step | 360 |
| Parallel Independent Tools | 120 |
| Tool Failure Recovery | 240 |
| Grounding / Approval / Security | 240 |
| 合计 | 2,400 |

样本语言按条数为英文 1,680 条、中文 720 条，即 70%/30%。该比例只描述自建来源的样本
分布；M10 完整混合仍必须在 Tokenization 后按实际监督 Token 重新平衡，不能用样本条数替代
最终语言和来源 Token 门禁。

全部样本共包含 4,320 条受监督 Assistant 消息、6,840 条屏蔽 Loss 的
System/User/Tool 消息和 2,040 次工具调用。所有 Assistant 内容均使用 Non-thinking 模式，
生成与校验过程拒绝可见 `<think>` 轨迹。

## 工具与安全边界

轨迹只使用 M8 已公开并受本地策略约束的七个工具：

```text
search_evidence
list_runs
get_run
read_log_excerpt
query_metrics
inspect_config
apply_sandbox_config_patch
```

每个调用在构建时执行工具名、必填参数、额外参数和顶层类型检查。Tool Call ID 必须唯一，
每次调用必须恰好匹配一个 Tool Result；写轨迹只描述 Agent 沙箱副本，并明确保留运行时审批
和源配置不变约束。未注册 MCP、任意 Shell、路径逃逸、主机重启和生产数据库修改被训练为
拒绝或改为只读证据检查。

## 去重与分组

使用 `minhash-5gram-lsh-v1`、128 个排列和 0.85 阈值执行 Prompt 近重复扫描：

| 指标 | 实际结果 |
| -- | --: |
| Canonical Exact Duplicate Pair | 0 |
| 同模板组内近重复 Pair | 9,228 |
| 跨模板组近重复 Pair | 0 |
| LSH 候选中的最高 Prompt 相似度 | 95.35% |
| 去重门禁 | 通过 |

同一生成配方的结构化变体被绑定到同一个 `group_id`，共 48 个组；它们后续不能跨数据切分。
跨组近重复和 Canonical Exact Duplicate 都会使构建失败。七个样本共享相同 MCP Tool Catalog
属于协议复用，Tool Schema 相同本身不计为数据重复。

## 污染检查

自建数据完成以下真实边界扫描：

| 边界 | 项目数 | Exact | Near | 结果 |
| -- | --: | --: | --: | -- |
| M9 Public Dev `f958bcc6` | 80 | 0 | 0 | 通过 |
| M9 Sealed Release `1ae9b75b` | 160 | 0 | 0 | 通过 |
| BFCL v1.3 Offline Core | 1,840 | 0 | 0 | 通过 |
| M6 Domain v7 `b82cbca1` | 300 | 0 | 0 | 通过 |

Release 扫描只输出版本、输入内容哈希、数量和命中计数，公开报告不包含 Release 正文、匹配
片段或可逆样本标识。Release 内容没有参与轨迹生成、模板选择或参数设计。

## 可复现与 Artifact

完整数据位于：

```text
$TINYLLM_ARTIFACT_ROOT/datasets/m10-agent/devops/m10-devops-training-v1-2ac97fcd/
├── items.jsonl
├── manifest.json
├── duplicate-report.json
├── contamination-report.json
└── COMMITTED.json
```

构建先写 staging 目录，再生成所有文件 SHA256 和 Commit Marker，最后原子 Rename。重复写入
只有在每个已提交文件逐字节一致时才返回成功；同版本内容漂移会被拒绝。

内容审查包位于：

```text
$TINYLLM_ARTIFACT_ROOT/reviews/m10-devops-training-v1-2ac97fcd/review_packet.md
```

它按八个类别、两种语言各固定抽取五条，共 80 条，并隐藏重复的完整 Tool Schema 以便人工
核对 Prompt、调用参数、Tool Result、最终结论和安全边界。

审批记录将原始 Pending Manifest、Review Packet、Items、Duplicate Report 与 Contamination
Report 的 SHA256 绑定到独立 Approved Manifest。原始数据目录保持不变；批准制品以原子目录
提交并保留 Commit Marker。公开审批记录 SHA256 为
`53c6ef815795010dc38c26ac618bbf92390c30efe73ab0bcda74bdea5ec577ed`。

## 当前门禁与下一步

| 检查 | 状态 |
| -- | -- |
| 2,400 条确定性重建 | 通过 |
| Schema、消息哈希与调用配对 | 通过 |
| Exact / 分组 Near Dedup | 通过 |
| 四边界污染扫描 | 通过 |
| 80 条分层内容审查 | 80/80 通过 |
| 允许进入完整 M10 混合 | 是 |
| 完整 M10 混合冻结 | 待完成 |
| 允许启动 M10 训练 | 否 |

下一步进入两个外部来源的 Canonical Import、跨来源去重、Replay 接入、Tokenizer 实测监督
Token 配比和最终 Dataset Registry；在这些步骤完成前不启动 GPU 训练。
