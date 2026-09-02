# TinyLLM Agent Production Model Card

## 模型身份

| 项目 | 值 |
|---|---|
| Production Version | `qwen3-8b-m10-agent-production-b2d88493` |
| Source Subject | `qwen3-8b-m10-agent-lora-5m-3e8bf1dd` |
| Base Model | `Qwen/Qwen3-8B`，Revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Architecture | Qwen3 dense Transformer，GQA，8,234,382,336 参数 |
| Adaptation | BF16 LoRA，Rank 16，Alpha 32，Dropout 0.05 |
| Training | 5,000,000 Supervised Token，Checkpoint `checkpoint-tokens-0005000000` |
| Effective Artifact SHA256 | `05077ebd567eca7ffaaa6504927a25dbc36951caacae33c2149979e54fe56d81` |

## 用途与运行方式

模型用于 TinyLLM-System 的 DevOps 诊断单 Agent：检索训练 Run、配置、日志和指标，给出带
证据引用的诊断结论，并在需要时提出受审批的沙箱配置修改。Agent 默认使用 Non-thinking；
Thinking 是显式可选模式。Adapter 只对完整的 TinyLLM 七工具目录启用，普通对话和其他工具
目录使用 Base 路径。

运行时通过 `agent-production` Alias 解析私有 Artifact Store。启动前会校验模型、Tokenizer、
Adapter、路由策略、评测和平台门禁 SHA256；公开仓库不包含权重、提示词、工具结果或服务器
身份信息。

## 真实评测摘要

- 160 条密封 Release：Task Success 93.12%，父模型 74.38%，提升 18.74pp；Cluster Bootstrap
  95% CI `[+3.47, +36.07]pp`。
- Schema Valid、No-tool、Multi-step、Error Recovery、Grounding 均为 100%；Tool Hallucination
  为 0%。
- BFCL v1.3 Offline Core Profile：39.29%（723/1840），父模型 39.18%（721/1840）。
- M6 通用任务聚合：62.64%，与父模型相比回归 0pp。
- 未审批写入、路径逃逸、任意命令执行：0 / 0 / 0。

完整门禁、配置身份和证据哈希见 [M10 总验收报告](../../reports/m10/m10_acceptance.md)。

## 限制与安全边界

模型能力集中在已注册的 TinyLLM DevOps 工具和证据格式；它不是通用运维自动化系统。工具
Allowlist、路径根目录、审批和幂等策略由运行时强制执行，模型输出不能扩大权限。原始 CoT、
完整 Prompt、工具参数和工具结果默认不写入公开日志。任何写操作仅作用于 Agent 专属沙箱副本。

## 许可证

基础模型和 Adapter 的许可证分别遵循 Qwen3 模型条款与本项目发布策略；数据集许可证和固定
Revision 记录在 M10 Dataset Manifest 中。
