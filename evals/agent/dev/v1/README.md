# TinyLLM DevOps Agent Dev v1

该目录保存 M9 的 80 条公开开发集，用于在模型后训练前验证 Agent Runtime、工具选择、参数、
多步执行、失败恢复、证据引用和审批安全。任务由 TinyLLM-System 原创，按 Apache-2.0 发布。

固定分布：

| 类别 | 数量 |
|---|---:|
| Single Tool | 13 |
| No-tool | 10 |
| Wrong-tool / Irrelevance | 10 |
| Missing Argument / Clarification | 10 |
| Sequential Multi-step | 15 |
| Parallel Independent Tools | 5 |
| Tool Failure Recovery | 10 |
| Grounding / Approval / Security | 7 |

语言比例为英文 56 条、中文 24 条。每条任务绑定初始状态、完整工具 Schema、允许轨迹、状态
转移、最终答案断言、失败注入和三类内容哈希。该目录明确排除在 M10 训练数据之外。

160 条 Release 集正文仅保存在私有 Artifact Store，正式评测前不会提交至公开仓库。Dev 与
Release 的 Prompt、任务 ID 和内容哈希互不重叠。

确定性重建与检查：

```bash
python scripts/build_m9_agent_suite.py \
  --release-root "$TINYLLM_ARTIFACT_ROOT/agent-evaluations/m9/suites"
python scripts/build_m9_agent_suite.py --check
```
