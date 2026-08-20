# M9 Agent Readiness Evaluation

M9 在模型训练前冻结 DevOps Agent 的任务、评分和晋级边界。它回答三个独立问题：模型是否会
选择正确工具并填写正确参数，Agent Runtime 是否能安全执行多步轨迹，能力提升是否同时满足
BFCL、M6 质量回归和 M7 Serving 血缘约束。

## 评测组成

自建套件共 240 条原创任务：

- 80 条 Dev 随仓库公开，用于开发和错误分析。
- 160 条 Release 保存在私有 Artifact Store；M10 正式评测前不公开正文。
- 英文占 70%，中文占 30%。
- Dev、Release、M6 领域集和后续 M10 训练集必须保持内容隔离。

每条任务保存初始环境状态、七个固定工具及其完整 JSON Schema、允许轨迹、并行组、状态
转移、最终答案断言、失败注入和内容哈希。Release 分布为：

| 类别 | 数量 |
|---|---:|
| Single Tool | 25 |
| No-tool | 20 |
| Wrong-tool / Irrelevance | 20 |
| Missing Argument / Clarification | 20 |
| Sequential Multi-step | 30 |
| Parallel Independent Tools | 10 |
| Tool Failure Recovery | 20 |
| Grounding / Approval / Security | 15 |

任务运行复用 M8 的 LangGraph、Tool Schema 校验、权限策略和安全节点恢复。评测工具返回由
任务身份绑定的确定性夹具结果；失败恢复题在第一次底层读取时注入可重试错误。评测过程不会
提供任意 Shell、任意文件读取或免审批写入能力。

## 指标

事实源保存每题的 Run ID、工具轨迹、参数、Schema 状态、重试次数、证据引用、最终答案、
Token 用量和端到端耗时。汇总指标包括：

```text
Tool Selection Accuracy
Argument Accuracy
Schema Valid Rate
No-tool Accuracy
Multi-step Success Rate
Task Success Rate
Tool Hallucination Rate
Error Recovery Rate
Grounding Accuracy
Approval Safety
Average Tool Calls
Tokens per Task
P95 End-to-End Latency
```

`tinyllm agent eval` 按题原子落盘。相同 Suite、配置、模型 Artifact 和 Git Commit 可以从已有
题目继续；任一身份变化都会拒绝续跑。正式结果要求干净 Git，开发 Smoke 可显式使用
`--allow-dirty`，但该标记会保留在结果中。

每次评测还会在开始前验证 Gateway `/version` 返回的模型版本与 Artifact 哈希，并保存
`config.resolved.json`、`suite.manifest.json`、`environment.json` 和 `hardware.json`。Summary
引用软件环境与 GPU 身份文件的 SHA256；配置中的 `physical_gpu_index` 必须与承载 Gateway
Backend 的物理卡一致。

## BFCL Core Profile

外部对照固定为 `TinyLLM BFCL v1.3 Offline Core Profile`：

```text
BFCL tag:    v1.3
BFCL commit: ea13468e4423454d0c213704fb87cf7cb3990433
任务数:       1840
```

只包含 Simple、Multiple、Parallel、Parallel Multiple、Irrelevance、Multi-turn Base、
Multi-turn Missing Function 和 Multi-turn Missing Parameter。Live、Java、JavaScript、
Multi-turn Long Context、Agentic Web Search 和外部 Memory 均排除。报告名称不得缩写为
BFCL 官方 Overall 或用于官方排行榜比较。

BFCL 在独立环境运行。TinyLLM Endpoint Handler 仅连接环回 Gateway，通过环境变量读取
Bearer Token，并发送 OpenAI Chat Completions Tool Calling 请求；它不会修改固定 BFCL 源码。
依赖安装和审计分别使用 `make bootstrap-bfcl` 与 `make audit-bfcl`。上游固定依赖的适用边界
记录在 `requirements/m9_bfcl_security_exceptions.md`，审计例外不适用于任何线上服务进程。

## M10 Agent Model Gate

M9 提前冻结 M10 门禁：

- Release Task Success 至少 70%。
- 相对父模型提升至少 5pp，配对 Cluster Bootstrap 95% CI 下界大于 0。
- Schema Valid Rate 至少 98%，No-tool Accuracy 至少 90%。
- Tool Hallucination Rate 不高于 2%。
- Grounding Accuracy 至少 90%，Error Recovery Rate 至少 70%。
- 未审批写入、路径逃逸和任意命令执行均为 0。
- BFCL Core 总分不得低于父模型，任一类别回退不超过 2pp。
- M6 能力相对父模型回退不超过 2pp，Serving 与完整血缘继续有效。

门禁比较使用相同 Release Task ID 的配对 Cluster Bootstrap，Seed 为 20260820，重复 10,000
次。单项失败会保留完整证据并返回 `rejected`，不会自动调整阈值。

## 重建与运行

```bash
python scripts/build_m9_agent_suite.py \
  --release-root "$TINYLLM_ARTIFACT_ROOT/agent-evaluations/m9/suites"

tinyllm agent eval \
  --suite evals/agent/dev/v1 \
  --config configs/eval/m9_agent_dev.yaml \
  --model production \
  --output "$TINYLLM_ARTIFACT_ROOT/agent-evaluations/m9/runs/<evaluation-id>" \
  --json
```

BFCL 命令在独立环境执行，原始 `result/`、`score/` 和 TinyLLM 汇总均写入 `/data`：

```bash
python scripts/run_m9_bfcl.py \
  --bfcl-checkout "$TINYLLM_ARTIFACT_ROOT/cache/bfcl/gorilla-v1.3" \
  --output "$TINYLLM_ARTIFACT_ROOT/agent-evaluations/m9/bfcl/<evaluation-id>" \
  --model-id <immutable-model-id> \
  --model-artifact-sha256 <sha256>
```
