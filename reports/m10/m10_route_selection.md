# M10 Agent 模型路线选择报告

## 结论

最终路线选择 `Qwen3-8B Base + 5M LoRA + 确定性 Tool-Catalog 路由`。候选在 160 条密封
Release 上达到 93.12% Task Success，相对同协议 8B 父模型提升 18.74pp；Cluster Bootstrap
95% CI 为 `[+3.47, +36.07]pp`。它同时满足 BFCL、M6 回归、Serving 和安全门禁，已注册为
`qwen3-8b-m10-agent-production-b2d88493`。

## 选择依据

M10 使用 0.6B Full SFT 与 8B LoRA 两类实验验证训练路径。最终生产路线采用 8B，原因包括：

- M9 冻结基线中，8B Base 的 BFCL Offline Core 为 39.18%，高于 0.6B 的 24.24%；
- 8B LoRA 可在单张 24 GiB RTX 3090 上完成 BF16 训练，正式 Probe 的 Peak Reserved 为
  22.54 GiB；
- 5M Adapter 保留 Base 权重身份，便于独立注册、加载和回滚；
- 专用 Adapter 只对完整 TinyLLM DevOps Tool Catalog 生效，其他请求回到 Base 路径。

训练损失只用于优化过程诊断，路线选择使用端到端 Agent 任务、通用能力和 Serving 证据。

## Release 配对结果

| 指标 | 8B Base | 8B Agent Production | 变化 |
|---|---:|---:|---:|
| Task Success | 74.38% | **93.12%** | **+18.74pp** |
| Tool Selection | 83.12% | **98.12%** | +15.00pp |
| Argument Accuracy | 79.38% | **98.12%** | +18.74pp |
| Schema Valid | 100% | **100%** | 0pp |
| No-tool Accuracy | 66.67% | **100%** | +33.33pp |
| Tool Hallucination | 16.88% | **0%** | -16.88pp |
| Multi-step Success | 78.33% | **100%** | +21.67pp |
| Error Recovery | 100% | **100%** | 0pp |
| Grounding | 100% | **100%** | 0pp |

候选在未审批写操作、路径逃逸和任意命令执行三类安全计数上均为 0。Release 使用固定
`tinyllm-devops-agent-release-v8-a7969931`，候选与父模型均在相同 Runtime 和评分协议下执行。

## 外部与通用回归

| 门禁 | 父模型 | 候选 | 判定 |
|---|---:|---:|---|
| TinyLLM BFCL v1.3 Offline Core Profile | 39.18% | **39.29%** | 总体 +0.11pp |
| BFCL 最差类别变化 | — | -0.50pp | 在 -2pp 边界内 |
| M6 三任务聚合 | 62.64% | **62.64%** | 0pp 回归 |
| M7 Serving 平台门禁 | 9/9 | 复用并绑定精确模型 | 通过 |

## 生产身份

```text
Source Subject:     qwen3-8b-m10-agent-lora-5m-3e8bf1dd
Production Version: qwen3-8b-m10-agent-production-b2d88493
Alias:              agent-production
Effective Model:    05077ebd567eca7ffaaa6504927a25dbc36951caacae33c2149979e54fe56d81
Final Gate:         b2d88493e308ea93507f069a447920ed956cdfdd1d55ec7b288a33b9b52bfd63
Production Record:  eccccd83402254a8626527f647c28a76a765f3a4feaa0ef21023595a2495d78c
```

Registry 记录不包含服务器路径；运行时从私有 Artifact Store 解析 Subject，逐项校验 Base、
Tokenizer、Adapter 和路由策略哈希后才允许启动。
