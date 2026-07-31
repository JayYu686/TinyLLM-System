# M5.2-R3 正式来源扩展准备报告

## 1. 结论

维护者内容审查 33/33 通过后，`m5-r3-formal-source-v1` 的 240→160 正式来源契约、
确定性任务生成器、分片 GPU Runner、Finalizer、CPU 合成 Smoke 和失败路径已经实现。
当前状态为 `READY_FOR_GPU_EXPANSION`。

CPU Smoke 的 240/240 接受与 160 条选择是合成契约结果，明确标记
`model_generated=false`、`quality_metric=false`。它只授权真实 Qwen3-8B 来源生成，
不代表真实来源质量，也不授权 Mixture 或训练。

## 2. 协议身份

| 项目 | 固定值 |
| -- | -- |
| Expansion | `m5-r3-formal-source-v1` |
| Config SHA256 | `08ccc14ca01173df853b60065aad978833dd617fc5ae38c01263e2023f5d8eba` |
| Parent Content Review SHA256 | `ec42e7a3f62d5db7953677a75960e3c7e3bd6a328782e2353ea0130ddf4211ae` |
| Task Set SHA256 | `7bd4af40ac3325c77612ed41c9edb54bbe2ed6c8b10f97c5da9b6d522f3336fd` |
| Teacher | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` |
| 架构 / 精度 | GQA / BF16 |
| Policy Tokenizer | Qwen3-0.6B Revision；Tokenizers 0.21.4 |

## 3. 任务与选择

正式输入固定为：

| 任务族 | 英文 | 中文 | 合计 |
| -- | --: | --: | --: |
| Config | 84 | 36 | 120 |
| Log Diagnosis | 84 | 36 | 120 |
| 合计 | 168 | 72 | 240 |

四个标签在每个任务族内各出现 30 次。Evidence Anchor、Case Reference、Prompt SHA256
和 Task ID 均唯一。最终按以下稳定键选择：

```text
reasoning_tokens
→ repeated_8gram_basis_points
→ sample_id
```

选择配额为每个任务族英文 56、中文 24，共 160 条。任何分层不足都会让整个正式来源
结果失败，不从旧 Pilot 补齐，也不降低 Trace Policy。

## 4. 生成协议

每条任务使用：

1. Qwen3-8B 原生 Thinking、最大 896 New Tokens 完成一次简洁 Solver；
2. Exact Answer Verifier 验证答案；
3. 隔离的 Non-thinking Compressor 只接收已验证答案、Evidence 和 Evidence Anchor；
4. Qwen3-0.6B Tokenizer 重验 192 Token、重复、唯一性和 1024 序列上限；
5. 按四个分层执行确定性选择。

GPU 任务支持 1–8 个独立分片。每个分片绑定 Config、Git Commit、Shard Index、物理 GPU、
环境版本、任务列表和生成记录；Finalizer 要求所有分片来自同一干净提交并完整覆盖 240
个 Task ID。分片可按当前空闲 GPU 并发运行，不要求八卡同时空闲。

## 5. CPU 合成证据

| 检查 | 结果 |
| -- | -- |
| 合成输入 / 接受 / 选择 | 240 / 240 / 160 |
| Config 英文 / 中文选择 | 56 / 24 |
| Log 英文 / 中文选择 | 56 / 24 |
| Dev、历史 Pilot、P0、P0-R1、P1 污染 | 全部为 0 |
| 语言分层不足 | 拒绝 |
| Solver Seed 漂移 | 拒绝 |
| 父任务污染 | 拒绝 |
| 真实 GPU 扩展授权 | `true` |
| Mixture / 训练授权 | `false / false` |

机器证据见
[m5_r3_formal_source_cpu_smoke.json](raw/m5_r3_formal_source_cpu_smoke.json)。

## 6. 下一步

在通过 Preflight 的空闲 RTX 3090 上运行全部分片。Finalizer 只有在四个分层全部满足
56/24、污染为 0、来源和软件身份一致时，才会写入
`r3_mixture_authorized=true`。失败样本和原因必须保留，不重写为成功。

相关入口：

- [正式配置](../../configs/data/m5_r3_formal_source.yaml)
- [分片与 Finalizer Runner](../../scripts/run_m5_r3_formal_source.py)
- [CPU Smoke](../../scripts/run_m5_r3_formal_source_cpu_smoke.py)
- [内容审查结果](m5_r3_content_review.md)
