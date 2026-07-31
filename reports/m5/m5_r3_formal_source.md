# M5.2-R3 正式来源扩展报告

## 1. 结论

`m5-r3-formal-source-v1` 已在 GPU 4、7 两张 RTX 3090 上完成真实运行，状态为
`COMPLETED_GATE_PASSED`：

- 240/240 个任务完成 Solver，218 条通过完整 Trace Policy，接受率为 90.83%；
- 四个任务族/语言分层均满足预注册配额，确定性选择 160/160 条；
- 与 Dev、历史 Pilot、P0、P0-R1、P1 的所有污染计数均为 0；
- `formal_source_expansion_complete=true`、`r3_mixture_authorized=true`；
- `r3_training_authorized=false`，正式来源通过不等于训练门禁通过。

门禁后的可行性审计发现，当前 160 条来源每轮只有 4,933 个监督 Token。按照
`max_source_reuse=4`，最多只能提供 19,732 个监督 Token，无法满足 R3 Mixture 的
150,000 Targeted Thinking Token。即使将“复用四次”解释为初次使用之外再复用四次，
也只有 24,665 Token。因此 Mixture 实现继续阻断，必须先发布一个前瞻、版本化的新契约，
不能静默提高复用上限。

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
| 生成 Git Commit | `84c21864446cbc2f99679a579c534a58c7583c55` |
| Finalizer Git Commit | `4aa3c9ff0c2e087266c06f2cae9e734a80f53b9a` |
| GPU | RTX 3090 物理卡 4、7 |
| PyTorch / Transformers | `2.7.1+cu118` / `4.57.6` |

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

## 5. 真实运行结果

| 分层 | 输入 | 接受 | 接受率 | 选择 / 要求 | 结果 |
| -- | --: | --: | --: | --: | -- |
| Config 英文 | 84 | 63 | 75.00% | 56 / 56 | 通过 |
| Config 中文 | 36 | 35 | 97.22% | 24 / 24 | 通过 |
| Log Diagnosis 英文 | 84 | 84 | 100.00% | 56 / 56 | 通过 |
| Log Diagnosis 中文 | 36 | 36 | 100.00% | 24 / 24 | 通过 |
| 合计 | 240 | 218 | 90.83% | 160 / 160 | 通过 |

22 条拒绝全部是 `solver_length_limit`。218 个进入 Compressor 的样本全部通过后续
答案一致性、Evidence Anchor、其他标签、重复、推理长度和训练序列检查。最终 160 条的
聚合质量如下：

| 指标 | 结果 |
| -- | --: |
| 推理 Token，最小 / P50 / P95 / 最大 | 15 / 17 / 19 / 19 |
| 推理 Token 均值 | 16.82 |
| 训练序列 Token，最小 / P50 / P95 / 最大 | 113 / 122 / 128 / 129 |
| 重复 8-gram 最大值 | 0 bp |
| Evidence Anchor 命中 | 160 / 160 |
| 其他标签提及最大值 | 0 |
| 相同行最大重复次数 | 1 |
| Dev、历史 Pilot、P0、P0-R1、P1 污染 | 全部为 0 |

真实公开机器证据见
[m5_r3_formal_source.json](raw/m5_r3_formal_source.json)。CPU 合成证据仍保留在
[m5_r3_formal_source_cpu_smoke.json](raw/m5_r3_formal_source_cpu_smoke.json)，
但不作为本次质量结论。

## 6. 选择分布与限制

语言和任务族严格满足 112 英文、48 中文以及 Config/Log 各 80 条。当前稳定排序没有将
标签作为配额维度，最终标签分布为：

| 任务族 | 标签 | 条数 |
| -- | -- | --: |
| Config | `forbidden_truncation` / `missing_checkpoint` / `unsupported_precision` / `world_size_mismatch` | 14 / 29 / 7 / 30 |
| Log Diagnosis | `collective_timeout` / `cuda_oom` / `disk_full` / `non_finite_gradient` | 30 / 10 / 30 / 10 |

该分布满足已冻结的门禁，但明显偏向更短的标签 Trace。它和 4 次复用上限必须在构建 R3
Mixture 前一并解决，并通过新版本配置记录，不能在看到训练结果后回改。

## 7. 血缘与完整性

| Artifact | SHA256 |
| -- | -- |
| Shard 0 | `5b2b018ef271e02efe7292757ad71e79917542a3c3471d78bcc5308c7ed336fb` |
| Shard 1 | `a2763e2d7ff6bdb5dc35dd16eccc4dfd9a20718741a057c64ab98e3962c625ad` |
| Private Raw | `a71c68f51761f8f041d0961bbe419d2f045efaac15706ad598ad3a2446ac05f9` |
| Private Selected | `f4dd371f2c89094b716c14602dab9157efa0a7fbd573776e08375caf49204c10` |
| Selected Content | `eed02200e9d4f00693d1ba0e67cce716bdeb02f95b71cbe9e00aa78324eba179` |

分片由生成提交产生。首次 Finalizer 暴露出严格 Tuple 不能从 JSON Array 恢复的
Schema 缺陷；修复提交只增加 JSON 往返兼容和双提交血缘，原分片未修改。Finalizer 要求
生成提交是自身提交的祖先，并同时公开两者。

## 8. 下一步

先冻结 R3 Mixture 修订契约，解决：

1. 150K Targeted Token 与单来源最多四次之间的不可满足约束；
2. 最短优先选择造成的标签分布偏斜；
3. 修订后重新执行离线选样和精确 Token 可行性测试。

不需要重新运行本轮 240 条 Qwen3-8B 生成。只有新 Mixture Manifest 通过精确 Token、
复用、分层和血缘验证后，才授权两个 Seed 的 R3 训练。

相关入口：

- [正式配置](../../configs/data/m5_r3_formal_source.yaml)
- [分片与 Finalizer Runner](../../scripts/run_m5_r3_formal_source.py)
- [CPU Smoke](../../scripts/run_m5_r3_formal_source_cpu_smoke.py)
- [内容审查结果](m5_r3_content_review.md)
