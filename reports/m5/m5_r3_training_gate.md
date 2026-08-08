# M5.2-R3 训练与门禁报告

## 1. 结论

R3 模型质量门禁已拒绝。Seed42 完成 1,000,000 Supervised Tokens 后，Thinking 格式率为
92.5%，低于冻结的 99% 门槛；由于门禁要求两个 Seed 全部通过，继续 Seed20260727 已无法
改变 R3 结论，因此在 672,024 Token 时停止，保留通过完整性校验的 500,721 Token
Checkpoint。

R3 的正式来源和 Mixture 构建仍然有效；它们证明数据血缘和 Token 配比正确，不代表模型
质量通过。

## 2. Seed42 真实结果

| 项目 | 结果 |
| -- | --: |
| 训练 Token | 1,000,000 |
| Global Step | 1,037 |
| 初始 / 最终 Loss | 1.7468 / 0.7124 |
| 峰值 Allocated / Reserved | 7.03 GB / 8.19 GB |
| Thinking 格式率 | 92.5% |
| Thinking 最终答案分数 | 89.5% |
| Thinking 长度触顶 | 13 / 200 |
| Non-thinking 分数 | 74.0% |

15 条 Thinking 格式失败中，Config 9 条、Log Diagnosis 1 条、Python 3 条、JSON 2 条；
13 条触及 896 Token 上限，2 条在 EOS 前没有闭合。与 R1 Seed42 对照，R3 修复 4 个旧
失败项，但新增 8 个失败项，未形成稳定改善。

## 3. 训练完整性

Seed42 的 501,881 Token 与 1,000,000 Token Checkpoint 均包含完整模型、Optimizer、RNG、
数据游标、配置、Git 和 Mixture 身份。状态文件大小、SHA256、Manifest SHA256 与
`COMMITTED` 标记均通过重新校验；Safetensors 导出目录组合哈希与 Run Result 一致。

两组训练都使用物理 GPU 7。该卡在持续训练时散热较差，因此运行器外部使用只作用于本作业
进程组的 84°C 暂停 / 74°C 恢复策略，并保留 88°C 硬停止线。未修改共享 GPU 功耗上限，
也未终止其他用户进程。

## 4. 为什么停止第二 Seed

R3 Gate 是逻辑 AND：

```text
Seed42 Thinking 格式率 >= 99%
且
Seed20260727 Thinking 格式率 >= 99%
```

Seed42 已得到 92.5%，因此第二 Seed 无论取得什么结果都不能让 R3 通过。继续剩余训练和
评测只会增加 GPU 时间，不会增加决策信息。停止点已有有效 Checkpoint，失败证据可复查。

## 5. 后续纠偏

连续使用小规模短 Trace 调整配比没有解决 Qwen3 的长 Thinking 闭合机制。下一协议采用
[ADR-0006](../../docs/adr/0006-qwen3-thinking-budget-controller.md)：

- 保持 99% 格式门槛；
- 使用 Qwen 官方两阶段 Thinking Budget；
- 分开报告自然闭合和控制器强制收束；
- 增加强制收束率、最终答案质量和运行成本约束；
- 使用已经完成的 R1 双 Seed 作为首个协议 v2 Candidate。

机器可读结果：
[m5_r3_training_gate.json](raw/m5_r3_training_gate.json)。
