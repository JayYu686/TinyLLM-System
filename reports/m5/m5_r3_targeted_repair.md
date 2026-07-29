# M5.2-R3 Config/Log 定向修复准备报告

## 1. 当前状态

R3 的问题边界、Trace Policy、来源数量门禁和现有 Pilot 真实 CPU 审计已完成。当前状态为
`SOURCE_AUDIT_REJECTED_NEW_TEACHER_REQUIRED`：现有 Pilot 不能直接构建 R3 Mixture，下一步
必须先实现新的 Config/Log 任务集和 R3-P0 Teacher Pilot。

本阶段没有训练模型、没有加载 GPU、没有修改正式评测协议，也没有读取 M6 冻结结果。

## 2. 真实来源审计

审计器重新加载私有 Teacher Pilot，校验 Raw Artifact、Dataset、Reasoning Config、Tokenizer
和 R2 Decision 的固定 SHA256，再使用固定 Qwen3-0.6B Tokenizer 统计可见推理。公开结果只
包含计数、分位数、比例和哈希。

R3 Trace Policy 要求：

- 可见推理不超过 192 Token；
- Repeated 8-gram Ratio 不超过 5%；
- 无重复非空规范化行；
- 规范化 Trace 唯一；
- 原 Teacher 样本已通过格式、Final JSON、答案和污染验证。

真实结果：

| 任务族 | 来源数 | 英文/中文 | Token Min/P50/P90/Max | 192 Token 超限 | 8-gram 超限 | 可用数 | 可用英文/中文 |
| -- | --: | -- | -- | --: | --: | --: | -- |
| Config | 19 | 14 / 5 | 170 / 323 / 764 / 800 | 17 | 2 | 2 | 2 / 0 |
| Log Diagnosis | 20 | 14 / 6 | 163 / 242 / 272 / 392 | 16 | 0 | 4 | 3 / 1 |

39 条目标任务 Trace 中只有 6 条通过，距离 160 条门禁仍缺 154 条；Config 没有可用中文
样本。现有 Trace 虽无 Exact Normalized Duplicate，但数量、长度和语言覆盖均不足。

机器结果为：

```text
status: insufficient_requires_new_source
eligible_source_items: 6
required_source_items: 160
new_teacher_source_required: true
decision_reason: existing_pilot_lacks_concise_diverse_config_log_traces
exit code: 6
```

## 3. 为什么不继续复用旧数据

R1 的 150K Repair Token 只来自 40 条五类通用短样本，最终产生 608 个 Repair Sequence；
两个 Seed 的格式率反而从原 30%消融的 95.5%/97.0%下降到 94.5%/93.5%。R2 又证明更长
解码只能部分恢复。

当前审计进一步说明，旧 Pilot 中大部分 Config/Log Trace 连 Prompt 已声明的 192 Token
要求都没有满足。继续从这批数据中重复选择，会同时保留任务不聚焦、来源少和长度不受控
三个问题，无法形成清晰的因果实验。

## 4. 已冻结的下一步

R3 采用两阶段数据生成：

1. R3-P0：40 个 Config/Log 任务验证 Qwen3-8B 能否稳定生成不超过 192 Token 的原生
   Thinking Trace；
2. P0 通过后扩展到 240 个任务，确定性选择 160 条：每类 80 条、英文 56 条、中文 24 条。

R3 Mixture 仍为 700K Non-thinking、150K 原完整 Thinking、150K 新 Targeted Thinking，
总体 Thinking 比例和总训练预算不变。同一来源最多复用四次。

## 5. P0 实现进展

以下 P0 前置工作已经完成：

- 新 Config/Log 任务生成器和独立身份；
- 至少六种证据变体/标签；
- Dev 与历史 Pilot 的污染检查；
- R3-P0 严格 Schema、CLI、CPU Fixture 和失败路径；
- 公开结果与私有 Raw Artifact 分离。

CPU Fixture 为合成契约 Smoke，不加载模型、不使用 GPU，也不构成质量指标。真实
Qwen3-8B Teacher Pilot 的结果和中文汇总独立记录在
[R3-P0 实验报告](m5_r3_p0.md)。P0 未通过前不构建正式 Mixture，不启动两个 Seed 训练。

## 6. 证据索引

- 机器审计：[m5_r3_source_audit.json](raw/m5_r3_source_audit.json)
- P0 CPU Smoke：[m5_r3_p0_cpu_smoke.json](raw/m5_r3_p0_cpu_smoke.json)
- 冻结配置：[m5_r3_targeted_repair.yaml](../../configs/data/m5_r3_targeted_repair.yaml)
- 设计契约：[m5_r3_targeted_repair_design.md](../../docs/m5_r3_targeted_repair_design.md)
- R2 诊断：[m5_r2_diagnostic.md](m5_r2_diagnostic.md)
