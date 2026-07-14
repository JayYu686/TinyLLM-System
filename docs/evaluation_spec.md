# 评测规范

## 1. 目标

评测用于回答：

- 模型是否提升目标任务。
- 模型是否回退通用能力。
- 模型输出是否满足格式。
- 模型部署性能是否可接受。
- 模型是否可以晋级。

## 2. 评测层次

### 通用能力

- ARC Easy。
- HellaSwag。
- PIQA。

### 自建任务

固定 300 条，英文/中文目标 70%/30%：

- Python。
- Linux。
- JSON。
- 配置修改。
- 日志分析。
- 简短代码。
- 无依据拒答。

### 系统性能

- TTFT。
- P50/P95。
- Tokens/s。
- Peak Memory。
- Failure Rate。
- JSON Valid Rate。

## 3. 评测记录

每次评测必须记录：

- 模型版本。
- Checkpoint。
- Tokenizer。
- Prompt Template。
- 评测集版本。
- 解码参数。
- 硬件。
- 软件版本。
- Seed。
- 原始输出。
- 汇总结果。

## 4. 模型比较

报告必须同时包含：

- 提升项。
- 回退项。
- 不显著变化项。
- 失败样例。
- 性能变化。
- 不确定性说明。

## 5. Promotion Gate 示例

```yaml
required:
  json_valid_rate:
    min: 0.98
  domain_accuracy:
    min_delta: 0.03

regression_limits:
  general_eval:
    max_drop: 0.02
  p95_latency:
    max_increase: 0.15
```

## 6. 晋级条件

候选模型必须满足：

- 评测完整。
- 数据血缘完整。
- 无严重通用能力回退。
- 格式有效率达标。
- 性能回归在阈值内。
- Checkpoint 可加载。
- 推理 Smoke Test 通过。

M6 Candidate 的量化门禁为：领域总分相对 Baseline 至少提升 3pp 且 Bootstrap
95% CI 下界大于 0；ARC-Easy/HellaSwag/PIQA 聚合回退不超过 2pp；JSON Valid Rate
至少 98%；血缘完整。M7 前不以未实测的推理性能项阻止或通过 Candidate。

门禁分阶段启用：

- M6 先启用评测完整性、数据/模型血缘、能力回退和格式有效率门禁。
- M7 在真实推理服务与 Benchmark 可用后，启用延迟、吞吐、显存和失败率门禁。
- 尚未具备实测基础设施的性能项必须标记为 `not_evaluated`，不得以估算值通过门禁。

## 7. 禁止事项

- 只展示最好的一次结果。
- 删除失败样例。
- 使用测试集调参。
- 省略解码参数。
- 混用不同硬件结果。
- 将估算值当实测值。
