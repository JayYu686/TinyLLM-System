# M6.1 Base 证据复用与执行准备报告

## 1. 当前结论

Qwen3-0.6B 的 M2 正式 Non-thinking 领域结果和通用评测能够进入 M6：源 Run、配置、模型文件、
300 条原始输出、40 条维护者判断、lm-eval 原始结果树、软件环境与硬件快照均通过重新校验。

M6.1 尚未完成。Base Thinking 必须等待本批实现合并到 clean `main` 后独立运行；其 40 条
Human Rubric 完成审查以前，不能生成完整 Base `evaluation.json`，也不能执行 Candidate Gate。

## 2. 已验证的历史真实结果

| 指标 | M2 Base Non-thinking 实际结果 | M6 用途 |
| -- | --: | -- |
| 领域正确 | 16 / 300（5.33%） | Non-thinking Base 配对基线 |
| JSON Valid | 32 / 80（40.00%） | Non-thinking JSON 基线 |
| 可见推理泄漏 | 0 / 300 | Non-thinking 泄漏基线 |
| 通用任务等权 `acc_norm` | 51.80% | 通用能力回归基线 |

通用任务原始值保持 M2 正式报告中的 ARC-Easy 47.26%、HellaSwag 42.07% 和 PIQA 66.05%。
本批没有重新运行这些任务，也没有改变或补填指标。

## 3. 新增契约

- `tinyllm eval m6-import-base`：验证源证据并原子生成 M6 Base Import；
- `tinyllm eval m6-domain`：按冻结配置运行单个 Thinking/Non-thinking 领域 Pass；
- `tinyllm eval m6-domain-review`：提交完整 40 条人工判断并生成 Content-free Mode Result；
- Base Import 保存源评测、领域输出、人工判断、通用原始结果树、环境和硬件 SHA256；
- Thinking Transcript 分离模型原始输出、最终答案和控制器注入，评分器只读取最终答案；
- 生成前使用按物理 GPU 索引查询的 Preflight，单张故障卡不会再阻断其他健康卡检查。

## 4. CPU 与真实 Artifact 验证

- 300 条领域项映射为 90 个双语 Pair Cluster 和 120 个英文 Singleton；
- 合成完整源 Run 可以导入，缺失通用原始文件树时 Fail Closed；
- Thinking 最终答案解析、客观评分、40 条人工审查和 300 条 Mode Result 闭环通过；
- 对私有 M2 正式 Run 执行真实导入 Smoke，复现本报告第 2 节的全部聚合值；
- GPU 生成尚未执行，因此没有 Base Thinking 分数、格式率、干预率或显存指标。

## 5. 下一步验收

1. 合并 M6.1 实现并确认远端 CI；
2. 在空闲 RTX 3090 上执行 Base Thinking 300 条生成；
3. 生成中文人工审查包，维护者确认 40 条 Human Rubric；
4. 固化 Base Thinking Mode Result；
5. 组装完整 Base Release Evaluation，随后进入 10M Candidate 双模式评测。
