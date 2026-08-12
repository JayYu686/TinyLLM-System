# TinyLLM-System 系统能力与证据映射

本文说明各项系统能力、对应实现和公开证据，便于版本验收、演示和外部技术审查。

## 1. 核心能力域

### 分布式训练

- DDP。
- FSDP2。
- ZeRO-3 对照。
- Sharded Checkpoint。
- 多卡故障恢复。
- 扩展效率与拓扑分析。

### LLM 数据工程

- 数据 Schema 与许可策略。
- 规范化、过滤和去重。
- 数据版本与 Manifest。
- Tokenization 与 Packing。
- 训练/评测污染检查。
- 端到端数据血缘。

### LLM 后训练

- Full SFT。
- LoRA 与显式 QLoRA 回退。
- Thinking/Non-thinking 双模式。
- Base/Candidate 对比。
- 可选偏好优化研究。

### LLMOps

- Experiment Tracking。
- Model Registry。
- Promotion Gate。
- Deployment 与 Rollback。
- 训练和推理 Benchmark。

### 系统工程

- 配置驱动与稳定 CLI。
- 版本化 Schema。
- 运行状态和错误恢复。
- 结构化可观测性。
- 自动化测试与真实验收报告。

## 2. 与应用层 Agent 项目的职责边界

| 应用层 Agent | TinyLLM-System |
| -- | -- |
| 企业业务工作流 | 模型训练与生命周期 |
| Agent 编排 | 分布式训练策略 |
| RAG/MCP | Dataset/Checkpoint |
| 运营控制台 | 评测和模型晋级 |
| 业务安全 | 数值正确性和实验复现 |

## 3. 端到端演示

1. 查看 Dataset Manifest 和固定 Revision。
2. 执行 `tinyllm doctor` 与资源 Preflight。
3. 解析训练 YAML 并创建 Run。
4. 模拟中断、校验 Checkpoint 并恢复。
5. 展示真实 1/2/4 卡 DDP 报告及拓扑边界。
6. 展示 FSDP2 分片状态和 DCP Resume。
7. 比较 Base 与 Fine-tuned Candidate。
8. 执行 Promotion Gate 并展示 M6 的 11/11 真实门禁结果。
9. 使用 `tinyllm run rebuild|list|show` 反向查询数据、配置、Checkpoint 和评测血缘。
10. 在 M7 部署 Candidate 并执行推理压测与 Production Gate。

完整演示流程见 [10 分钟中文演示](demo_m6.md)，M6 质量与晋级结果见
[M6 验收报告](../reports/m6/m6_acceptance.md)。

## 4. 对外发布原则

公开说明只引用已经合并且可复现的结果：

- 实际模型规模。
- 实际 GPU 数量与索引。
- 实际吞吐与扩展效率。
- 实际显存占用。
- 实际恢复时间。
- 实际评测变化。
- 实际推理 P50/P95。

待测能力使用计划中、进行中或 `not_evaluated` 状态。所有数字链接到真实报告和原始证据身份。

## 5. 项目标题

**TinyLLM-System——面向消费级多 GPU 工作站的硬件感知大语言模型训练、评测与部署平台**

## 6. 能力展示结构

- 平台闭环和统一 CLI。
- 数据版本与实验血缘。
- DDP、FSDP2 与可选 ZeRO-3 对照。
- 开源模型 SFT 和自动评测。
- 模型晋级与推理压测。

每项能力均由真实操作、机器可读 Artifact 和审查报告共同支撑。
