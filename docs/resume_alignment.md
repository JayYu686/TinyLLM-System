# 简历能力对齐

## 1. 项目要证明的能力

### 分布式训练

- DDP。
- FSDP2。
- ZeRO-3。
- Sharded Checkpoint。
- 多卡恢复。
- 扩展效率分析。

### LLM 数据工程

- 数据 Schema。
- 清洗。
- 去重。
- 版本。
- Tokenization。
- Packing。
- 数据血缘。

### LLM 后训练

- Full SFT。
- LoRA。
- 可选 DPO。
- Base/Candidate 对比。

### LLMOps

- Experiment Tracking。
- Model Registry。
- Promotion Gate。
- Deployment。
- Rollback。
- Benchmark。

### 系统工程

- 配置驱动。
- CLI。
- 状态机。
- 错误恢复。
- 可观测性。
- 自动化测试。

## 2. 与 CommerceFlow Agent 的区别

| CommerceFlow Agent | TinyLLM-System |
|---|---|
| 企业业务 Agent | 模型训练与生命周期 |
| LangGraph 工作流 | 分布式训练策略 |
| RAG/MCP | Dataset/Checkpoint |
| 运营控制台 | 评测和模型晋级 |
| 业务安全 | 数值正确性和复现 |

## 3. 面试演示

1. 查看 Dataset Manifest。
2. 执行 `tinyllm plan`。
3. 启动训练。
4. 模拟中断和恢复。
5. 展示 1/2/4/8 卡报告。
6. 比较 Base 与 Fine-tuned。
7. 展示 Promotion Gate。
8. 部署 production 模型。
9. 压测并查看指标。
10. 根据 Run ID 展示完整血缘。

## 4. 简历表述原则

只有项目完成后才能写：

- 实际模型规模。
- 实际 GPU 数量。
- 实际吞吐提升。
- 实际显存下降。
- 实际恢复时间。
- 实际评测提升。
- 实际 P95。

项目未完成前，README 中全部写 TBD。

## 5. 预期项目标题

**TinyLLM-System——面向消费级多 GPU 集群的硬件感知大语言模型训练、评测与部署平台**

## 6. 完成后的简历结构

- 第一条：平台闭环和统一 CLI。
- 第二条：数据版本与实验血缘。
- 第三条：DDP/FSDP2/ZeRO-3。
- 第四条：开源模型 SFT 和自动评测。
- 第五条：模型晋级与推理压测。

每条都必须有真实动作和真实指标。
