# M10.5 Repair v4 准备度与阶段诊断报告

## 1. 结论

Repair v3 的 1M Checkpoint 是当前最佳候选。继续在同一份 1M Token 混合上重复训练至 3M、
4M 和 5M 后，Agent Dev 的任务成功率与证据 Grounding 均下降，因此现有 Run 不继续训练至
10M。下一轮改用 9,600 条唯一、与 Agent Runtime System Policy 对齐的 DevOps 轨迹，减少重复
样本导致的过拟合，并重新从冻结的 Qwen3-8B Base 开始 1M Token 对照实验。

## 2. Repair v3 阶段对照

所有阶段使用同一 `m10-agent-scoring-v3` 协议和同一 80 条 Agent Dev。

| 阶段 | Task Success | Grounding | Tool Hallucination | 判断 |
| -- | --: | --: | --: | -- |
| 1M | 63.75% | 89.13% | 10.00% | 当前最佳，保留为历史候选 |
| 3M | 52.50% | 47.83% | 1.25% | 明显退化 |
| 4M | 50.00% | 58.70% | 1.25% | 明显退化 |
| 5M | 56.25% | 76.09% | 1.25% | 未恢复至 1M 水平 |

1M 的主要剩余缺口是顺序多步任务、中文参数绑定和基于工具结果的最终回答。3M–5M 虽然降低了
工具幻觉率，但没有换来总体任务成功率提升，说明简单增加 Epoch 不能解决这些缺口。

原始阶段摘要和内容哈希见
[`raw/m10_repair_v4_stage_diagnostic.json`](raw/m10_repair_v4_stage_diagnostic.json)。

## 3. Repair v4 数据准备

Repair v4 已完成机器质量门禁，尚未通过维护者内容审查。

| 检查项 | 实际结果 |
| -- | --: |
| 数据版本 | `m10-devops-training-v4-f13ae053` |
| 完整轨迹 | 9,600 条 |
| 英文 / 中文 | 6,720 / 2,880（70% / 30%） |
| 唯一最终回答 | 8,688 |
| 单一最终回答最大频次 | 32 |
| Tool-grounded 轨迹 | 5,760 |
| Sequential 两步轨迹 | 1,920 |
| Parallel 双调用轨迹 | 480 |
| Failure Recovery 轨迹 | 1,440 |
| Missing Argument 轨迹 | 960 |
| 精确重复 / 跨组近重复 | 0 / 0 |
| Dev、Release、BFCL、M6 污染 | 0 |
| 当前状态 | `review_pending` |

机器门禁证据见
[`raw/m10_repair_v4_training_build.json`](raw/m10_repair_v4_training_build.json) 和
[`raw/m10_repair_v4_content_quality.json`](raw/m10_repair_v4_content_quality.json)。完整 9,600 条轨迹
保留在私有 Artifact Store；维护者只需审查按类别和语言分层抽取的 80 条内容。

## 4. 后续执行门禁

维护者确认 80 条抽样轨迹后，依次执行：

1. 生成不可变的 `approval-v4` 内容审查记录；
2. 构建无样本复用的 1M Supervised Token Repair v4 混合，并验证所有来源的复用计数；
3. 执行 10 Step 显存探测；
4. 从冻结 Qwen3-8B Base 训练新的 1M LoRA；
5. 运行 Agent Dev，并与 Repair v3 1M 及父模型进行同一 v3 协议比较；
6. 只有 Dev 指标证明值得晋级时，才消耗密封 Release、BFCL、M6 和 Serving 门禁。

在内容确认与冻结混合生成前，Repair v4 保持 `training_permitted=false`，不启动 GPU 训练。
