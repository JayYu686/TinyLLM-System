# M5.3 Qwen3-8B BF16 LoRA 正式验收报告

## 1. 结论

Qwen3-8B 单卡 BF16 LoRA 路线已通过 M5 系统验收。固定 Revision 的 GQA 模型在 RTX 3090
上完成 10M Supervised Token 双模式训练，并在 5,000,444 Token 处执行受控中断，再由全新
进程 Exact Resume 到 10M。训练产生五个评测 Pin 点、最终 Adapter Safetensors 和 Model
Card；训练 Checkpoint 与部署 Adapter 保持独立。

最终使用冻结的 Thinking Budget v2 协议真实评测 200 条 Thinking 和 200 条 Non-thinking
样本。Thinking 正确 198/200，格式与 JSON 均为 200/200；199 条由模型自然闭合，只有 1 条
触发显式预算收束。Non-thinking 正确 144/200，格式为 200/200，JSON 为 192/200；两种模式
均未观察到可见推理泄漏。

该结果证明 LoRA 训练、恢复、导出和双模式评测链路可用。M5 Dev 没有 Qwen3-8B 同协议 Base
结果，因此本报告不宣称相对 Base 提升；M6 将使用独立冻结套件决定 Candidate 资格。

## 2. 固定身份

| 项目 | 实际结果 |
| -- | -- |
| 模型 | `Qwen/Qwen3-8B` |
| Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Attention | 原生 GQA |
| 正式 Run | `20260731T125617Z-m5-formal-qwen3-8b-lora-cc363170-e922` |
| Campaign | `20260731T125549Z-m5-lora-campaign-gpu4` |
| 训练 Git Commit | `d0752e84a628032f95e409445716e24b466c173a`，clean |
| 训练配置 SHA256 | `cc363170bda3d7637124664a9b742c16dff6eea6a2030bdae98b76ea35efc85f` |
| 数据版本 | `m5-dual-sft-v1-b5b9e839` |
| 数据 Manifest SHA256 | `607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6` |
| Thinking 比例 | 30% Supervised Token |
| 软件 | Python 3.11.14、PyTorch 2.7.1+cu118、Transformers 4.57.6、PEFT 0.19.1 |

完整私有 Artifact Store 保存绝对路径、主机身份、逐步 Metrics、模型输出和训练状态；公开
[机器可读摘要](raw/m5_lora_formal.json)只保留脱敏身份、聚合值和 Hash。

## 3. 训练配置与资源

| 项目 | 配置或实测 |
| -- | -- |
| 精度 | BF16，允许 TF32，不使用 GradScaler |
| LoRA | Rank 16、Alpha 32、Dropout 0.05、无 Bias |
| Target | Attention 与 MLP Linear |
| Sequence Length | 1024 |
| Micro Batch / Accumulation | 1 / 8 |
| Learning Rate | `2e-4`，Warmup 200K Token |
| 可训练参数 | 43,646,976 / 8,234,382,336（约 0.53%） |
| 物理 GPU | RTX 3090 GPU 4 |
| Peak Allocated | 19,774,459,392 B（约 18.42 GiB） |
| Peak Reserved | 21,002,977,280 B（约 19.56 GiB） |

BF16 路线已经真实容纳并完成训练，因此没有触发 NF4 QLoRA 回退。该结论只适用于本次固定
配置；增大 Sequence Length、Batch 或 LoRA Scope 后仍需重新 Probe。

## 4. 中断、恢复与温控

| 作业段 | 状态 | Token | Optimizer Step | 作业段时长 |
| -- | -- | --: | --: | --: |
| Fresh | 按计划中断 | 5,000,444 | 2,594 | 26,175.20 秒 |
| Exact Resume | 成功 | 10,000,000 | 5,188 | 25,954.74 秒 |

恢复点完整保存 Adapter、Optimizer、Scheduler、RNG、Sampler Cursor、数据版本、基座 Revision、
配置、环境与硬件 Hash。Resume 段验证相同身份后从下一序列继续，最终完成 10 个数据 Epoch。
训练 Loss 从首 Step 的 1.592890 降至末 Step 的 0.005099；这条曲线说明训练状态正常推进，
低末值也提示重复 10 个 Epoch 可能产生过拟合，不能单独作为模型质量证据。

共享服务器温控守护在 84°C 暂停进程、降至 62–74°C 后继续，共记录 16 次暂停，两个作业段
都在各自 12 小时上限内完成。没有终止其他用户进程，也没有把暂停时间从真实作业成本中
剔除。

## 5. Checkpoint 与 Adapter

训练在约 2M、4M、6M、8M 和 10M Token 保存五个永久评测点；5,000,444 Token 中断点也被
单独 Pin。每个训练 Checkpoint 约 500 MiB，包含完整可恢复状态，并通过 Manifest、Payload
SHA256 和 Commit Marker 校验。

最终 Adapter 目录树 SHA256 为
`a8326e16adbb5ebe9886ec587960c6d5f4a7888f0a884b4aeb8852615e8849d3`。其中
`adapter_model.safetensors` 为 174,655,536 Bytes，SHA256 为
`809cec8734fab9507ec3ffabbbc9650b879381d3b2f77f4f8ce782ba21d5c488`。Model Card 明确固定基座
Revision、数据版本、Run 和许可证边界，且不重新分发 Qwen3-8B 基座权重。

## 6. 冻结双模式评测

评测 ID 为 `20260803T045819Z-m5-thinking-budget-lora_candidate-088ca318`，使用
`m5-thinking-budget-v2` 和独立的 200 条 M5 Dev。原始 `results.jsonl` SHA256 为
`cf69aeaef263d0404b387be230ebd155dfd5cb056df922103772609f1cd8c2c3`。

| 指标 | Thinking | Non-thinking |
| -- | --: | --: |
| 样本数 | 200 | 200 |
| 格式有效 | 200（100.0%） | 200（100.0%） |
| 最终 JSON 有效 | 200（100.0%） | 192（96.0%） |
| 最终答案正确 | 198（99.0%） | 144（72.0%） |
| 自然闭合 | 199（99.5%） | 不适用 |
| 预算强制收束 | 1（0.5%） | 不适用 |
| 可见推理泄漏 | 0 | 0 |

评测启动时 GPU 4 已用显存 1,744 MiB、利用率 0%、温度 30°C，符合受控共享评测阈值；本进程
Peak Reserved 为 18,840,813,568 B，完整运行成功。Summary 显式记录
`shared_gpu_evaluation=true`，不把该结果描述为独占卡性能 Benchmark。

## 7. 验收边界

本路线已经覆盖真实 BF16 Probe、10M Token 训练、Exact Resume、温控恢复、五个评测
Checkpoint、Adapter、Model Card、严格血缘和最终双模式评测。M5 仍需等待 Qwen3-0.6B 四卡
50M Full SFT 与五阶段评测完成，之后才能形成总验收报告。

本报告不发布训练吞吐 Benchmark，不比较 8B Base，不消费 M6 冻结发布集，也不授予
Candidate 或 Production 状态。
