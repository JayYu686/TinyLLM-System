# M10.2 Qwen3-0.6B Full SFT 工程就绪报告

## 结论

M10.2 的单卡训练接口、阶段 Checkpoint、Exact Resume、父模型解析和数据 Preflight 已完成。
本报告保留启动前工程状态；后续 1M→5M 真实 GPU 阶段已经完成，最新训练、显存和恢复证据见
[`M10.2 5M 阶段报告`](m10_full_sft_5m.md)。5M Agent Dev 未通过阶段门禁，10M 已正式停止。

## 已冻结身份

| 项目 | 实际身份 |
| -- | -- |
| 父模型 | `qwen3-0-6b-m7-fa678d92` |
| Production Record SHA256 | `a83eaff5e9b91be403bddb7a4ac23926465e0a78872921429dad72b6ee7d0ca5` |
| 父模型 Artifact SHA256 | `63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6` |
| 数据版本 | `m10-agent-sft-v1-4655d3e3` |
| Dataset Manifest SHA256 | `6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490` |
| 模型结构 | Qwen3-0.6B、GQA、Full SFT |
| 训练精度 | RTX 3090 BF16，可启用 TF32，无 GradScaler |

真实只读 Preflight 已重新打开私有训练数组、复算 Manifest，并通过 M7 Production Alias
解析父模型；上述五项身份与冻结配置完全一致。

## 阶段与恢复设计

三个训练阶段属于同一个不可变配置和同一个 Run：

```text
Fresh → 1M Stage Export
      → Exact Resume → 5M Stage Export
                     → Agent Dev / M6 阶段门禁
                     → Exact Resume → 10M Final Export
```

- 每个逻辑 Epoch 恰好消费 1,000,000 个实际参与 Loss 的监督 Token。
- 数据顺序由 `seed + completed_epoch` 确定，Checkpoint 只在逻辑 Epoch 边界提交。
- Checkpoint 保存模型、AdamW、进度、Python/NumPy/PyTorch/CUDA RNG、配置、Git、数据和父模型
  身份。
- 写入采用临时目录、SHA256 校验、原子 Rename 和原子 `LATEST` 更新。
- 1M、5M 与 10M Checkpoint 永久 Pin；普通 Epoch Checkpoint 滚动保留最近两个。
- 每个阶段生成独立 Safetensors Export，评测不直接读取完整训练 Checkpoint。
- 训练顺序被强制为 `Fresh 1M → Resume 5M → Resume 10M`，无法跳过中间阶段。
- 5M→10M 默认拒绝；只有哈希绑定当前 Run、5M Export、Agent Dev 与 M6 证据的
  `accepted` Continuation Gate 才能解除阻断。

## 已验证范围

- 严格 YAML / Pydantic v2 配置与 JSON Schema；未知字段拒绝。
- 数据、父模型、Production Record 与 Artifact 哈希漂移拒绝。
- Token 阶段、优化器常量、单卡 BF16 和 GQA 身份冻结。
- CPU 线性模型 Checkpoint 保存、加载、参数恢复、Retention 和损坏检测。
- 固定数据数组到 Torch Dataset 的形状、Label 和 Attention Mask 读取。
- Result Schema 对阶段状态、Resume 来源、Checkpoint 与 Export 对齐进行校验。
- 全量 CPU/Mock 回归为 `1162 passed, 2 deselected`，分支覆盖率为 `85.18%`；受保护的
  CUDA Worker 延续项目既有 GPU Workflow 策略，在真实 RTX 3090 Smoke 中单独验收。

## 后续实测状态

启动前列出的 GPU 验收项现已得到以下真实结果：

- 1M 和 5M 阶段训练、峰值显存、Checkpoint 与 Stage Export 已完成；
- 1M→5M 使用全新进程完成 GPU Exact Resume；
- 同协议 Agent Dev 相对父模型下降 11.25pp；
- M6 通用聚合回退 1.78pp，单项满足阈值；
- Continuation Gate 决策为 `rejected`，10M 保持阻断；
- 未通过开发门禁的模型不执行 Release、BFCL、Serving 和最终 Agent Model Gate。

本路线已经收尾，完整结论纳入 [`M10 总验收`](m10_acceptance.md)。
