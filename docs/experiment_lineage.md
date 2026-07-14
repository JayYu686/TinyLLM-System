# 实验血缘规范

## 1. Run ID

每次训练创建唯一 Run ID。

格式固定为：

```text
<UTC时间>-<run-slug>-<resolved-config-hash前8位>-<随机4位>
```

Run ID 关联：

```text
Run
├── Config Snapshot
├── Git Commit
├── Environment
├── Hardware
├── Dataset Version
├── Tokenizer
├── Checkpoints
├── Logs
├── Evaluation Reports
└── Exported Models
```

每个 Run 的 JSON/JSONL 目录是事实源。M6 的 SQLite 索引必须可以完全从目录重建；
MLflow 若接入只能作为可选投影。

## 2. 必须记录的环境

- Python。
- PyTorch。
- CUDA。
- NCCL。
- Transformers。
- TRL。
- DeepSpeed。
- vLLM。
- Driver。
- OS。
- GPU 型号和数量。
- GPU 拓扑摘要。

## 3. 代码状态

记录：

- Git Commit。
- Branch。
- 是否 Dirty。
- Dirty Diff Hash。
- 启动命令。

Dirty 工作区允许 Debug，但不得晋级 production。

## 4. 数据血缘

记录：

- Dataset Name。
- Version。
- Manifest Hash。
- Tokenizer Revision。
- Packing 配置。
- Split Hash。

## 5. 模型血缘

记录：

- Base Model。
- Revision。
- Training Mode。
- Adapter。
- Parent Run。
- Export Format。

## 6. `tinyllm reproduce`

流程：

1. 读取 Run。
2. 校验代码版本。
3. 校验数据。
4. 校验模型。
5. 校验依赖。
6. 检查硬件兼容。
7. 重建配置。
8. 启动复现实验。
9. 输出差异报告。

## 7. 差异类型

- 完全一致。
- 软件版本差异。
- 硬件差异。
- 数据差异。
- 配置差异。
- 不可复现。

系统不能因为命令成功运行就宣称“复现成功”。
