# 系统架构设计

## 1. 设计原则

- 配置驱动。
- 模块解耦。
- 可替换后端。
- 正确性优先。
- 数据和模型血缘完整。
- 训练、评测、部署分离。
- 硬件差异显式建模。

## 2. 核心模块

### CLI

负责：

- 参数解析。
- 配置覆盖。
- 命令分发。
- 用户友好错误信息。

### Experiment Orchestrator

负责：

- Run 创建。
- 配置快照。
- 状态机。
- 资源规划。
- 训练启动。
- 自动评测触发。
- 失败恢复。
- 晋级动作。

### Dataset Registry

负责：

- 数据版本。
- Manifest。
- Hash。
- Schema。
- 统计和过滤原因。
- Tokenizer 与 Packing 信息。

### Training Runtime

统一接口支持：

- Single GPU。
- DDP。
- FSDP2。
- ZeRO-3。
- BF16/FP16。
- Checkpoint。
- Resume。
- Callback。

### Artifact Store

保存：

- 配置。
- 日志。
- Checkpoint。
- 数据 Manifest。
- 评测报告。
- Benchmark。
- 模型导出。

第一阶段使用本地文件系统，后续可接 MinIO。

### Evaluation Service

负责：

- 通用任务。
- 自建评测。
- 结果缓存。
- 模型比较。
- 回归检查。

### Model Registry

负责：

- 模型版本。
- 阶段。
- 别名。
- Run 血缘。
- 晋级记录。

### Inference Gateway

负责：

- Transformers/vLLM 后端。
- OpenAI-compatible API。
- 流式输出。
- 统一指标。
- 模型切换。

## 3. 状态机

### Run 状态

```text
CREATED
  → PLANNED
  → RUNNING
  → CHECKPOINTING
  → EVALUATING
  → SUCCEEDED
```

失败路径：

```text
RUNNING
  → FAILED
  → RESUMABLE / NON_RESUMABLE
  → RUNNING / TERMINATED
```

### 模型状态

```text
development
  → candidate
  → production
  → archived
```

## 4. 数据流

```text
Raw Dataset
  → Data Pipeline
  → Dataset Version
  → Training Run
  → Checkpoint
  → Evaluation
  → Promotion Gate
  → Model Version
  → Deployment
```

## 5. MVP 候选技术与引入阶段

- Python 3.11。
- PyTorch。
- Transformers。
- TRL。
- Typer 或 Click；M0 的最小 CLI 可以使用标准库实现，待命令面复杂后再引入框架。
- Pydantic。
- MLflow；实验追踪接入在血缘和评测阶段推进，不是 M0 依赖。
- FastAPI；仅在 M7 推理服务阶段引入。
- PostgreSQL 可后置。
- 本地文件系统优先。
- Docker Compose 用于后续服务组件，不作为 M0 和训练核心的前置条件。

## 6. 可替换接口

必须抽象：

- DatasetBackend。
- TrainingStrategy。
- CheckpointBackend。
- EvaluationTask。
- ModelRegistryBackend。
- InferenceBackend。
- MetricsBackend。

不要求第一版实现多个后端，但接口不能与单一框架强绑定。
