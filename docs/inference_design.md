# 推理服务设计

> M7 的冻结接口、安全边界、Benchmark 和 Production Gate 以
> [M7 在线推理与 Production 晋级契约](m7_serving_contract.md)为准。本文保留后端抽象与资源
> 策略说明。

## 1. 目标

将 candidate 或 production 模型统一部署，并提供可比较的性能结果。

## 2. 后端接口

```text
InferenceBackend
├── load
├── generate
├── stream
├── health
├── metrics
└── unload
```

实现：

- MockBackend。
- TransformersBackend。
- VLLMBackend。

## 3. API

第一阶段提供：

- `/v1/chat/completions`
- `/v1/models`
- `/health/live`
- `/health/ready`
- `/metrics`
- `/version`

## 4. 请求记录

记录：

- Request ID。
- Model Version。
- Backend。
- 输入长度。
- 输出长度。
- 排队时间。
- TTFT。
- 总耗时。
- 错误类型。

默认不永久保存完整用户内容。

## 5. 并行策略

### 单卡模型

优先验证单实例性能；多副本 Data Parallel 作为后续容量扩展策略。

### 单卡放不下

使用成熟推理框架提供的 Tensor Parallel。Pipeline Parallel 属于 Future Work，不进入当前实现范围。

### 推荐资源布局

M7 的 0.6B Candidate 使用一张经 Preflight 确认空闲的 RTX 3090；实际 GPU 索引写入硬件
证据。多副本和 Tensor Parallel 只在模型容量或真实负载提出需求后进入增强实验。

## 6. 部署约束

- 只有 candidate/production 可正式部署。
- development 只能用于本地测试。
- 服务启动时校验模型血缘。
- 模型切换必须可回滚。
- 后端版本必须固定。
- 量化能力必须与硬件兼容。

## 7. 失败处理

- OOM。
- Backend Crash。
- Tokenizer Error。
- 超时。
- 非法请求。
- 流式连接中断。

必须返回标准错误码，并记录可诊断信息。
