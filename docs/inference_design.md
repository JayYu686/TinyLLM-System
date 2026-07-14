# 推理服务设计

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
- `/health`
- `/models`
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

优先多副本 Data Parallel，提高吞吐。

### 单卡放不下

MVP 使用成熟推理框架提供的 Tensor Parallel。Pipeline Parallel 属于 Future Work，不进入当前实现范围。

### 推荐资源布局

```text
GPU 0–1：副本 A，TP=2
GPU 2–3：副本 B，TP=2
GPU 4–5：副本 C，TP=2
GPU 6–7：副本 D，TP=2
GPU 8：评测
GPU 9：备用
```

仅作为后期测试模板，不代表所有模型都适合 TP=2。

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
