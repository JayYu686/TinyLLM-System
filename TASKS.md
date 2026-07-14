# TinyLLM-System 任务清单

> 规则：只有完成测试、文档和验收后，任务才能勾选。

---

## M0：文档与项目初始化

- [x] 创建根目录与 Python 包结构。
- [x] 完成 README.md。
- [x] 完成 AGENTS.md。
- [x] 完成 PLANS.md。
- [x] 完成产品范围文档。
- [x] 完成架构文档。
- [x] 完成数据契约。
- [x] 完成训练设计。
- [x] 完成评测规范。
- [x] 完成实验血缘规范。
- [x] 完成推理设计。
- [x] 完成 Benchmark 计划。
- [x] 完成简历对齐文档。
- [x] 创建 ADR 目录。
- [x] 创建 Future Work 目录。
- [x] 初始化 Git 仓库和 `main` 分支。
- [x] 配置 Git author 并创建首个审查提交。
- [x] 初始化 pyproject.toml。
- [x] 创建隔离 `.venv` 和固定的 3090 PyTorch/CUDA Profile。
- [x] 配置 Ruff、MyPy、Pytest。
- [x] 建立基础 CI 配置。
- [x] 在远程仓库执行一次基础 CI。
- [x] 设计 `tinyllm doctor` 输出 Schema。
- [x] 实现 `tinyllm doctor` 只读采集和稳定 JSON。
- [x] 验证 doctor Smoke Test 和失败路径。
- [x] 收集 3090 服务器硬件信息。
- [ ] 收集 V100 服务器硬件信息（取得访问方式后执行，不阻塞 3090 M0）。
- [x] 运行 `nvidia-smi topo -m` 并记录 NUMA/P2P/NVLink。
- [x] 验证 RTX 3090 CUDA/BF16 单卡 Smoke。
- [x] 运行 1/2/4 卡 NCCL All-Reduce、All-Gather、Reduce-Scatter。
- [x] 根据实时空闲状态选择 GPU 4–9，运行 6 卡三种 NCCL Collective。
- [ ] 可选运行 10 卡 NCCL 边界对照（不阻塞 M0）。
- [x] 生成 3090 Inventory 和 NCCL 原始报告。
- [x] 明确动态空闲卡用于开发 Smoke、固定 1/2/4/8 卡用于 M3 正式扩展实验。

---

## M1：单卡 Debug 训练闭环

- [x] 定义 TinyGPTConfig。
- [x] 实现 RMSNorm。
- [x] 实现 RoPE。
- [x] 实现 Causal Self-Attention。
- [x] 实现 SwiGLU。
- [x] 实现 Transformer Block。
- [x] 实现 TinyGPT。
- [x] 实现 Weight Tying。
- [x] 实现 Causal LM Loss。
- [x] 实现 Toy Dataset。
- [x] 实现固定 Seed。
- [x] 实现训练配置 Schema。
- [ ] 实现单卡训练器。
- [ ] 实现 BF16 Profile。
- [ ] 实现 FP16 + GradScaler Profile。
- [ ] 实现 Gradient Accumulation。
- [ ] 实现 Gradient Clipping。
- [ ] 实现 LR Scheduler。
- [ ] 实现 Checkpoint 保存。
- [ ] 实现 Checkpoint 载入。
- [ ] 保存 RNG 状态。
- [ ] 保存采样器状态。
- [ ] 实现损坏 Checkpoint 检测。
- [ ] 实现自动 Resume。
- [ ] 编写 CPU Smoke Test。
- [ ] 编写 3090 GPU Smoke Test。
- [ ] 编写 V100 GPU Smoke Test。
- [ ] 验证 Loss 下降。
- [ ] 验证恢复后 Step 连续。
- [ ] 生成 M1 报告。

---

## M2：数据版本化

- [ ] 定义 Dataset Schema。
- [ ] 实现 JSONL 导入。
- [ ] 实现 Schema 校验。
- [ ] 实现 Unicode 规范化。
- [ ] 实现空白规范化。
- [ ] 实现长度过滤。
- [ ] 实现规则过滤。
- [ ] 实现精确去重。
- [ ] 评估近似去重方案。
- [ ] 实现去重后数据切分。
- [ ] 实现 Tokenization。
- [ ] 实现 Sequence Packing。
- [ ] 生成 Dataset Manifest。
- [ ] 生成 Statistics。
- [ ] 生成 Rejected Samples。
- [ ] 实现数据版本注册。
- [ ] 实现数据版本读取。
- [ ] 实现数据哈希校验。
- [ ] 编写数据管线单元测试。
- [ ] 编写重复构建一致性测试。
- [ ] 生成 M2 报告。

---

## M3：DDP

- [ ] 实现 torchrun 启动器。
- [ ] 实现分布式环境初始化。
- [ ] 实现 DistributedSampler。
- [ ] 实现 Rank-aware Logging。
- [ ] 实现指标 All-Reduce。
- [ ] 实现 Global Batch 校验。
- [ ] 实现 DDP Checkpoint。
- [ ] 实现 DDP Resume。
- [ ] 验证 1 卡运行。
- [ ] 验证 2 卡运行。
- [ ] 验证 4 卡运行。
- [ ] 验证 8 卡运行。
- [ ] 测试异常 Rank 退出。
- [ ] 测试 Checkpoint 恢复。
- [ ] 记录 Tokens/s。
- [ ] 记录 Step Time。
- [ ] 记录 Peak Memory。
- [ ] 计算 Scaling Efficiency。
- [ ] 生成 1/2/4/8 卡对比报告。

---

## M4：FSDP2 与 ZeRO-3

- [ ] 设计统一 ShardingStrategy 接口。
- [ ] 实现 FSDP2 FULL_SHARD。
- [ ] 实现自动 Wrap Policy。
- [ ] 实现 Mixed Precision Policy。
- [ ] 实现 Activation Checkpointing。
- [ ] 实现 Sharded State Dict。
- [ ] 实现 Full State Dict 导出。
- [ ] 实现 FSDP2 Resume。
- [ ] 实现 ZeRO-3 配置适配。
- [ ] 实现 ZeRO-3 Resume。
- [ ] 对比 DDP/FSDP2/ZeRO-3 显存。
- [ ] 对比吞吐。
- [ ] 运行 3B Full SFT Smoke Test。
- [ ] 运行 7B Full SFT Smoke Test。
- [ ] 记录失败和限制。
- [ ] 生成 M4 报告。

---

## M5：正式训练实验

### TinyGPT-350M

- [ ] 确定模型配置。
- [ ] 确定语料来源。
- [ ] 构建数据版本。
- [ ] 完成短程稳定性测试。
- [ ] 确定训练 Token 预算。
- [ ] 执行正式训练。
- [ ] 记录 Loss 曲线。
- [ ] 记录吞吐和显存。
- [ ] 保存生成样例。
- [ ] 生成训练报告。

### 小模型 Full SFT

- [ ] 选择 0.5B–1.5B 模型。
- [ ] 构建 SFT 数据版本。
- [ ] 完成 Baseline Evaluation。
- [ ] 完成 Full SFT。
- [ ] 完成 Post-SFT Evaluation。
- [ ] 生成能力变化报告。

### 7B–8B LoRA

- [ ] 选择模型。
- [ ] 设计 LoRA 配置。
- [ ] 完成单卡或多卡 Smoke Test。
- [ ] 完成正式 LoRA SFT。
- [ ] 导出 Adapter。
- [ ] 测试合并权重。
- [ ] 生成对比报告。

---

## M6：评测与晋级

- [ ] 设计 EvalTask 接口。
- [ ] 接入 lm-eval 适配器。
- [ ] 构建自建评测集。
- [ ] 版本化评测集。
- [ ] 实现 JSON Valid Rate。
- [ ] 实现任务准确率。
- [ ] 实现拒答率。
- [ ] 实现性能测试调用。
- [ ] 实现结果缓存。
- [ ] 实现 Baseline/Candidate Compare。
- [ ] 定义 Promotion Gate Schema。
- [ ] 实现 development 阶段。
- [ ] 实现 candidate 阶段。
- [ ] 实现 production 阶段。
- [ ] 实现 archived 阶段。
- [ ] 实现晋级审计日志。
- [ ] 编写门禁集成测试。
- [ ] 生成 M6 报告。

---

## M7：推理服务

- [ ] 定义 InferenceBackend。
- [ ] 实现 MockBackend。
- [ ] 实现 TransformersBackend。
- [ ] 实现 VLLMBackend。
- [ ] 实现 OpenAI-compatible API。
- [ ] 实现 Streaming。
- [ ] 实现 Health API。
- [ ] 实现 Model Info API。
- [ ] 实现请求日志。
- [ ] 实现 Request ID。
- [ ] 实现错误映射。
- [ ] 实现单卡 Benchmark。
- [ ] 实现并发 Benchmark。
- [ ] 实现 TP 配置。
- [ ] 实现 DP 配置。
- [ ] 实现模型热切换。
- [ ] 实现实例故障摘除。
- [ ] 生成 3090 报告。
- [ ] 生成 V100 报告。

---

## M8：硬件感知规划器

- [ ] 解析模型参数规模。
- [ ] 估算参数内存。
- [ ] 估算梯度内存。
- [ ] 估算优化器状态。
- [ ] 估算激活内存。
- [ ] 估算通信缓冲。
- [ ] 生成 DDP 方案。
- [ ] 生成 ZeRO-2 方案。
- [ ] 生成 FSDP2 方案。
- [ ] 生成 ZeRO-3 方案。
- [ ] 检测 BF16/FP16 能力。
- [ ] 读取 GPU 拓扑。
- [ ] 生成 GPU 分组建议。
- [ ] 实现自动 Micro Batch Probe。
- [ ] 实现 OOM 后回退。
- [ ] 输出最终训练计划。
- [ ] 生成估算与实测误差报告。
