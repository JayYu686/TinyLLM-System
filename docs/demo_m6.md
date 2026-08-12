# TinyLLM-System 10 分钟中文演示

本流程展示 `v0.6.0-rc.1` 已交付的 M0–M6 闭环。演示前在私有服务器设置
`TINYLLM_ARTIFACT_ROOT`，并使用项目 Python 3.11 环境。屏幕共享时隐藏用户名、主机名和绝对
路径。

## 0:00–1:00：项目与硬件边界

讲解 README 架构图：固定输入经过硬件感知执行、Checkpoint/Resume、独立评测和 Candidate
晋级。展示主机只读体检：

```bash
tinyllm doctor --distributed --json
```

强调当前真实范围为单机 RTX 3090，DDP 有 1/2/4 卡扩展证据，FSDP2 有 Qwen3-8B 四卡
FULL_SHARD 与 DCP Resume 证据。

## 1:00–2:00：数据与配置身份

```bash
tinyllm data inspect \
  --dataset-version m2-sft-v1-f82ff32e \
  --artifact-root "$TINYLLM_ARTIFACT_ROOT" \
  --json
```

指出数据 Revision、许可过滤、分组切分、Tokenization、Packing、Manifest 和污染检查共同组成
不可变数据身份；正式实验由 Schema 校验的 YAML 启动。

## 2:00–3:00：Run 血缘查询

```bash
tinyllm run rebuild --artifact-root "$TINYLLM_ARTIFACT_ROOT" --json
tinyllm run list --artifact-root "$TINYLLM_ARTIFACT_ROOT" --limit 5 --json
tinyllm run show \
  20260811T024325Z-m6-domain-contract-r41-seed42-dce956b0-d5b6 \
  --artifact-root "$TINYLLM_ARTIFACT_ROOT" \
  --json
```

说明 JSON/JSONL 是事实源，SQLite 是可重建查询投影。展示 Run 如何连接配置、Git、数据版本、
Checkpoint 和最终模型导出。

## 3:00–4:30：Checkpoint 与中断恢复

打开 [M1 Exact Resume 报告](../reports/m1/exact_resume_report.md)和
[M1 原子 Checkpoint 报告](../reports/m1/atomic_checkpoint_report.md)，讲解临时目录写入、逐文件
SHA256、原子 Rename、`LATEST` 更新和完整状态恢复。展示 CPU 逐位一致以及 RTX 3090 上真实
SIGTERM/SIGKILL 恢复证据。

## 4:30–6:00：DDP 扩展与 FSDP2 分片

打开 [M3 DDP 扩展报告](../reports/m3/ddp_scaling.md)，展示 1/2/4 卡真实重复结果、拓扑边界和
异常保留策略。随后打开 [M4 Qwen3-8B 四卡报告](../reports/m4/fsdp2_qwen3_8b_formal.md)与
[DCP 恢复报告](../reports/m4/fsdp2_dcp_recovery.md)，说明：

- DDP 用于模型完整状态能装入单卡时的吞吐扩展；
- FSDP2 分片参数、梯度和优化器状态；
- Step 25 分片 Checkpoint 由全新进程恢复到 Step 50，并独立加载 Safetensors 导出。

## 6:00–8:30：M6 Base/Candidate 比较

打开 [M6 验收报告](../reports/m6/m6_acceptance.md)，先展示 160/160 人工审查已完成，再展示：

- Thinking：34.33% → 41.67%，+7.34pp，95% CI `[+0.33, +14.29]pp`；
- Non-thinking：22.33% → 40.67%，+18.34pp，95% CI `[+12.46, +24.40]pp`；
- 通用三任务聚合：51.80% → 54.48%；
- Candidate 双模式 JSON 100%，Thinking 格式 100%，强制闭合 1.67%，Non-thinking 泄漏 0。

可在私有主机幂等复查门禁和注册：

```bash
M6_ROOT="$TINYLLM_ARTIFACT_ROOT/evaluations/m6/v7/final"
tinyllm compare \
  --config configs/eval/m6_release_v7.yaml \
  --baseline "$M6_ROOT/base/evaluation.json" \
  --candidate "$M6_ROOT/candidate/evaluation.json" \
  --output "$M6_ROOT/comparison.demo.json" \
  --json
tinyllm promote \
  --comparison "$M6_ROOT/comparison.json" \
  --registry-root "$TINYLLM_ARTIFACT_ROOT/registry" \
  --json
```

## 8:30–10:00：晋级边界与下一阶段

展示 Candidate 版本 `qwen3-0-6b-m6-d16c2357` 和 11/11 门禁。解释 M6 只授予 Candidate，
`production_eligible=false`。M7 将集成 vLLM，并在固定硬件与并发矩阵下测量 TTFT、TPOT、
吞吐、P50/P95、稳定性和回滚；只有推理门禁通过后才考虑 Production。

演示结束时回到架构图，总结从硬件、数据、训练、恢复、评测到模型注册的可追溯闭环。
