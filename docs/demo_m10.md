# M10 Agent Production 中文演示

下面流程在配置好私有 Artifact Store、Serving 环境和 Bearer Token 后执行。命令默认只读，
不会把权重或请求级日志写入公开仓库。

## 1. 校验主机与 Production 身份

```bash
tinyllm doctor --json
tinyllm deploy show --model agent-production --artifact-root /data/yujielun/tinyllm --json
```

输出应包含 `status=Production`、模型版本
`qwen3-8b-m10-agent-lora-5m-3e8bf1dd` 和 Production Record SHA256。

## 2. 启动在线服务

```bash
export TINYLLM_GATEWAY_BEARER_TOKEN='使用至少32字符的私有Token'
tinyllm serve \
  --config configs/serving/m7_gateway.yaml \
  --model agent-production \
  --artifact-root /data/yujielun/tinyllm \
  --agent-config configs/agent/m8_devops.yaml \
  --evidence-index /data/yujielun/tinyllm/agent-evidence/index
```

服务提供 OpenAI-compatible Chat API 和 Agent API。`/version` 会返回 Agent Production 的
Evaluation Subject 与 Production Record 身份。

## 3. 发起一次诊断

```bash
tinyllm agent run \
  '检查最近一次 M3 DDP 运行的吞吐、数据等待和恢复记录，并引用证据路径。' \
  --mode nonthinking --max-steps 8 --json
```

演示重点观察：`run.started`、证据检索、工具调用、工具结果、最终回答和 `run.completed`。
审批型请求会进入 `waiting_approval`，使用 `tinyllm agent approve` 显式批准；断开 SSE
不会取消 Run，取消必须调用 `tinyllm agent cancel`。

## 4. 复核与回滚能力

```bash
tinyllm deploy resolve --model agent-production --artifact-root /data/yujielun/tinyllm --json
```

当 Alias 存在上一版本时可执行 `tinyllm deploy rollback-agent`；它只切换不可变 Alias，不修改
训练 Run、评测结果或历史 Production 记录。当前首个 Agent Production 版本没有历史目标，
因此不会人为制造回滚结果。所有请求级证据继续保存在私有 Artifact Store。
