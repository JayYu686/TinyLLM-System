# M9 BFCL 依赖审计例外

审查日期：2026-08-20

适用范围仅限隔离的 BFCL v1.3 Offline Core Profile 环境。该环境只读取固定 Commit
`ea13468e4423454d0c213704fb87cf7cb3990433` 中的官方离线任务，通过环回 TinyLLM Gateway
发出 JSON 请求，并把结果写入私有 Artifact Store。它不参与 Gateway、Agent Runtime、训练或
Production 服务进程。

`pip-audit` 在 115 个发行包中发现 1 个受影响包：BFCL 上游明确固定的
`datamodel-code-generator==0.25.7`。审计原始 JSON 和完整 `pip freeze` 保存于：

```text
$TINYLLM_ARTIFACT_ROOT/agent-evaluations/m9/bfcl/environment/
```

| Advisory | 受影响能力 | 本项目边界与控制 | 移除条件 |
| -- | -- | -- | -- |
| `PYSEC-2026-3555`、`PYSEC-2026-3557`、`PYSEC-2026-3561`、`PYSEC-2026-3566` | 从恶意 Schema 或模板生成并导入 Python 代码 | 不接受用户 Schema、模板或生成代码；只执行固定 BFCL 数据与本项目 Endpoint Handler | BFCL 上游允许 `datamodel-code-generator>=0.64.0` 后完成兼容性验证 |
| `PYSEC-2026-3560`、`PYSEC-2026-3563`、`PYSEC-2026-3564`、`PYSEC-2026-3565` | 远程 `$ref`、URL、重定向与 DNS 重绑定导致 SSRF 或凭据泄露 | 评测输入固定且经过哈希校验；不调用远程 Schema 生成入口，不向代码生成器提供认证头 | 同上 |

原始审计包含同一 Advisory 的重复记录，因此工具报告 9 条 Finding，对应 8 个唯一 Advisory。
`setuptools` 已升级至 84.0.0，消除了初次环境审计中的 `PYSEC-2026-3447`。若 BFCL 输入来源、
网络边界、代码生成路径或依赖 Commit 发生变化，本例外立即失效并阻止正式 M9 结果发布。
