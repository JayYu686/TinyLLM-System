# `tinyllm doctor` 接口契约

## 1. 目标

`tinyllm doctor` 是 TinyLLM-System 的只读环境体检入口。它用于在训练、评测或推理开始前采集可追溯的主机、软件、GPU、拓扑和存储信息，并明确区分：

- 已实测且通过。
- 已实测但存在风险。
- 已实测且失败。
- 因命令、权限或运行环境缺失而无法验证。

doctor 不修改驱动、CUDA、GPU 状态、系统配置或 Python 环境，也不自动安装依赖。

## 2. 命令面

```bash
tinyllm --help
tinyllm doctor
tinyllm doctor --json
tinyllm --json doctor
tinyllm doctor --distributed
tinyllm doctor --distributed --json --output report.json
```

- 默认输出面向人的摘要。
- `--json` 输出稳定、版本化的 JSON。
- `--distributed` 增加 NUMA、GPU 拓扑、P2P/NVLink 和 NCCL 可用性检查，但不自动运行高负载 NCCL Benchmark。
- `--output` 将与标准输出相同的结果写入指定文件；父目录必须已经存在。

M0 不提供训练、评测、部署或任意写操作命令。

## 3. JSON Envelope

```json
{
  "schema_version": "1.0",
  "command": "tinyllm doctor",
  "status": "pass",
  "generated_at": "2026-01-01T00:00:00Z",
  "inventory": {
    "host": {},
    "python": {},
    "cuda": {},
    "gpus": [],
    "topology": {},
    "storage": []
  },
  "checks": [
    {
      "id": "python.import",
      "status": "pass",
      "summary": "Python executable is available",
      "required": true,
      "evidence": {},
      "remediation": null
    }
  ],
  "errors": []
}
```

字段规则：

- `schema_version`：doctor 输出 Schema 版本；破坏性变化提升 Major。
- `status`：`pass`、`warn` 或 `fail`。
- `inventory`：采集到的事实；未知值使用 `null`，不得猜测。
- `checks[].status`：`pass`、`warn`、`fail` 或 `unavailable`。
- `errors`：命令执行、解析或文件输出错误；不得包含密钥、完整环境变量或完整用户请求。

## 4. 状态聚合

- 任一必需检查为 `fail`：总状态为 `fail`。
- 无 `fail`，但存在 `warn` 或 `unavailable`：总状态为 `warn`。
- 所有已要求检查通过：总状态为 `pass`。

初始必需检查：

- Linux 主机信息可读取。
- Python 可执行文件可用。
- PyTorch 可正常导入并报告版本。
- CUDA 可用性可以由 PyTorch 明确报告。
- 至少发现一张 GPU，且 GPU 查询可解析。
- 项目所在文件系统可读取剩余空间。

`--distributed` 的附加检查在 M0 作为风险检查；缺少 `numactl`、P2P 状态或 NCCL 工具时报告 `unavailable`，不伪装成通过。

## 5. 退出码

| 退出码 | 语义 |
|---:|---|
| 0 | doctor 完成，总状态为 `pass` 或 `warn` |
| 2 | 参数或输出路径错误 |
| 3 | doctor 完成，总状态为 `fail` |
| 4 | 未处理的采集或序列化错误 |

doctor 在发现缺失工具时应尽量完成其余采集，不能因单个可选命令缺失而崩溃。

## 6. 安全与隐私

- 不输出环境变量全集。
- 不输出完整进程命令行。
- 不读取或输出 API Token、SSH Key、Cookie 和 Docker 凭据。
- GPU 占用只记录 PID、显存和进程名的安全摘要；默认不记录参数。
- JSON 标准输出只包含 JSON；诊断信息写入标准错误。

## 7. M0 验收

- CLI 能从项目目录外通过已安装命令调用。
- 人类可读和 JSON 输出均通过 Smoke Test。
- JSON 可被标准 JSON 解析器读取，并包含固定顶层字段。
- 命令缺失、坏 PyTorch、无 GPU、坏 GPU 输出、输出目录不存在均有测试。
- 真实服务器结果写入硬件报告；未运行的检查明确标记为未验证。
- CPU CI 不执行 GPU Benchmark。

## 8. M0 实现选择

- CLI 使用 Python 标准库 `argparse`，M0 不为一个只读命令引入额外 CLI 框架。
- 项目环境使用 `.venv`，避免复用已检测到损坏的 Conda base。
- RTX 3090 Profile 使用 PyTorch 2.7.1 + CUDA 11.8 官方 wheel。
- `tinyllm doctor` 是高层只读命令；NCCL Benchmark 使用独立、显式的 `scripts/run_nccl_matrix.py`，避免 doctor 意外启动高负载任务。
