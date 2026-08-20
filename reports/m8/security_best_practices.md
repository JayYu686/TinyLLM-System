# M8 Agent 安全实践审查

## 结论

M8 的默认部署边界适合当前单机、回环地址、固定模型和受控 Artifact Store 场景。运行时不提供
任意 Shell、任意文件读取、任意 MCP Endpoint 或源配置覆盖；唯一写工具只能在显式审批后修改
Agent 专属沙箱副本。审查未发现需要阻止 M8 兼容性验证的新增依赖公告。

本结论不覆盖公网直连、恶意管理员、任意第三方 MCP Server、多租户隔离或容器逃逸。此类部署
变化需要新的威胁建模和安全审查。

## 1. 信任边界与权限

- Agent API 只接受 User/Assistant 文本，System、Tool Call 和 Tool Result 权限保留在服务端；
- MCP Server 与工具由管理员 YAML 注册，调用方不能提供 URL 或进程启动参数；
- MCP Annotation 和检索结果只作为不可信内容，不能修改本地 Allowlist；
- 模型提出的 Server、Tool 和参数依次通过本地注册、发现 Schema 和 JSON Schema 校验；
- 未注册工具、重复工具循环和超出 Step/Tool 上限会失败关闭。

## 2. 网络、认证与敏感信息

- Gateway 与模型 Adapter 只接受 `127.0.0.1`/`localhost` HTTP 地址；
- Bearer Token 长度至少 32 字符，仅从环境变量读取，比较使用常量时间函数；
- HTTP Client 关闭环境代理继承与重定向，降低凭据被代理或重定向转发的风险；
- API 不返回内部路径和异常详情；公开事件拒绝 `reasoning_content/raw_cot` 等私有推理字段；
- 日志和 Run Projection 不保存完整 Prompt、工具结果、Secret 或原始 CoT。

## 3. 文件与写操作

- 读取路径要求落在允许根目录，逐段拒绝绝对路径、`..`、NUL 和软链接；
- 文件大小、文本编码、JSON 深度、返回记录数和摘录长度均有上限；
- 配置与指标递归脱敏 Token、Secret、Password、API Key 和 Authorization 字段；
- 写操作仅支持 YAML 顶层键更新，拒绝 Secret 键和过深/过大结构；
- 审批记录绑定 Tool Call SHA256，写入使用临时文件、`0600` 权限和原子替换；
- 源配置从不修改，目标只位于 `agent-sandboxes/<run-id>/`，参数漂移和目标冲突均拒绝。

## 4. 超时、重试与恢复

- 只读工具最多两次自动重试，退避 250/500 ms；
- 写工具不自动重试，通过审批与输出内容哈希实现幂等；
- 单工具默认 10 秒、整次 Run 默认 120 秒；
- SSE 断开不改变 Run 状态，取消使用显式接口；
- LangGraph 只从持久化安全节点恢复，审批状态由 SQLite Checkpoint 与 Artifact Store 交叉校验。

## 5. 依赖审计

2026-08-20 对隔离 M8 CI 环境执行 `pip-audit --skip-editable --format json`，原始结果保存在私有
Artifact Store。结果包含 8 条记录、6 个唯一公告：1 条 protobuf 和 7 条 Starlette 记录（部分
公告被重复报告），与 M7 已审查集合一致，LangGraph/MCP 没有引入新的审计发现。

限定 Profile 的逐项适用性与缓解控制见
[`requirements/m8_security_exceptions.md`](../../requirements/m8_security_exceptions.md)。CI 和
`make audit-agent` 只忽略这 6 个精确 ID；任何新公告仍会失败。依赖或部署边界变化后不得沿用
本次结论。

## 6. 后续安全工作

- M9 Release Suite 增加 Prompt Injection、错误 Tool、缺参、Tool Failure、Grounding、未审批
  写入和路径逃逸样例；
- M9 Agent Gate 要求未审批写入、路径逃逸和任意命令执行为 0；
- 接入真实远程 MCP 前增加证书、凭据轮换、Server 身份和网络出口策略；
- 公网部署前由反向代理提供 TLS、用户身份、分布式限流和审计日志保留策略。
