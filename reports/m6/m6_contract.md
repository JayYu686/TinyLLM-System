# M6.0 评测与晋级契约验收报告

## 1. 结论

M6.0 已建立独立 Release 评测的机器可执行契约。该批次冻结双模式比较、配对 Cluster
Bootstrap、通用能力回归、格式与血缘门禁，以及 Candidate Registry 原子写入接口。

本报告不包含真实 M6 模型质量结果。Qwen3-0.6B Base 与 M5 10M 最优快照尚未在该 Release
协议下完成双模式领域评测和完整通用评测，因此 M6 当前状态保持 `IN_PROGRESS`，模型保持
`Development`。

## 2. 冻结范围

- Release 配置：`configs/eval/m6_release.yaml`；
- 领域集：`tinyllm-domain-v1-83bdd8ef`，共 300 条；
- 重采样单元：90 个中英翻译对 Cluster 与 120 个英文单例 Cluster；
- Bootstrap：10,000 次，Seed `20260809`，`nearest-rank-v1` 95% 区间；
- 通用任务：ARC-Easy、HellaSwag、PIQA，三项 `acc_norm` 等权平均；
- 生成协议：Thinking 1,536+512 Token 有界生成与固定采样；Non-thinking 512 Token Greedy；
- 可比性：双模式 Template、通用 Chat Template、Task Adapter 和 Dataset Revision 均绑定配置 Hash；
- 晋级上限：M6 只能写入 `Candidate`，不能授予 `Production`。

## 3. 已实现能力

`tinyllm compare` 会验证 Base/Candidate 的协议、模型 Revision、Tokenizer、Prompt、评测项与
Cluster 身份一致性，然后执行预注册 AND Gate。任何质量、格式、人工审查或血缘检查失败均
保存拒绝结果并返回退出码 6。

`tinyllm promote` 只接受已通过的比较结果，以 Comparison SHA256 形成模型版本，在私有
Registry 中原子写入不可变 Candidate 记录。相同证据重复执行保持幂等，冲突血缘拒绝覆盖。

## 4. 验收方法

- 严格 Pydantic Schema 与公开 JSON Schema Snapshot；
- 300 条固定语言、类别、评分器和 Cluster 结构校验；
- 确定性 Bootstrap 与完整 AND Gate 单元测试；
- 无领域增益、通用回退、血缘不完整和协议不兼容失败路径；
- CLI 稳定 JSON、退出码 2/6、原子写入与幂等晋级测试；
- Ruff、MyPy、Pytest、Schema Snapshot 和文档链接检查。

## 5. 证据边界与下一步

M6.0 证明评测判定逻辑已经在看到 Candidate Release 输出前冻结，尚不证明任何模型通过
门禁。下一批 M6.1 将复用符合新契约的历史 Non-thinking Base 原始证据，并补跑 Base
Thinking；M6.2 再评测 10M Candidate，完成人工审查和完整通用任务，最后才能生成正式比较。
