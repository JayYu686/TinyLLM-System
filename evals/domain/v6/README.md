# TinyLLM M6 v6 密封输出边界审计集

该目录在实现 Non-thinking Thinking-tag 停止策略之前冻结。冻结时只知道 v5 Candidate
Non-thinking 的汇总诊断及其单条机制；尚未执行或读取 v5 的其余三路结果。v6 保持
既有 300 条规模、类别、语言比例、评分器、双语 Cluster Bootstrap 和全部门禁阈值，并使用
新的完整指令与任务参数。v6 与 v1–v5 的完整 Prompt 交集为 0。

v6 只用于验证版本化输出边界和既有 JSON 结构化解码能否跨独立内容共同泛化。模型权重保持
v4/v5 Candidate 不变；v5 失败正文不得进入训练数据、v6 Prompt、Reference Answer 或人工
Rubric。

事实身份：

```text
suite_version: tinyllm-domain-output-boundary-audit-v1-c34f63a8
content_sha256: c34f63a87c05910f421db19c71eede7368328028f81bbf08870070bb2fba6002
```

重建检查：

```bash
python scripts/build_m2_domain_eval.py --suite-version v6 --check
```
