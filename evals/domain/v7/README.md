# TinyLLM M6 v7 密封 Thinking 边界审计集

该目录在实现 Thinking 最终答案停止策略之前冻结。冻结时 v6 Candidate Non-thinking 已通过
机械门禁，Candidate Thinking 的 JSON、强制闭合比例和可见推理泄漏均通过，但格式有效率为
296/300，低于 297/300 门槛。4 条失败均表现为最终答案后再次生成 `</think>` 与重复答案。
失败正文不得进入训练数据、v7 Prompt、Reference Answer 或人工 Rubric。

v7 保持既有 300 条规模、类别、语言比例、评分器、双语 Cluster Bootstrap 和全部门禁阈值，
使用新的完整指令与任务参数，并要求与 v1–v6 的完整 Prompt 交集为 0。模型权重保持 v4–v6
Candidate 不变；v7 只验证版本化 Thinking 最终答案边界与既有结构化 JSON 解码是否共同泛化。

事实身份：

```text
suite_version: tinyllm-domain-thinking-boundary-audit-v1-b82cbca1
content_sha256: b82cbca1821cadbaf4872636e89c61cef730ebe09413f9c63f34993302b6f955
```

重建检查：

```bash
python scripts/build_m2_domain_eval.py --suite-version v7 --check
```
