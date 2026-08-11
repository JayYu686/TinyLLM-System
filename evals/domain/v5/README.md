# TinyLLM M6 v5 密封 JSON 解码审计集

该目录在读取 v4 JSON 失败正文之前冻结。v5 保持既有 300 条规模、类别、语言比例、评分器、
双语 Cluster Bootstrap 和门禁阈值，使用新的完整指令与任务参数。v5 与 v1–v4 的完整 Prompt
交集为 0。

v5 只用于验证版本化 JSON 约束解码是否跨独立内容泛化。模型权重保持 v4 Candidate 不变；
v4 的失败正文不得进入训练数据、v5 Prompt、Reference Answer 或人工 Rubric。

事实身份：

```text
suite_version: tinyllm-domain-json-audit-v1-3e5fffd7
content_sha256: 3e5fffd7d408a6d2d237f4da7f5e3ecfb72523bd5f9e42b6e74f24e9199b1bfe
```

重建检查：

```bash
python scripts/build_m2_domain_eval.py --suite-version v5 --check
```
