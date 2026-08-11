# TinyLLM M6 v4 密封终审领域集

该目录保存 R3 在 v3 暴露任务族泛化不足后、R4 训练开始前冻结的 300 条终审题。v4 保持
既有类别、语言比例、评分器、双语 Cluster Bootstrap 和门禁阈值，并使用新的任务参数与
终审指令。v4 与 v1、v2、v3 的完整 Prompt 交集均为 0。

## 内容边界

- 语言：英文 210 条、中文 90 条；
- 类别：Python 50、Linux 45、日志诊断 45、JSON 40、配置 40、短代码 40、无依据拒答 40；
- 自动评分 260 条，人工 Rubric 40 条；
- 中英文语义配对 90 组，其余 120 条为英文 Singleton；
- R4 训练集不读取任何 v1–v4 Reference Answer、模型输出或逐项得分；
- R4 与 v4 共享公开的任务类别契约，但使用互不重合的参数区间与完整 Prompt。

事实身份：

```text
suite_version: tinyllm-domain-final-audit-v1-bac25144
content_sha256: bac25144d53d186693514f6a421e3894a820bddb039c75ca29c2484190b7913a
```

重建检查：

```bash
python scripts/build_m2_domain_eval.py --suite-version v4 --check
```

v4 沿用 v1–v3 全部门禁阈值。v3 仅作为失败诊断与开发验证证据，不能替代 v4 终审结果。
