# TinyLLM M6 v3 独立领域评测集

该目录保存 M6 v2 暴露“思考过长”和“证据拒答未泛化”后，在 R2 数据构建及训练开始前冻结的
300 条独立领域评测。v3 保持既有类别、语言比例、评分类型、生成参数和 Cluster Bootstrap
策略；它与 v1、v2 的精确 Prompt 交集均为 0。

## 内容边界

- 语言：英文 210 条、中文 90 条；
- 类别：Python 50、Linux 45、日志诊断 45、JSON 40、配置 40、短代码 40、无依据拒答 40；
- 自动评分 260 条，人工 Rubric 40 条；
- 中英文语义配对 90 组，其余 120 条为英文 Singleton；
- 所有内容由 TinyLLM-System 生成并按 Apache-2.0 公开；
- R2 训练数据不读取 v3 Reference Answer、模型输出或逐项得分。

事实身份以 `manifest.json` 的 `suite_version` 与 `content_sha256` 为准：

```text
suite_version: tinyllm-domain-holdout-v1-2b167ce6
content_sha256: 2b167ce67a3761558bf2c556131d86eb572dc5d36e533a668a539a78eb86d6e2
```

可使用以下命令重建并验证：

```bash
python scripts/build_m2_domain_eval.py --suite-version v3 --check
```

M6 v3 沿用 v1/v2 的全部门禁阈值，不因前两次拒绝降低标准。v2 结果只作为失败诊断证据，
不能参与 v3 Candidate 选优。
