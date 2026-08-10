# TinyLLM M6 v2 独立领域评测集

该目录保存 M6 v1 门禁拒绝并完成根因诊断后重新冻结的 300 条独立领域评测。v2 保持与 v1
一致的类别、语言比例、评分类型和 Cluster Bootstrap 单元，但 300 条 Prompt 全部重新生成，
与 v1 的精确 Prompt 交集为 0。

## 内容边界

- 语言：英文 210 条、中文 90 条；
- 类别：Python 50、Linux 45、日志诊断 45、JSON 40、配置 40、短代码 40、无依据拒答 40；
- 自动评分 260 条，人工 Rubric 40 条；
- 中英文语义配对 90 组，其余 120 条为英文 Singleton；
- 所有内容由 TinyLLM-System 生成并按 Apache-2.0 公开；
- 未读取 M6 v1 的模型输出，也不作为修复训练数据。

事实身份以 `manifest.json` 的 `suite_version` 与 `content_sha256` 为准。可使用以下命令验证：

```bash
python scripts/build_m2_domain_eval.py --suite-version v2 --check
```

M6 v2 继续使用原门禁阈值。v1 结果仅作为历史拒绝与根因证据，v2 只在修复策略、训练配置和
Candidate 选择均冻结后执行。
