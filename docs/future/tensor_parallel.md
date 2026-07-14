# Future Work：自研 Tensor Parallel

MVP 只调用成熟推理框架的 Tensor Parallel。

不自行实现：

- Column Parallel Linear。
- Row Parallel Linear。
- Attention Head 切分。
- 自定义 Collective。
- Pipeline Schedule。

后续进入条件：

- 多卡推理 Benchmark 完成。
- 已定位成熟框架无法解释或解决的问题。
- 自研具有明确学习或性能价值。
