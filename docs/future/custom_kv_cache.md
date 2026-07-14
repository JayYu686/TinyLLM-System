# Future Work：自研 KV Cache

自研 KV Cache 不进入 MVP。

MVP 使用推理后端现有实现，并重点测试：

- 不同上下文长度。
- 显存占用。
- Continuous Batching。
- Prefix Caching。
- TP/DP。

后续若需要深入推理系统，再考虑：

- Paged KV Cache。
- Block Allocator。
- Eviction。
- Prefix Sharing。
- Cache Quantization。
