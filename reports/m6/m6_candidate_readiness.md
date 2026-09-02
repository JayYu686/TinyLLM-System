# M6 Candidate 正式评测就绪报告

> 历史状态说明：本报告记录评测启动前的输入就绪状态。该 Candidate 后续已完成 M6 v1
> 评测，实际结果与修复进展见 [M6 v1 诊断分析](m6_v1_diagnostic_analysis.md)。

## 结论

M5 正式 Full SFT Run 的 10M Token 快照已通过 M6 Candidate 血缘校验，可以进入冻结发布集的双模式评测。当前结论仅代表评测输入就绪，不包含尚未运行的 M6 Candidate 指标。

## 冻结身份

- 训练 Run：`20260807T071224Z-m5-formal-qwen3-0-6b-d39dad35-3d15`
- Checkpoint：`checkpoint-tokens-0010000532`
- 实际监督 Token：`10,000,532`
- 训练配置 SHA256：`d39dad3534730dfde08d526f24a69344d3be1341a097e610eaec7038041ad676`
- 数据版本：`m5-dual-sft-v1-b5b9e839`
- 数据 Manifest SHA256：`607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6`
- Checkpoint Manifest SHA256：`e42b4b587abcbdebe7b729189e77fc567a8534347ecf596aa1e92161f35350b9`
- 模型导出 SHA256：`b894b6ea081bd174ef0132182c231afea491ced2e4593c61cf1ef103447e3c5c`
- 基础模型 Revision：`c1899de289a04d12100db370d81485cdf75e47ca`
- 注意力架构：GQA

## 已完成校验

1. 正式 M5 Run 已成功完成 50M Token 训练，并包含按 10M Token 固定的五个评测快照。
2. 10M 快照与 Checkpoint ID、实际 Token 数、训练配置、数据版本和 Git Commit 一致。
3. Checkpoint Manifest、软件环境和硬件环境的文件哈希与训练结果一致。
4. 模型目录按 M5 导出算法重新计算，结果与快照记录一致。
5. Candidate 固定为 Qwen3-0.6B Full SFT，参数规模 `596,049,920`，保持 GQA 路线。
6. M6 Release 配置哈希固定为 `056e035e695492825ac781280162771fb944744870c73740e02ac213257e3447`。

## 下一步

在干净 Git Commit 上分别运行 Candidate Thinking 与 Non-thinking 300 条正式领域评测。两种模式完成 40 条人工评分后，再运行完整 ARC-Easy、HellaSwag 和 PIQA 通用回归评测，随后组装 Base/Candidate 证据并执行 Candidate Gate。
