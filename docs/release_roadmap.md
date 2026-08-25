# TinyLLM-System release roadmap

## Objective

The core release establishes a reviewable training-systems baseline across native PyTorch
single-GPU training, DDP, FSDP2, Exact Resume, hardware-aware analysis, complete lineage,
Qwen post-training, regression evaluation, and Candidate promotion.

The schedule is ten core weeks plus two buffer weeks. Calendar labels express dependency
order, not permission to skip milestone gates when shared GPUs are unavailable.

## Core release train

| Window | Milestone | Required outcome | Release |
| -- | -- | -- | -- |
| Week 1 | Professional foundation | Apache-2.0, public README, governance, Typer/Pydantic contracts, CI | — |
| Weeks 2–3 | M1 | Native single-GPU trainer and verified Exact Resume/failure paths | `v0.1.0-alpha.1` |
| Week 4 | M2 | Deterministic licensed dataset version plus frozen baseline evaluation | — |
| Weeks 5–6 | M3 | Correct DDP and controlled 1/2/4-GPU scaling evidence | `v0.3.0-beta.1` |
| Week 7 | M4 | Qwen3-8B FSDP2 four-GPU sharded checkpoint/resume smoke after memory probe | — |
| Weeks 8–10 | M5 | Qwen3 GQA dual-mode data, Qwen3-0.6B Full SFT, and Qwen3-8B LoRA with real evaluations | — |
| After M5 | M6 | Compare, Candidate gate, reports, release notes, and end-to-end demo | `v0.6.0-rc.1` |

M0–M6 are complete. M5 produced a four-GPU Qwen3-0.6B Full-SFT run over 50M supervised
tokens and a single-GPU Qwen3-8B LoRA run over 10M tokens, both with fresh-process Exact
Resume and real dual-mode evaluation. M6 then evaluated the final 0.6B artifact on an independent
300-item bilingual suite, completed 160 human judgments, passed all 11 Candidate checks, and
registered `qwen3-0-6b-m6-d16c2357` as Candidate. M7 subsequently completed its 18,000-request
formal serving matrix, recovery, rollback, and security gates, promoting
`qwen3-0-6b-m7-fa678d92` to Production.

M8 then completed its clean-commit 8/8 OpenAI Tool Calling matrix, bounded MCP tools, durable
approval/restart recovery, FTS5 grounding, and sandbox-only idempotent write contract. The runtime
is complete. M9 subsequently froze the 80/160 Dev/Release split, completed three real Agent Dev
baselines, and ran 5,520/5,520 BFCL Core items with zero inference failures. M10 then froze its
five-source training mixture and completed real 0.6B Full-SFT and 8B LoRA routes. Both routes were
stopped by their preregistered Agent Dev continuation gates, so M7 Production remains unchanged
and the project publishes `v1.0.0-rc.1` without an Agent-model Production claim.

## Agent release train

| Window | Milestone | Required outcome | Release |
| -- | -- | -- | -- |
| Weeks 1–2 | M7 | vLLM serving, authenticated Gateway, real inference benchmark, recovery/rollback, Production Gate | `v0.7.0` |
| Week 3 | M8 | Qwen tool calling, MCP client/reference server, LangGraph DevOps agent, approval and evidence retrieval | `v0.8.0-beta.1` |
| Week 4 | M9 | BFCL v1.3 Offline Core Profile plus frozen public/hidden DevOps Agent Evaluation | `v0.9.0-rc.1` |
| Weeks 5–6 | M10 | 0.6B Agent Full SFT, 8B Agent LoRA, fail-closed model selection | `v1.0.0-rc.1` |

M7 Production proves model quality, lineage, serving correctness, performance, recovery and
rollback. M9/M10 add an independent Agent capability gate; an M7 Production model cannot claim
Agent readiness from serving evidence alone.

M10 is closed under its registered failure branch. The 0.6B 5M stage scored 10.00% against its
21.25% parent, and the 8B LoRA 1M stage scored 32.50% against its 45.00% parent. Neither model
advanced to the sealed Release suite or the unified Production gate. A future data revision may
target final-answer grounding, irrelevance boundaries, and failure recovery without rewriting
these results.

## Frozen experiment targets

- TinyGPT DDP target: hidden 768, 12 layers, 12 heads, intermediate 2304, vocabulary
  32768, sequence length 1024, weight tying. It is named `TinyGPT-Target-120M` until the
  instantiated parameter count is recorded.
- FSDP2 smoke: pinned Qwen3-8B revision, BF16, activation checkpointing, FULL_SHARD,
  sequence length 512, 50 optimizer steps, checkpoint at step 25, and resume to step 50.
- Full SFT: pinned Qwen3-0.6B revision, native GQA, explicit Thinking/Non-thinking modes,
  assistant-only loss, BF16, sequence length 1024, gradient checkpointing, and staged
  10M-token gates. The formal Thinking-token ratio is selected by a preregistered short ablation.
- LoRA: pinned Qwen3-8B revision, BF16 LoRA rank 16/alpha 32/dropout 0.05 over attention
  and MLP linear layers with the same explicit dual-mode contract. NF4 is an explicit fallback
  only after the defined BF16 setup produces a recorded OOM.

Model and dataset revision changes require an ADR. Revision availability and licensing
must be re-verified when M2/M4/M5 implementation starts; this roadmap does not claim the
remote artifacts have already been downloaded or tested.

## Core versus buffer

The `v0.6.0-rc.1` core ends at Candidate. Production requires M7 inference performance
evidence. Buffer work is prioritized as:

1. M7 vLLM serving, Gateway, inference benchmark and Production Gate;
2. M8 Tool Calling, MCP and DevOps Agent;
3. M9 BFCL plus DevOps Agent Evaluation;
4. M10 Agent Full SFT/LoRA and unified model selection;
5. minimal static estimator plus 10–20-step probe for `tinyllm plan`;
6. FSDP2-versus-ZeRO-3 short comparison;
7. optional MLflow projection, GPU container validation, V100 FP16 smoke, and TinyGPT-350M.

MoE, custom KV cache, custom tensor parallel, custom FlashAttention/CUDA kernels,
multi-node training, pipeline parallelism, full RLHF, Kubernetes, billing, and complex
frontends remain Future Work and stay outside the core release gate.

## Release evidence gate

Each milestone remains `IN_PROGRESS` until design, interface, tests, smoke, failure path,
integration, real report, Issue synchronization, and merged PR all exist. Code completion
alone is not a release condition.
