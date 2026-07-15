# M2 Acceptance: Versioned Data and Frozen Evaluation

## Decision

M2 is accepted when the PR containing this report is merged. The evidence demonstrates a
deterministic licensed data product, grouped split isolation, content-addressed registration,
frozen evaluation identity, zero Exact train/evaluation matches, and a complete pre-training
Qwen3-0.6B Baseline with private raw outputs and maintainer judgments.

## Evidence chain

| Gate | Result | Evidence |
| -- | -- | -- |
| Pinned sources and licenses | Pass | OASST1 and CommitPackFT revisions and Dataset Card hashes verified |
| Import and filtering | Pass | Ready/Positive/non-deleted OASST path and CommitPackFT license allowlist |
| Normalization and Exact Dedup | Pass | Deterministic normalization, rejection accounting, and Exact components |
| Grouped split | Pass | Conversation Tree and Repository groups do not cross splits |
| Tokenizer and labels | Pass | Pinned Qwen3 tokens, non-thinking ChatML, Assistant-only labels |
| Balance, packing, and Manifest | Pass | Split-local packing and content-derived identity |
| Immutable Registry | Pass | Atomic publish, complete hashes, idempotency, corruption rejection |
| Full build and rebuild | Pass | `m2-sft-v1-f82ff32e` reproduced offline with the same content identity |
| Frozen Domain suite | Pass | 300 reviewed items; 210 English, 90 Chinese; seven categories |
| Exact contamination | Pass | 4,597 Train samples; zero full-sequence and zero Prompt-prefix matches |
| Formal Base Model Baseline | Pass | 300 Domain and 14,256 general samples from clean `main` |
| Human judgment | Pass | 40/40 rubric items reviewed, hashed, and atomically committed |
| Public/private boundary | Pass | Aggregates and failed IDs public; prompts, outputs, samples, and judgments private |

Detailed reports:

- [Pinned-source verification](source_verification.md)
- [Deterministic processing Smoke](deterministic_pipeline_smoke.md)
- [Qwen3 Tokenizer Smoke](qwen3_tokenizer_smoke.md)
- [Packing and Manifest Smoke](packing_manifest_smoke.md)
- [Immutable Registry Smoke](registry_smoke.md)
- [Full pinned-source build](full_dataset_build.md)
- [Domain content review](domain_eval_content_review.md)
- [Formal Exact contamination](domain_eval_contamination.md)
- [Formal Qwen3-0.6B Baseline](baseline_formal.md)

## Accepted identities

| Artifact | Identity |
| -- | -- |
| Dataset | `m2-sft-v1-f82ff32e` |
| Dataset content | `f82ff32ee98cb852fe6779774d9cce75a71e9430da72a6e5e1f4e3f7c2efd108` |
| Domain evaluation | `tinyllm-domain-v1-83bdd8ef` |
| Domain content | `83bdd8ef24dfa2bae0a997570594e7243f81ec3891a420458dd29b10f5e7af27` |
| Base Model | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` |
| Baseline config | `a2bae098181959c2f9799cc85d3404875ca015b61454f055c304b87033a936be` |
| Baseline code | `388241db0fd375939da9f47d2b1a3ac282819cbf` |

## Explicit boundaries and next dependency

M2 does not claim semantic decontamination, candidate improvement, distributed scaling,
distributed recovery, training throughput, inference latency, or production readiness. The
formal Baseline is intentionally weak on several TinyLLM domain categories; those failures define
the fixed comparison point and cannot be hidden or replaced after seeing a candidate result.

M1 and M2 now satisfy the strict prerequisites for M3 DDP correctness work. M5 may consume the
registered data and frozen Baseline only through their recorded identities. M3 still must establish
real torch.distributed correctness, rank-failure behavior, DDP Resume, and controlled scaling; M2
does not provide any distributed-training evidence.
