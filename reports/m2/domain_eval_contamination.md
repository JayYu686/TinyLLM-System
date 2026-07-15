# M2 Domain Evaluation Exact Contamination Report

Execution date: 2026-07-15 (Asia/Shanghai)

Status: **PASS**

This is the formal M2.4b Exact contamination result for the reviewed 300-item domain suite. It ran
from a clean `main` checkout after PR #49 was squash-merged. It is not a Near-Dedup or semantic
contamination claim, and it does not contain a model-quality result.

## Frozen inputs

| Input | Identity |
| -- | -- |
| Code | `c944cdb633c4d13f2183c82b418b33e0c1364ef6`; clean worktree |
| Evaluation | `tinyllm-domain-v1-83bdd8ef` |
| Evaluation content | `83bdd8ef24dfa2bae0a997570594e7243f81ec3891a420458dd29b10f5e7af27` |
| Dataset | `m2-sft-v1-f82ff32e` |
| Dataset content | `f82ff32ee98cb852fe6779774d9cce75a71e9430da72a6e5e1f4e3f7c2efd108` |
| Tokenizer | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` |
| Template | `qwen3-chatml-nonthinking-v1` |

The content review was approved and recorded in the
[M2 content-review report](domain_eval_content_review.md) before this run.

## Observed results

| Item | Actual result |
| -- | --: |
| Evaluation items checked | 300 |
| Verified Train samples checked | 4597 |
| Full-sequence Exact matches | 0 |
| Prompt-prefix Exact matches | 0 |
| Contaminated evaluation items | 0 |
| Exit status | 0 |
| Wall time | 23.97 seconds |
| Peak RSS | 495656 KiB |

The registered dataset contains Train, Validation, and Test samples; the scanner intentionally
indexed only the 4597 Train samples. It reconstructed Sample boundaries from verified Packs before
hashing, and published no raw training Sample ID or text. The stable raw evidence is retained in
[raw/domain_eval_contamination.json](raw/domain_eval_contamination.json).

## Reproduce

Run from the recorded commit with the private Artifact Root that contains the immutable registered
dataset and pinned Tokenizer cache:

```bash
.venv/bin/tinyllm eval contamination \
  --evaluation-set evals/domain/v1/items.jsonl \
  --config configs/eval/m2_domain_v1.yaml \
  --dataset-version m2-sft-v1-f82ff32e \
  --artifact-root "$TINYLLM_ARTIFACT_ROOT" \
  --json
```

The required success result is exit code 0, suite version `tinyllm-domain-v1-83bdd8ef`, dataset
version `m2-sft-v1-f82ff32e`, and zero matches of both kinds.

## Boundaries and next gate

- `near_dedup=not_evaluated`; paraphrases and semantic overlap were not ruled out.
- This run loaded Tokenizer files and verified data Packs but did not load model weights or use a
  GPU.
- The suite is frozen before formal post-training, but M2 is not complete until the pre-training
  Qwen3-0.6B Baseline in issue #47 is executed and its raw outputs are retained.
