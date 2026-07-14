# M2 full pinned-source dataset build

Execution window: 2026-07-14 to 2026-07-15 (Asia/Shanghai)

Status: **PASS**

This report records the first complete M2 build from the pinned OASST1 and CommitPackFT
artifacts through import, normalization, Exact Dedup, grouped Split, Qwen3 Tokenization,
Train-only Token balancing, split-local Packing, and immutable Dataset Registration. It is a data
correctness and reproducibility result, not a model-quality or training-throughput result.

## Dataset identity

| Item | Actual result |
| -- | -- |
| Dataset version | `m2-sft-v1-f82ff32e` |
| Content SHA256 | `f82ff32ee98cb852fe6779774d9cce75a71e9430da72a6e5e1f4e3f7c2efd108` |
| Code revision | `2ab550cdd2369f09472d604a53375884637928a3`; clean worktree |
| Source rows | 88,838 OASST1 messages; 56,025 CommitPackFT records |
| Imported samples | 26,172 OASST1; 50,149 CommitPackFT |
| Processed / Tokenized | 75,260 / 74,665 samples |
| Balanced dataset | 6,183 samples; 1,983 Packs |
| Token payload | 2,026,977 total; 1,412,707 supervised |
| Storage | `numpy-sharded-v1`; 32 verified files; 56,138,828 bytes |
| Packing efficiency | 9,982 basis points (99.82%) |

The fixed upstream revisions are:

- `OpenAssistant/oasst1@fdf72ae0827c1cda404aff25b6603abec9e3399b`;
- `bigcode/commitpackft@fc56fe33c030c6daa414c2b112c932b8eed085e6`;
- `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` Tokenizer files.

## Split and mixture results

| Split | Samples | Packs | Tokens | Supervised Tokens |
| -- | --: | --: | --: | --: |
| Train | 4,597 | 1,419 | 1,452,021 | 1,047,193 |
| Validation | 944 | 341 | 347,816 | 220,460 |
| Test | 642 | 223 | 227,140 | 145,054 |

The final source Token counts are 1,015,650 CommitPackFT and 1,011,327 OASST1. The final language
Token counts are 1,588,122 English and 438,855 Chinese. Train strata measured 40.00% CommitPackFT
English, 29.99% OASST1 English, and 29.99% OASST1 Chinese after integer rounding. These are
observed data-build distributions, not accuracy metrics.

## Independent rebuild evidence

| Run | Mode | Result | Wall clock | Peak RSS |
| -- | -- | -- | --: | --: |
| 1 | Online first build from fixed artifacts | `created=true`; verified | 1:25:23 | 2,443,512 KiB |
| 2 | Offline full rebuild from verified cache | `created=false`; verified | 1:03:58 | 2,444,640 KiB |

An independent `tinyllm data inspect` verified every declared file, size, and SHA-256 in 1.89
seconds and reported the same version, content hash, lineage, and counts. The offline rebuild did
not perform network access, produced the same content identity, returned the original registration
timestamp, and reported zero filesystem outputs. This proves idempotent reuse of the committed
version rather than overwrite.

The path-free machine-readable evidence is retained in
[raw/full_dataset_build.json](raw/full_dataset_build.json). Original data, rejection records,
registered arrays, and unsanitized execution context remain in the private Artifact Store.

## Rejections and license boundary

Observed rejection counts are preserved in the raw evidence. The largest categories were 68,482
samples removed by deterministic Train balancing, 27,510 OASST paths outside the English/Chinese
language allowlist, 5,873 CommitPackFT records outside the license allowlist, 1,040 Exact
duplicates, and 595 sequences above the 1,024-Token limit. No rejected sample content is published.

The 6,183 selected samples retain these normalized licenses: 4,181 Apache-2.0, 1,414 MIT, 441
BSD-3-Clause, 93 BSD-2-Clause, 29 ISC, 14 Unlicense, and 11 CC0-1.0.

## Limitations and next gate

- The current full build is single-process and emits only a final summary. The observed 64–85
  minute runtime makes stage progress, stage timing, and deterministic performance optimization
  worthwhile follow-up work; no optimized runtime is claimed here.
- Exact Dedup is complete. Near-Dedup and train/evaluation contamination checks remain M2.4.
- The Validation/Test splits are versioned data products, but the formal frozen evaluation suite
  and pre-training Baseline Evaluation are not yet complete.
- This report does not claim model quality, GPU training throughput, DDP scaling, or deployment
  readiness.

Reproduce in an authorized environment with access to the private Artifact Root:

```bash
tinyllm data prepare --artifact-root <private-artifact-root> --json
tinyllm data inspect \
  --artifact-root <private-artifact-root> \
  --dataset-version m2-sft-v1-f82ff32e \
  --json
tinyllm data prepare --artifact-root <private-artifact-root> --offline --json
```
