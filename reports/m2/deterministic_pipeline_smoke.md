# M2.2 deterministic pipeline smoke

Execution date: 2026-07-14 (Asia/Shanghai)

Status: **PASS**

This is a correctness smoke over the repository's synthetic CC0-1.0 fixtures. It proves the
import-to-normalize-to-deduplicate-to-grouped-split code path and its deterministic manifests; it
does not claim full-dataset counts, target language/token ratios, model quality, or training
readiness.

## Observed result

| Stage | Input | Accepted/output | Rejected | Notes |
| -- | --: | --: | --: | -- |
| OASST1 import fixture | 3 rows / 2 candidates | 1 sample | 1 | negative review rejected |
| CommitPackFT import fixture | 3 candidates | 1 sample | 2 | non-Python and unknown license rejected |
| M2.2 processing | 2 imported samples | 2 samples | 0 | 2 connected components; no exact duplicate in this fixture |

Both output components mapped to Train under the frozen 98/1/1, Seed 42 hash policy. That is valid
for a two-sample smoke and is not a distribution claim. Empty Test and Validation splits have the
deterministic SHA256 of an empty length-framed sequence.

The processing output hash was
`83bcce87a849e81a09c761a022075dfba97d3eddccfc42317a4ebcd6bca6024c` and remained identical when
the input iterator order was reversed by the integration test. Full machine-readable evidence is
stored in [raw/deterministic_pipeline_smoke.json](raw/deterministic_pipeline_smoke.json).

## Boundaries

- The fixture contains no upstream OASST1 or CommitPackFT records.
- Tokenization, Packing, final Dataset Manifest/registration, contamination checks, and frozen
  Baseline Evaluation remain M2.3/M2.4 work.
- These processed samples are intentionally not exposed to the Trainer as registered data.
