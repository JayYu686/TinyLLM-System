# M2.3b balancing, packing, and Manifest smoke

Execution date: 2026-07-14 (Asia/Shanghai)

Status: **PASS**

This smoke used public synthetic Token ID arrays to test the deterministic M2.3b boundary. It
did not download or process the full upstream datasets and is not evidence of model quality. Its
purpose is to verify source/language balancing, split-local Packing, explicit sample boundaries,
content-addressed lineage, and exact rebuild behavior before the full-source M2.3c build.

## Observed results

| Item | Actual result |
| -- | -- |
| Input | 17 synthetic samples; 15 Train, 1 Validation, 1 Test |
| Train selection | 13 retained; 2 `balance_downsampled` records |
| Train Token mix | CommitPackFT English 1,000 / OASST1 English 800 / OASST1 Chinese 800 |
| Train mix in basis points | 3,846 / 3,076 / 3,076; every Stratum within the configured 300 bp tolerance |
| Train Packing | 2,600 Tokens in 3 × 1,024 capacity; 84.63% utilization |
| Whole synthetic build | 2,900 Tokens in 5 packs; 56.64% utilization including small Validation/Test splits |
| Boundary checks | Position IDs reset and Segment IDs aligned in all 5 packs |
| Duplicate/cross-split check | 15 retained samples appeared exactly once; every pack contains one split only |
| Rebuild | Reversing input order produced an identical Manifest and identical packs |
| Dataset identity | `m2-sft-v1-b606b6d3` |
| Content SHA256 | `b606b6d326eaf859a6b968c68752c8d480ff33b5291191478d80051be28bf38d` |

The complete non-sensitive evidence is retained in
[raw/packing_manifest_smoke.json](raw/packing_manifest_smoke.json). It contains lineage hashes,
per-split hashes, Pack IDs, sample boundaries, factual counts, and the two downsampling records;
it does not contain text, user paths, hostnames, or timestamps in the dataset identity.

Reproduce from a normal project environment:

```bash
.venv/bin/python scripts/run_m2_packing_smoke.py
```

## Failure paths covered

- Missing `oasst1:zh`, `oasst1:en`, or `commitpackft:en` Train strata abort the build.
- Unsupported source/language combinations abort instead of being silently reassigned.
- Duplicate sample IDs, Tokenization/Packing length drift, incomplete source lineage, and stage
  count mismatches abort the build.
- A changed Token ID changes the Pack hash, final content hash, and dataset version.
- Corrupted Pack content, Position IDs, Segment IDs, labels, or boundary counts fail Schema
  validation.

## Boundaries

- The smoke starts from synthetic `TokenizedSample` records; full-source download, Tokenization,
  durable Artifact Store writes, Registry inspection, and end-to-end CLI execution remain M2.3c
  (#40).
- Packing emits Segment IDs and reset Position IDs. The future Trainer must enforce a
  block-diagonal causal attention mask before packed examples are used for training.
- The reported 84.63%/56.64% values describe this synthetic fixture only. They are not projected
  full-dataset utilization or training throughput.
