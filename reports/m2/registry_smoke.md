# M2.3c immutable Dataset Registry smoke

Execution date: 2026-07-14 (Asia/Shanghai)

Status: **PASS**

This smoke registered the public M2 synthetic Token-array fixture into a temporary private-style
Artifact Root. It verifies storage and failure semantics only; it did not download or process the
full OASST1/CommitPackFT snapshots and is not a model-quality or full-dataset storage result.

## Observed results

| Item | Actual result |
| -- | -- |
| Dataset identity | `m2-sft-v1-b606b6d3` |
| Content SHA256 | `b606b6d326eaf859a6b968c68752c8d480ff33b5291191478d80051be28bf38d` |
| Storage | `numpy-sharded-v1`; 44 registered files including root metadata |
| Payload | 5 Packs; 2,900 synthetic Tokens |
| First registration | Created atomically and reopened after complete hash verification |
| Second identical registration | Idempotent; returned the existing committed version |
| Reader | Reconstructed all 5 Packs exactly, including IDs, arrays, boundaries, and hashes |
| Corruption | Appended array bytes were refused with `DATASET_CORRUPT` |

The stable path-free evidence is retained in
[raw/registry_smoke.json](raw/registry_smoke.json). The temporary dataset directory is removed after
the smoke; no synthetic or upstream dataset payload is committed to Git.

Reproduce locally:

```bash
.venv/bin/python -m scripts.run_m2_registry_smoke
```

## Failure paths covered by the test suite

- missing `COMMITTED`, partial existing versions, file-size/SHA mismatch, unknown files, symlinks,
  path traversal, invalid Version strings, lineage drift, and undersized shards are refused;
- simulated write failure removes the temporary directory and never publishes the destination;
- existing versions are never overwritten, including incomplete or corrupt versions;
- JSONL parse errors report only a line number and do not echo source content;
- bad/missing offline cache entries and pinned size/SHA mismatches fail before import.

## Boundaries

- The source and Tokenizer acquisition identities are frozen. The subsequent
  [full pinned-source build](full_dataset_build.md) was executed from the merged clean commit and
  reproduced offline with the same content identity.
- `tinyllm data inspect --dataset-version ...` verifies every registered file before reporting the
  version. Future Trainers must use the same Reader instead of opening shard paths directly.
- The 44-file count is a consequence of the smoke's deliberately small 1,024-Token shard limit;
  the formal default is 4,194,304 Tokens per shard.
