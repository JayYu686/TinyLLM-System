# Changelog

All notable changes are recorded here. Versions follow Semantic Versioning; Python
package pre-release notation uses PEP 440 while Git tags use the public release name.

## Unreleased

- started M2 with pinned OASST1 and CommitPackFT identities, explicit per-source-license policy,
  strict import schemas, deterministic input/config hashes, and privacy-preserving rejection
  summaries;
- added synthetic CC0 fixtures and `tinyllm data inspect` for the public import contract.
- added conservative NFC/LF normalization, content-addressed Exact Dedup, connected Tree/Repository
  grouping, deterministic hash-based splitting, and reproducible synthetic M2.2 evidence.
- pinned the Qwen3-0.6B tokenizer artifacts and Non-thinking ChatML subset, added integrity-checked
  `tokenizers` loading, offset-aligned Assistant-only labels, and real synthetic-fixture evidence.
- added deterministic Train-only source/language Token balancing, split-local boundary-aware
  Best-Fit Decreasing Packing, versioned Pack/Manifest schemas, content-addressed dataset identity,
  and reproducible synthetic rebuild/failure evidence.
- added pinned atomic Artifact acquisition, strict JSONL readers, deterministic NumPy-sharded
  storage, immutable Dataset Registration/commit markers, complete file verification, safe Pack
  reconstruction, and `tinyllm data prepare|inspect` Registry contracts.
- raised the isolated Setuptools build/development constraint to 83.0.0 after the dependency audit
  identified `PYSEC-2026-3447` in the previous local build tool.
- completed the real pinned-source M2 build as `m2-sft-v1-f82ff32e`, independently verified every
  registered artifact, and reproduced the same content identity through a full offline rebuild
  without overwriting the immutable Registry version.
- added strict versioned evaluation-item/config/manifest/report schemas, deterministic evaluation
  content identity, privacy-preserving full-sequence and Prompt-prefix Train fingerprints, and the
  `tinyllm eval contamination` JSON/exit-code contract.

## 0.1.0-alpha.1

First M1 correctness release:

- native PyTorch TinyGPT single-device Trainer with AdamW, Warmup/Cosine scheduling,
  Gradient Accumulation/Clipping, finite-value guards, and structured metrics;
- deterministic CPU Loss-decrease evidence;
- atomic, integrity-checked full-state checkpoints with retention and pinned points;
- explicit Exact, Warm, and Transfer restore semantics;
- CPU bit-for-bit Exact Resume evidence and compatibility failure matrix;
- RTX 3090 BF16 repeat baseline and real SIGTERM/SIGKILL recovery evidence;
- stable Ruff, MyPy, Pytest, schema, link, public-artifact, audit, and Docker CI gates.

This release does not claim DDP throughput, FSDP2, real-data training, model quality,
inference performance, V100 compatibility, or distributed checkpoint recovery.
