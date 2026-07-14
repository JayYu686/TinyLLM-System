# Changelog

All notable changes are recorded here. Versions follow Semantic Versioning; Python
package pre-release notation uses PEP 440 while Git tags use the public release name.

## Unreleased

- started M2 with pinned OASST1 and CommitPackFT identities, explicit per-source-license policy,
  strict import schemas, deterministic input/config hashes, and privacy-preserving rejection
  summaries;
- added synthetic CC0 fixtures and `tinyllm data inspect` for the public import contract.

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
