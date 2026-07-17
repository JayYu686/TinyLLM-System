# Changelog

All notable changes are recorded here. Versions follow Semantic Versioning; Python
package pre-release notation uses PEP 440 while Git tags use the public release name.

## Unreleased

- added non-reentrant Transformer-block activation checkpointing, strict wrapped-module evidence,
  and a forced nonzero-Rank exit diagnostic; passed real two-RTX-3090 BF16 CUDA/NCCL FSDP2
  correctness and Rank-failure runs while keeping DCP, Qwen3-8B, memory, and throughput unclaimed;
- froze an isolated M4 dependency profile with a network-free Tiny Qwen autograd gate, dedicated
  audit policy, schema snapshot, and CI job; revalidated CPU/Gloo and passed a real single-RTX-3090
  BF16 CUDA/NCCL FSDP2 Smoke while retaining a two-GPU busy-card preflight refusal;
- started M4 with strict FSDP2 correctness schemas, explicit CPU DeviceMesh selection, two-process
  Gloo/DTensor sharding evidence, full-state reconstruction, rank-zero-only artifacts, and
  fail-closed World Size/config/numerical guards; multi-GPU CUDA, DCP, Qwen3-8B, and four-GPU
  support remain explicitly unevaluated;
- made the complete Chinese `README.md` the primary public entrypoint while retaining a complete
  mutually linked English version in `README.en.md`.

## 0.3.0-beta.1

- added a fail-closed, YAML-driven M3 DDP benchmark harness with per-rank CUDA timings,
  data-wait and peak-memory metrics, PyTorch Profiler traces, live GPU telemetry, retained
  failure evidence, and strict repeat/matrix aggregation;
- completed real RTX 3090 BF16 Strong/Weak Scaling runs for 1/2/4 GPUs with three independent
  repeats per cell, published the raw repeat-level summary, and documented the observed
  non-linear scaling without extrapolation;
- adopted ADR-0004 so the shared-server release gate uses reproducible 1/2/4-GPU evidence while
  eight-GPU and controlled cross-NUMA runs remain explicitly uncollected optional enhancements;
- started M3 with strict torchrun/DDP configuration, deterministic initialization and Sampler
  evidence, exact Global Batch and reduced-Loss validation, rank-zero-only durable artifacts, and
  real one-/two-RTX-3090 NCCL/BF16 correctness runs; distributed Resume and scaling remain open.
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
- added the reproducible 300-item TinyLLM domain evaluation candidate with seven fixed categories,
  explicit objective/human scorers, Apache-2.0 provenance, and 90 tagged bilingual task pairs;
  formal clean-Train evidence and the Base Model Baseline are separate acceptance artifacts.
- recorded the reviewed domain set's clean-main scan against all 4597 registered Train samples:
  zero full-sequence and zero Prompt-prefix Exact matches; Near-Dedup remains not evaluated;
- completed the clean-main Qwen3-0.6B pre-training Baseline over 300 Domain items and 14,256
  general-task samples, atomically committed 40/40 maintainer judgments, retained private raw
  outputs, and published redacted aggregates, failed Item IDs, integrity hashes, and M2 acceptance.

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
