# Changelog

All notable changes are recorded here. Versions follow Semantic Versioning; Python
package pre-release notation uses PEP 440 while Git tags use the public release name.

## Unreleased

- preregistered the M10 Agent training mixture, language and supervision policy, fixed external
  revisions, replay lineage, deduplication, sealed-Release contamination boundary, and fail-closed
  training-readiness contract;
- added safe ToolACE/Hermes source parsers, immutable JSON Schemas, and a real content-free profile
  of 13,193 pinned external rows, accepting 12,592 structural candidates and isolating 601 rows
  under stable rejection reasons.
- built 2,400 deterministic bilingual DevOps Agent trajectories with message-level supervision
  masks, strict tool-call/result pairing, grouped MinHash deduplication, and content-free scans
  against M9 Dev/Release, BFCL Core, and M6 Domain; training remains blocked pending content review.

## 0.9.0-rc.1 - 2026-08-20

- froze an original bilingual 240-task DevOps Agent Suite with 80 public Dev tasks and 160 sealed
  Release tasks, deterministic tool environments, failure injection, trace assertions, and an
  independently preregistered M10 Agent Model Gate;
- registered immutable evaluation-only Qwen3-8B Base and historical LoRA subjects without
  changing the M6 Candidate or M7 Production records;
- completed real RTX 3090 Agent Dev baselines for the 0.6B Production, 8B Base, and historical
  8B LoRA subjects, measuring Task Success of 20.00%, 36.25%, and 36.25%, respectively;
- completed three fail-closed `TinyLLM BFCL v1.3 Offline Core Profile` runs over 5,520/5,520
  items with zero inference failures; measured scores were 24.24%, 39.18%, and 36.25%;
- hardened the offline BFCL adapter against host proxies, non-standard response schemas,
  incomplete generations, swallowed endpoint errors, excessive multi-turn message counts, and
  insufficient context limits while keeping the Agent API's own safety limits unchanged.

## 0.8.0-beta.1 - 2026-08-20

- delivered the bounded LangGraph DevOps agent runtime, OpenAI tool calling, an allowlisted MCP
  client/reference server, SQLite FTS5 evidence retrieval, explicit approval, cancellation,
  idempotency, SSE replay, and safe-node restart recovery;
- passed the real Qwen tool-calling protocol matrix, MCP security checks, approval/resume smoke,
  and public-artifact security review without exposing arbitrary shell or unrestricted file I/O.

## 0.7.0 - 2026-08-13

- delivered the lineage-aware vLLM/FastAPI Gateway, model resolver, observability, deployment
  registry, atomic Production alias, rollback, and inference benchmark contracts;
- completed the 18,000-request formal Direct/Gateway matrix with 18,000 successes and passed all
  nine Production checks before promoting the M6 0.6B Candidate.

## 0.6.0-rc.1 - 2026-08-12

- completed the independent M6 v7 release evaluation with four 300-item dual-mode passes,
  160/160 maintainer-reviewed judgments, three full general benchmarks, and complete lineage;
- passed all 11 Candidate checks: Thinking improved by 7.34pp with a paired-bootstrap 95% CI of
  `[+0.33, +14.29]pp`, Non-thinking improved by 18.34pp with `[+12.46, +24.40]pp`, and the
  equal-task general aggregate improved by 2.68pp;
- atomically registered the 596,049,920-parameter Qwen3-0.6B Full-SFT artifact as Candidate
  `qwen3-0-6b-m6-d16c2357`, retaining `production_eligible=false` until M7;
- added `tinyllm run rebuild|list|show` and strict public schemas for an atomic SQLite v1 query
  index, then successfully rebuilt it from all 57 private Run manifests;
- published redacted Chinese/English M6 acceptance material and a 10-minute Chinese demo flow.

- froze an independent 300-item M6 v7 audit after v6 Candidate Thinking missed the fixed format
  gate by one item, then added an auditable Thinking final-answer stop boundary that preserves raw
  responses while preventing repeated Thinking tags from invalidating an already emitted answer;
- added the M6.1 Base-evidence bridge and independent domain runner: the formal M2
  Non-thinking/general evidence is reused only after full protocol, model, file, human-review,
  environment, and raw-tree verification; Base Thinking remains a separate clean-main GPU run
  with private transcripts, explicit controller actions, and a 40-item review gate;
- started M6 by preregistering the isolated release-evaluation protocol before Candidate
  inference: separate Thinking/Non-thinking comparisons, a deterministic 10,000-replicate
  paired cluster bootstrap, equal-task `acc_norm` regression, strict lineage and format checks,
  stable JSON CLI outputs, and atomic Candidate-only registry promotion;
- completed M5 with a real four-RTX-3090 Qwen3-0.6B BF16 Full-SFT campaign over 50M
  supervised tokens, an independent-process Exact Resume from 2,002,739 tokens, five
  immutable 10M–50M snapshots, thermal pause/resume evidence, and verified checkpoint,
  evaluation, and export hashes; the 10M development snapshot led the curve at 95.0%
  Thinking and 47.5% Non-thinking accuracy and is prioritized for the independent M6 gate;
- completed the real single-RTX-3090 Qwen3-8B BF16 LoRA route with 10M supervised
  tokens, a fresh-process Exact Resume at 5,000,444 tokens, five pinned evaluation
  checkpoints, thermal pause/resume evidence, an adapter-only Safetensors export and
  Model Card; the final 200-item dual-mode evaluation reached 99.0% Thinking and 72.0%
  Non-thinking accuracy with zero visible-reasoning leakage, without claiming a Base gain;
- built the corrected `m5-r3-mixture-v2-b47723e1` before any R3 training or Dev
  evaluation: exact 700K/150K/150K supervised-token strata, label-aware 160-source
  selection, a measured 29–30 total uses per targeted source, immutable lineage, and
  explicit training authorization for only the two fixed 1M-token Seeds;
- completed the real two-RTX-3090 M5 R3 formal-source expansion with 218/240 accepted,
  all four family/language strata passing, a deterministic 160/160 selection, and zero
  frozen-source contamination; a post-gate exact-token audit keeps training blocked because
  the selected sources cannot satisfy both the 150K targeted-token budget and four-use cap;
- added the sharded M5 R3 formal-source expansion contract: 240 deterministic bilingual
  Config/Log tasks, isolated Qwen3-8B solve/compress generation, fail-closed shard lineage,
  and stable 56/24-per-family selection into 160 sources;
- added and completed a fail-closed private maintainer review for all 33 accepted R3 P2
  samples; all 33 judgments passed and authorized formal source expansion while the public
  summary remains free of prompts and distilled rationales and training stays blocked;
- implemented and ran the parent-bound M5 R3 P2 protocol, which retries only rejected P1
  solvers and prevents raw solver reasoning or alternative labels from entering the isolated
  compressor input; the real Qwen3-8B pilot accepted 33/40, passed both family/language gates
  and authorized content review plus formal source expansion without authorizing training;
- implemented and ran the bounded M5 R3 P1 solve/compress pipeline with four-way contamination
  checks, rule-trace control, strict lineage, and CPU contract Smoke; the real Qwen3-8B Pilot
  accepted 11/40 and was gate-rejected without authorizing source expansion or training;
- selected a versioned two-stage solve/compress Teacher-source strategy for the next bounded
  M5 R3 P1 contract, with deterministic rule traces retained as a control-only baseline;
- merged the full M5.2/R1/R2/R3-P0 evidence stack, then completed an independently versioned
  P0-R1 bilingual prompt-control diagnostic; its real Qwen3-8B run accepted 12/40 and was
  gate-rejected, keeping the 240-task expansion blocked and ending same-family prompt-only repair;
- completed the preregistered M5.2 dual-mode ablation with a 96/100 Qwen3-8B Teacher
  Pilot, three exact 1M-token mixtures, six Qwen3-0.6B runs, one real Exact Resume, and
  frozen two-mode evaluation; all arms passed the Non-thinking regression gate while the
  99% Thinking-format gate rejected every arm, leaving the formal ratio unselected;
- completed M4 with a real four-RTX-3090 Qwen3-8B BF16 FULL_SHARD run: a strict idle/NUMA
  memory probe, 50 optimizer steps, atomic Step 25 and Step 50 DCP checkpoints, fresh-process
  Step 25→50 resume, per-Rank peak-memory evidence, and an independently loaded Safetensors export;
- added a pinned Qwen3-8B artifact/data-view contract, strict four-GPU supervisor, versioned result
  schemas, public redacted evidence, and explicit limits excluding throughput, quality, eight-GPU,
  changed-World-Size, and ZeRO-3 claims;
- added atomic FSDP2 DCP sharded checkpoints with complete optimizer/scheduler/RNG/Sampler lineage,
  same-World-Size Exact Resume, integrity validation, retention, and CPU/Gloo bitwise recovery tests;
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
