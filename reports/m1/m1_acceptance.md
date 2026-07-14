# M1 Acceptance: Native Single-Device Training and Recovery

## Decision

M1 is accepted for the `v0.1.0-alpha.1` release when the PR containing this report is
merged. The evidence demonstrates the native PyTorch single-device correctness base,
full-state Checkpoint contract, CPU Exact Resume, RTX 3090 BF16 repeatability, and real
process-interruption recovery. It does not make a throughput or model-quality claim.

## Evidence chain

| Gate | Result | Evidence |
| -- | -- | -- |
| TinyGPT-Debug model | Pass | 1,820,352 instantiated parameters; CPU forward/backward |
| Native CPU Trainer | Pass | 30 Optimizer Steps and observed Loss decrease |
| Atomic Checkpoint | Pass | Full state, SHA256, `COMMITTED`, `LATEST`, Retention and corruption rejection |
| CPU Exact Resume | Pass | Batch/Loss/LR/parameters/Optimizer/Scheduler/RNG/State bit-for-bit equal |
| Warm and Transfer | Pass | Explicit reset/partial-load semantics and incompatible-key reporting |
| RTX 3090 BF16 | Pass | Two 40-Step repeat baselines on the same idle GPU |
| SIGTERM | Pass | Interruption point committed at Step 7; resumed with Step 8 |
| SIGKILL | Pass | Actual return code -9; resumed from Step 5 and re-computed two lost Steps |
| Failure paths | Pass | NaN/Inf, OOM mapping, disk-write failure, corruption, config/data/environment/World Size drift |
| Public CLI | Pass | `tinyllm train` writes the Run layout and supports Exact/Warm/Transfer source Runs |
| Quality gate | Pass | 91 CPU tests, 2 GPU tests, 85.42% CPU-testable branch coverage |

Detailed prerequisite reports:

- [TinyGPT foundation](foundation_report.md)
- [Native CPU Trainer](native_cpu_trainer_report.md)
- [Atomic Checkpoint](atomic_checkpoint_report.md)
- [CPU Exact Resume](exact_resume_report.md)
- [M1.4 raw sanitized result](raw/m1_4_gpu_interruption_smoke.json)

## M1.4 configuration and environment

The formal process-interruption result came from clean source commit
`3300ea6a58c4f558fc07c74f5d4a4bb0b85b7812` and resolved config SHA256
`7c11f1edb0cc79e2b879cc4ad88e2470ba56b13a9bf5297e17f01ac1b2696808`.

| Field | Actual value |
| -- | -- |
| GPU | NVIDIA GeForce RTX 3090 24,576 MiB, physical index 5 |
| Preflight | 4 MiB used, 0% utilization, 29°C |
| Python / PyTorch | 3.11.14 / 2.7.1+cu118 |
| CUDA runtime in PyTorch | 11.8 |
| Precision | BF16 Autocast, FP32 master weights, no GradScaler, TF32 enabled |
| Model | TinyGPT-Debug, 1,820,352 trainable parameters |
| Training | 40 Optimizer Steps, Micro Batch 8, Sequence Length 128 |
| Checkpoint interval | 5 Optimizer Steps |
| Signal injection | after Step 7 |

Doctor's required host, Python, PyTorch/CUDA, GPU inventory, storage, Git and topology
checks passed immediately before the run. It returned `warn`, not `pass`, because other
shared-server GPUs were busy, one unrelated GPU was hot, P2P/NVLink were unavailable,
and the active environment did not expose `nccl-tests`. Those warnings did not affect the
selected idle single GPU and are not reinterpreted as success.

## Baseline before tolerance

Two uninterrupted runs used the same config, Seed, physical GPU and software environment.
Both produced 40 metrics and the same final model hash:

`b5b82eda5cca984ead2de6f3ee870a0a95af644b34d32a64e231e36502379304`

Observed maximum absolute differences were 0 for both per-Step Loss and final parameters.
The pre-declared rule therefore selected a Loss absolute tolerance of `1e-6` and parameter
absolute tolerance of `1e-7` before either interrupted comparison began. LR, Step,
Trainer State and Sampler State still required exact equality.

## Real process recovery

SIGTERM was sent to the live worker after its Step 7 metric. The Handler set a flag; after
the optimizer boundary the worker atomically committed a pinned Step 7 checkpoint and
returned 143. The next process restored Step 7 and first executed Step 8. Its Loss, LR,
final parameters, Trainer State and Sampler State matched the uninterrupted control with
zero observed difference.

SIGKILL was sent to a separate live worker after Step 7 and produced return code -9. As
expected, no Handler ran. The latest committed periodic point was Step 5, so the restarted
process first executed Step 6 and recomputed Steps 6–7. This is a two-Step rollback, not a
claim to recover the uncommitted Kill instant. The canonical Step 1–40 result again matched
the uninterrupted control with zero observed difference.

The private evidence bundle retains four Run directories, full `events.jsonl` and
`metrics.jsonl`, and 13 integrity-checked committed Checkpoints. Its public group label is
`m1-4-20260714T091528Z-7c11f1ed-1d4e`; the private absolute path and host identity are not
published.

## Failure-path accounting

- Non-finite Loss and Gradient fail before an Optimizer Step.
- Simulated accelerator OOM maps to `ACCELERATOR_OUT_OF_MEMORY` and clears partial grads.
- Simulated Checkpoint write failure leaves no published directory or `LATEST` pointer.
- Missing markers and size/SHA256 drift fail closed.
- Exact Resume rejects config, precision, data, Git, environment and World Size drift.
- SIGKILL only promises recovery from the latest committed point and reports rollback.

## Explicit limitations and next dependency

M1 does not evaluate throughput, DDP/FSDP2/ZeRO-3, distributed recovery, real datasets,
model quality, V100 FP16 + GradScaler, or cross-GPU/cross-version reproducibility. M2 may
now build versioned licensed data and freeze evaluation. M3 remains blocked on both M1 and
M2 and cannot use this single-GPU Smoke as a DDP scaling result.
