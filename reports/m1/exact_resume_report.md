# M1.3 CPU Resume Semantics Report

## Scope

This report validates explicit Exact, Warm, and Transfer restore semantics on the CPU
test fixture. Exact Resume is compared with an uninterrupted control in the same software
environment. It does not claim process-signal recovery, CUDA BF16 equivalence, or
distributed checkpoint recovery.

## Reproduction

```bash
.venv/bin/python scripts/run_m1_resume_smoke.py \
  --config configs/pretrain/tinygpt_debug_cpu_smoke.yaml
```

Machine-readable evidence:
[`raw/m1_3_resume_smoke.json`](raw/m1_3_resume_smoke.json).

## Environment and lineage

| Field | Actual value |
| -- | -- |
| Source commit | `3a077aeae7c661c7b77efe11d9256cd0a0dfe32e` |
| Git dirty at execution | `false` |
| Run ID | `20260714T084736Z-m1-exact-resume-smoke-1dc53763-9b38` |
| Resolved config SHA256 | `1dc537638ea9984943a4423c38400073735d331c39885fd2d657bd822160fbd7` |
| Python | 3.11.14 |
| PyTorch build | 2.7.1+cu118 |
| CUDA runtime in build | 11.8 |
| Execution device | CPU / FP32 |

## Exact Resume result

The control trained without process interruption from Step 0 to Step 30. The resumed
path saved an interruption-pinned checkpoint after Step 10, constructed a new Trainer,
restored the checkpoint, and continued with Step 11 through Step 30.

| Check | Actual result |
| -- | -- |
| First resumed Optimizer Step | 11 |
| Resumed Optimizer Steps | 20 |
| Sampler boundary | Bit-for-bit equal |
| Next Batch indices | 8, 9, 10, 11 |
| Next Batch SHA256 | `66baa66fe7bbef392ab80deb9438c9cf15f16e2330fb16fc83ee839004c9bba4` |
| RNG at restore | Python/NumPy/PyTorch/CUDA state equal |
| Loss, LR, Gradient Norm, Step metrics | Bit-for-bit equal |
| Final Trainer state | Bit-for-bit equal |
| Final model parameters | Bit-for-bit equal |
| Final Optimizer state | Bit-for-bit equal |
| Final Scheduler state | Bit-for-bit equal |
| Control/resumed model SHA256 | `50d4056e082dd1f444b59c6b2de251871e1845a47e755ca873651d277032fca4` |

The resumed path began at Step 11 and emitted exactly 20 Optimizer-Step records, so it
did not repeat Step 10.

## Warm and Transfer boundaries

Warm Resume loaded all 10 model-state keys and reproduced the source weights exactly,
while the target remained at Global Step 0 with fresh Optimizer, Scheduler, Sampler, and
RNG state.

Transfer Resume into a model with a different vocabulary loaded 8 compatible keys and
reported `token_embeddings.weight` and `lm_head.weight` as incompatible rather than
silently reshaping them. The target also remained at Global Step 0.

## Observed failure matrix

| Probe | Actual error | Reason |
| -- | -- | -- |
| Learning-rate drift | `CHECKPOINT_INCOMPATIBLE` | `config:training.learning_rate` |
| Dataset version drift | `CHECKPOINT_INCOMPATIBLE` | `lineage:dataset_version` |
| World Size drift | `CHECKPOINT_INCOMPATIBLE` | `lineage:world_size` |
| Deliberate payload hash corruption | `CHECKPOINT_CORRUPT` | Integrity validation |

Automated tests additionally reject precision, Git Commit, software environment, and
non-pristine-target drift. After the Step 10 payload was deliberately corrupted, automatic
selection reported it as skipped and selected the still-valid Step 5 checkpoint. Explicit
selection of the corrupt Step 10 checkpoint failed instead of falling back.

At evidence preparation time, the complete CPU suite passed 83 tests with one GPU test
deselected and 86.21% branch coverage. Dependency auditing found no known vulnerability
in auditable packages; the pinned CUDA PyTorch wheel is not present on the configured
PyPI index and remains covered by the repository's documented audit limitation.

## Not evaluated

- Real SIGTERM or SIGKILL process interruption and automatic relaunch.
- RTX 3090 BF16 repeat-baseline tolerance and Resume.
- V100 FP16 + GradScaler.
- DDP, FSDP2, or ZeRO-3 checkpoint recovery.
