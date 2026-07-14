# M1.2 Atomic Checkpoint Smoke Report

## Scope

This report validates full-state checkpoint publication, integrity checks, stateful
sampler capture, `LATEST`, retention, and corruption detection on CPU. It does not load
state into a new Trainer or claim Exact Resume equivalence; that is the separate M1.3
gate.

The smoke uses the 11,360-parameter test fixture and ephemeral checkpoint files. Model
weights and pickle-based training state are not committed to the public repository.

## Reproduction

```bash
.venv/bin/python scripts/run_m1_checkpoint_smoke.py \
  --config configs/pretrain/tinygpt_debug_cpu_smoke.yaml
```

Machine-readable evidence:
[`raw/m1_2_checkpoint_smoke.json`](raw/m1_2_checkpoint_smoke.json).

## Environment and lineage

| Field | Actual value |
| -- | -- |
| Source commit | `aa9b29b5dd80d679f676f54c89b9275951ff561e` |
| Git dirty at execution | `false` |
| Run ID | `20260714T082527Z-m1-checkpoint-smoke-1dc53763-db92` |
| Resolved config SHA256 | `1dc537638ea9984943a4423c38400073735d331c39885fd2d657bd822160fbd7` |
| Dataset version | `toy-checkpoint-1dc53763` |
| Python | 3.11.14 |
| PyTorch build | 2.7.1+cu118 |
| Execution device | CPU |
| Strategy / World Size | single / 1 |

## Published Step 4 manifest

| File | Bytes | SHA256 |
| -- | --: | -- |
| `training_state.pt` | 166,409 | `9918e9d348c0585c2cc85277ad07cdb3bfd04472f454c4d9128a660b7d1aeb52` |
| `config.resolved.json` | 921 | `56c4b1db8517a8bf4eb9483b29f7defb7004d7f61ed7aa2200e18d274731e71e` |
| `environment.json` | 97 | `5b4cd06e3fd9c5a1b4318cded6b82dfec29929954fc7c1fa8efaf17004213828` |

The completion marker binds the manifest SHA256. The manifest binds the Run/config/data/
Git identity, Global/Micro Step 4, Epoch 0, and every payload size/hash. The training
payload accounts for model, optimizer, scheduler, GradScaler (explicitly not applicable),
Python/NumPy/PyTorch/CUDA RNG (CUDA explicitly not applicable), sampler, config, and
environment state.

## Observed retention and failure behavior

Four checkpoints were saved. Step 2 was pinned as an interruption point and ordinary
retention was set to two:

```text
retained: checkpoint-step-00000002  (pinned)
          checkpoint-step-00000003
          checkpoint-step-00000004  (LATEST)
removed:  checkpoint-step-00000001
```

The saved Step 4 sampler state was `epoch=0`, `cursor=16`, `num_samples=32`; Trainer state
was Global Step 4, Micro Step 4, and 176 predicted tokens. Deliberately appending bytes to
`training_state.pt` caused validation to fail with `CHECKPOINT_CORRUPT`.

Automated tests also verify:

- a simulated `torch.save`/disk failure removes the temporary directory and never writes
  `LATEST`;
- missing `COMMITTED`, unsafe checkpoint IDs, duplicate destinations, and size/hash drift
  fail closed;
- even after file hashes and `COMMITTED` are recomputed, config or payload lineage drift
  is rejected;
- the stateful sampler restores the next Batch and rejects dataset-cardinality mismatch.

The complete CPU suite passed 69 tests with one GPU test deselected and 87.41% branch
coverage when this evidence was prepared.

## Not evaluated

- Applying the payload to a new model, optimizer, scheduler, RNG, and sampler.
- Batch/Loss/LR/parameter equality after Resume.
- Exact, Warm, or Transfer Resume compatibility decisions.
- SIGTERM/SIGKILL recovery, RTX 3090 BF16, V100 FP16, or distributed checkpoints.
