# M1.1 Native CPU Trainer Smoke Report

## Scope

This report validates the native PyTorch Trainer correctness path added in M1.1. It uses
the dedicated test-scale TinyGPT fixture, not the 1,820,352-parameter default
TinyGPT-Debug model. It is not a throughput benchmark or a claim about model quality.

Checkpoint/Resume, CUDA BF16, and distributed training are not evaluated in this report;
M1 remains `IN_PROGRESS` until their separate milestone gates pass.

## Reproduction

```bash
.venv/bin/python scripts/run_m1_cpu_smoke.py \
  --config configs/pretrain/tinygpt_debug_cpu_smoke.yaml
```

The committed machine-readable evidence is
[`raw/m1_1_cpu_smoke.json`](raw/m1_1_cpu_smoke.json).

## Environment and lineage

| Field | Actual value |
| -- | -- |
| Source commit | `be0f7e339de8fb2a14bc5a4b1de594b352cb9a53` |
| Git dirty at execution | `false` |
| Resolved config SHA256 | `1dc537638ea9984943a4423c38400073735d331c39885fd2d657bd822160fbd7` |
| Python | 3.11.14 |
| PyTorch build | 2.7.1+cu118 |
| PyTorch CUDA runtime metadata | 11.8 |
| Execution device | CPU |
| Test fixture parameters | 11,360 |

The CUDA value describes the installed PyTorch build. This smoke executed on CPU and did
not allocate or validate a GPU.

## Observed result

| Measurement | Actual value |
| -- | --: |
| Optimizer steps | 30 |
| Micro steps | 30 |
| Predicted tokens | 1,320 |
| First 5-step mean loss | 3.1249433994 |
| Last 5-step mean loss | 1.4603532314 |
| Last/first loss ratio | 0.4673214983 |
| Status | Pass |

All 30 per-step Loss, LR, pre-clip Gradient Norm, clip decision, Epoch, Step, and cumulative
Token values are retained in the raw JSON. This single deterministic run demonstrates
that the test fixture can learn; it is not a statistical training comparison.

## Automated correctness and failure paths

The same PR runs the following CPU gates:

- Gradient Accumulation and Optimizer/Micro Step accounting.
- Warmup/Cosine LR values and decay/no-decay AdamW groups.
- Same-seed, same-environment metric and parameter equality.
- Loss reduction smoke from the tracked YAML configuration.
- Stable failures for vector Loss, NaN Loss, NaN Gradient, empty DataLoader, and unsupported
  CPU precision.
- Strict versioned Trainer State and Step Metrics JSON Schemas.

The complete CPU suite passed 60 tests with one GPU test deselected and 87.93% branch
coverage when this evidence was prepared.

## Not evaluated

- Atomic Checkpoint, retention, corruption detection, or Resume.
- SIGTERM/SIGKILL interruption recovery.
- RTX 3090 BF16 or V100 FP16.
- Tokens/s, Step Time, memory, scaling efficiency, or any other performance metric.
