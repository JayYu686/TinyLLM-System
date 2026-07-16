# M3.1 DDP Correctness Report

## Decision

M3.1 passes when this PR is merged. Real one- and two-GPU RTX 3090 runs demonstrate bounded
native PyTorch DDP launch, deterministic parameter initialization, complete non-overlapping
`DistributedSampler` partitions, exact Global Batch accounting, consistent reduced Loss,
synchronized final parameters, and rank-zero-only durable artifacts.

M3 as a whole remains `IN_PROGRESS`. This report does not accept DDP Checkpoint/Resume, Rank
Failure recovery, performance, scaling, or the TinyGPT-Target-120M benchmark.

## Reproduction

After selecting idle physical GPUs with `tinyllm doctor --distributed --json`, the accepted runs
used:

```bash
.venv/bin/python scripts/run_m3_ddp_smoke.py \
  --config configs/pretrain/tinygpt_debug_ddp_1gpu_bf16_smoke.yaml \
  --output-root "$TINYLLM_ARTIFACT_ROOT/runs" \
  --evidence-dir "$PRIVATE_EVIDENCE_ROOT/1gpu" \
  --gpu-indices 4

.venv/bin/python scripts/run_m3_ddp_smoke.py \
  --config configs/pretrain/tinygpt_debug_ddp_2gpu_bf16_smoke.yaml \
  --output-root "$TINYLLM_ARTIFACT_ROOT/runs" \
  --evidence-dir "$PRIVATE_EVIDENCE_ROOT/2gpu" \
  --gpu-indices 4,5
```

The redacted machine-readable facts are in
[`raw/ddp_correctness.json`](raw/ddp_correctness.json). Full Doctor output, torchrun stdout/stderr,
preflight snapshots, Run directories, metrics, and environments remain private. The public JSON
records SHA256 digests for the private Doctor, stdout/stderr, supervisor summaries, and Run
correctness artifacts.

## Environment and lineage

| Field | Actual value |
| -- | -- |
| Source commit | `9bd5e06caa1e6b217c531fdeabd826e2e148fb3c` |
| Git dirty at execution | `false` |
| Python | 3.11.14 |
| PyTorch | 2.7.1+cu118 |
| CUDA runtime in PyTorch | 11.8 |
| NVIDIA driver | 535.261.03 |
| NCCL | 2.21.5 |
| GPU | RTX 3090 24 GiB, compute capability 8.6 |
| Precision | BF16, TF32 enabled, no GradScaler |

The Doctor report was `warn`, not `pass`: unrelated GPUs were busy, GPU 2 was hot, P2P reads
reported `CNS`, NVLink was inactive, and standalone nccl-tests binaries were unavailable in the
active environment. The explicitly selected GPUs passed the stricter launch preflight. The
two-GPU pair was GPU 4 on NUMA 0 and GPU 5 on NUMA 1 with a `SYS` topology path. Its successful
run is correctness evidence across that path, not a communication-performance result.

## Observed results

| Measurement | One GPU | Two GPUs |
| -- | --: | --: |
| Physical GPU indices | 4 | 4, 5 |
| World Size | 1 | 2 |
| Micro Batch / Rank | 4 | 4 |
| Gradient Accumulation | 2 | 1 |
| Global Batch | 8 | 8 |
| Optimizer Steps | 8 | 8 |
| Durable metric rows | 8 | 8 |
| Durable writer Rank | 0 | 0 |
| Sampler union / dataset | 256 / 256 | 256 / 256 |
| Per-Rank sample counts | 256 | 128, 128 |
| Maximum Loss-reduction difference | 0 | 0 |
| Maximum cross-Rank Gradient Norm difference | 0 | 0 |
| stderr bytes | 0 | 0 |
| Status | Pass | Pass |

Both runs instantiated the 1,820,352-parameter TinyGPT-Debug model and produced the same initial
parameter SHA256. Each run also required all Rank-local final parameter hashes to be identical.
The one- and two-GPU final hashes differ from each other, which is allowed: M3.1 does not claim
cross-World-Size bitwise equivalence under BF16 and different accumulation/reduction order.

The recorded orchestration durations include Python and torchrun startup and are deliberately not
reported as throughput or scaling measurements.

## Automated correctness and failure paths

The same change includes:

- a real two-process CPU/Gloo integration run;
- malformed or mismatched torchrun World Size rejection before artifact creation;
- missing/invalid Rank-coordinate rejection;
- overlapping, duplicate, incomplete, empty, and out-of-range Sampler partition rejection;
- fixed `1e-12` Loss-reduction and `1e-6` Gradient Norm thresholds that cannot be loosened by YAML;
- strict DDP result, summary, and partition JSON Schema snapshots;
- a single-device CLI refusal for DDP YAML until the full #13 Checkpoint/Resume contract exists;
- backward-compatible canonical serialization preserving published M1 config hashes.

An initial diagnostic execution passed numerically but emitted PyTorch device-binding warnings.
The implementation was changed to bind NCCL initialization and barriers to each local CUDA device,
then both accepted runs were repeated from the new clean commit with empty stderr. The earlier
private runs were retained and excluded from this report rather than deleted.

## Remaining M3 gates

M3.1 closes only Issue #12. The following remain mandatory:

1. #13: complete DDP Checkpoint, Exact Resume constraints, real Rank Failure, and recovery.
2. #14: benchmark harness with warmup, measurement windows, three repetitions, memory, data wait,
   communication, and Profiler evidence.
3. #15: controlled 1/2/4 Strong and Weak Scaling; 8-GPU and cross-NUMA evidence is optional
   under ADR-0004 and cannot be claimed unless actually measured.

No Checkpoint was written by these runs; the empty directory and
`checkpoint_status=not_evaluated_m3_1` prevent a partial state from being mislabeled as resumable.
