# M3.2 DDP Checkpoint and Recovery Report

## Decision

M3.2 passes when this PR is merged. Real two-GPU RTX 3090 BF16 runs demonstrate atomic complete
DDP Checkpoints, same-World-Size Exact Resume, canonical metric continuity after coordinated
interruption, and recovery after a forced nonzero Rank exit.

M3 as a whole remains `IN_PROGRESS`. This report does not accept throughput, scaling efficiency,
Profiler evidence, changed-World-Size Resume, FSDP2, ZeRO-3, elastic membership, or node-failure
tolerance. Issues #14 and #15 remain mandatory before M3 can be marked complete.

## Reproduction

After selecting two idle same-NUMA GPUs with `tinyllm doctor --distributed --json`, the accepted
gate used:

```bash
.venv/bin/python scripts/run_m3_ddp_recovery_smoke.py \
  --config configs/pretrain/tinygpt_debug_ddp_recovery_2gpu_bf16_smoke.yaml \
  --output-root "$TINYLLM_ARTIFACT_ROOT/runs" \
  --evidence-dir "$PRIVATE_EVIDENCE_ROOT/m3-ddp-recovery" \
  --gpu-indices 5,6
```

The supervisor ran two uninterrupted baselines before freezing the recovery tolerance. It then
ran a coordinated interruption at Step 6 and a separate Rank 1 exit immediately after the Step 8
Checkpoint, restarting both Runs through the same Exact Resume interface.

Path-free machine-readable facts are committed in
[`raw/ddp_recovery.json`](raw/ddp_recovery.json). Full Doctor output, preflight snapshots,
torchrun stdout/stderr, Run directories, model/optimizer state, and failure diagnostics remain in
the private Artifact Store. Public evidence records SHA256 digests for the private summary,
tolerance, Doctor report, and accepted Checkpoint Manifests/commit markers.

## Environment and lineage

| Field | Actual value |
| -- | -- |
| Source commit | `03e1cf3b4da11470ff8fa7c74f9a5713105ef823` |
| Git dirty at execution | `false` |
| Python | 3.11.14 |
| PyTorch | 2.7.1+cu118 |
| CUDA runtime in PyTorch | 11.8 |
| NVIDIA driver | 535.261.03 |
| NCCL | 2.21.5 |
| GPU | 2 × RTX 3090 24 GiB, physical indices 5 and 6 |
| Topology | Same NUMA node, `PXB` path |
| Precision | BF16, TF32 enabled, no GradScaler |
| Model | TinyGPT-Debug, 1,820,352 parameters |

The Doctor result was `warn`: other GPUs were busy, P2P reads reported `CNS`, NVLink was inactive,
and standalone nccl-tests binaries were unavailable. Every accepted phase independently repeated
the stricter launch preflight. Both selected GPUs had 1 MiB reported usage and 0% utilization;
recorded preflight temperatures ranged from 29°C to 46°C.

These warnings and topology facts bound the claim: this is real single-host DDP correctness and
recovery evidence, not a communication-performance benchmark.

## Observed results

| Measurement | Baseline repeats | Coordinated interruption | Rank failure |
| -- | --: | --: | --: |
| World Size | 2 | 2 | 2 |
| Global Batch | 8 | 8 | 8 |
| Final Optimizer Step | 12 | 12 | 12 |
| Interruption/failure boundary | — | Step 6 | Step 8 |
| Forced exit | — | Coordinated worker exit | Rank 1, code 17 |
| Exact Resume source | — | Step 6 | Step 8 |
| Canonical metric rows | 12 each | 12 | 12 |
| Missing/repeated Step | 0 | 0 | 0 |
| Maximum Loss difference from baseline | 0 | 0 | 0 |
| Final parameter SHA256 equals baseline | Yes | Yes | Yes |
| Checkpoint integrity | Pass | Pass | Pass |
| Final status | Pass | Pass | Pass |

The two uninterrupted repeats had a maximum per-Step Loss difference of `0.0`, so the
predeclared rule `max(1e-6, 2 × baseline difference)` froze the recovery tolerance at `1e-6`
before any interrupted comparison. Both recovered Loss series differed from the baseline by
`0.0`; all accepted final model states shared SHA256
`d0be4763d8682db8eaa634d01e5c7794d09a98341270a53377a2527d3e3c4c16`.

The recorded 48.924-second supervisor duration covers six torchrun invocations and failure
orchestration. It is retained as operational evidence and is not reported as training throughput.

## Checkpoint and failure evidence

Each accepted Checkpoint contains one shared full training state plus contiguous `rank-00000.pt`
and `rank-00001.pt` files. The Manifest covers model, AdamW, Scheduler, Trainer progress, resolved
config, environment, and every Rank-local Sampler and RNG state. Publication uses a temporary
directory, file hashes, a final `COMMITTED` marker, atomic rename, and atomic `LATEST` update.

The coordinated Run preserved its Run ID across restart and emitted only Steps 7–12 after loading
Step 6. The forced-failure Run committed Step 8, recorded Rank 1 and exit code 17, and then let
torchrun terminate the remaining process group. Its restart selected the valid Step 8 Checkpoint
and emitted only Steps 9–12. Both final `metrics.jsonl` files contain exactly Steps 1–12.

Automated tests also reject wrong World Size, partial or corrupt Rank state, invalid Sampler cursor,
environment/config drift, duplicate Checkpoint destinations, non-pristine restore targets, and
failed optimizer-state application. CPU/Gloo integration executes both failure paths with real
two-process torchrun launches.

## Remaining M3 gates

M3.2 closes only Issue #13. The remaining work is:

1. #14: benchmark harness with warmup, measurement windows, three repetitions, memory, data wait,
   communication, and Profiler evidence.
2. #15: controlled 1/2/4/8 Strong and Weak Scaling plus same-NUMA/cross-NUMA comparison.

No M3.2 recovery duration, checkpoint write time, Tokens/s, or scaling claim is promoted into a
benchmark result.
