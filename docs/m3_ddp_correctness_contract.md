# M3.1 DDP Correctness Contract

## 1. Scope

M3.1 proves native PyTorch DDP launch and data-parallel semantics before any scaling claim. It
covers torchrun initialization/teardown, deterministic parameter initialization,
`DistributedSampler` partitioning, Global Batch calculation, reduced Loss, synchronized final
parameters, and rank-zero-only durable logging.

M3.1 does not implement or claim DDP Checkpoint/Resume, Rank Failure recovery, performance,
scaling efficiency, Profiler traces, FSDP2, or ZeRO-3. Those remain #13, #14, #15, and M4.

## 2. Minimal interface

Formal correctness runs use a strict YAML configuration with an explicit `distributed` block:

```yaml
distributed:
  strategy: ddp
  backend: nccl
  world_size: 2
  timeout_seconds: 120
  broadcast_buffers: false
  find_unused_parameters: false
```

The worker is launched only through torchrun:

```bash
CUDA_VISIBLE_DEVICES="$PHYSICAL_GPUS" \
  .venv/bin/torchrun \
  --standalone \
  --nproc-per-node "$WORLD_SIZE" \
  -m tinyllm.training.ddp_worker \
  --config "$CONFIG" \
  --output-root "$TINYLLM_ARTIFACT_ROOT/runs"
```

Formal GPU evidence uses `scripts/run_m3_ddp_smoke.py`, which performs the same launch after
rejecting dirty Git state and any selected GPU above the fixed memory, utilization, or temperature
threshold. It retains private command, preflight, stdout, stderr, and summary files. There is no
busy-GPU override for this gate, and the explicit physical GPU order defines local Rank order.

`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and `LOCAL_WORLD_SIZE` must all be present and valid. The
resolved World Size must equal the YAML value; the worker never guesses or silently overrides it.
The existing single-device `tinyllm train` path rejects a DDP config until #13 supplies the full
DDP Checkpoint/Resume contract.

For backward lineage compatibility, an omitted `distributed` block resolves internally to the
single-process defaults but remains omitted from canonical JSON. This preserves the published M1
configuration hashes and Exact Resume compatibility. Any explicit DDP block is included in the
canonical snapshot and hash.

## 3. Fixed correctness configurations

| Run | Backend | World Size | Micro Batch | Accumulation | Global Batch | Steps |
| -- | -- | --: | --: | --: | --: | --: |
| CPU integration | Gloo | 2 | 4 | 1 | 8 | 2 |
| RTX 3090 single-rank DDP | NCCL | 1 | 4 | 2 | 8 | 8 |
| RTX 3090 two-rank DDP | NCCL | 2 | 4 | 1 | 8 | 8 |

The fixed GPU configs keep Global Batch at 8 while changing World Size. These are bounded
correctness Smokes, not Strong Scaling measurements. Formal 1/2/4/8 Benchmark configs will use
the separate #14/#15 protocol with warmup, measurement windows, repetition, and Profiler evidence.

Global Batch is resolved once from:

```text
micro_batch_size × gradient_accumulation_steps × world_size
```

Per-rank token counts are converted to global token counts once when rank zero creates the durable
metric. No downstream logger multiplies them again.

## 4. Deterministic initialization and synchronization

Every rank receives the same Python, NumPy, PyTorch, and CUDA Seed before model construction. Each
rank hashes the ordered parameter names, dtypes, shapes, and raw bytes before DDP wrapping. The run
fails if any initialization hash differs.

After the last optimizer step, every rank hashes the underlying unwrapped model again. The run
fails if any final hash differs. Matching hashes prove rank synchronization for this run; they do
not prove cross-World-Size numerical identity or long-training reproducibility.

## 5. DistributedSampler contract

M3.1 uses PyTorch `DistributedSampler` with the fixed Run Seed, `shuffle=true`, `drop_last=true`,
and `epoch=0`. The YAML validator requires:

- dataset size divisible by World Size;
- samples per rank divisible by Micro Batch;
- all configured optimizer steps fit inside one sampler epoch.

Before training, every rank gathers its complete ordered partition. The validator rejects empty,
duplicate, overlapping, out-of-range, or incomplete partitions. Public evidence stores only each
rank's sample count and ordered-ID SHA256, not the ID list.

Multi-epoch `set_epoch`, stateful DDP Sampler Resume, and changed-World-Size behavior belong to
#13 and cannot be inferred from this one-epoch gate.

## 6. Loss and metric reduction

DDP averages gradients across ranks. At every optimizer step, the worker also:

1. gathers the rank-local accumulated Loss values;
2. independently performs a float64 `all_reduce(SUM) / world_size`;
3. compares the reduced value with the arithmetic mean of gathered local values;
4. gathers post-sync Gradient Norm values and records their maximum cross-rank difference;
5. lets only rank zero append one validated metric row.

The fixed pass thresholds are `1e-12` absolute difference for float64 Loss reduction and `1e-6`
for the cross-rank Gradient Norm. The validated correctness artifact fails closed when either
threshold is exceeded; thresholds cannot be changed from YAML or CLI for M3.1.

The correctness summary records the maximum observed reduction difference. It is not throughput,
communication-time, or scaling evidence. Gradient accumulation currently synchronizes every
micro-step; `no_sync()` optimization is deferred to the benchmark batch so performance changes
cannot precede this correctness baseline.

## 7. Artifact and logging boundary

Only global rank zero may create or update the shared Run directory:

```text
run.json
events.jsonl
metrics.jsonl
config.original.yaml
config.resolved.json
environment.json
hardware.json
correctness.json
checkpoints/
evaluations/
exports/
```

`metrics.jsonl` must contain exactly one row per optimizer step, not one row per rank. Environment
and hardware files contain a gathered record for every rank. M3.1 creates an empty Checkpoint
directory and records `checkpoint_status=not_evaluated_m3_1`; it does not publish a partial model
state as a training Checkpoint.

Private torchrun/NCCL stderr is retained with the supervising Smoke evidence. Public reports may
contain config and result hashes, GPU indices, counts, reduction differences, and failure types,
but not usernames, hostnames, absolute paths, or unrelated process details.

## 8. Failure paths

The worker fails closed when:

- it is not launched by torchrun;
- rank variables are missing, non-integer, or out of range;
- torchrun World Size differs from YAML;
- NCCL/BF16/CUDA is unavailable;
- sampler partitions overlap or do not cover the dataset;
- initial or final parameter hashes differ across ranks;
- optimizer-step counts diverge;
- a shared artifact path already exists.

M3.1 does not convert NCCL timeout or one-rank death into a resumable Run. Real Rank Failure and
recovery are mandatory #13 evidence.

## 9. Acceptance

M3.1 is complete only after:

1. strict Schema snapshots and CPU unit tests pass;
2. a two-process Gloo integration test passes in CI;
3. malformed launch and sampler-overlap paths fail closed;
4. an idle RTX 3090 passes one-rank NCCL/BF16 torchrun;
5. two idle RTX 3090s pass two-rank NCCL/BF16 torchrun;
6. both GPU Runs prove initialization identity, complete sampler partitioning, exact durable metric
   cardinality, reduced Loss consistency, and final cross-rank parameter identity;
7. private raw logs and a redacted public report are retained;
8. documentation and Issue #12 are synchronized and the PR is merged.
