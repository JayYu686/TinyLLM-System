# M3.2 DDP Checkpoint, Resume, and Rank-Failure Contract

## 1. Scope

M3.2 extends the accepted M3.1 DDP correctness path with complete same-World-Size Exact Resume.
It covers atomic DDP Checkpoint publication, per-Rank state, coordinated interruption, nonzero
Rank exit diagnostics, rollback to the latest committed optimizer boundary, and canonical metric
continuity after restart.

M3.2 does not measure throughput, scaling efficiency, communication time, peak memory, or
Profiler data. It does not implement changed-World-Size Resume, FSDP2/DCP, ZeRO-3, elastic
membership, node failure, or multi-node training. Those boundaries cannot be inferred from a
successful same-host DDP restart.

## 2. Public launch boundary

Formal M3.2 runs use a strict DDP YAML and an explicit supervisor. The supervisor selects physical
GPUs, rejects a dirty Git tree or busy/hot devices, launches torchrun, retains stdout/stderr, and
reuses the same Run ID and Artifact directory for a restart.

The worker modes are deliberately explicit:

```text
fresh
resume-exact
coordinated-stop-after-step N
fail-rank R after committed step N
```

Failure injection arguments are private Smoke interfaces, not general training configuration and
not permitted in benchmark YAML. Formal configs keep `checkpoint.resume: auto`; the supervisor
still chooses whether a launch is fresh or Exact Resume so a missing Checkpoint cannot silently
turn an intended restart into a new Run.

## 3. DDP Checkpoint layout

Single-node DDP uses a full Rank-zero training state plus one state file per Rank:

```text
checkpoint-step-00000008/
├── training_state.pt
├── rank-00000.pt
├── rank-00001.pt
├── config.resolved.json
├── environment.json
├── manifest.json
└── COMMITTED
```

`training_state.pt` contains:

- unwrapped model state;
- AdamW state;
- Scheduler state;
- explicit GradScaler not-applicable marker under BF16/FP32;
- global Trainer State;
- resolved config and hash;
- dataset version, Git commit, gathered environment, strategy, and World Size.

Each `rank-xxxxx.pt` contains that Rank's identity, World Size, Trainer State, stateful distributed
Sampler state, and Python/NumPy/PyTorch/local-CUDA RNG state. The Manifest must list exactly
`world_size` contiguous Rank files. A missing, extra, renamed, corrupt, duplicated, or structurally
invalid Rank state rejects the whole Checkpoint.

## 4. Atomic distributed publication

Checkpointing occurs only after a successful optimizer step:

1. all Ranks expose the same Global/Micro Step;
2. every Rank captures local RNG and Sampler state;
3. small Rank-local state is gathered to Rank 0;
4. Rank 0 writes all files into a unique temporary directory and fsyncs them;
5. Rank 0 writes the Manifest and final `COMMITTED` marker;
6. Rank 0 atomically renames the directory, validates every size/SHA256, and atomically updates
   `LATEST`;
7. success or a sanitized error is broadcast before any Rank continues.

Only Rank 0 mutates the shared Artifact Store. A failure before `COMMITTED` never advances
`LATEST`. Ordinary retention keeps the latest two points; interruption and final points are
pinned. A full DDP Checkpoint is not a model export and must not be labeled Safetensors.

## 5. Stateful distributed Sampler

M3.2 replaces the stateless M3.1 `DistributedSampler` training iterator with a deterministic
stateful equivalent. Its state binds:

- dataset cardinality;
- World Size and Rank;
- Seed, shuffle, and drop-last policy;
- Epoch;
- cursor within the Rank-local ordered partition.

The permutation is derived only from `seed + epoch`. Resume restores the next local sample, not
the last consumed sample. Dataset size, Rank, World Size, policy, or cursor mismatch fails before
model/optimizer mutation. M3.2 formal runs stay within a bounded dataset but tests cover Epoch
rollover and state restoration.

## 6. Exact Resume preflight and application

Exact Resume requires all of the following to match:

- Run ID and resolved config hash;
- dataset version and Git commit;
- `strategy=ddp` and the same World Size;
- gathered software/hardware environment;
- complete model keys and shapes;
- Trainer Step and every Rank-local Sampler/RNG state.

Every Rank validates the committed Checkpoint before state application. State is then restored in
this order: unwrapped model, Optimizer, Scheduler, Sampler, Trainer progress, process-group
barrier, and finally local RNG. Any compatibility failure uses the stable Checkpoint/Resume error
boundary and exit code 5 at the supervising CLI layer.

Changed-World-Size Exact Resume always fails. Warm/Transfer Resume is not added to the M3.2 DDP
worker; M3.2 must not weaken the Exact contract merely to load weights.

## 7. Metric and Run continuity

Rank 0 remains the only durable metric writer. Before restart it validates existing JSONL and
atomically removes metric rows after the selected Checkpoint Step. Rows at or before the Step are
retained, and new rows must begin at `checkpoint_step + 1`. Duplicate or non-monotonic optimizer
steps fail closed.

The append-only event log records launch attempts, Checkpoint commits, interruption intent,
Resume source, discarded uncommitted metric rows, Rank-failure injection, and terminal status.
The same Run ID is preserved across attempts. A Run may move from `running` to `resumable` and
back to `running`; it becomes `succeeded` only after the final pinned Checkpoint validates.

## 8. Failure scenarios

### Coordinated interruption

After the configured safe optimizer boundary, all Ranks publish an `interruption`-pinned
Checkpoint, synchronize, and exit with worker code 143. A new torchrun invocation resumes the same
Run from that exact Step. The final canonical metrics, model, Optimizer, Scheduler, Trainer, and
Sampler states are compared with an uninterrupted baseline.

### Nonzero Rank exit

Immediately after a periodic Checkpoint is committed, Rank 0 records the armed failure and all
Ranks synchronize. The selected nonzero Rank then exits with code 17 without cleanup. The torchrun
agent must terminate the remaining process group and return nonzero. Private stderr and exit
diagnostics are retained. Restart uses the last committed point; no partial directory may be
selected.

The injected Rank exit is a controlled single-host process failure, not evidence of transparent
elastic recovery or node-failure tolerance.

## 9. Fixed Smoke configurations

| Run | Backend | World Size | Precision | Steps | Save interval | Global Batch |
| -- | -- | --: | -- | --: | --: | --: |
| CPU integration | Gloo | 2 | FP32 | 6 | 2 | 8 |
| RTX 3090 formal | NCCL | 2 | BF16 | 12 | 4 | 8 |

The formal GPU sequence is:

1. two uninterrupted repeats to freeze same-hardware tolerance;
2. coordinated stop after Step 6, then Exact Resume to Step 12;
3. Rank 1 exit immediately after the committed Step 8, then Exact Resume to Step 12.

Loss tolerance is frozen before interrupted comparisons using the M1 rule:
`max(1e-6, 2 × baseline repeat Loss difference)`. Final parameter state must match the baseline by
exact SHA256; Step, LR, Run ID, final Trainer state, and per-Rank Sampler state must also match
exactly. Orchestration wall time is retained privately but is not a benchmark result.

## 10. Acceptance

M3.2 is complete only after:

1. Checkpoint/Rank-state schemas and deterministic Sampler unit tests pass;
2. a two-process CPU/Gloo uninterrupted-versus-resumed integration comparison passes;
3. wrong World Size, missing Rank state, corrupt hash, invalid cursor, and config drift fail closed;
4. a real two-RTX-3090 coordinated interruption resumes without duplicate canonical Steps;
5. a real Rank 1 exit terminates torchrun, retains diagnostics, and resumes from the last valid
   Checkpoint;
6. both recovered Runs match the predeclared uninterrupted baseline tolerances;
7. private raw evidence and a path-free public failure/recovery report are retained;
8. Issue #13, milestone documents, CI, PR review, and Squash Merge are complete.
