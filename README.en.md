# TinyLLM-System

[简体中文](README.md) | **English**

> A hardware-aware LLM training, evaluation, and deployment platform for consumer
> multi-GPU workstations.

[![CI](https://github.com/JayYu686/TinyLLM-System/actions/workflows/ci.yml/badge.svg)](https://github.com/JayYu686/TinyLLM-System/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

TinyLLM-System organizes data processing, training strategy, failure recovery, model
evaluation, and version promotion into one reproducible engineering lifecycle. Native
PyTorch provides the training core. The primary environment is a shared workstation with
10 × RTX 3090 24GB GPUs, where execution adapts to memory capacity, topology, and available
resources through single-device training, DDP, or FSDP2.

Every experiment receives complete lineage, from pinned dataset/model revisions and
schema-validated YAML to the Git commit, software and hardware environment, checkpoints,
evaluation results, and deployment exports. Published conclusions are backed by reviewable
run evidence, while pending capabilities keep an explicit status.

## Problems addressed

| Engineering question | TinyLLM-System approach |
| -- | -- |
| What produced a training run? | A Run Manifest binds data, tokenizer, config, code, environment, and hardware identity |
| How can a job start safely on shared GPUs? | `tinyllm doctor`, utilization/temperature checks, topology capture, and strategy preflight |
| How does interrupted training continue? | Atomic checkpoints, complete training state, integrity checks, and explicit resume semantics |
| Which multi-GPU strategy fits the workload? | DDP for throughput when one device holds full state; FSDP2 when state sharding is required |
| How is a trained model accepted? | Frozen evaluation, Base/Candidate comparison, regression analysis, and a promotion gate |
| How can an experiment or deployment be reproduced? | The Artifact Store preserves facts and links deployment exports back to training and evaluation |

## Architecture

```mermaid
flowchart LR
    subgraph Inputs["Immutable inputs"]
        D[Dataset version]
        M[Model and tokenizer revision]
        Y[Schema-validated YAML]
    end

    subgraph Planning["Hardware-aware execution"]
        H[Doctor and hardware snapshot]
        P[Preflight and strategy selection]
        T[Single / DDP / FSDP2]
    end

    subgraph Reliability["Training reliability"]
        C[Atomic checkpoint]
        R[Exact / Warm / Transfer resume]
        A[Run artifact store]
    end

    subgraph Delivery["Quality and delivery"]
        E[Frozen evaluation]
        G[Candidate gate]
        I[Inference performance gate]
        V[Registry and promotion]
    end

    D --> P
    M --> P
    Y --> P
    H --> P
    P --> T
    T --> C
    C --> R
    R --> T
    T --> A
    D --> A
    H --> A
    A --> E
    E --> G
    G --> I
    I --> V
```

The complete lifecycle is:

```text
data versioning
  → hardware inspection and training planning
  → single-device or distributed training
  → checkpoint and failure recovery
  → automated evaluation and model comparison
  → candidate quality gate
  → inference performance gate
  → model registry, deployment, and experiment reproduction
```

## Core capabilities

### Traceable training

- Formal experiments start from Pydantic-schema-validated YAML.
- Run IDs combine UTC time, a slug, the resolved-config hash, and a random suffix.
- `run.json`, `config.resolved.json`, `environment.json`, `hardware.json`, and
  `metrics.jsonl` form the durable run record.
- Datasets, models, and tokenizers use pinned revisions; configuration drift changes
  experiment identity.

### Reliable checkpoint and resume

- Single-device/DDP runs save complete PyTorch state; FSDP2 uses
  `torch.distributed.checkpoint` for sharded state.
- Checkpoints cover model, optimizer, scheduler, scaler, step/epoch, RNG state, sampler
  cursor, dataset version, config hash, Git identity, environment, world size, and
  per-file SHA256.
- Publication uses a temporary directory, integrity validation, atomic rename, and an
  atomic `LATEST` update.
- Exact, Warm, and Transfer resume have separate validation rules. Safetensors serves as
  the deployment export format.

```mermaid
sequenceDiagram
    participant T as Trainer
    participant TMP as Temporary checkpoint
    participant V as Integrity validator
    participant CKPT as Committed checkpoint
    participant L as LATEST

    T->>TMP: Write state and manifest
    TMP->>V: Validate files, SHA256, and identity
    V-->>TMP: Validation passes
    TMP->>CKPT: Atomic rename
    CKPT->>L: Atomic pointer update
    L-->>T: Publish resumable location
```

### Hardware-aware distributed training

| Strategy | State placement | Primary use | Project status |
| -- | -- | -- | -- |
| Single device | Full state on one GPU | Correctness baseline, small models, rapid debugging | Complete |
| DDP | Full state on every rank | Throughput scaling when full state fits one GPU | Verified on 1/2/4 GPUs |
| FSDP2 | Sharded parameters, gradients, and optimizer state | Training models constrained by per-device memory | Qwen3-8B verified on four GPUs |
| ZeRO-3 | DeepSpeed sharding with optional offload | Engineering and performance comparison after FSDP2 | Enhancement stage |

### Data, evaluation, and promotion

- Raw data passes normalization, license filtering, deduplication, grouped splitting,
  tokenization, packing, and registration.
- Dataset manifests record input hashes, processing config, output hashes, rejection
  statistics, and license distribution.
- Evaluation suites are independently versioned, frozen before training, and checked for
  train/evaluation contamination.
- Base and Candidate use identical prompt templates, tokenizer revisions, and decoding
  configuration.
- Promotion evaluates target-task gains, general regressions, structured-output quality,
  and lineage completeness together.

## Current status

| Milestone | Status | Verified result |
| -- | -- | -- |
| M0 host readiness | Complete | 10 RTX 3090s inventoried; CUDA/BF16 smoke; 1/2/4/6-GPU NCCL correctness |
| M1 single-device training | Complete | TinyGPT-Debug CPU forward/backward; CPU Exact Resume; RTX 3090 BF16 SIGTERM/SIGKILL recovery |
| M2 data and evaluation | Complete | Pinned-source full build and offline rebuild; frozen 300-item suite; Exact contamination scan; Qwen3-0.6B Baseline |
| M3 DDP | Complete | Initialization, sampler, loss reduction, rank-failure recovery, and real 1/2/4-GPU scaling |
| M4 FSDP2 | Complete | Qwen3-8B four-GPU BF16 FULL_SHARD; Step 25→50 DCP resume; independent Safetensors load |
| M5 dual-mode SFT | In progress | M5.2 gate rejection closed; R1 failure analysis, repair mixture, and two-seed configs are frozen and awaiting single-GPU runs |
| M6 evaluation and promotion | Planned | Base/Candidate comparison, regression analysis, and Candidate Gate |
| M7 inference | Planned | vLLM serving, throughput/latency benchmark, and Production Gate |
| M8 planner | Enhancement | Static memory estimation and short probe |

Milestone status represents a combined gate across implementation, tests, smoke runs,
failure paths, real reports, and documentation. Evidence entry points:

- [M0 acceptance](reports/m0/m0_acceptance.md),
  [RTX 3090 inventory](reports/hardware/rtx3090_inventory.md), and
  [topology/NCCL](reports/hardware/nccl_topology.md)
- [M1 acceptance](reports/m1/m1_acceptance.md),
  [atomic checkpoint](reports/m1/atomic_checkpoint_report.md), and
  [Exact Resume](reports/m1/exact_resume_report.md)
- [M2 acceptance](reports/m2/m2_acceptance.md),
  [full dataset build](reports/m2/full_dataset_build.md), and
  [formal Baseline](reports/m2/baseline_formal.md)
- [M3 DDP correctness](reports/m3/ddp_correctness.md),
  [failure recovery](reports/m3/ddp_recovery.md), and
  [scaling](reports/m3/ddp_scaling.md)
- [M4 FSDP2 correctness](reports/m4/fsdp2_cpu_correctness.md),
  [DCP recovery](reports/m4/fsdp2_dcp_recovery.md), and
  [Qwen3-8B four-GPU run](reports/m4/fsdp2_qwen3_8b_formal.md)
- [M5 dual-mode contract](docs/m5_sft_contract.md),
  [M5.0 review](reports/m5/m5_dual_mode_contract.md), and
  [M5.1 data report](reports/m5/m5_reasoning_data.md), and
  [M5.2 ablation/selection report](reports/m5/m5_ablation_selection.md), and
  [M5.2-R1 format-reliability report](reports/m5/m5_format_repair_r1.md)

Each report states its evidence boundary. M0 NCCL runs cover collective correctness, M3
owns training throughput evidence, and multi-GPU results are published at their measured
world size.

## Run lifecycle

```mermaid
flowchart TD
    A[Select pinned data and model revisions] --> B[Resolve and validate YAML]
    B --> C[Doctor / GPU preflight]
    C --> D[Create Run ID and environment snapshot]
    D --> E[Train and record metrics]
    E --> F{Save point or interruption signal}
    F --> G[Publish atomic checkpoint]
    G --> H{Training complete}
    H -- Continue --> E
    H -- Complete --> I[Export Safetensors / adapter]
    I --> J[Independent evaluation]
    J --> K[Base / Candidate comparison]
    K --> L{Promotion gate}
    L -- Pass --> M[Register Candidate]
    L -- Reject --> N[Retain Development state and evidence]
```

Failed runs remain useful engineering evidence: exit cause, last valid checkpoint, resume
mode, and configuration differences stay in structured artifacts.

## Hardware and resource strategy

The primary workstation contains 10 × RTX 3090 24GB GPUs across two NUMA nodes and is
shared by multiple users. Formal scaling uses coordinated idle 1/2/4-GPU sets. Dynamically
available GPUs 4–9 can serve smoke runs, short training, and evaluation. Eight-GPU,
ten-GPU, and controlled cross-NUMA comparisons remain enhancement experiments. Reports
record actual GPU indices, world size, topology, temperature, and background load.

The auxiliary compatibility target is an 8 × V100 32GB host:

| Platform | Default precision | Numeric policy | Role |
| -- | -- | -- | -- |
| RTX 3090 | BF16, configurable TF32 | GradScaler usually unnecessary | Primary development, training, evaluation, and serving |
| V100 | FP16 | GradScaler | Compatibility validation after host access is available |

GPU jobs on the shared host pass resource preflight first. Busy-device refusals are
retained as failure-path evidence and protect other users' memory allocations.

## Quickstart

Python 3.11 is the supported development runtime. Default CI and core logic run in the
CPU profile:

```bash
git clone https://github.com/JayYu686/TinyLLM-System.git
cd TinyLLM-System
make bootstrap-cpu
source .venv/bin/activate

tinyllm --help
tinyllm doctor --json
tinyllm train \
  --config configs/pretrain/tinygpt_debug_cpu_smoke.yaml \
  --device cpu \
  --output /tmp/tinyllm-runs \
  --json

make check
```

The RTX 3090 host uses the isolated CUDA 11.8 dependency profile:

```bash
make bootstrap-gpu
source .venv/bin/activate
tinyllm doctor --distributed --json
```

`tinyllm doctor` collects read-only environment information. High-load NCCL smoke and
training use separate commands after reviewing utilization, temperature, topology,
storage, and dependency compatibility. See
[requirements/README.md](requirements/README.md).

## CLI and configuration contracts

The public command surface is delivered incrementally:

```text
tinyllm doctor
tinyllm data prepare|inspect
tinyllm train
tinyllm run list|show|reproduce
tinyllm benchmark train
tinyllm eval
tinyllm compare
tinyllm promote
tinyllm plan
tinyllm serve
tinyllm benchmark inference
```

Commands expose stable `--json` output for shell, CI, and service integration:

| Code | Meaning |
| --: | -- |
| 0 | Success |
| 2 | Invalid config or user input |
| 3 | Environment, hardware, or resource preflight failure |
| 4 | Training run failure |
| 5 | Checkpoint or resume integrity failure |
| 6 | Evaluation failure or promotion rejection |

CLI overrides focus on GPU selection, output location, resume mode, and a small set of
runtime fields. YAML stores the experiment definition. Public schemas carry version fields,
use `extra="forbid"`, and publish snapshots under [schemas/](schemas/README.md).

## Artifact Store

The private server-side Artifact Store is configured through `$TINYLLM_ARTIFACT_ROOT`:

```text
$TINYLLM_ARTIFACT_ROOT/
├── cache/       # dataset, model, and evaluation caches
├── datasets/    # registered immutable dataset versions
├── models/      # model inputs and deployment exports
├── runs/        # training runs and checkpoints
└── registry/    # rebuildable index and promotion records
```

A typical run directory:

```text
<run-id>/
├── run.json
├── events.jsonl
├── config.original.yaml
├── config.resolved.json
├── environment.json
├── hardware.json
├── metrics.jsonl
├── checkpoints/
├── evaluations/
└── exports/
```

JSON/JSONL artifacts are the fact source. SQLite provides a rebuildable query index, and
MLflow can serve as an observability projection. The public repository contains redacted
reports and configurations; raw logs, weights, datasets, and server identity stay in the
private Artifact Store.

## Repository layout

```text
TinyLLM-System/
├── configs/       # data, training, evaluation, and benchmark YAML
├── docs/          # architecture, contracts, ADRs, and design notes
├── evals/         # versioned domain evaluation suites
├── reports/       # redacted real-run and acceptance reports
├── schemas/       # public Pydantic JSON Schema snapshots
├── scripts/       # reviewable build, evaluation, and evidence tools
├── src/tinyllm/   # Python package, CLI, and core implementation
└── tests/         # unit, integration, failure-path, and GPU-marker tests
```

## Release roadmap

M0–M8 advance in dependency order, and each stage delivers one independently reviewable
system capability:

| Stage | Delivered capability | Release role |
| -- | -- | -- |
| M0 | Hardware inventory, topology, Doctor, and NCCL readiness | Execution baseline |
| M1 | Native trainer, atomic checkpoint, and Exact Resume | Training correctness |
| M2 | Deterministic data pipeline, contamination checks, and frozen evaluation | Data and evaluation lineage |
| M3 | Native DDP, rank-failure recovery, and 1/2/4-GPU scaling | `v0.3.0-beta.1` distributed baseline |
| M4 | Qwen3-8B FSDP2 sharded training and DCP resume | Large-model sharding |
| M5 | Qwen3 dual-mode Full SFT and LoRA | Practical post-training |
| M6 | Base/Candidate comparison and Candidate Gate | `v0.6.0-rc.1` candidate release |
| M7 | vLLM serving and measured inference gate | Production prerequisite |
| M8 | Static estimate and short-probe planner | Resource-planning enhancement |

M7/M8, ZeRO-3, MLflow, V100 compatibility, and TinyGPT-350M enter later iterations
according to core lifecycle dependencies and resource availability. See the
[release roadmap](docs/release_roadmap.md).

## Evaluation and model promotion

M6 compares Base and trained models on ARC-Easy, HellaSwag, PIQA, and a frozen 300-item
domain suite spanning Python, Linux, JSON/configuration, log diagnosis, and unsupported
claim refusal. The evaluation records prompt templates, tokenizer revision, decoding
configuration, raw outputs, scoring evidence, and bootstrap 95% confidence intervals.

The preregistered Candidate Gate targets:

- at least +3 percentage points on the domain aggregate, with a bootstrap 95% confidence
  interval lower bound above zero;
- general-task aggregate regression within 2 percentage points;
- JSON Valid Rate of at least 98%;
- complete data, model, checkpoint, environment, and evaluation lineage.

Rejected models retain Development status with regression metrics and failure examples.
A Candidate becomes eligible for Production after the measured M7 inference performance
gate passes.

## Core boundary and future research

The current core covers single-host single/multi-GPU training, data versioning,
checkpointing, automated evaluation, model promotion, and inference deployment. The
following directions live in the future research queue:

- MoE, pipeline parallelism, and multi-node training;
- custom KV cache, tensor parallelism, FlashAttention, and CUDA kernels;
- full RLHF;
- Kubernetes, multi-tenant billing, and complex management frontends.

M7 integrates vLLM's native OpenAI-compatible API with lineage-aware launch and benchmark
wrappers. Scope references live under [Future Work](docs/future/) and
[ADRs](docs/adr/).

## Documentation

The Chinese [README](README.md) is the primary public entry point, with this English
version maintained alongside it. Design documents and human-review reports use Chinese
by default; CLI, schema, and machine-readable JSON fields remain English.

- [Contribution, PR, and review workflow](CONTRIBUTING.md)
- [Release roadmap](docs/release_roadmap.md) and
  [capability/evidence map](docs/capability_map.md)
- [Architecture](docs/architecture.md), [training design](docs/training_design.md), and
  [M5 SFT contract](docs/m5_sft_contract.md)
- [Data contract](docs/dataset_contract.md), [evaluation spec](docs/evaluation_spec.md),
  and [experiment lineage](docs/experiment_lineage.md)
- [Hardware strategy](docs/hardware_strategy.md) and
  [benchmark policy](docs/benchmark_plan.md)
- [Public reporting policy](docs/public_reporting.md) and
  [security policy](SECURITY.md)

## License

Licensed under the [Apache License 2.0](LICENSE). Dataset and model licenses remain
independent; each registered dataset and published adapter preserves its source, pinned
revision, and license metadata.
