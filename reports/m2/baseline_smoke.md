# M2 Qwen3-0.6B Baseline Smoke Report

Execution date: 2026-07-15 (Asia/Shanghai)

Status: **PASS**

This run verifies the M2.4c model-evaluation path with two Domain items and two samples from each
general task. It is deliberately too small to support a model-quality claim. The formal 300-item
Domain and full ARC-Easy/HellaSwag/PIQA Baseline remains a separate clean-`main` gate.

## Frozen inputs

| Input | Identity |
| -- | -- |
| Model | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` |
| Actual parameters | 596,049,920 |
| Evaluation suite | `tinyllm-domain-v1-83bdd8ef` |
| Baseline config | `3457c91740c26013a8408cad9b38e2e970a66dace26dbd1acb2c52d5bca5c72c` |
| Run | `20260715T031711Z-qwen3-0-6b-baseline-smoke-3457c917-bc1b` |
| Code | `98cbfe42464983a06b6342576be5c21ca867b096`; dirty implementation branch |

The dirty state is intentional and makes this evidence ineligible as the formal Baseline. It was
captured before the implementation PR so the complete CLI and Artifact path could be tested.

## Runtime

| Item | Actual result |
| -- | -- |
| GPU | Physical GPU 5, NVIDIA GeForce RTX 3090 |
| Start state | 4 MiB used, 0% utilization, 31 C |
| Compute / precision | Compute Capability 8.6; BF16 supported |
| Driver / CUDA Runtime | 535.261.03 / 11.8 |
| PyTorch / Transformers | 2.7.1+cu118 / 4.57.6 |
| Tokenizers | 0.22.2 in isolated `.venv-baseline` |
| lm-eval / Datasets | 0.4.12 / 4.8.5 |
| Mode | Offline model and dataset caches |
| End-to-end wall time | 34.854161 seconds |

## Observed Smoke results

| Scope | Samples | Actual result |
| -- | --: | -- |
| Domain JSON | 2 | 2 valid JSON; 1 strict match |
| ARC-Easy | 2 | `acc=0.5`, `acc_norm=0.5` |
| HellaSwag | 2 | `acc=0.0`, `acc_norm=0.5` |
| PIQA | 2 | `acc=0.5`, `acc_norm=0.5` |

The Domain responses and every lm-eval sample are retained only in the private Run. The public
evidence contains aggregates and response-integrity state, not raw prompts or model output. The
stable redacted record is [raw/baseline_smoke.json](raw/baseline_smoke.json).

## Verified behavior

- Every model file was checked against its frozen size and SHA256 before loading.
- The local Qwen Template matched the frozen non-thinking renderer byte-for-byte.
- GPU 5 passed memory, utilization, and temperature preflight immediately before visibility was
  restricted to that card.
- Domain raw responses round-tripped through strict result Schema and their SHA256 values were
  rechecked after the Run.
- `lm-eval validate` and the evaluation subprocess both exited with status 0.
- `make audit-baseline` reported no unhandled known vulnerability; five reviewed exceptions and
  the unauditable non-PyPI CUDA wheel are documented in the repository security policy.
- The private Run retained original/resolved config, Git state, software environment, hardware,
  events, metrics, commands, logs, raw samples, and summaries. The environment contains all 115
  `pip freeze --all` entries, not only direct dependencies.

## Reproduce

After installing the constrained Baseline extra and filling the pinned private caches:

```bash
.venv-baseline/bin/tinyllm eval baseline \
  --config configs/eval/m2_baseline_smoke.yaml \
  --artifact-root "$TINYLLM_ARTIFACT_ROOT" \
  --project-root . \
  --device cuda \
  --gpu-index 5 \
  --offline \
  --json
```

The physical GPU index is a runtime override, not a permanent allocation. A reproduction should
select a currently idle RTX 3090; the CLI rejects materially busy or hot cards.

## Boundary and next gate

- Values above are six total general-task examples and two Domain examples, not publishable model
  quality metrics.
- No human-rubric item is included in the first-two-item Smoke subset.
- The formal Baseline must run from a clean `main`, remove all sample limits, evaluate all 300
  Domain items and 14,256 general-task examples, and complete 40 maintainer refusal judgments.
