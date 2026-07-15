# M2 Qwen3-0.6B Formal Pre-training Baseline

Execution date: 2026-07-15 (Asia/Shanghai)

Status: **PASS**

This report records the frozen Qwen3-0.6B Base Model result before TinyLLM formal
post-training. It is a baseline for later candidate comparison, not a claim that the Base
Model meets a quality gate. All values below come from one real offline run on clean `main`.

## Frozen identities

| Input | Identity |
| -- | -- |
| Model | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` |
| Actual parameters | 596,049,920 |
| Evaluation suite | `tinyllm-domain-v1-83bdd8ef` |
| Baseline config | `a2bae098181959c2f9799cc85d3404875ca015b61454f055c304b87033a936be` |
| Run | `20260715T063643Z-qwen3-0-6b-pretraining-baseline-a2bae098-fdd0` |
| Code | `388241db0fd375939da9f47d2b1a3ac282819cbf`; clean `main` |
| Precision / prompt | BF16; frozen non-thinking ChatML; greedy decoding |

The model files, Domain suite, Prompt renderer, local lm-eval adapters, configuration, and
cached dataset revisions passed their frozen size/SHA256 checks before evaluation.

## Runtime

| Item | Actual result |
| -- | -- |
| GPU | Physical GPU 5, NVIDIA GeForce RTX 3090 |
| Start state | 4 MiB used, 0% utilization, 28 C |
| Driver / CUDA Runtime | 535.261.03 / 11.8 |
| Python / PyTorch | 3.11.14 / 2.7.1+cu118 |
| Transformers / Tokenizers | 4.57.6 / 0.22.2 |
| lm-eval / Datasets | 0.4.12 / 4.8.5 |
| Model-evaluation wall time | 371.24934 seconds |
| General-task evaluation time | 262.142773 seconds |
| Mode | Offline model and dataset caches |

An operator spot check observed 83 C during lm-eval. At the detailed 82 C check, NVIDIA-SMI
reported no hardware or software thermal slowdown and a target temperature of 83 C. This was
not continuous telemetry and must not be interpreted as a measured peak-temperature result.

## Domain baseline

| Measure | Actual result |
| -- | --: |
| Evaluated | 300 / 300 |
| Objective strict pass | 16 / 260 (6.15%) |
| JSON parse valid | 32 / 80 (40.00%) |
| Maintainer human-rubric pass | 0 / 40 (0.00%) |
| Derived overall pass | 16 / 300 (5.33%) |
| Failed Item IDs retained in public record | 284 |

The JSON validity rate is diagnostic and independent of strict answer correctness. All 40
evidence-grounding/refusal responses were reviewed against three frozen Boolean criteria. The
maintainer approved the item-level judgments; none explicitly requested every evidence artifact
required by its rubric, so none passed all three criteria.

| Category | Items | Passed |
| -- | --: | --: |
| Config | 40 | 16 |
| JSON | 40 | 0 |
| Linux | 45 | 0 |
| Logs | 45 | 0 |
| Python | 50 | 0 |
| Refusal | 40 | 0 |
| Short code | 40 | 0 |

| Language | Items | Passed |
| -- | --: | --: |
| English | 210 | 13 |
| Chinese | 90 | 3 |

The public raw record lists every failed Item ID but excludes prompts, model responses, and
judgment text. Those remain in the private Artifact Store.

## General-task baseline

All tasks used zero-shot local adapters, the pinned Qwen non-thinking Chat Template, and complete
sample logging.

| Task | Samples | Acc | Acc stderr | Acc norm | Acc norm stderr |
| -- | --: | --: | --: | --: | --: |
| ARC-Easy | 2,376 | 0.537458 | 0.010231 | 0.472643 | 0.010244 |
| HellaSwag | 10,042 | 0.361382 | 0.004794 | 0.420733 | 0.004927 |
| PIQA | 1,838 | 0.664309 | 0.011018 | 0.660501 | 0.011049 |

The stable redacted result and all public integrity hashes are in
[raw/baseline_formal.json](raw/baseline_formal.json). The private Run retains 300 Domain outputs,
14,256 general-task sample records, commands, logs, complete environment, hardware, and 40
maintainer judgments.

## Reproduce

After installing the constrained Baseline environment and filling the pinned private caches:

```bash
.venv-baseline/bin/tinyllm eval baseline \
  --config configs/eval/m2_baseline.yaml \
  --artifact-root "$TINYLLM_ARTIFACT_ROOT" \
  --project-root . \
  --device cuda \
  --gpu-index "$IDLE_PHYSICAL_GPU" \
  --offline \
  --json
```

After generation, a maintainer must submit all 40 private judgments with
`tinyllm eval baseline-review`. Partial or changed post-commit judgments fail closed.

## Limitations

- This is one Base Model run, not a repeated-run variance study or a candidate comparison.
- The Domain suite is TinyLLM-authored and frozen, but only one maintainer judged the 40 subjective
  items; inter-rater agreement is not measured.
- Exact contamination is zero; Near-Dedup and semantic contamination remain unevaluated.
- HellaSwag and PIQA Hub mirrors did not declare a license, so their sample-level records stay
  private even though aggregate metrics are public.
- Temperature was spot-checked rather than continuously sampled.
- No DDP, FSDP2, ZeRO-3, inference latency, Promotion Gate, or deployment claim follows from this
  result.
