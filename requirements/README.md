# Dependency profiles

TinyLLM-System separates Python dependency constraints from hardware-specific PyTorch
wheels. This avoids silently replacing the CUDA build when a general dependency is
updated.

| File | Purpose | Validation state |
| -- | -- | -- |
| `constraints/runtime.txt` | Direct runtime dependency versions | CPU CI and RTX 3090 development environment |
| `constraints/dev.txt` | Direct quality-tool versions | CPU CI and RTX 3090 development environment |
| `constraints/baseline.txt` | Qwen3 and lm-eval Baseline dependencies | RTX 3090 M2.4c compatibility Smoke |
| `constraints/m4.txt` | FSDP2/DCP and Qwen3 training dependencies | Isolated M4 CPU/API compatibility Smoke |
| `constraints/m5.txt` | Qwen3 Full SFT and reviewed PEFT LoRA dependencies | M5 CPU/API and RTX 3090 compatibility gates |
| `constraints/serving.txt` | FastAPI Gateway and observability dependencies | M7 CPU/Mock contract and RTX 3090 service validation |
| `serving-cu118.txt` | vLLM 0.8.5.post1 CUDA 11.8 dependencies | M7 Qwen3 RTX 3090 compatibility Smoke |
| `torch-cpu.txt` | CPU-only CI and local smoke tests | CPU CI |
| `torch-cu118.txt` | RTX 3090 CUDA 11.8 profile | M0 hardware smoke |
| `torch-v100-cu118.txt` | Prospective V100 FP16 profile | Not validated; cannot be used for a release claim |

These files constrain direct dependencies; they are not a fully locked transitive
environment. Each run must still capture a complete `pip freeze`, PyTorch/CUDA
versions, and hardware inventory in `environment.json` and `hardware.json`.

The build backend is also constrained through the development profile. `setuptools==83.0.0`
is the minimum patched line for `PYSEC-2026-3447`; older environments must be upgraded inside
their virtual environment before the dependency audit can pass.

`pip-audit` audits packages resolvable from PyPI. Hardware-specific PyTorch wheels use
the PyTorch index and may be reported as unauditable; this is a recorded audit limitation,
not evidence that the wheel has no vulnerabilities. PyTorch revisions remain pinned and
must be reviewed against upstream security advisories before a release.

Install the CPU profile with `make bootstrap-cpu`, the main RTX 3090 profile with
`make bootstrap-gpu`, the M2.4c model-evaluation profile with `make bootstrap-baseline`, or the
isolated FSDP2/Qwen profile with `make bootstrap-m4`.
The Baseline uses `.venv-baseline` because its reviewed Transformers 4.57 line requires
Tokenizers 0.22, while deterministic M2 data builds remain pinned to Tokenizers 0.21.4 in the
default `.venv`. Run Baseline commands through `.venv-baseline/bin/tinyllm`; do not reuse that
environment to rebuild M2 data. V100 remains a conditional compatibility target until access to
the auxiliary host is provided and a real FP16 + GradScaler smoke test passes.

The Baseline dependency audit and its narrowly scoped, time-bounded advisory exceptions are
documented in [baseline_security_exceptions.md](baseline_security_exceptions.md). Run it with
`make audit-baseline`; an exception is not a claim that the dependency is vulnerability-free.

M4 FSDP2 dependencies are not treated as validated by either the core or Baseline profile. M4
uses a separate `.venv-m4`. The committed direct constraints passed the PyTorch FSDP2/DCP,
Transformers Qwen, Safetensors, Tiny Qwen forward/backward, and CPU/Gloo compatibility gates
described in [the M4 contract](../docs/m4_fsdp2_contract.md). This is dependency-readiness
evidence only: it does not prove that the fixed Qwen3-8B revision has been acquired or fits on
four RTX 3090 GPUs. Run `make m4-dependency-smoke` and `make audit-m4`; the scoped audit exceptions
are documented in [m4_security_exceptions.md](m4_security_exceptions.md).

M5 uses a separate `.venv-m5` profile. This keeps PEFT isolated from the frozen Baseline
environment, so installing LoRA support cannot silently change evaluation dependencies or
invalidate an in-progress Exact Resume.

M7 uses `.venv-serving`. `make bootstrap-serving` installs the Gateway, HTTP client,
Prometheus and OpenTelemetry stack only. The CUDA-specific vLLM wheel is deliberately installed
as a second, explicit step after the Gateway dependency audit, so dependency resolution cannot
silently replace the reviewed training environments or claim CUDA compatibility before a real
RTX 3090 Smoke Test.

The pinned CUDA 11.8 serving profile has narrowly scoped dependency-audit exceptions documented
in [m7_security_exceptions.md](m7_security_exceptions.md). GitHub CI ignores only those exact
advisory identifiers; the M7 Production Gate independently rejects unreviewed Critical or High
findings.

`make bootstrap-serving-vllm` installs the frozen CUDA 11.8 dependency set and the reviewed
official Wheel. Set `PIP_CACHE_DIR` to an Artifact Store cache when home-disk pressure matters.

The vLLM profile omits Ray's `cgraph` extra because that optional dependency path introduces
CUDA 12 CuPy into the resolver while this profile is fixed to CUDA 11.8. M7 uses one CUDA 11.8
device and does not use Ray pipeline parallelism; the plain Ray dependency still satisfies
vLLM's local execution import path without mixing CUDA major versions.
