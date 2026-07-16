# Dependency profiles

TinyLLM-System separates Python dependency constraints from hardware-specific PyTorch
wheels. This avoids silently replacing the CUDA build when a general dependency is
updated.

| File | Purpose | Validation state |
| -- | -- | -- |
| `constraints/runtime.txt` | Direct runtime dependency versions | CPU CI and RTX 3090 development environment |
| `constraints/dev.txt` | Direct quality-tool versions | CPU CI and RTX 3090 development environment |
| `constraints/baseline.txt` | Qwen3 and lm-eval Baseline dependencies | RTX 3090 M2.4c compatibility Smoke |
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
`make bootstrap-gpu`, or the M2.4c model-evaluation profile with `make bootstrap-baseline`.
The Baseline uses `.venv-baseline` because its reviewed Transformers 4.57 line requires
Tokenizers 0.22, while deterministic M2 data builds remain pinned to Tokenizers 0.21.4 in the
default `.venv`. Run Baseline commands through `.venv-baseline/bin/tinyllm`; do not reuse that
environment to rebuild M2 data. V100 remains a conditional compatibility target until access to
the auxiliary host is provided and a real FP16 + GradScaler smoke test passes.

The Baseline dependency audit and its narrowly scoped, time-bounded advisory exceptions are
documented in [baseline_security_exceptions.md](baseline_security_exceptions.md). Run it with
`make audit-baseline`; an exception is not a claim that the dependency is vulnerability-free.

M4 FSDP2 dependencies are intentionally not treated as validated by either the core or Baseline
profile. M4 uses a separate `.venv-m4`; its constraints file will be committed only after the
PyTorch FSDP2/DCP, Transformers Qwen, Safetensors, and CPU/Gloo compatibility smoke described in
[the M4 contract](../docs/m4_fsdp2_contract.md) passes. Until then, the presence of an importable
API is readiness evidence, not a model-training support claim.
