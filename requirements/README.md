# Dependency profiles

TinyLLM-System separates Python dependency constraints from hardware-specific PyTorch
wheels. This avoids silently replacing the CUDA build when a general dependency is
updated.

| File | Purpose | Validation state |
| -- | -- | -- |
| `constraints/runtime.txt` | Direct runtime dependency versions | CPU CI and RTX 3090 development environment |
| `constraints/dev.txt` | Direct quality-tool versions | CPU CI and RTX 3090 development environment |
| `torch-cpu.txt` | CPU-only CI and local smoke tests | CPU CI |
| `torch-cu118.txt` | RTX 3090 CUDA 11.8 profile | M0 hardware smoke |
| `torch-v100-cu118.txt` | Prospective V100 FP16 profile | Not validated; cannot be used for a release claim |

These files constrain direct dependencies; they are not a fully locked transitive
environment. Each run must still capture a complete `pip freeze`, PyTorch/CUDA
versions, and hardware inventory in `environment.json` and `hardware.json`.

`pip-audit` audits packages resolvable from PyPI. Hardware-specific PyTorch wheels use
the PyTorch index and may be reported as unauditable; this is a recorded audit limitation,
not evidence that the wheel has no vulnerabilities. PyTorch revisions remain pinned and
must be reviewed against upstream security advisories before a release.

Install the CPU profile with `make bootstrap-cpu`, or the main RTX 3090 profile with
`make bootstrap-gpu`. V100 remains a conditional compatibility target until access to
the auxiliary host is provided and a real FP16 + GradScaler smoke test passes.
