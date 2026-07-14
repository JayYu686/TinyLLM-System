# Contributing to TinyLLM-System

TinyLLM-System is developed as an evidence-first training-systems project. Small,
reviewable changes are preferred over broad framework additions.

## Development workflow

1. Open or select a GitHub Issue with a defined acceptance criterion.
2. Create a focused branch from `main`.
3. Open a Draft pull request early and keep its checklist current.
4. Follow the required progression in [AGENTS.md](AGENTS.md): design, interface,
   unit test, smoke test, failure path, integration test, real benchmark, and docs.
5. Run `make check` before requesting review.
6. Convert the pull request to ready only when all applicable evidence is attached.
7. Squash-merge after required checks pass.

Direct pushes to `main` and force pushes are not part of the project workflow.

## Local setup

Python 3.11 is the supported development version.

```bash
make bootstrap-cpu       # CPU development and CI-equivalent tests
make bootstrap-gpu       # RTX 3090 / CUDA 11.8 profile
source .venv/bin/activate
make check
```

The prospective V100 profile is not a release target until it passes a real FP16 +
GradScaler smoke test. See [requirements/README.md](requirements/README.md).

## Pull request boundaries

- Formal experiments must start from validated YAML configuration.
- CLI flags may only override documented runtime fields.
- GPU tests require the `gpu` marker and do not run in public fork CI.
- A benchmark claim must include raw machine-readable results, environment, topology,
  GPU indices, warm-up, repetitions, and anomaly notes.
- Model, dataset, dependency, or evaluation revisions cannot change silently.
- Generated JSON Schema snapshots must be updated with
  `python scripts/export_schemas.py`.

## Public artifacts

Never commit credentials, downloaded datasets, model weights, private run directories,
or unsanitized machine logs. Follow [docs/public_reporting.md](docs/public_reporting.md)
before publishing evidence.

## Reporting security issues

Do not open a public issue for a suspected vulnerability or exposed credential. Follow
[SECURITY.md](SECURITY.md) instead.
