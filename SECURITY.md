# Security policy

## Supported versions

Security fixes are applied to the latest commit on `main`. Pre-release tags are
research snapshots and are not maintained as long-term support branches.

## Private reporting

Use the repository's **Security → Report a vulnerability** workflow to submit a private
GitHub Security Advisory. Include the affected commit, reproduction steps, impact, and
any proposed mitigation. Do not include live credentials in the report.

If a credential or private dataset was committed, revoke access first, then report the
incident privately. Removing a value from the latest commit does not remove it from Git
history.

## Project security boundaries

TinyLLM-System is research and portfolio software, not a hosted multi-tenant service.
Training configs, datasets, model artifacts, and checkpoints are untrusted inputs until
validated. PyTorch checkpoint files may execute code when loaded; only project-created,
integrity-checked training checkpoints should enter Exact Resume flows. Public model
exports use Safetensors when that feature is implemented.

The project does not promise security support for models, datasets, or dependencies
outside revisions explicitly recorded in a run manifest.

Reviewed dependency-audit exceptions are recorded in
[requirements/baseline_security_exceptions.md](requirements/baseline_security_exceptions.md).
They must name the constrained execution path and a removal condition; suppressing an advisory
without that record is not allowed.
