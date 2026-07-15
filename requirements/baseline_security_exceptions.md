# Baseline dependency-audit exceptions

Reviewed: 2026-07-15

Scope: the isolated `.venv-baseline` M2.4c evaluation environment only. These exceptions do not
apply to arbitrary model loading, Transformers Trainer checkpoints, hosted inference, or the
default data-development environment.

TinyLLM pins and verifies every Qwen3-0.6B model file by size and SHA256 before loading, runs the
Baseline offline, sets `trust_remote_code=false`, passes `attn_implementation=sdpa` explicitly,
and loads only Safetensors weights. The Baseline does not use X-CLIP, LightGlue, Transformers
Trainer, pickle checkpoints, or model-supplied code. lm-eval creates its own local SQLite cache;
TinyLLM does not open an externally supplied sqlitedict database.

| Advisory | Package | Why temporarily accepted | Removal condition |
| -- | -- | -- | -- |
| `PYSEC-2025-217` | Transformers 4.57.6 | Affects untrusted X-CLIP checkpoint conversion, which is outside the pinned Qwen path. No fixed 4.x release is published. | Remove when a compatible fixed Transformers release exists. |
| `PYSEC-2026-2290` | Transformers 4.57.6 | Affects LightGlue nested remote-code loading, which is outside the pinned Qwen path. No fixed release is published. | Remove when upstream publishes a fix. |
| `PYSEC-2026-2288` | Transformers 4.57.6 | Affects Transformers Trainer RNG checkpoint loading; TinyLLM does not use Trainer and runs PyTorch 2.7.1. The first fix is Transformers 5.0. | Remove after the Transformers 5 compatibility gate passes. |
| `PYSEC-2026-2289` | Transformers 4.57.6 | Affects model-config-selected remote kernels. The exact local config is hash-pinned, contains no dynamic-kernel field, and execution is offline with explicit SDPA. The first fix is Transformers 5.3. | Remove after the Transformers 5.3+ compatibility gate passes. |
| `PYSEC-2026-1939` | sqlitedict 2.1.0 | No fixed release exists. It is a transitive lm-eval dependency and only project-created local cache state is accepted. | Remove when lm-eval removes the dependency or sqlitedict publishes a fix. |

Before M5, re-run `make audit-baseline`, check for fixed compatible releases, and review this table.
Any change to model source, online loading, model architecture, Attention backend, or serialized
cache/checkpoint input invalidates these exceptions and requires a new security review.
