# M4 dependency-audit exceptions

Reviewed: 2026-07-16

Scope: only the isolated `.venv-m4` environment and the pinned Qwen3-8B FSDP2 path. These
exceptions do not apply to arbitrary model repositories, remote code, Transformers Trainer,
hosted inference, pickle checkpoints, or user-supplied serialized state.

The dependency gate performs no network access and constructs a synthetic Qwen3 model from a
local config. The later model gate must verify the fixed Qwen3-8B revision and every downloaded
file before loading, run with `trust_remote_code=false`, select SDPA explicitly, accept only
Safetensors model weights, and use TinyLLM's native FSDP2/DCP path rather than Transformers
Trainer. Any failure to enforce those controls invalidates these exceptions.

| Advisory | Package | Why temporarily accepted | Removal condition |
| -- | -- | -- | -- |
| `PYSEC-2025-217` | Transformers 4.57.6 | The affected untrusted X-CLIP conversion path is outside the fixed Qwen3 path. No fixed 4.x release is published. | Remove when a compatible fixed Transformers release exists. |
| `PYSEC-2026-2290` | Transformers 4.57.6 | The affected LightGlue remote-code path is outside M4 and remote code is forbidden. No fixed release is published. | Remove when upstream publishes a compatible fix. |
| `PYSEC-2026-2288` | Transformers 4.57.6 | The issue affects Transformers Trainer RNG checkpoint loading; M4 uses native PyTorch FSDP2/DCP and does not call Trainer. The first fix is Transformers 5.0. | Remove after the Transformers 5 compatibility gate passes. |
| `PYSEC-2026-2289` | Transformers 4.57.6 | Model-selected remote kernels are forbidden; the fixed config must be hash-verified and M4 selects local SDPA. The first fix is Transformers 5.3. | Remove after the Transformers 5.3+ compatibility gate passes. |

Re-run `make audit-m4` before every M4 model run and release. A model revision, architecture,
attention backend, serialization format, or dependency change requires a fresh review.
