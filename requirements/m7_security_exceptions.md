# M7 Serving dependency security exceptions

M7 freezes an older CUDA 11.8 vLLM stack because the workstation driver remains on the 535 line.
The fixes listed below require dependency versions outside the reviewed FastAPI/vLLM profile.
CI therefore ignores only these exact identifiers. This is an applicability decision for the
loopback, text-only Qwen3 serving profile, not a claim that the packages are vulnerability-free.

| pip-audit ID | Canonical advisory | Profile decision | Enforced controls |
| -- | -- | -- | -- |
| `PYSEC-2026-1805` | `GHSA-7gcm-g887-7qv7` | Not affected: no inbound protobuf parser | JSON-only HTTP schema; OTel protobuf is outbound and optional |
| `PYSEC-2026-161` | `GHSA-86qp-5c8j-p5mr` | Mitigated: security decisions do not use reconstructed request URLs | trusted-host allowlist; ASGI raw-path validation; endpoint allowlist |
| `PYSEC-2026-248` | `GHSA-jp82-jpqv-5vv3` | Mitigated: malformed non-slash paths are rejected before routing | ASGI raw-path validation; endpoint allowlist |
| `PYSEC-2026-249` | `GHSA-82w8-qh3p-5jfq` | Not affected: no form parser is exposed | JSON-only request contract and bounded request body |
| `PYSEC-2026-2280` | `GHSA-x746-7m8f-x49c` | Not affected: no `HTTPEndpoint` subclass is registered | FastAPI function routes with explicit methods |
| `PYSEC-2026-2281` | `GHSA-wqp7-x3pw-xc5r` | Not affected: Linux service mounts no `StaticFiles` | no static-file route and no request-controlled filesystem path |
| `CVE-2026-9856` | `GHSA-xrqw-3rrv-vx5w` | Not affected: Serving is read-only with respect to model artifacts and never calls tokenizer or processor `save_pretrained()` | pinned local Qwen3 text tokenizer; artifact SHA256 verification; offline loading; `trust_remote_code=false` |

Critical and High advisories are additionally captured from OSV, bound to the serving environment,
and evaluated by `scripts/build_m7_security_audit.py`. Any new unreviewed Critical or High advisory
rejects Production promotion. A future driver and CUDA upgrade must remove these exceptions rather
than carrying them into the new profile without review.
