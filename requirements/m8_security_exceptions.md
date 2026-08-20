# M8 Agent dependency security exceptions

M8 extends the reviewed M7 loopback Gateway with LangGraph, MCP, and a local DevOps Agent. A clean
M8 environment audit on 2026-08-20 reported the same six unique advisory IDs as the M7 profile and
no new finding attributable to LangGraph or MCP. Duplicate Starlette findings in the raw audit are
preserved as emitted by `pip-audit`.

| pip-audit ID | Component | M8 applicability and enforced control |
| -- | -- | -- |
| `PYSEC-2026-1805` | protobuf | No inbound protobuf parser; Agent and MCP public contracts use bounded JSON, and OTel protobuf remains optional outbound telemetry |
| `PYSEC-2026-161` | Starlette | Security decisions never depend on a reconstructed request URL; trusted-host and raw-path checks run before routing |
| `PYSEC-2026-248` | Starlette | Malformed non-slash paths are rejected by the Gateway raw-path allowlist |
| `PYSEC-2026-249` | Starlette | No form or multipart endpoint is registered; Agent writes accept strict JSON only |
| `PYSEC-2026-2280` | Starlette | No `HTTPEndpoint` subclass is registered; FastAPI function routes have explicit methods |
| `PYSEC-2026-2281` | Starlette | No static-file mount exists; user input cannot select a local filesystem route |

Additional M8 controls include a loopback-only model Gateway, environment-only Bearer secrets,
administrator-owned MCP registrations, tool allowlists, JSON Schema validation, bounded retries,
zero automatic retry for writes, explicit idempotent approval, and sandbox-only YAML copies with
path traversal and symlink rejection. MCP annotations and retrieved evidence never grant authority.

These are deployment-profile decisions, not claims that the dependencies are vulnerability-free.
`make audit-agent` ignores only the identifiers above. A new finding, public-network deployment,
remote MCP Server, dependency upgrade, or expanded filesystem capability requires a fresh review.
