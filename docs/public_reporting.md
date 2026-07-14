# Public reporting and redaction policy

## Purpose

Public reports must preserve enough evidence to reproduce a claim without publishing
machine identity, user-specific filesystem layout, credentials, private data, or model
artifacts that cannot legally be redistributed.

## Fact sources

- The private Artifact Store is the source of truth for complete run directories and
  unsanitized logs.
- JSON and JSONL files inside a run are the source of truth; Markdown is a rendered
  explanation.
- Public reports are sanitized snapshots. A public snapshot must retain its Run ID,
  config hash, code revision, dependency versions, hardware class, test protocol, raw
  measurements, and failure records.
- MLflow, when added, is a rebuildable projection and never the only copy of lineage.

The default private root is `/data/yujielun/tinyllm/`. That path is a configured storage
contract, not a statement that the directory currently exists or has passed capacity
checks.

## Required redaction

Before committing a report, replace:

- login names, real hostnames, IP addresses, SSH aliases, and user-specific Home paths;
- credentials, environment-variable values, cookies, signed URLs, and private registry
  endpoints;
- private dataset contents, unlicensed model files, and personally identifying data;
- process command lines or exception messages that expose any item above.

Stable placeholders include `<redacted-host>`, `<project-root>`, `<private-artifact-root>`,
and `<redacted-address>`. System paths such as `/usr/local/cuda` may remain when they are
necessary to explain a compatibility result and do not reveal user identity.

`reports/hardware/raw/` contains legacy-named **sanitized evidence snapshots**, not the
private originals. New run systems must write originals under the private Artifact Store
and publish only explicitly exported snapshots.

## Benchmark publication gate

Every public benchmark must include:

1. model/data/config/code revisions and environment capture;
2. hardware class, visible GPU indices, topology, temperature, frequency, and background
   load notes;
3. warm-up, measurement steps, independent repetitions, aggregation method, and raw
   machine-readable results;
4. failed and anomalous runs with an explanation instead of deletion;
5. a clear distinction between measured, estimated, and not evaluated values.

Run `make public-check` and `make links` before requesting review. These checks reduce
accidental disclosure but do not replace human review.
