# M2.4a exact-contamination contract smoke

Execution date: 2026-07-15 (Asia/Shanghai)

Status: **PASS**

This smoke uses only public synthetic Apache-2.0 messages and generated Token arrays. It validates
the M2.4a identity, redaction, and Exact contamination boundary; it is not the result for the
formal 300-item domain evaluation set and does not claim that the real M2 dataset is contamination
free.

## Observed results

| Item | Actual result |
| -- | -- |
| Registry input | 10 synthetic Train samples; atomically registered, reopened, and verified |
| Evaluation input | 3 synthetic items |
| Exact full-sequence matches | 1 |
| Exact Prompt-prefix matches | 2 |
| Unique contaminated items | 2 |
| Clean control | `clean`; 0 matches |
| Fingerprint | `token-sequence-sha256-v1` |
| Published training Sample IDs | None; only SHA256 identities are allowed |
| Near-Dedup | `not_evaluated` |

The full-sequence item reused both the synthetic Prompt and Reference. A second item reused only the
Prompt and changed the Reference, so it produced only a Prompt-prefix match. A third item used a
different Prompt and produced no match. The stable machine-readable result is retained in
[raw/contamination_smoke.json](raw/contamination_smoke.json).

Reproduce locally:

```bash
.venv/bin/python -m scripts.run_m2_contamination_smoke
```

## Failure paths covered by the test suite

- noncanonical Unicode/line endings/outer whitespace, invalid Prompt roles, scorer/reference drift,
  unknown Schema fields, duplicate IDs, wrong item counts, and version-prefix drift are refused;
- empty or out-of-range Token fingerprints are refused;
- duplicate Train Sample IDs, Tokenizer/Template/maximum-length incompatibility, malformed JSONL,
  and invalid YAML are refused;
- JSONL errors report only a line number and do not echo rejected content;
- contaminated CLI results return exit code 6; config/input failures return 2; Registry/cache
  failures return 3.

## Boundaries

- The formal set remains exactly 300 items with the frozen 210/90 language and per-category counts.
- The real `m2-sft-v1-f82ff32e` Train Split will be checked only after the 300 items are reviewed
  and frozen in M2.4b.
- Exact Token fingerprints do not detect paraphrases or semantic overlap. Near-Dedup remains an
  explicit non-blocking enhancement and cannot be reported as complete.
- Baseline model accuracy remains `not_evaluated`; no model weights were loaded in this smoke.
