# M6 public summary

TinyLLM-System completed M6 and registered a 596,049,920-parameter Qwen3-0.6B
Full-SFT artifact as Candidate `qwen3-0-6b-m6-d16c2357`. The decision used an independently
frozen 300-item bilingual release suite, four Base/Candidate × Thinking/Non-thinking passes,
160 maintainer-reviewed rubric judgments, three complete general benchmarks, and 10,000 paired
cluster-bootstrap replicates.

| Metric | Base | Candidate | Delta |
| -- | --: | --: | --: |
| Thinking domain score | 34.33% | 41.67% | +7.34pp; 95% CI `[+0.33, +14.29]pp` |
| Non-thinking domain score | 22.33% | 40.67% | +18.34pp; 95% CI `[+12.46, +24.40]pp` |
| Equal-task general `acc_norm` | 51.80% | 54.48% | +2.68pp |
| Candidate JSON validity | — | 100% in both modes | pass |
| Candidate Thinking format validity | — | 100% | pass |
| Candidate Thinking forced-close rate | — | 1.67% | pass |
| Candidate Non-thinking visible-reasoning leakage | — | 0/300 | pass |

All 11 preregistered checks passed. Model, data, checkpoint, environment, evaluation, and
human-review lineage were complete and the Candidate was produced from a clean Git state. M6
also shipped a SQLite v1 query index that was rebuilt from all 57 private Run manifests; JSON and
JSONL artifacts remain the fact source.

The Candidate is not Production-eligible. M7 must add measured vLLM latency, throughput,
concurrency, stability, and rollback evidence before a Production promotion can be considered.
The content-free machine summary is available in
[M6 v7 acceptance data](raw/m6_v7_acceptance.json); the full Chinese report is
[M6 acceptance](m6_acceptance.md).
