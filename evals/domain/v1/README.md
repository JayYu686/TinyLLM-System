# TinyLLM Domain Evaluation v1

This directory contains the public, frozen-content candidate for the M2 domain evaluation set.
It is authored for TinyLLM-System and redistributed under Apache-2.0.

## Contents

- `items.jsonl`: 300 strict evaluation items in stable ID order.
- `manifest.json`: timestamp-free content, configuration, Tokenizer, Template, and decoding identity.
- `configs/eval/m2_domain_v1.yaml`: the build contract stored at the repository root.
- `scripts/build_m2_domain_eval.py`: the deterministic generator and drift checker.

The manifest is the source of truth for the suite version and hashes. Do not edit generated JSONL
or Manifest files directly.

## Distribution and pairing

| Category | Total | English | Chinese | Scorer |
| -- | --: | --: | --: | -- |
| Python | 50 | 35 | 15 | Exact Match |
| Linux | 45 | 32 | 13 | Exact Match |
| JSON | 40 | 28 | 12 | Canonical JSON Object |
| Configuration editing | 40 | 28 | 12 | Canonical JSON Object |
| Log diagnosis | 45 | 31 | 14 | Multiple Choice |
| Short code | 40 | 28 | 12 | Exact Match |
| Evidence-grounded refusal | 40 | 28 | 12 | Human Rubric |

Each of the 90 Chinese items is paired with one English item of the same task and difficulty. The
pair is identified by category plus its `bilingual-pair-NNN` tag. The remaining 120 English items
carry `english-only`. Pair tags make language-slice analysis explicit without treating translations
as independent evidence.

## Scoring contract

The 260 objective items freeze their answer normalization, option order, or canonical JSON target.
The 40 refusal items require all three binary rubric criteria and retention of the reviewer's
item-level rationale. They must not be silently converted to automatic keyword scoring.

All content is authored from versioned project templates. No external benchmark question or dataset
sample is copied into this directory. Evaluation content must never be included in a training or
validation dataset.

## Rebuild and verify

```bash
python scripts/build_m2_domain_eval.py
python scripts/build_m2_domain_eval.py --check
```

Any Prompt, Reference, scoring, pairing, distribution, Tokenizer, Template, or decoding change
creates a new content identity. A frozen published version must not be edited in place.

## Limitations

- This is a focused engineering-domain set, not a general intelligence benchmark.
- Exact contamination scanning does not detect semantic paraphrases; Near-Dedup remains explicitly
  `not_evaluated` in M2.
- The public prompts can be optimized against, so future training data must continue to exclude them
  and reported results must identify the exact suite version.
- Human-rubric results require reviewer evidence and are not available until a real model evaluation.
- A clean pre-training Baseline is a separate M2.4c artifact and is not implied by these files.
