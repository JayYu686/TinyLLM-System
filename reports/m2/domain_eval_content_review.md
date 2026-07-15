# M2 Domain Evaluation Content Review

## Status

`PASS`

This report covers the content candidate only. It does not claim a clean formal contamination scan,
a Base Model result, or completion of M2.4. Maintainer approval of the content PR is the required
human-review record.

## Maintainer review

- Date: 2026-07-15
- Scope: technical correctness, bilingual-pair fidelity, log distractors, exact-answer constraints,
  refusal rubrics, provenance, and public-data safety
- Disposition: approved without requested content changes
- Evidence: [PR #49 maintainer approval](https://github.com/JayYu686/TinyLLM-System/pull/49#issuecomment-4975715508)

## Candidate identity

- Suite: `tinyllm-domain`
- Items: 300
- Languages: 210 English / 90 Chinese
- Pairing: 90 bilingual task pairs / 120 English-only items
- Categories: Python 50, Linux 45, JSON 40, configuration 40, logs 45, short code 40,
  evidence-grounded refusal 40
- Scorers: Exact Match 135, JSON Object 80, Multiple Choice 45, Human Rubric 40
- License: Apache-2.0 for every item; redistribution allowed for every item

The exact version and content hashes are recorded in `evals/domain/v1/manifest.json` and are not
duplicated here to avoid stale review text.

## Automated review completed

- Strict Schema parsing succeeds for all 300 items.
- Item IDs and serialized Prompts are unique and committed in stable ID order.
- Language, category, and scorer counts match the frozen configuration.
- Every bilingual tag resolves to exactly one English and one Chinese item in the same category.
- All 300 provenance records use `tinyllm-authored`, Apache-2.0, and redistribution enabled.
- Objective references are bound to their explicit scoring contract.
- Every refusal item requires all three rubric criteria and retained judgment rationale.
- Regeneration is deterministic and `scripts/build_m2_domain_eval.py --check` detects drift.

## Human review checklist applied

The maintainer review checked:

1. Prompts and References are technically correct and unambiguous.
2. Chinese items preserve the intent and difficulty of their tagged English pair.
3. Multiple-choice distractors are plausible but not also supported by the shown log.
4. Exact answers do not exclude a materially equivalent answer unless the Prompt explicitly fixes
   the output form.
5. Refusal rubrics reward evidence-grounded uncertainty without rewarding generic non-answers.
6. No item reveals private data, credentials, host identity, or copied external benchmark content.

## Acceptance evidence completed

After PR #49 was approved and squash-merged, a clean `main` checkout ran `tinyllm eval
contamination` against registered dataset `m2-sft-v1-f82ff32e`. The separate
[formal contamination report](domain_eval_contamination.md) records the exact suite identity,
checked Train sample count, exit status, and `near_dedup=not_evaluated`. The pre-training
Qwen3-0.6B Baseline remains a later M2.4c task.
