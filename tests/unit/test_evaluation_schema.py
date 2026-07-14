from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tinyllm.data import load_m2_tokenization_config
from tinyllm.evaluation import (
    AuthoredProvenance,
    CategoryCounts,
    ContaminationPolicy,
    DecodingConfig,
    EvaluationBuildConfig,
    EvaluationItem,
    EvaluationPromptMessage,
    ExactMatchScorer,
    HumanRubricScorer,
    JsonObjectScorer,
    LanguageCounts,
    MultipleChoiceScorer,
    RequiredTermsScorer,
)

TOKENIZATION_CONFIG = Path("configs/data/m2_tokenization.yaml")


def evaluation_item(
    suffix: int = 1,
    *,
    prompt: str = "What is 2 + 2?",
    answer: str = "4",
) -> EvaluationItem:
    return EvaluationItem(
        id=f"domain-python-{suffix:03d}",
        language="en",
        category="python",
        prompt_messages=(EvaluationPromptMessage(role="user", content=prompt),),
        reference_answer=answer,
        scorer=ExactMatchScorer(
            kind="exact_match",
            accepted_answers=(answer,),
            case_sensitive=True,
            strip_outer_whitespace=True,
        ),
        provenance=AuthoredProvenance(
            origin="tinyllm-authored",
            license="Apache-2.0",
            redistribution_allowed=True,
            source_note="Authored for the TinyLLM public evaluation set.",
        ),
    )


def evaluation_config(*, expected_items: int = 1) -> EvaluationBuildConfig:
    tokenization = load_m2_tokenization_config(TOKENIZATION_CONFIG)
    return EvaluationBuildConfig(
        suite_name="tinyllm-smoke",
        version_prefix="tinyllm-smoke-v1",
        expected_items=expected_items,
        language_counts=LanguageCounts(en=expected_items, zh=0),
        category_counts=CategoryCounts(
            config=0,
            json_items=0,
            linux=0,
            logs=0,
            python=expected_items,
            refusal=0,
            short_code=0,
        ),
        tokenizer=tokenization.tokenizer,
        template=tokenization.template,
        max_sequence_length=tokenization.max_sequence_length,
        decoding=DecodingConfig(
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=256,
            seed=42,
        ),
        contamination=ContaminationPolicy(
            split="train",
            full_sequence=True,
            prompt_prefix=True,
            near_dedup=False,
            fingerprint_algorithm="token-sequence-sha256-v1",
        ),
    )


def test_formal_300_item_distribution_is_exact() -> None:
    raw = evaluation_config().to_dict()
    raw.update(
        {
            "suite_name": "tinyllm-domain",
            "version_prefix": "tinyllm-domain-v1",
            "expected_items": 300,
            "language_counts": {"en": 210, "zh": 90},
            "category_counts": {
                "config": 40,
                "json_items": 40,
                "linux": 45,
                "logs": 45,
                "python": 50,
                "refusal": 40,
                "short_code": 40,
            },
        }
    )

    config = EvaluationBuildConfig.model_validate(raw)

    assert config.expected_items == 300
    assert sum(config.language_counts.to_dict().values()) == 300
    assert sum(config.category_counts.to_dict().values()) == 300
    assert config.contamination.near_dedup is False


def test_evaluation_item_refuses_noncanonical_or_inconsistent_content() -> None:
    item = evaluation_item()

    with pytest.raises(ValidationError, match="outer whitespace"):
        EvaluationPromptMessage(role="user", content=" padded ")
    with pytest.raises(ValidationError, match="ID must match category"):
        EvaluationItem.model_validate({**item.to_dict(), "category": "linux"})
    with pytest.raises(ValidationError, match="must be user or system/user"):
        EvaluationItem.model_validate(
            {
                **item.to_dict(),
                "prompt_messages": [
                    {"role": "user", "content": "first"},
                    {"role": "user", "content": "second"},
                ],
            }
        )
    with pytest.raises(ValidationError, match="reference answer must be accepted"):
        EvaluationItem.model_validate({**item.to_dict(), "reference_answer": "5"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        EvaluationItem.model_validate({**item.to_dict(), "unknown": True})


def test_all_scorer_contracts_reject_ambiguous_references() -> None:
    choice = MultipleChoiceScorer(
        kind="multiple_choice",
        choices=("A", "B"),
        answer_index=1,
    )
    assert choice.choices[choice.answer_index] == "B"
    with pytest.raises(ValidationError, match="outside choices"):
        MultipleChoiceScorer(kind="multiple_choice", choices=("A", "B"), answer_index=2)

    expected = json.dumps({"ok": True}, separators=(",", ":"), sort_keys=True)
    json_scorer = JsonObjectScorer(
        kind="json_object",
        expected_json=expected,
        required_keys=("ok",),
    )
    assert json_scorer.expected_json == '{"ok":true}'
    with pytest.raises(ValidationError, match="canonical encoding"):
        JsonObjectScorer(
            kind="json_object",
            expected_json='{ "ok": true }',
            required_keys=("ok",),
        )

    with pytest.raises(ValidationError, match="disjoint"):
        RequiredTermsScorer(
            kind="required_terms",
            required_terms=("unknown",),
            forbidden_terms=("unknown",),
            case_sensitive=False,
        )
    with pytest.raises(ValidationError, match="exceeds criteria"):
        HumanRubricScorer(
            kind="human_rubric",
            criteria=("states uncertainty",),
            pass_threshold=2,
            retain_judgment_rationale=True,
        )


def test_build_config_rejects_count_or_version_drift() -> None:
    raw = evaluation_config().to_dict()
    raw["expected_items"] = 2
    with pytest.raises(ValidationError, match="language counts"):
        EvaluationBuildConfig.model_validate(raw)

    raw = evaluation_config().to_dict()
    raw["version_prefix"] = "other-suite-v1"
    with pytest.raises(ValidationError, match="must match suite"):
        EvaluationBuildConfig.model_validate(raw)
