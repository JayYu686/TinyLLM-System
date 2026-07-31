from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tinyllm.data import (
    M5MixtureSequence,
    M5R3MixtureError,
    M5R3MixtureManifest,
    M5R3TargetedSource,
    ReasoningSample,
    load_m5_r3_mixture_config,
    m5_r3_mixture_config_sha256,
    select_exact_supervised_tokens_capped,
    select_m5_r3_targeted_sources,
)
from tinyllm.data import m5_r3_mixture as mixture_module
from tinyllm.data.m5_r3_schema import M5R3TargetFamily
from tinyllm.data.reasoning_schema import ReasoningLanguage, canonical_json, content_sha256

CONFIG = Path("configs/data/m5_r3_mixture_v2.yaml")


def _sequence(supervised_tokens: int) -> M5MixtureSequence:
    labels = [-100] * 1024
    for index in range(1, supervised_tokens + 1):
        labels[index] = index
    return M5MixtureSequence(
        input_ids=tuple(range(1024)),
        labels=tuple(labels),
        attention_mask=(1,) * 1024,
        mode=2,
    )


def _source(
    *,
    family: M5R3TargetFamily,
    language: ReasoningLanguage,
    label: str,
    index: int,
    reasoning_tokens: int,
) -> M5R3TargetedSource:
    short_family = "config" if family == "config" else "log"
    task_id = f"m5-reasoning:pilot:r3formal-{short_family}-{language}-{index:03d}"
    prompt = f"prompt-{family}-{language}-{label}-{index}"
    reasoning = f"evidence supports {label}"
    label_key = "issue" if family == "config" else "root_cause"
    final_answer = canonical_json({label_key: label})
    sample = ReasoningSample(
        id=f"m5-reasoning-sample:r3formal-{short_family}-{language}-{index:03d}",
        task_id=task_id,
        task_family=family,
        language=language,
        split="pilot_train",
        template_family=f"pilot.{family}.r3-two-stage-formal.v1",
        prompt=prompt,
        reasoning_content=reasoning,
        final_answer=final_answer,
        generation_id=f"{task_id}:candidate-0",
        verification_id=f"{task_id}:verify-0",
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        raw_output_sha256="a" * 64,
        content_sha256=content_sha256(
            {
                "final_answer": final_answer,
                "prompt": prompt,
                "reasoning_content": reasoning,
            }
        ),
        observed_token_count=100,
    )
    return M5R3TargetedSource(
        sample=sample,
        label=label,
        reasoning_tokens=reasoning_tokens,
        repeated_8gram_basis_points=0,
    )


def _quota_sources() -> tuple[M5R3TargetedSource, ...]:
    config = load_m5_r3_mixture_config(CONFIG)
    sources: list[M5R3TargetedSource] = []
    index = 0
    for family, languages in config.selection.quotas.items():
        for language, labels in languages.items():
            for label, required in labels.items():
                for offset in range(required + 1):
                    sources.append(
                        _source(
                            family=family,
                            language=language,
                            label=label,
                            index=index,
                            reasoning_tokens=10 + offset,
                        )
                    )
                    index += 1
    return tuple(sources)


def test_m5_r3_mixture_v2_policy_is_frozen() -> None:
    config = load_m5_r3_mixture_config(CONFIG)

    assert (
        m5_r3_mixture_config_sha256(config)
        == "68fe849f097baa2c60660d2db7a45af55b0338e7aeb91c12d0df38653b3b16a7"
    )
    assert config.selection.max_source_uses == 30
    assert (
        sum(
            count
            for family in config.selection.quotas.values()
            for language in family.values()
            for count in language.values()
        )
        == 160
    )


def test_m5_r3_mixture_config_rejects_bad_path_or_payload(tmp_path: Path) -> None:
    with pytest.raises(M5R3MixtureError, match="YAML"):
        load_m5_r3_mixture_config(tmp_path / "config.json")
    with pytest.raises(M5R3MixtureError, match="cannot be read"):
        load_m5_r3_mixture_config(tmp_path / "missing.yaml")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: ['wrong']\n", encoding="utf-8")
    with pytest.raises(M5R3MixtureError, match="violates"):
        load_m5_r3_mixture_config(invalid)


def test_formal_source_loader_fails_closed_on_hash_or_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_m5_r3_mixture_config(CONFIG)
    public = tmp_path / "public.json"
    raw = tmp_path / "raw.json"
    public.write_text("{}\n", encoding="utf-8")
    raw.write_text("{}\n", encoding="utf-8")
    with pytest.raises(M5R3MixtureError, match="public result SHA256"):
        mixture_module.load_verified_m5_r3_sources(
            config=config,
            formal_result_path=public,
            formal_raw_artifact=raw,
        )

    monkeypatch.setattr(
        mixture_module,
        "_sha256_file",
        lambda path: (
            config.formal_source.result_sha256
            if path == public
            else config.formal_source.raw_artifact_sha256
        ),
    )
    with pytest.raises(M5R3MixtureError, match="cannot be parsed"):
        mixture_module.load_verified_m5_r3_sources(
            config=config,
            formal_result_path=public,
            formal_raw_artifact=raw,
        )


def test_m5_r3_real_manifest_satisfies_exact_budget_and_exposure_cap() -> None:
    manifest = M5R3MixtureManifest.model_validate_json(
        Path("reports/m5/raw/m5_r3_mixture.json").read_text(encoding="utf-8")
    )

    assert manifest.mixture_version == "m5-r3-mixture-v2-b47723e1"
    assert manifest.target_supervised_tokens == 1_000_000
    assert manifest.nonthinking_supervised_tokens == 700_000
    assert manifest.general_thinking_supervised_tokens == 150_000
    assert manifest.targeted_thinking_supervised_tokens == 150_000
    assert manifest.targeted_thinking_source_sequences == 160
    assert manifest.targeted_source_use_min == 29
    assert manifest.targeted_source_use_max == 30
    assert manifest.r3_training_authorized is True
    assert manifest.consume_m6_frozen_results is False


def test_label_aware_selection_is_deterministic_and_satisfies_every_quota() -> None:
    config = load_m5_r3_mixture_config(CONFIG)
    sources = _quota_sources()

    selected = select_m5_r3_targeted_sources(sources, config=config)

    assert selected == select_m5_r3_targeted_sources(sources, config=config)
    assert len(selected) == len({source.sample.id for source in selected}) == 160
    for family, languages in config.selection.quotas.items():
        for language, labels in languages.items():
            for label, required in labels.items():
                assert (
                    sum(
                        source.sample.task_family == family
                        and source.sample.language == language
                        and source.label == label
                        for source in selected
                    )
                    == required
                )


def test_label_aware_selection_rejects_missing_stratum() -> None:
    config = load_m5_r3_mixture_config(CONFIG)
    sources = tuple(
        source for source in _quota_sources() if source.label != "unsupported_precision"
    )

    with pytest.raises(M5R3MixtureError, match="insufficient"):
        select_m5_r3_targeted_sources(sources, config=config)


def test_capped_selector_reaches_exact_target_without_exceeding_limit() -> None:
    selected, reuse_count, partial, uses = select_exact_supervised_tokens_capped(
        (_sequence(7), _sequence(5), _sequence(3)),
        target=28,
        seed=42,
        max_source_uses=3,
    )

    assert sum(item.supervised_tokens for item in selected) == 28
    assert reuse_count == sum(max(count - 1, 0) for count in uses)
    assert partial == 1
    assert min(uses) > 0
    assert max(uses) <= 3


def test_capped_selector_rejects_infeasible_budget() -> None:
    with pytest.raises(M5R3MixtureError, match="exceeds"):
        select_exact_supervised_tokens_capped(
            (_sequence(7), _sequence(5)),
            target=25,
            seed=42,
            max_source_uses=2,
        )
