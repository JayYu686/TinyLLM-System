from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts.audit_m5_r3_sources import build_parser
from tinyllm.data.m5_r3_audit import (
    M5R3AuditError,
    _family_audit,
    audit_m5_r3_sources,
    load_m5_r3_source_audit_config,
)
from tinyllm.data.reasoning_schema import (
    ReasoningLanguage,
    ReasoningSample,
    ReasoningTaskFamily,
    content_sha256,
)
from tinyllm.data.tokenization import TokenEncoding, TokenizersBackend

CONFIG = Path("configs/data/m5_r3_targeted_repair.yaml")


class _FixtureTokenizer:
    def encode(self, text: str) -> TokenEncoding:
        length = 193 if text.startswith("long") else 100
        return TokenEncoding(
            ids=tuple(range(length)),
            offsets=tuple((index, index + 1) for index in range(length)),
        )


def _sample(
    *,
    family: ReasoningTaskFamily,
    language: ReasoningLanguage,
    index: int,
    reasoning: str,
) -> ReasoningSample:
    task_id = f"m5-reasoning:pilot:{family}-{language}-{index:03d}"
    prompt = f"private fixture prompt {index}"
    final_answer = '{"result":1}'
    return ReasoningSample(
        id=f"m5-reasoning-sample:{family}-{language}-{index:03d}",
        task_id=task_id,
        task_family=family,
        language=language,
        split="pilot_train",
        template_family=f"pilot.{family}.fixture.v2",
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
        observed_token_count=200,
    )


def test_r3_config_freezes_targeted_source_gate() -> None:
    config = load_m5_r3_source_audit_config(CONFIG)

    assert config.target_families == ("config", "log_diagnosis")
    assert config.trace_policy.max_reasoning_tokens == 192
    assert config.source_requirement.selected_per_family == {
        "config": 80,
        "log_diagnosis": 80,
    }
    assert config.source_requirement.selected_languages_per_family["config"] == {
        "en": 56,
        "zh": 24,
    }
    assert config.consume_m6_frozen_results is False


def test_family_audit_excludes_overlong_trace_without_exposing_content() -> None:
    config = load_m5_r3_source_audit_config(CONFIG)
    samples = (
        _sample(family="config", language="en", index=0, reasoning="short private trace"),
        _sample(family="config", language="zh", index=1, reasoning="long private trace"),
    )

    result = _family_audit(
        samples,
        family="config",
        tokenizer=cast(TokenizersBackend, cast(Any, _FixtureTokenizer())),
        config=config,
    )

    assert result.source_items == 2
    assert result.eligible_items == 1
    assert result.eligible_language_counts == {"en": 1, "zh": 0}
    assert result.exclusion_reason_counts["reasoning_over_192_tokens"] == 1
    assert result.normalized_unique_traces == 2
    public = result.model_dump_json()
    assert "private trace" not in public
    assert "prompt" not in public


def test_family_audit_rejects_missing_target_family() -> None:
    config = load_m5_r3_source_audit_config(CONFIG)

    with pytest.raises(M5R3AuditError, match="lacks a targeted task family"):
        _family_audit(
            (),
            family="log_diagnosis",
            tokenizer=cast(TokenizersBackend, cast(Any, _FixtureTokenizer())),
            config=config,
        )


def test_r3_audit_rejects_frozen_source_hash_drift(tmp_path: Path) -> None:
    wrong_raw = tmp_path / "raw.json"
    wrong_raw.write_text("{}", encoding="utf-8")

    with pytest.raises(M5R3AuditError, match="SHA256 differs"):
        audit_m5_r3_sources(
            config_path=CONFIG,
            raw_pilot_artifact=wrong_raw,
            reasoning_config_path=Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml"),
            tokenization_config_path=Path("configs/data/m2_tokenization.yaml"),
            tokenizer_dir=tmp_path,
            r2_decision_path=Path("reports/m5/raw/m5_r2_length_diagnostic.json"),
        )


def test_r3_audit_builds_path_free_two_family_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_m5_r3_source_audit_config(CONFIG)
    paths = {
        tmp_path / "pilot.json": config.source_raw_artifact_sha256,
        tmp_path / "reasoning.yaml": config.source_reasoning_config_sha256,
        tmp_path / "tokenization.yaml": config.tokenization_config_sha256,
        tmp_path / "r2.json": config.r2_decision_sha256,
    }
    samples = (
        _sample(family="config", language="en", index=0, reasoning="short config trace"),
        _sample(
            family="log_diagnosis",
            language="zh",
            index=1,
            reasoning="short log trace",
        ),
    )
    pilot = SimpleNamespace(
        manifest=SimpleNamespace(
            dataset_version=config.source_pilot_dataset_version,
            content_sha256=config.source_pilot_content_sha256,
        ),
        samples=samples,
    )
    tokenization = SimpleNamespace(
        tokenizer=SimpleNamespace(
            tokenizer_file="tokenizer.json",
            tokenizer_config_file="tokenizer_config.json",
        )
    )
    monkeypatch.setattr(
        "tinyllm.data.m5_r3_audit._sha256_file",
        lambda path: paths[path],
    )
    monkeypatch.setattr(
        "tinyllm.data.m5_r3_audit.load_verified_reasoning_pilot",
        lambda **_kwargs: pilot,
    )
    monkeypatch.setattr(
        "tinyllm.data.m5_r3_audit.load_m2_tokenization_config",
        lambda _path: tokenization,
    )
    monkeypatch.setattr(
        "tinyllm.data.m5_r3_audit.TokenizersBackend.from_files",
        lambda *_args, **_kwargs: _FixtureTokenizer(),
    )

    result = audit_m5_r3_sources(
        config_path=CONFIG,
        raw_pilot_artifact=tmp_path / "pilot.json",
        reasoning_config_path=tmp_path / "reasoning.yaml",
        tokenization_config_path=tmp_path / "tokenization.yaml",
        tokenizer_dir=tmp_path / "tokenizer",
        r2_decision_path=tmp_path / "r2.json",
    )

    assert result.status == "insufficient_requires_new_source"
    assert result.eligible_source_items == 2
    assert result.new_teacher_source_required is True
    assert tuple(item.task_family for item in result.family_audits) == (
        "config",
        "log_diagnosis",
    )
    public = result.model_dump_json()
    assert str(tmp_path) not in public
    assert "short config trace" not in public


def test_r3_audit_cli_requires_private_source_and_tokenizer() -> None:
    args = build_parser().parse_args(
        [
            "--raw-pilot-artifact",
            "/private/pilot.json",
            "--tokenizer-dir",
            "/private/tokenizer",
            "--output",
            "reports/m5/raw/m5_r3_source_audit.json",
        ]
    )

    assert args.config == CONFIG
    assert args.raw_pilot_artifact == Path("/private/pilot.json")
    assert args.output == Path("reports/m5/raw/m5_r3_source_audit.json")
