"""Content-free audit of existing Teacher traces for M5.2-R3 reuse."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from tinyllm.data.m5_mixture import load_verified_reasoning_pilot
from tinyllm.data.m5_r3_schema import (
    M5R3FamilySourceAudit,
    M5R3SourceAudit,
    M5R3SourceAuditConfig,
    M5R3TargetFamily,
)
from tinyllm.data.reasoning_schema import ReasoningLanguage, ReasoningSample, content_sha256
from tinyllm.data.tokenization import (
    TokenizersBackend,
    load_m2_tokenization_config,
)


class M5R3AuditError(ValueError):
    """Raised when private R3 audit inputs fail closed validation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise M5R3AuditError("M5 R3 audit input cannot be read") from exc
    return digest.hexdigest()


def load_m5_r3_source_audit_config(path: Path) -> M5R3SourceAuditConfig:
    """Load the strict R3 audit policy without allowing unknown fields."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise M5R3AuditError("M5 R3 source audit config must use YAML")
    try:
        decoded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        return M5R3SourceAuditConfig.model_validate(decoded)
    except OSError as exc:
        raise M5R3AuditError("M5 R3 source audit config cannot be read") from exc
    except yaml.YAMLError as exc:
        raise M5R3AuditError("M5 R3 source audit config is invalid YAML") from exc
    except ValidationError as exc:
        raise M5R3AuditError("M5 R3 source audit config violates its schema") from exc


def _repetition_metrics(token_ids: tuple[int, ...], text: str) -> tuple[int, int]:
    windows = tuple(tuple(token_ids[index : index + 8]) for index in range(len(token_ids) - 7))
    repeated_basis_points = (
        round((len(windows) - len(set(windows))) * 10_000 / len(windows)) if windows else 0
    )
    normalized_lines = tuple(line.strip().casefold() for line in text.splitlines() if line.strip())
    maximum_line_repeat = max(Counter(normalized_lines).values()) if normalized_lines else 1
    return repeated_basis_points, maximum_line_repeat


def _family_audit(
    samples: tuple[ReasoningSample, ...],
    *,
    family: M5R3TargetFamily,
    tokenizer: TokenizersBackend,
    config: M5R3SourceAuditConfig,
) -> M5R3FamilySourceAudit:
    selected = tuple(sample for sample in samples if sample.task_family == family)
    if not selected:
        raise M5R3AuditError("M5 R3 source Pilot lacks a targeted task family")
    token_ids = tuple(tokenizer.encode(sample.reasoning_content).ids for sample in selected)
    if any(not ids for ids in token_ids):
        raise M5R3AuditError("M5 R3 source Pilot contains empty tokenized reasoning")
    normalized = tuple(" ".join(sample.reasoning_content.split()).casefold() for sample in selected)
    normalized_counts = Counter(normalized)
    repeated = tuple(
        _repetition_metrics(ids, sample.reasoning_content)
        for ids, sample in zip(token_ids, selected, strict=True)
    )
    reasons: Counter[str] = Counter()
    eligible: list[ReasoningSample] = []
    for sample, ids, normalized_trace, (repeated_bp, maximum_line_repeat) in zip(
        selected,
        token_ids,
        normalized,
        repeated,
        strict=True,
    ):
        excluded = False
        if normalized_counts[normalized_trace] != 1:
            reasons["duplicate_normalized_trace"] += 1
            excluded = True
        if maximum_line_repeat > config.trace_policy.max_identical_line_hash_repetitions:
            reasons["identical_line_repetition"] += 1
            excluded = True
        if len(ids) > config.trace_policy.max_reasoning_tokens:
            reasons["reasoning_over_192_tokens"] += 1
            excluded = True
        if repeated_bp > config.trace_policy.max_repeated_8gram_basis_points:
            reasons["repeated_8gram_over_500bp"] += 1
            excluded = True
        if not excluded:
            eligible.append(sample)
    lengths = sorted(len(ids) for ids in token_ids)
    p90_index = math.ceil(0.9 * len(lengths)) - 1
    source_languages: Counter[ReasoningLanguage] = Counter(sample.language for sample in selected)
    eligible_languages: Counter[ReasoningLanguage] = Counter(sample.language for sample in eligible)
    return M5R3FamilySourceAudit(
        task_family=family,
        source_items=len(selected),
        source_language_counts={
            "en": source_languages["en"],
            "zh": source_languages["zh"],
        },
        reasoning_tokens_min=lengths[0],
        reasoning_tokens_p50=float(statistics.median(lengths)),
        reasoning_tokens_p90=lengths[p90_index],
        reasoning_tokens_max=lengths[-1],
        repeated_8gram_ratio_mean_basis_points=round(
            sum(item[0] for item in repeated) / len(repeated)
        ),
        normalized_unique_traces=len(normalized_counts),
        eligible_items=len(eligible),
        eligible_language_counts={
            "en": eligible_languages["en"],
            "zh": eligible_languages["zh"],
        },
        exclusion_reason_counts={
            "duplicate_normalized_trace": reasons["duplicate_normalized_trace"],
            "identical_line_repetition": reasons["identical_line_repetition"],
            "reasoning_over_192_tokens": reasons["reasoning_over_192_tokens"],
            "repeated_8gram_over_500bp": reasons["repeated_8gram_over_500bp"],
        },
    )


def audit_m5_r3_sources(
    *,
    config_path: Path,
    raw_pilot_artifact: Path,
    reasoning_config_path: Path,
    tokenization_config_path: Path,
    tokenizer_dir: Path,
    r2_decision_path: Path,
) -> M5R3SourceAudit:
    """Verify private inputs and publish only aggregate R3 reuse evidence."""

    config = load_m5_r3_source_audit_config(config_path)
    expected_hashes = (
        (raw_pilot_artifact, config.source_raw_artifact_sha256),
        (reasoning_config_path, config.source_reasoning_config_sha256),
        (tokenization_config_path, config.tokenization_config_sha256),
        (r2_decision_path, config.r2_decision_sha256),
    )
    for path, expected in expected_hashes:
        if _sha256_file(path) != expected:
            raise M5R3AuditError("M5 R3 frozen input SHA256 differs")
    pilot = load_verified_reasoning_pilot(
        raw_artifact=raw_pilot_artifact,
        reasoning_config=reasoning_config_path,
    )
    if (
        pilot.manifest.dataset_version != config.source_pilot_dataset_version
        or pilot.manifest.content_sha256 != config.source_pilot_content_sha256
    ):
        raise M5R3AuditError("M5 R3 source Pilot identity differs")
    tokenization = load_m2_tokenization_config(tokenization_config_path)
    tokenizer = TokenizersBackend.from_files(
        tokenizer_dir / tokenization.tokenizer.tokenizer_file,
        tokenizer_dir / tokenization.tokenizer.tokenizer_config_file,
        tokenization.tokenizer,
    )
    audits = tuple(
        _family_audit(
            pilot.samples,
            family=family,
            tokenizer=tokenizer,
            config=config,
        )
        for family in config.target_families
    )
    typed_audits = cast(
        tuple[M5R3FamilySourceAudit, M5R3FamilySourceAudit],
        audits,
    )
    requirements = config.source_requirement.selected_languages_per_family
    sufficient = all(
        item.eligible_items >= config.source_requirement.selected_per_family[item.task_family]
        and all(
            item.eligible_language_counts[language] >= required
            for language, required in requirements[item.task_family].items()
        )
        for item in typed_audits
    )
    return M5R3SourceAudit(
        status=(
            "sufficient_for_targeted_mixture" if sufficient else "insufficient_requires_new_source"
        ),
        audit_version=config.audit_version,
        audit_config_sha256=content_sha256(config.to_dict()),
        source_pilot_dataset_version=pilot.manifest.dataset_version,
        source_pilot_content_sha256=pilot.manifest.content_sha256,
        source_raw_artifact_sha256=config.source_raw_artifact_sha256,
        tokenizer_revision=config.tokenizer_revision,
        r2_decision_sha256=config.r2_decision_sha256,
        target_families=config.target_families,
        family_audits=typed_audits,
        eligible_source_items=sum(item.eligible_items for item in typed_audits),
        required_source_items=160,
        new_teacher_source_required=not sufficient,
        decision_reason=(
            "existing_pilot_meets_targeted_source_gate"
            if sufficient
            else "existing_pilot_lacks_concise_diverse_config_log_traces"
        ),
    )
