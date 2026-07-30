"""Evidence-bound decision logic for the M5.2-R3 Teacher-source strategy."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml
from pydantic import ValidationError

from tinyllm.data.m5_r3_p0_schema import M5R3P0Result
from tinyllm.data.m5_r3_source_strategy_schema import (
    M5R3StrategyAlternative,
    M5R3StrategyObservation,
    M5R3TeacherSourceStrategyConfig,
    M5R3TeacherSourceStrategyReview,
)
from tinyllm.data.reasoning_schema import ReasoningLanguage, content_sha256

if TYPE_CHECKING:
    from tinyllm.evaluation.m5_r2_schema import M5R2DiagnosticDecision


class M5R3SourceStrategyError(ValueError):
    """Raised when source-strategy inputs drift or contradict real evidence."""


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise M5R3SourceStrategyError("M5 R3 source-strategy input cannot be read") from exc


def load_m5_r3_teacher_source_strategy_config(
    path: Path,
) -> M5R3TeacherSourceStrategyConfig:
    """Load a strict YAML source-strategy contract."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise M5R3SourceStrategyError("M5 R3 source-strategy config must use YAML")
    try:
        decoded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        return M5R3TeacherSourceStrategyConfig.model_validate(decoded)
    except OSError as exc:
        raise M5R3SourceStrategyError("M5 R3 source-strategy config cannot be read") from exc
    except yaml.YAMLError as exc:
        raise M5R3SourceStrategyError("M5 R3 source-strategy config is invalid YAML") from exc
    except ValidationError as exc:
        raise M5R3SourceStrategyError("M5 R3 source-strategy config violates its schema") from exc


def m5_r3_teacher_source_strategy_config_sha256(
    config: M5R3TeacherSourceStrategyConfig,
) -> str:
    """Hash the canonical parsed strategy configuration."""

    return content_sha256(config.model_dump(mode="json"))


def _load_parent_result(path: Path, expected_sha256: str) -> M5R3P0Result:
    if _sha256_file(path) != expected_sha256:
        raise M5R3SourceStrategyError("M5 R3 parent P0 result SHA256 differs")
    try:
        return M5R3P0Result.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise M5R3SourceStrategyError("M5 R3 parent P0 result is invalid") from exc


def _load_r2_decision(path: Path, expected_sha256: str) -> M5R2DiagnosticDecision:
    from tinyllm.evaluation.m5_r2_schema import M5R2DiagnosticDecision

    if _sha256_file(path) != expected_sha256:
        raise M5R3SourceStrategyError("M5 R3 parent R2 decision SHA256 differs")
    try:
        return M5R2DiagnosticDecision.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise M5R3SourceStrategyError("M5 R3 parent R2 decision is invalid") from exc


def _p0_observation(
    result: M5R3P0Result,
    *,
    experiment: str,
) -> M5R3StrategyObservation:
    accepted_languages: dict[ReasoningLanguage, int] = {"en": 0, "zh": 0}
    for family in result.family_results:
        for language, count in family.accepted_language_counts.items():
            accepted_languages[language] += count
    return M5R3StrategyObservation(
        experiment=cast(Any, experiment),
        status="fail",
        accepted_samples=result.accepted_samples,
        accepted_per_family=(
            result.family_results[0].accepted_items,
            result.family_results[1].accepted_items,
        ),
        accepted_languages=accepted_languages,
        reasoning_over_192_tokens=result.rejection_counts.get(
            "reasoning_over_192_tokens",
            0,
        ),
        teacher_length_limit=result.rejection_counts.get("teacher_length_limit", 0),
    )


def review_m5_r3_teacher_source_strategy(
    *,
    config_path: Path,
    r2_decision_path: Path,
    p0_result_path: Path,
    p0_r1_result_path: Path,
) -> M5R3TeacherSourceStrategyReview:
    """Select the next bounded source experiment from committed parent evidence."""

    config = load_m5_r3_teacher_source_strategy_config(config_path)
    r2 = _load_r2_decision(r2_decision_path, config.parent_r2_decision_sha256)
    p0 = _load_parent_result(p0_result_path, config.parent_p0_result_sha256)
    p0_r1 = _load_parent_result(p0_r1_result_path, config.parent_p0_r1_result_sha256)
    if (
        r2.status != "length_ceiling_insufficient"
        or p0.status != "fail"
        or p0.pilot_version != "m5-r3-p0-v1"
        or p0_r1.status != "fail"
        or p0_r1.pilot_version != "m5-r3-p0-r1-v1"
    ):
        raise M5R3SourceStrategyError("M5 R3 parent evidence does not support strategy review")

    observations = (
        M5R3StrategyObservation(
            experiment="r2",
            status=r2.status,
            projected_format_basis_points=(
                r2.projections[0].projected_format_basis_points,
                r2.projections[1].projected_format_basis_points,
            ),
            unresolved_format_items=(
                r2.projections[0].unresolved_format_items,
                r2.projections[1].unresolved_format_items,
            ),
        ),
        _p0_observation(p0, experiment="p0"),
        _p0_observation(p0_r1, experiment="p0_r1"),
    )
    alternatives = (
        M5R3StrategyAlternative(
            strategy="single_stage_prompt_control",
            disposition="rejected",
            evidence_reason="p0_and_p0_r1_failed_same_gate",
        ),
        M5R3StrategyAlternative(
            strategy="higher_generation_ceiling_only",
            disposition="rejected",
            evidence_reason="r2_1536_projection_failed_99_percent",
        ),
        M5R3StrategyAlternative(
            strategy="two_stage_solve_compress",
            disposition="selected_for_p1",
            evidence_reason="separates_correctness_from_length_control",
        ),
        M5R3StrategyAlternative(
            strategy="deterministic_rule_trace",
            disposition="control_only",
            evidence_reason="deterministic_control_not_formal_teacher_source",
        ),
    )
    return M5R3TeacherSourceStrategyReview(
        status="two_stage_contract_authorized",
        evidence_kind="deterministic_review_of_real_public_results",
        quality_metric=False,
        review_version=config.review_version,
        review_config_sha256=m5_r3_teacher_source_strategy_config_sha256(config),
        parent_r2_decision_sha256=config.parent_r2_decision_sha256,
        parent_p0_result_sha256=config.parent_p0_result_sha256,
        parent_p0_r1_result_sha256=config.parent_p0_r1_result_sha256,
        observations=observations,
        alternatives=alternatives,
        selected_strategy=config.selected_strategy,
        controlled_baseline=config.controlled_baseline,
        next_pilot_version=config.pilot.pilot_version,
        p1_contract_implementation_authorized=True,
        p1_gpu_pilot_authorized=False,
        formal_source_expansion_authorized=config.formal_source_expansion_authorized,
        r3_mixture_authorized=config.r3_mixture_authorized,
        r3_training_authorized=config.r3_training_authorized,
        decision_reason=("single_stage_length_control_failed_select_two_stage_with_rule_control"),
    )
