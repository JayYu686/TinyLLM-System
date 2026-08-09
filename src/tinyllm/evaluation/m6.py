"""M6 paired comparison and atomic Candidate promotion."""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from tinyllm.evaluation.m6_schema import (
    M6BootstrapInterval,
    M6ComparisonResult,
    M6DomainModeResult,
    M6EvaluationResult,
    M6GateCheck,
    M6GeneralComparison,
    M6ModeComparison,
    M6PromotionRecord,
    M6ReleaseConfig,
)
from tinyllm.schemas import canonical_config_hash


class M6ContractError(RuntimeError):
    """Raised when M6 configuration or persisted evidence is invalid."""


class M6ComparisonError(RuntimeError):
    """Raised when two evaluations cannot be compared safely."""


class M6PromotionError(RuntimeError):
    """Raised when a model cannot be atomically promoted to Candidate."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    payload = _json_bytes(value)
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_m6_release_config(path: Path) -> M6ReleaseConfig:
    """Load the immutable M6 release policy from YAML."""

    try:
        decoded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        return M6ReleaseConfig.model_validate(decoded)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise M6ContractError("M6 release config is invalid") from exc


def load_m6_evaluation(path: Path) -> tuple[M6EvaluationResult, str]:
    """Load one complete private evaluation and return its file identity."""

    try:
        decoded: Any = json.loads(path.read_text(encoding="utf-8"))
        return M6EvaluationResult.model_validate(decoded), _sha256_file(path)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise M6ContractError("M6 evaluation result is invalid") from exc


def load_m6_comparison(path: Path) -> tuple[M6ComparisonResult, str]:
    """Load one comparison and return the exact persisted-file identity."""

    try:
        decoded: Any = json.loads(path.read_text(encoding="utf-8"))
        return M6ComparisonResult.model_validate(decoded), _sha256_file(path)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise M6ContractError("M6 comparison result is invalid") from exc


def _mode(result: M6EvaluationResult, name: str) -> M6DomainModeResult:
    for mode in result.domain_modes:
        if mode.mode == name:
            return mode
    raise M6ComparisonError(f"M6 evaluation is missing {name} mode")


def _validate_pair(
    config: M6ReleaseConfig,
    base: M6EvaluationResult,
    candidate: M6EvaluationResult,
) -> None:
    config_sha256 = canonical_config_hash(config)
    if base.model.role != "base" or candidate.model.role != "candidate":
        raise M6ComparisonError("M6 comparison requires one Base and one Candidate")
    if base.evaluation_id == candidate.evaluation_id:
        raise M6ComparisonError("M6 Base and Candidate evaluation identities must differ")
    if base.config_sha256 != config_sha256 or candidate.config_sha256 != config_sha256:
        raise M6ComparisonError("M6 evaluation config identity differs from the release policy")
    shared = (
        (base.protocol_version, candidate.protocol_version, config.protocol_version),
        (base.suite_version, candidate.suite_version, config.suite_version),
        (base.model.repository, candidate.model.repository, base.model.repository),
        (base.model.base_revision, candidate.model.base_revision, base.model.base_revision),
        (
            base.model.attention_architecture,
            candidate.model.attention_architecture,
            "gqa",
        ),
        (
            base.tokenizer_revision,
            candidate.tokenizer_revision,
            base.tokenizer_revision,
        ),
        (
            base.thinking_template_sha256,
            candidate.thinking_template_sha256,
            config.domain_execution.thinking.template_sha256,
        ),
        (
            base.nonthinking_template_sha256,
            candidate.nonthinking_template_sha256,
            config.domain_execution.nonthinking.template_sha256,
        ),
        (
            base.general_chat_template_sha256,
            candidate.general_chat_template_sha256,
            config.general_execution.tokenizer_chat_template_sha256,
        ),
    )
    if any(left != right or right != expected for left, right, expected in shared):
        raise M6ComparisonError("M6 Base and Candidate protocol or model lineage differs")
    for mode_name in ("thinking", "nonthinking"):
        base_mode = _mode(base, mode_name)
        candidate_mode = _mode(candidate, mode_name)
        left = tuple((item.item_id, item.cluster_id) for item in base_mode.items)
        right = tuple((item.item_id, item.cluster_id) for item in candidate_mode.items)
        if left != right:
            raise M6ComparisonError("M6 paired domain item or cluster identity differs")
    if tuple(task.task for task in base.general.tasks) != tuple(
        task.task for task in candidate.general.tasks
    ):
        raise M6ComparisonError("M6 general task identities differ")


def _nearest_rank(values: Sequence[int], probability: float) -> int:
    if not values:
        raise M6ComparisonError("M6 Bootstrap produced no replicates")
    rank = max(1, min(len(values), int(probability * len(values) + 0.999999999999)))
    return values[rank - 1]


def _bootstrap_interval(
    config: M6ReleaseConfig,
    base: M6DomainModeResult,
    candidate: M6DomainModeResult,
) -> M6BootstrapInterval:
    cluster_values: dict[str, list[int]] = {}
    for base_item, candidate_item in zip(base.items, candidate.items, strict=True):
        cluster_values.setdefault(base_item.cluster_id, []).append(
            int(candidate_item.correct) - int(base_item.correct)
        )
    clusters = tuple((sum(values), len(values)) for _, values in sorted(cluster_values.items()))
    rng = random.Random(config.bootstrap.seed + (0 if base.mode == "thinking" else 1))
    replicates: list[int] = []
    for _ in range(config.bootstrap.replicates):
        delta_sum = 0
        item_count = 0
        for _ in clusters:
            cluster_delta, cluster_size = clusters[rng.randrange(len(clusters))]
            delta_sum += cluster_delta
            item_count += cluster_size
        replicates.append(round(delta_sum * 10000 / item_count))
    replicates.sort()
    tail = (1.0 - config.bootstrap.confidence_basis_points / 10000) / 2
    return M6BootstrapInterval(
        replicates=config.bootstrap.replicates,
        confidence_basis_points=config.bootstrap.confidence_basis_points,
        point_delta_basis_points=(candidate.score_basis_points - base.score_basis_points),
        lower_basis_points=_nearest_rank(replicates, tail),
        upper_basis_points=_nearest_rank(replicates, 1.0 - tail),
    )


def _mode_comparison(
    config: M6ReleaseConfig,
    base: M6DomainModeResult,
    candidate: M6DomainModeResult,
) -> M6ModeComparison:
    return M6ModeComparison(
        mode=base.mode,
        base_score_basis_points=base.score_basis_points,
        candidate_score_basis_points=candidate.score_basis_points,
        delta_basis_points=candidate.score_basis_points - base.score_basis_points,
        bootstrap=_bootstrap_interval(config, base, candidate),
    )


def _numeric_check(
    name: str,
    actual: int,
    threshold: int,
    comparison: str,
    detail: str,
) -> M6GateCheck:
    operators = {
        "gte": actual >= threshold,
        "gt": actual > threshold,
        "lte": actual <= threshold,
    }
    return M6GateCheck(
        name=cast(Any, name),
        passed=operators[comparison],
        actual_basis_points=actual,
        threshold_basis_points=threshold,
        comparison=cast(Any, comparison),
        detail=detail,
    )


def _boolean_check(name: str, passed: bool, detail: str) -> M6GateCheck:
    return M6GateCheck(
        name=cast(Any, name),
        passed=passed,
        comparison="boolean",
        detail=detail,
    )


def compare_m6_evaluations(
    config: M6ReleaseConfig,
    base: M6EvaluationResult,
    candidate: M6EvaluationResult,
    *,
    base_evaluation_sha256: str,
    candidate_evaluation_sha256: str,
) -> M6ComparisonResult:
    """Apply the preregistered paired quality and lineage Gate."""

    _validate_pair(config, base, candidate)
    thinking = _mode_comparison(
        config,
        _mode(base, "thinking"),
        _mode(candidate, "thinking"),
    )
    nonthinking = _mode_comparison(
        config,
        _mode(base, "nonthinking"),
        _mode(candidate, "nonthinking"),
    )
    general = M6GeneralComparison(
        metric=config.general_metric,
        aggregation=config.general_aggregation,
        base_basis_points=base.general.aggregate_basis_points,
        candidate_basis_points=candidate.general.aggregate_basis_points,
        delta_basis_points=(
            candidate.general.aggregate_basis_points - base.general.aggregate_basis_points
        ),
    )
    candidate_thinking = _mode(candidate, "thinking")
    candidate_nonthinking = _mode(candidate, "nonthinking")
    json_valid = min(
        candidate_thinking.json_valid_basis_points,
        candidate_nonthinking.json_valid_basis_points,
    )
    evaluation_integrity = base.human_review_complete and candidate.human_review_complete
    lineage = (
        base.lineage_complete
        and candidate.lineage_complete
        and not base.git_dirty
        and not candidate.git_dirty
    )
    gate = config.gate
    checks = (
        _numeric_check(
            "thinking_domain_delta",
            thinking.delta_basis_points,
            gate.domain_min_delta_basis_points,
            "gte",
            "Thinking domain score must improve by at least 3pp.",
        ),
        _numeric_check(
            "thinking_domain_ci",
            thinking.bootstrap.lower_basis_points,
            gate.domain_ci_lower_min_exclusive_basis_points,
            "gt",
            "Thinking paired Bootstrap 95% CI lower bound must exceed zero.",
        ),
        _numeric_check(
            "nonthinking_domain_delta",
            nonthinking.delta_basis_points,
            gate.domain_min_delta_basis_points,
            "gte",
            "Non-thinking domain score must improve by at least 3pp.",
        ),
        _numeric_check(
            "nonthinking_domain_ci",
            nonthinking.bootstrap.lower_basis_points,
            gate.domain_ci_lower_min_exclusive_basis_points,
            "gt",
            "Non-thinking paired Bootstrap 95% CI lower bound must exceed zero.",
        ),
        _numeric_check(
            "general_regression",
            general.delta_basis_points,
            -gate.general_max_drop_basis_points,
            "gte",
            "Equal-task acc_norm mean may regress by at most 2pp.",
        ),
        _numeric_check(
            "json_valid_rate",
            json_valid,
            gate.json_valid_min_basis_points,
            "gte",
            "Both modes must retain at least 98% valid JSON on JSON-object items.",
        ),
        _numeric_check(
            "thinking_format",
            candidate_thinking.format_valid_basis_points,
            gate.thinking_format_min_basis_points,
            "gte",
            "Thinking controlled-format validity must reach 99%.",
        ),
        _numeric_check(
            "thinking_forced_close",
            candidate_thinking.forced_close_basis_points,
            gate.thinking_forced_close_max_basis_points,
            "lte",
            "Thinking controller forced-close rate must stay at or below 10%.",
        ),
        _numeric_check(
            "nonthinking_leakage",
            candidate_nonthinking.visible_reasoning_leakage_basis_points,
            gate.nonthinking_leakage_max_basis_points,
            "lte",
            "Non-thinking visible-reasoning leakage must remain zero.",
        ),
        _boolean_check(
            "evaluation_integrity",
            evaluation_integrity,
            "Both 300-item evaluations require complete human-rubric review.",
        ),
        _boolean_check(
            "lineage",
            lineage,
            "Base and Candidate require clean Git and complete evaluation/training lineage.",
        ),
    )
    accepted = all(check.passed for check in checks)
    return M6ComparisonResult(
        status="accepted" if accepted else "rejected",
        protocol_version=config.protocol_version,
        config_sha256=canonical_config_hash(config),
        base_evaluation_id=base.evaluation_id,
        base_evaluation_sha256=base_evaluation_sha256,
        candidate_evaluation_id=candidate.evaluation_id,
        candidate_evaluation_sha256=candidate_evaluation_sha256,
        base_model=base.model,
        candidate_model=candidate.model,
        mode_comparisons=(thinking, nonthinking),
        general_comparison=general,
        checks=checks,
        candidate_eligible=accepted,
        production_eligible=False,
    )


def write_m6_comparison(path: Path, result: M6ComparisonResult) -> None:
    """Atomically persist one comparison result."""

    _atomic_json(path, result.to_dict())


def promote_m6_candidate(
    comparison: M6ComparisonResult,
    *,
    comparison_sha256: str,
    registry_root: Path,
    now: datetime | None = None,
) -> M6PromotionRecord:
    """Atomically register an accepted M6 model as Candidate."""

    if not comparison.candidate_eligible or comparison.status != "accepted":
        raise M6PromotionError("M6 Promotion Gate rejected this model")
    if comparison.production_eligible:
        raise M6PromotionError("M6 cannot grant Production status")
    size = "0-6b" if comparison.candidate_model.repository.endswith("0.6B") else "8b"
    model_version = f"qwen3-{size}-m6-{comparison_sha256[:8]}"
    record = M6PromotionRecord(
        status="Candidate",
        model_version=model_version,
        promoted_at=now or datetime.now(UTC),
        comparison_sha256=comparison_sha256,
        comparison_config_sha256=comparison.config_sha256,
        candidate_evaluation_id=comparison.candidate_evaluation_id,
        candidate_evaluation_sha256=comparison.candidate_evaluation_sha256,
        model=comparison.candidate_model,
        production_eligible=False,
    )
    target = registry_root / "candidates" / model_version
    record_path = target / "model.json"
    if target.exists():
        try:
            existing = M6PromotionRecord.model_validate_json(
                record_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise M6PromotionError("existing M6 Candidate record is invalid") from exc
        if (
            existing.comparison_sha256 != comparison_sha256
            or existing.model != comparison.candidate_model
        ):
            raise M6PromotionError("M6 Candidate version already exists with different lineage")
        return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{model_version}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.mkdir()
        _atomic_json(temporary / "model.json", record.to_dict())
        os.replace(temporary, target)
    except OSError as exc:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise M6PromotionError("cannot atomically publish M6 Candidate record") from exc
    return record
