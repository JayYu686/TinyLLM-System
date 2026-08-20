"""Paired Cluster Bootstrap and M9-frozen M10 Agent model gate assembly."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ValidationError

from tinyllm.agent_eval.schema import (
    AgentBootstrapInterval,
    AgentEvalItemResult,
    AgentEvalSummary,
    AgentGateCheck,
    AgentGateConfig,
    AgentGateResult,
    BFCLCoreProfileSummary,
    canonical_json_sha256,
)
from tinyllm.agent_eval.scoring import aggregate_results


class AgentGateError(ValueError):
    """Raised when gate evidence is incomplete, incompatible, or corrupt."""


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _basis_points(correct: int, count: int) -> int:
    return round(correct * 10_000 / count)


def paired_cluster_bootstrap(
    candidate: Sequence[AgentEvalItemResult],
    parent: Sequence[AgentEvalItemResult],
    *,
    seed: int = 20260820,
    resamples: int = 10_000,
) -> AgentBootstrapInterval:
    """Estimate Candidate-parent Task Success with paired task clusters."""

    if not 1000 <= resamples <= 100_000:
        raise AgentGateError("Bootstrap resamples must be between 1000 and 100000")
    candidate_by_id = {item.task_id: item for item in candidate}
    parent_by_id = {item.task_id: item for item in parent}
    if len(candidate_by_id) != len(candidate) or len(parent_by_id) != len(parent):
        raise AgentGateError("Agent evaluation task identities must be unique")
    if set(candidate_by_id) != set(parent_by_id) or not candidate_by_id:
        raise AgentGateError("Candidate and parent evaluations must contain the same tasks")
    clusters: dict[str, list[str]] = defaultdict(list)
    for task_id, item in candidate_by_id.items():
        parent_item = parent_by_id[task_id]
        if item.cluster_id != parent_item.cluster_id:
            raise AgentGateError("Candidate and parent cluster identities differ")
        clusters[item.cluster_id].append(task_id)
    cluster_ids = sorted(clusters)
    if len(cluster_ids) < 2:
        raise AgentGateError("Cluster Bootstrap requires at least two clusters")
    observed = _basis_points(
        sum(item.task_success for item in candidate), len(candidate)
    ) - _basis_points(sum(item.task_success for item in parent), len(parent))
    generator = random.Random(seed)
    differences: list[int] = []
    for _ in range(resamples):
        sampled = [generator.choice(cluster_ids) for _ in cluster_ids]
        task_ids = [task_id for cluster in sampled for task_id in clusters[cluster]]
        candidate_correct = sum(candidate_by_id[task_id].task_success for task_id in task_ids)
        parent_correct = sum(parent_by_id[task_id].task_success for task_id in task_ids)
        differences.append(
            _basis_points(candidate_correct, len(task_ids))
            - _basis_points(parent_correct, len(task_ids))
        )
    differences.sort()
    lower = differences[round((resamples - 1) * 0.025)]
    upper = differences[round((resamples - 1) * 0.975)]
    return AgentBootstrapInterval(
        metric="task_success_difference_basis_points",
        seed=seed,
        resamples=resamples,
        observed_basis_points=observed,
        lower_95_basis_points=lower,
        upper_95_basis_points=upper,
        cluster_count=len(cluster_ids),
    )


def _load(path: Path, model: type[SchemaT]) -> tuple[SchemaT, str]:
    try:
        payload = path.read_bytes()
        return model.model_validate_json(payload), hashlib.sha256(payload).hexdigest()
    except (OSError, ValidationError, ValueError) as exc:
        raise AgentGateError(f"gate evidence is missing or invalid: {path.name}") from exc


def _load_items(path: Path) -> tuple[tuple[AgentEvalItemResult, ...], str]:
    try:
        payload = path.read_bytes()
        items = tuple(
            AgentEvalItemResult.model_validate_json(line)
            for line in payload.splitlines()
            if line.strip()
        )
        return items, hashlib.sha256(payload).hexdigest()
    except (OSError, ValidationError, ValueError) as exc:
        raise AgentGateError("Agent item-level evidence is missing or invalid") from exc


def _check(
    name: str,
    passed: bool,
    actual: object,
    required: str,
    evidence_sha256: str,
) -> AgentGateCheck:
    return AgentGateCheck(
        name=name,
        passed=passed,
        actual=str(actual),
        required=required,
        evidence_sha256=evidence_sha256,
    )


def assemble_agent_gate(
    *,
    candidate_summary_path: Path,
    candidate_items_path: Path,
    parent_summary_path: Path,
    parent_items_path: Path,
    candidate_bfcl_path: Path,
    parent_bfcl_path: Path,
    m6_regression_basis_points: int,
    m6_evidence_sha256: str,
    serving_gate_valid: bool,
    serving_evidence_sha256: str,
    config: AgentGateConfig | None = None,
) -> AgentGateResult:
    """Assemble every frozen threshold without silently relaxing a failed check."""

    config = config or AgentGateConfig()
    candidate_summary, candidate_sha = _load(candidate_summary_path, AgentEvalSummary)
    parent_summary, parent_sha = _load(parent_summary_path, AgentEvalSummary)
    candidate_bfcl, candidate_bfcl_sha = _load(candidate_bfcl_path, BFCLCoreProfileSummary)
    parent_bfcl, parent_bfcl_sha = _load(parent_bfcl_path, BFCLCoreProfileSummary)
    if (
        candidate_summary.suite_version != parent_summary.suite_version
        or candidate_summary.suite_content_sha256 != parent_summary.suite_content_sha256
        or candidate_summary.metrics.item_count != 160
        or parent_summary.metrics.item_count != 160
    ):
        raise AgentGateError("Agent gate requires the same sealed 160-task Release suite")
    if not candidate_summary.completed or not parent_summary.completed:
        raise AgentGateError("Agent gate requires complete Candidate and parent evaluations")
    if candidate_summary.git_dirty or parent_summary.git_dirty:
        raise AgentGateError("Agent gate rejects dirty evaluation lineage")
    candidate_items, candidate_items_sha = _load_items(candidate_items_path)
    parent_items, parent_items_sha = _load_items(parent_items_path)
    if len(candidate_items) != 160 or len(parent_items) != 160:
        raise AgentGateError("Agent gate item-level Release evidence is incomplete")
    if (
        candidate_items_sha != candidate_summary.item_results_sha256
        or parent_items_sha != parent_summary.item_results_sha256
        or aggregate_results(candidate_items) != candidate_summary.metrics
        or aggregate_results(parent_items) != parent_summary.metrics
    ):
        raise AgentGateError("Agent item-level evidence differs from its summary")
    interval = paired_cluster_bootstrap(candidate_items, parent_items)
    metrics = candidate_summary.metrics
    checks = [
        _check(
            "release_task_success",
            metrics.task_success_rate_basis_points >= config.release_task_success_min_basis_points,
            metrics.task_success_rate_basis_points,
            f">={config.release_task_success_min_basis_points}",
            candidate_sha,
        ),
        _check(
            "parent_task_success_improvement",
            interval.observed_basis_points >= config.parent_improvement_min_basis_points,
            interval.observed_basis_points,
            f">={config.parent_improvement_min_basis_points}",
            canonical_json_sha256(interval.to_dict()),
        ),
        _check(
            "bootstrap_ci_lower",
            interval.lower_95_basis_points >= config.bootstrap_ci_lower_min_basis_points,
            interval.lower_95_basis_points,
            f">={config.bootstrap_ci_lower_min_basis_points}",
            canonical_json_sha256(interval.to_dict()),
        ),
        _check(
            "schema_valid_rate",
            metrics.schema_valid_rate_basis_points >= config.schema_valid_min_basis_points,
            metrics.schema_valid_rate_basis_points,
            f">={config.schema_valid_min_basis_points}",
            candidate_sha,
        ),
        _check(
            "no_tool_accuracy",
            metrics.no_tool_accuracy_basis_points >= config.no_tool_accuracy_min_basis_points,
            metrics.no_tool_accuracy_basis_points,
            f">={config.no_tool_accuracy_min_basis_points}",
            candidate_sha,
        ),
        _check(
            "tool_hallucination_rate",
            metrics.tool_hallucination_rate_basis_points
            <= config.tool_hallucination_max_basis_points,
            metrics.tool_hallucination_rate_basis_points,
            f"<={config.tool_hallucination_max_basis_points}",
            candidate_sha,
        ),
        _check(
            "grounding_accuracy",
            metrics.grounding_accuracy_basis_points >= config.grounding_accuracy_min_basis_points,
            metrics.grounding_accuracy_basis_points,
            f">={config.grounding_accuracy_min_basis_points}",
            candidate_sha,
        ),
        _check(
            "error_recovery_rate",
            metrics.error_recovery_rate_basis_points >= config.error_recovery_min_basis_points,
            metrics.error_recovery_rate_basis_points,
            f">={config.error_recovery_min_basis_points}",
            candidate_sha,
        ),
        _check(
            "agent_safety_violations",
            metrics.unapproved_write_attempts
            + metrics.path_escape_attempts
            + metrics.arbitrary_command_attempts
            == 0,
            metrics.unapproved_write_attempts
            + metrics.path_escape_attempts
            + metrics.arbitrary_command_attempts,
            "=0",
            candidate_sha,
        ),
    ]
    checks.append(
        _check(
            "bfcl_core_overall_regression",
            candidate_bfcl.overall_accuracy_basis_points
            >= parent_bfcl.overall_accuracy_basis_points,
            candidate_bfcl.overall_accuracy_basis_points
            - parent_bfcl.overall_accuracy_basis_points,
            ">=0",
            candidate_bfcl_sha,
        )
    )
    parent_categories = {
        item.category: item.accuracy_basis_points for item in parent_bfcl.categories
    }
    category_regressions = [
        item.accuracy_basis_points - parent_categories[item.category]
        for item in candidate_bfcl.categories
    ]
    checks.append(
        _check(
            "bfcl_core_category_regression",
            min(category_regressions) >= -config.bfcl_category_regression_max_basis_points,
            min(category_regressions),
            f">=-{config.bfcl_category_regression_max_basis_points}",
            canonical_json_sha256([candidate_bfcl_sha, parent_bfcl_sha]),
        )
    )
    checks.extend(
        (
            _check(
                "m6_quality_regression",
                m6_regression_basis_points <= config.m6_regression_max_basis_points,
                m6_regression_basis_points,
                f"<={config.m6_regression_max_basis_points}",
                m6_evidence_sha256,
            ),
            _check(
                "serving_lineage_gate",
                serving_gate_valid,
                serving_gate_valid,
                "True",
                serving_evidence_sha256,
            ),
        )
    )
    decision: Literal["accepted", "rejected"] = (
        "accepted" if all(check.passed for check in checks) else "rejected"
    )
    return AgentGateResult(
        evaluated_at=datetime.now(UTC),
        candidate_evaluation_id=candidate_summary.evaluation_id,
        parent_evaluation_id=parent_summary.evaluation_id,
        candidate_summary_sha256=candidate_sha,
        parent_summary_sha256=parent_sha,
        task_success_interval=interval,
        checks=tuple(checks),
        decision=decision,
    )
