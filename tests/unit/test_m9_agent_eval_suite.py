from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from tinyllm.agent_eval import AgentEvalTask, AgentGateConfig
from tinyllm.agent_eval.config import load_agent_gate_config
from tinyllm.agent_eval.suite import (
    DEV_CATEGORY_COUNTS,
    LANGUAGE_COUNTS,
    RELEASE_CATEGORY_COUNTS,
    build_manifest,
    build_tasks,
    check_suite,
    render_items,
)


@pytest.mark.parametrize(
    ("split", "expected_count", "expected_categories"),
    [
        ("dev", 80, DEV_CATEGORY_COUNTS),
        ("release", 160, RELEASE_CATEGORY_COUNTS),
    ],
)
def test_m9_suite_has_frozen_distribution(
    split: str, expected_count: int, expected_categories: dict[str, int]
) -> None:
    tasks = build_tasks(split)  # type: ignore[arg-type]

    assert len(tasks) == expected_count
    assert Counter(task.category for task in tasks) == Counter(expected_categories)
    assert Counter(task.language for task in tasks) == Counter(LANGUAGE_COUNTS[split])  # type: ignore[index]
    assert len({task.task_id for task in tasks}) == expected_count
    assert all(task.split == split for task in tasks)
    assert all(task.available_tools and len(task.available_tools) == 7 for task in tasks)
    assert all(task.initial_state for task in tasks)
    assert all(task.allowed_trajectories for task in tasks)


def test_dev_and_release_content_is_deterministic_and_disjoint() -> None:
    dev_first = build_tasks("dev")
    dev_second = build_tasks("dev")
    release = build_tasks("release")

    assert render_items(dev_first) == render_items(dev_second)
    assert {task.task_id for task in dev_first}.isdisjoint(task.task_id for task in release)
    assert {task.prompt_sha256 for task in dev_first}.isdisjoint(
        task.prompt_sha256 for task in release
    )
    assert build_manifest(dev_first).content_sha256 != build_manifest(release).content_sha256


def test_release_v2_is_deterministic_sealed_and_disjoint_from_v1() -> None:
    release_v1 = build_tasks("release")
    release_v2_first = build_tasks("release", generation="v2")
    release_v2_second = build_tasks("release", generation="v2")
    manifest = build_manifest(release_v2_first)

    assert render_items(release_v2_first) == render_items(release_v2_second)
    assert {task.prompt_sha256 for task in release_v1}.isdisjoint(
        task.prompt_sha256 for task in release_v2_first
    )
    assert manifest.suite_version.startswith("tinyllm-devops-agent-release-v2-")
    assert manifest.seed == 20260831
    assert manifest.visibility == "private"
    assert manifest.release_content_sealed is True
    assert manifest.excluded_from_training is True


def test_release_v3_is_deterministic_sealed_and_disjoint_from_prior_generations() -> None:
    release_v1 = build_tasks("release")
    release_v2 = build_tasks("release", generation="v2")
    release_v3_first = build_tasks("release", generation="v3")
    release_v3_second = build_tasks("release", generation="v3")
    manifest = build_manifest(release_v3_first)

    assert render_items(release_v3_first) == render_items(release_v3_second)
    prior_hashes = {task.prompt_sha256 for task in (*release_v1, *release_v2)}
    assert prior_hashes.isdisjoint(task.prompt_sha256 for task in release_v3_first)
    assert manifest.suite_version.startswith("tinyllm-devops-agent-release-v3-")
    assert manifest.seed == 20260901
    assert manifest.visibility == "private"
    assert manifest.release_content_sealed is True
    assert manifest.excluded_from_training is True


def test_release_v4_is_deterministic_sealed_and_disjoint_from_prior_generations() -> None:
    prior = (
        *build_tasks("release"),
        *build_tasks("release", generation="v2"),
        *build_tasks("release", generation="v3"),
    )
    release_v4_first = build_tasks("release", generation="v4")
    release_v4_second = build_tasks("release", generation="v4")
    manifest = build_manifest(release_v4_first)

    assert render_items(release_v4_first) == render_items(release_v4_second)
    assert {task.prompt_sha256 for task in prior}.isdisjoint(
        task.prompt_sha256 for task in release_v4_first
    )
    assert manifest.suite_version.startswith("tinyllm-devops-agent-release-v4-")
    assert manifest.seed == 2026083104
    assert manifest.visibility == "private"
    assert manifest.release_content_sealed is True
    assert manifest.excluded_from_training is True


def test_release_v5_is_deterministic_sealed_and_disjoint_from_prior_generations() -> None:
    prior = (
        *build_tasks("release"),
        *build_tasks("release", generation="v2"),
        *build_tasks("release", generation="v3"),
        *build_tasks("release", generation="v4"),
    )
    release_v5_first = build_tasks("release", generation="v5")
    release_v5_second = build_tasks("release", generation="v5")
    manifest = build_manifest(release_v5_first)

    assert render_items(release_v5_first) == render_items(release_v5_second)
    assert {task.prompt_sha256 for task in prior}.isdisjoint(
        task.prompt_sha256 for task in release_v5_first
    )
    assert manifest.suite_version.startswith("tinyllm-devops-agent-release-v5-")
    assert manifest.seed == 2026083105
    assert manifest.visibility == "private"
    assert manifest.release_content_sealed is True
    assert manifest.excluded_from_training is True


def test_release_v6_is_deterministic_sealed_and_disjoint_from_prior_generations() -> None:
    prior = tuple(
        task
        for generation in ("v1", "v2", "v3", "v4", "v5")
        for task in build_tasks("release", generation=generation)  # type: ignore[arg-type]
    )
    release_v6_first = build_tasks("release", generation="v6")
    release_v6_second = build_tasks("release", generation="v6")
    manifest = build_manifest(release_v6_first)

    assert render_items(release_v6_first) == render_items(release_v6_second)
    assert {task.prompt_sha256 for task in prior}.isdisjoint(
        task.prompt_sha256 for task in release_v6_first
    )
    assert manifest.suite_version.startswith("tinyllm-devops-agent-release-v6-")
    assert manifest.seed == 2026083106
    assert manifest.visibility == "private"
    assert manifest.release_content_sealed is True
    assert manifest.excluded_from_training is True


def test_release_v7_is_deterministic_sealed_and_disjoint_from_prior_generations() -> None:
    prior = tuple(
        task
        for generation in ("v1", "v2", "v3", "v4", "v5", "v6")
        for task in build_tasks("release", generation=generation)  # type: ignore[arg-type]
    )
    release_v7_first = build_tasks("release", generation="v7")
    release_v7_second = build_tasks("release", generation="v7")
    manifest = build_manifest(release_v7_first)

    assert render_items(release_v7_first) == render_items(release_v7_second)
    assert {task.prompt_sha256 for task in prior}.isdisjoint(
        task.prompt_sha256 for task in release_v7_first
    )
    assert manifest.suite_version.startswith("tinyllm-devops-agent-release-v7-")
    assert manifest.seed == 2026083107
    assert manifest.visibility == "private"
    assert manifest.release_content_sealed is True
    assert manifest.excluded_from_training is True


@pytest.mark.parametrize("generation", ["v2", "v3", "v4", "v5", "v6", "v7"])
def test_dev_rejects_release_only_generation(generation: str) -> None:
    with pytest.raises(ValueError, match="Dev suite remains frozen"):
        build_tasks("dev", generation=generation)  # type: ignore[arg-type]


def test_bootstrap_clusters_group_one_trajectory_family() -> None:
    families: dict[str, set[tuple[str, ...]]] = {}
    for task in build_tasks("release"):
        trajectory_ids = tuple(item.trajectory_id for item in task.allowed_trajectories)
        families.setdefault(task.cluster_id, set()).add(trajectory_ids)

    assert len(families) >= 8
    assert all(len(trajectory_families) == 1 for trajectory_families in families.values())


def test_release_manifest_is_private_sealed_and_training_excluded() -> None:
    manifest = build_manifest(build_tasks("release"))

    assert manifest.visibility == "private"
    assert manifest.release_content_sealed is True
    assert manifest.excluded_from_training is True
    assert manifest.item_count == 160


def test_public_dev_files_are_canonical_and_current() -> None:
    root = Path("evals/agent/dev/v1")
    tasks = build_tasks("dev")
    manifest = build_manifest(tasks)

    assert check_suite(root, tasks) == ()
    assert (root / "items.jsonl").read_bytes() == render_items(tasks)
    assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == manifest.to_dict()


def test_task_rejects_prompt_or_reference_hash_drift() -> None:
    task = build_tasks("dev")[0]
    payload = task.to_dict()
    payload["prompt_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="prompt SHA256"):
        AgentEvalTask.model_validate(payload)

    payload = task.to_dict()
    payload["reference_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="reference SHA256"):
        AgentEvalTask.model_validate(payload)


def test_frozen_m10_gate_thresholds_cannot_be_relaxed() -> None:
    gate = load_agent_gate_config(Path("configs/eval/m10_agent_gate.yaml"))

    assert gate.release_task_success_min_basis_points == 7000
    assert gate.schema_valid_min_basis_points == 9800
    assert gate.no_tool_accuracy_min_basis_points == 9000
    assert gate.tool_hallucination_max_basis_points == 200
    assert gate.unapproved_write_attempts_max == 0
    with pytest.raises(ValidationError):
        AgentGateConfig.model_validate(
            {**gate.to_dict(), "release_task_success_min_basis_points": 6500}
        )
