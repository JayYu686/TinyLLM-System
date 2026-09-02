from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tinyllm.agent_eval.gate import AgentGateError, assemble_agent_gate, paired_cluster_bootstrap
from tinyllm.agent_eval.schema import (
    AgentEvalItemResult,
    AgentEvalSummary,
    BFCLCategoryResult,
    BFCLCoreProfileSummary,
)
from tinyllm.agent_eval.scoring import aggregate_results
from tinyllm.agent_eval.suite import build_manifest, build_tasks


def _items(*, candidate: bool) -> tuple[AgentEvalItemResult, ...]:
    tasks = build_tasks("release")
    cluster_ordinals: dict[str, int] = {}
    results: list[AgentEvalItemResult] = []
    for index, task in enumerate(tasks):
        ordinal = cluster_ordinals.get(task.cluster_id, 0)
        cluster_ordinals[task.cluster_id] = ordinal + 1
        success = candidate or ordinal < 4
        results.append(
            AgentEvalItemResult(
                task_id=task.task_id,
                cluster_id=task.cluster_id,
                category=task.category,
                language=task.language,
                run_id=f"agent-gate-{index:03d}",
                status="succeeded",
                final_answer="evidence-grounded answer",
                duration_milliseconds=10,
                input_tokens=10,
                output_tokens=5,
                tool_selection_correct=True,
                argument_correct=True,
                schema_valid=True,
                task_success=success,
                tool_hallucination=False,
            )
        )
    return tuple(results)


def _jsonl(items: tuple[AgentEvalItemResult, ...]) -> bytes:
    return b"".join(
        json.dumps(
            item.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        + b"\n"
        for item in items
    )


def _summary(items: tuple[AgentEvalItemResult, ...], *, candidate: bool) -> AgentEvalSummary:
    payload = _jsonl(items)
    suite = build_manifest(build_tasks("release"))
    return AgentEvalSummary(
        evaluation_id=f"m9-agent-eval-{'c' if candidate else 'a'}1234567",
        evaluated_at=datetime(2026, 8, 20, tzinfo=UTC),
        suite_version=suite.suite_version,
        suite_content_sha256=suite.content_sha256,
        model_id="candidate" if candidate else "parent",
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        model_artifact_sha256=("c" if candidate else "a") * 64,
        parent_model_id="Qwen/Qwen3-0.6B",
        deployment_record_sha256="d" * 64,
        environment_sha256="1" * 64,
        hardware_sha256="2" * 64,
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        driver_version="535.261.03",
        gateway_version="0.9.0rc1",
        agent_runtime_version="0.9.0rc1",
        git_commit="e" * 40,
        git_dirty=False,
        metrics=aggregate_results(items),
        item_results_sha256=hashlib.sha256(payload).hexdigest(),
        completed=True,
    )


def _bfcl(*, candidate: bool) -> BFCLCoreProfileSummary:
    counts = {
        "simple": 400,
        "multiple": 200,
        "parallel": 200,
        "parallel_multiple": 200,
        "irrelevance": 240,
        "multi_turn_base": 200,
        "multi_turn_miss_func": 200,
        "multi_turn_miss_param": 200,
    }
    categories = tuple(
        BFCLCategoryResult(
            category=category,  # type: ignore[arg-type]
            item_count=count,
            correct_items=count // 2,
            accuracy_basis_points=5000,
            source_score_sha256=hashlib.sha256(category.encode()).hexdigest(),
        )
        for category, count in counts.items()
    )
    return BFCLCoreProfileSummary(
        profile_name="TinyLLM BFCL v1.3 Offline Core Profile",
        bfcl_tag="v1.3",
        bfcl_commit="ea13468e4423454d0c213704fb87cf7cb3990433",
        evaluated_at=datetime(2026, 8, 20, tzinfo=UTC),
        model_id="candidate" if candidate else "parent",
        model_artifact_sha256=("c" if candidate else "a") * 64,
        endpoint_handler="tinyllm-openai-chat-completions-v1",
        categories=categories,
        total_items=1840,
        correct_items=920,
        overall_accuracy_basis_points=5000,
        raw_results_sha256=("f" if candidate else "b") * 64,
        completed=True,
    )


def _write(path: Path, value: object) -> None:
    data = value.to_dict()  # type: ignore[attr-defined]
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_paired_cluster_bootstrap_is_deterministic_and_positive() -> None:
    candidate = _items(candidate=True)
    parent = _items(candidate=False)

    first = paired_cluster_bootstrap(candidate, parent, resamples=1000)
    second = paired_cluster_bootstrap(candidate, parent, resamples=1000)

    assert first == second
    assert first.observed_basis_points >= 500
    assert first.lower_95_basis_points > 0


def test_complete_candidate_passes_frozen_agent_gate(tmp_path: Path) -> None:
    candidate_items = _items(candidate=True)
    parent_items = _items(candidate=False)
    candidate_items_path = tmp_path / "candidate-items.jsonl"
    parent_items_path = tmp_path / "parent-items.jsonl"
    candidate_items_path.write_bytes(_jsonl(candidate_items))
    parent_items_path.write_bytes(_jsonl(parent_items))
    candidate_summary_path = tmp_path / "candidate-summary.json"
    parent_summary_path = tmp_path / "parent-summary.json"
    candidate_bfcl_path = tmp_path / "candidate-bfcl.json"
    parent_bfcl_path = tmp_path / "parent-bfcl.json"
    _write(candidate_summary_path, _summary(candidate_items, candidate=True))
    _write(parent_summary_path, _summary(parent_items, candidate=False))
    _write(candidate_bfcl_path, _bfcl(candidate=True))
    _write(parent_bfcl_path, _bfcl(candidate=False))

    result = assemble_agent_gate(
        candidate_summary_path=candidate_summary_path,
        candidate_items_path=candidate_items_path,
        parent_summary_path=parent_summary_path,
        parent_items_path=parent_items_path,
        candidate_bfcl_path=candidate_bfcl_path,
        parent_bfcl_path=parent_bfcl_path,
        m6_regression_basis_points=0,
        m6_evidence_sha256="1" * 64,
        serving_gate_valid=True,
        serving_evidence_sha256="2" * 64,
    )

    assert result.decision == "accepted"
    assert len(result.checks) == 13
    assert all(check.passed for check in result.checks)


def test_agent_gate_rejects_bfcl_from_a_different_model(tmp_path: Path) -> None:
    candidate_items = _items(candidate=True)
    parent_items = _items(candidate=False)
    candidate_items_path = tmp_path / "candidate-items.jsonl"
    parent_items_path = tmp_path / "parent-items.jsonl"
    candidate_items_path.write_bytes(_jsonl(candidate_items))
    parent_items_path.write_bytes(_jsonl(parent_items))
    candidate_summary_path = tmp_path / "candidate-summary.json"
    parent_summary_path = tmp_path / "parent-summary.json"
    candidate_bfcl_path = tmp_path / "candidate-bfcl.json"
    parent_bfcl_path = tmp_path / "parent-bfcl.json"
    _write(candidate_summary_path, _summary(candidate_items, candidate=True))
    _write(parent_summary_path, _summary(parent_items, candidate=False))
    _write(
        candidate_bfcl_path,
        _bfcl(candidate=True).model_copy(update={"model_artifact_sha256": "9" * 64}),
    )
    _write(parent_bfcl_path, _bfcl(candidate=False))

    with pytest.raises(AgentGateError, match="BFCL evidence"):
        assemble_agent_gate(
            candidate_summary_path=candidate_summary_path,
            candidate_items_path=candidate_items_path,
            parent_summary_path=parent_summary_path,
            parent_items_path=parent_items_path,
            candidate_bfcl_path=candidate_bfcl_path,
            parent_bfcl_path=parent_bfcl_path,
            m6_regression_basis_points=0,
            m6_evidence_sha256="1" * 64,
            serving_gate_valid=True,
            serving_evidence_sha256="2" * 64,
        )
