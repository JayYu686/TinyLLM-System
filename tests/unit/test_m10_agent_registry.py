from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.unit.test_m10_lora_general import _resolved
from tinyllm.agent_eval.schema import (
    AgentBootstrapInterval,
    AgentGateCheck,
    AgentGateResult,
    M10ServingLineageEvidence,
)
from tinyllm.deployment import (
    DeploymentError,
    DeploymentErrorCode,
    promote_agent_production,
    resolve_agent_production,
    rollback_agent_production,
    show_deployment,
)

NOW = datetime(2026, 9, 2, tzinfo=UTC)


def _write(path: Path, value: object) -> str:
    payload = (
        json.dumps(value.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"  # type: ignore[attr-defined]
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _evidence(tmp_path: Path) -> tuple[Path, Path]:
    serving_path = (tmp_path / "serving.json").resolve()
    serving = M10ServingLineageEvidence(
        evaluated_at=NOW,
        candidate_subject_id="qwen3-8b-m10-agent-lora-5m-1234abcd",
        candidate_evaluation_subject_sha256="5" * 64,
        candidate_model_artifact_sha256="9" * 64,
        platform_gate_id="m7-production-gate-1234abcd",
        platform_gate_sha256="a" * 64,
        dev_evaluation_id="m9-agent-eval-11111111",
        dev_summary_sha256="b" * 64,
        release_evaluation_id="m9-agent-eval-22222222",
        release_summary_sha256="c" * 64,
        bfcl_summary_sha256="d" * 64,
        gateway_version="1.0.0",
        agent_runtime_version="1.0.0",
    )
    serving_sha256 = _write(serving_path, serving)
    names = (
        "release_task_success",
        "parent_task_success_improvement",
        "bootstrap_ci_lower",
        "schema_valid_rate",
        "no_tool_accuracy",
        "tool_hallucination_rate",
        "grounding_accuracy",
        "error_recovery_rate",
        "agent_safety_violations",
        "bfcl_core_overall_regression",
        "bfcl_core_category_regression",
        "m6_quality_regression",
        "serving_lineage_gate",
    )
    checks = tuple(
        AgentGateCheck(
            name=name,
            passed=True,
            actual="1",
            required=">=0",
            evidence_sha256=(
                serving_sha256
                if name == "serving_lineage_gate"
                else ("e" * 64 if name == "bfcl_core_category_regression" else "f" * 64)
            ),
        )
        for name in names
    )
    gate = AgentGateResult(
        evaluated_at=NOW,
        candidate_evaluation_id=serving.release_evaluation_id,
        parent_evaluation_id="m9-agent-eval-33333333",
        candidate_summary_sha256=serving.release_summary_sha256,
        parent_summary_sha256="1" * 64,
        task_success_interval=AgentBootstrapInterval(
            metric="task_success_difference_basis_points",
            observed_basis_points=1000,
            lower_95_basis_points=100,
            upper_95_basis_points=2000,
            cluster_count=22,
            resamples=10_000,
            seed=20260820,
        ),
        checks=checks,
        decision="accepted",
    )
    gate_path = (tmp_path / "gate.json").resolve()
    _write(gate_path, gate)
    return gate_path, serving_path


def test_agent_promotion_alias_and_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "artifacts").resolve()
    root.mkdir()
    resolved = _resolved(tmp_path, candidate=True, routed=True)
    monkeypatch.setattr(
        "tinyllm.deployment.agent_registry.resolve_m10_lora_stage_evaluation_subject",
        lambda *_args, **_kwargs: resolved,
    )
    gate_path, serving_path = _evidence(tmp_path)

    record = promote_agent_production(root, gate_path, serving_path, now=NOW)
    active = resolve_agent_production(root, now=NOW)

    assert record.production_version.startswith("qwen3-8b-m10-agent-production-")
    assert active.status == "Production"
    assert active.requested_ref == "agent-production"
    assert active.model_version == resolved.model_version
    assert active.production_record_sha256 is not None
    shown = show_deployment(root, "agent-production")
    assert shown["status"] == "Production"
    assert shown["evaluation_subject_sha256"] == "5" * 64
    assert (root / "registry" / "aliases" / "agent-production.json").stat().st_mode & 0o777 == 0o600


def test_agent_promotion_fails_closed_and_rollback_requires_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "artifacts").resolve()
    root.mkdir()
    resolved = _resolved(tmp_path, candidate=True, routed=True)
    monkeypatch.setattr(
        "tinyllm.deployment.agent_registry.resolve_m10_lora_stage_evaluation_subject",
        lambda *_args, **_kwargs: resolved,
    )
    gate_path, serving_path = _evidence(tmp_path)
    promote_agent_production(root, gate_path, serving_path, now=NOW)

    with pytest.raises(DeploymentError) as failure:
        rollback_agent_production(root)
    assert failure.value.code == DeploymentErrorCode.NOT_FOUND

    alias_path = root / "registry" / "aliases" / "agent-production.json"
    payload = json.loads(alias_path.read_text(encoding="utf-8"))
    payload["production_record_sha256"] = "0" * 64
    alias_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DeploymentError) as drift:
        resolve_agent_production(root)
    assert drift.value.code == DeploymentErrorCode.HASH_MISMATCH
