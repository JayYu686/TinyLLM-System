from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Final

from tests.unit.test_deployment_registry import _artifact_root, _gate
from tests.unit.test_m9_agent_gate import _bfcl, _items, _jsonl, _summary
from tinyllm.agent_eval.m10_gate import (
    assemble_m10_agent_model_gate,
    assemble_m10_serving_lineage,
)
from tinyllm.agent_eval.schema import AgentEvalSummary, M10ServingLineageEvidence
from tinyllm.training.m10_lora_schema import M10LoRAM6RegressionEvidence

NOW = datetime(2026, 8, 31, tzinfo=UTC)
CANDIDATE: Final = "qwen3-8b-m10-agent-lora-5m-1234abcd"
PARENT: Final = "qwen3-8b-m9-base-90587dd6"
CANDIDATE_ARTIFACT = "c" * 64
CANDIDATE_RECORD = "e" * 64
PARENT_ARTIFACT: Final = "81fec43ab8b1f03a158e39e50ec23d99cf8701144e8678aea3ca656d12d08de0"
PARENT_RECORD: Final = "9f72bba28bcfaed45f116080033cb9bc83be1632570e71623f2a5684350261d8"


def _write(path: Path, value: object) -> str:
    data = value.to_dict()  # type: ignore[attr-defined]
    payload = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _release_summaries() -> tuple[AgentEvalSummary, AgentEvalSummary]:
    candidate_items = _items(candidate=True)
    parent_items = _items(candidate=False)
    candidate = _summary(candidate_items, candidate=True).model_copy(
        update={
            "model_id": CANDIDATE,
            "model_artifact_sha256": CANDIDATE_ARTIFACT,
            "evaluation_subject_sha256": CANDIDATE_RECORD,
            "deployment_record_sha256": None,
        }
    )
    parent = _summary(parent_items, candidate=False).model_copy(
        update={
            "model_id": PARENT,
            "model_artifact_sha256": PARENT_ARTIFACT,
            "evaluation_subject_sha256": PARENT_RECORD,
            "deployment_record_sha256": None,
        }
    )
    return candidate, parent


def test_serving_lineage_binds_exact_candidate(tmp_path: Path, monkeypatch: object) -> None:
    artifact_root, record = _artifact_root(tmp_path / "artifacts")
    platform_gate_path = tmp_path / "platform-gate.json"
    _write(platform_gate_path, _gate(artifact_root, record))
    candidate, _ = _release_summaries()
    release_path = tmp_path / "release.json"
    _write(release_path, candidate)
    dev = candidate.model_copy(
        update={
            "evaluation_id": "m9-agent-eval-d1234567",
            "suite_version": "tinyllm-devops-agent-dev-v1-f958bcc6",
            "metrics": candidate.metrics.model_copy(update={"item_count": 80}),
        }
    )
    dev_path = tmp_path / "dev.json"
    _write(dev_path, dev)
    candidate_bfcl = _bfcl(candidate=True).model_copy(
        update={"model_id": CANDIDATE, "model_artifact_sha256": CANDIDATE_ARTIFACT}
    )
    bfcl_path = tmp_path / "bfcl.json"
    _write(bfcl_path, candidate_bfcl)
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "tinyllm.agent_eval.m10_gate.resolve_m10_lora_stage_evaluation_subject",
        lambda *_args, **_kwargs: SimpleNamespace(
            model_artifact_sha256=CANDIDATE_ARTIFACT,
            evaluation_subject_sha256=CANDIDATE_RECORD,
        ),
    )

    evidence = assemble_m10_serving_lineage(
        artifact_root=artifact_root,
        candidate_subject_id=CANDIDATE,
        platform_gate_path=platform_gate_path,
        dev_summary_path=dev_path,
        release_summary_path=release_path,
        bfcl_summary_path=bfcl_path,
        output_path=tmp_path / "serving.json",
        now=NOW,
    )

    assert evidence.status == "accepted"
    assert evidence.candidate_model_artifact_sha256 == CANDIDATE_ARTIFACT
    assert evidence.validated_dev_tasks + evidence.validated_release_tasks == 240


def test_final_gate_cross_checks_all_lineage_and_accepts(tmp_path: Path) -> None:
    candidate_items = _items(candidate=True)
    parent_items = _items(candidate=False)
    candidate_items_path = tmp_path / "candidate-items.jsonl"
    parent_items_path = tmp_path / "parent-items.jsonl"
    candidate_items_path.write_bytes(_jsonl(candidate_items))
    parent_items_path.write_bytes(_jsonl(parent_items))
    candidate, parent = _release_summaries()
    candidate_summary_path = tmp_path / "candidate-summary.json"
    parent_summary_path = tmp_path / "parent-summary.json"
    candidate_summary_sha = _write(candidate_summary_path, candidate)
    _write(parent_summary_path, parent)
    candidate_bfcl = _bfcl(candidate=True).model_copy(
        update={"model_id": CANDIDATE, "model_artifact_sha256": CANDIDATE_ARTIFACT}
    )
    parent_bfcl = _bfcl(candidate=False).model_copy(
        update={"model_id": PARENT, "model_artifact_sha256": PARENT_ARTIFACT}
    )
    candidate_bfcl_path = tmp_path / "candidate-bfcl.json"
    parent_bfcl_path = tmp_path / "parent-bfcl.json"
    candidate_bfcl_sha = _write(candidate_bfcl_path, candidate_bfcl)
    _write(parent_bfcl_path, parent_bfcl)
    m6 = M10LoRAM6RegressionEvidence(
        evaluated_at=NOW,
        protocol_version="m6-release-v7",
        parent_subject_id=PARENT,
        parent_evaluation_subject_sha256=PARENT_RECORD,
        parent_model_artifact_sha256=PARENT_ARTIFACT,
        parent_summary_sha256="1" * 64,
        parent_aggregate_basis_points=5000,
        candidate_subject_id=CANDIDATE,
        candidate_evaluation_subject_sha256=CANDIDATE_RECORD,
        candidate_model_artifact_sha256=CANDIDATE_ARTIFACT,
        candidate_summary_sha256="2" * 64,
        candidate_aggregate_basis_points=5000,
        regression_basis_points=0,
    )
    m6_path = tmp_path / "m6.json"
    _write(m6_path, m6)
    serving = M10ServingLineageEvidence(
        evaluated_at=NOW,
        candidate_subject_id=CANDIDATE,
        candidate_evaluation_subject_sha256=CANDIDATE_RECORD,
        candidate_model_artifact_sha256=CANDIDATE_ARTIFACT,
        platform_gate_id="m7-production-gate-1234abcd",
        platform_gate_sha256="3" * 64,
        dev_evaluation_id="m9-agent-eval-d1234567",
        dev_summary_sha256="4" * 64,
        release_evaluation_id=candidate.evaluation_id,
        release_summary_sha256=candidate_summary_sha,
        bfcl_summary_sha256=candidate_bfcl_sha,
        gateway_version=candidate.gateway_version,
        agent_runtime_version=candidate.agent_runtime_version,
    )
    serving_path = tmp_path / "serving.json"
    _write(serving_path, serving)

    gate = assemble_m10_agent_model_gate(
        candidate_summary_path=candidate_summary_path,
        candidate_items_path=candidate_items_path,
        parent_summary_path=parent_summary_path,
        parent_items_path=parent_items_path,
        candidate_bfcl_path=candidate_bfcl_path,
        parent_bfcl_path=parent_bfcl_path,
        m6_evidence_path=m6_path,
        serving_evidence_path=serving_path,
        output_path=tmp_path / "gate.json",
    )

    assert gate.decision == "accepted"
    assert all(check.passed for check in gate.checks)
