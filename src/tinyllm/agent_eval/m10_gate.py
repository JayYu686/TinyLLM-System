"""Strict M10 Serving-lineage and final Agent model gate assembly."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from tinyllm.agent_eval.gate import AgentGateError, assemble_agent_gate
from tinyllm.agent_eval.schema import (
    AgentEvalSummary,
    AgentGateResult,
    BFCLCoreProfileSummary,
    M10ServingLineageEvidence,
)
from tinyllm.deployment import M7ProductionGate, resolve_m10_lora_stage_evaluation_subject
from tinyllm.training.m10_lora_schema import M10LoRAM6RegressionEvidence


class M10AgentGateError(RuntimeError):
    """Raised when final M10 evidence is incomplete, incompatible, or corrupt."""


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, schema: type[SchemaT]) -> SchemaT:
    try:
        return schema.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise M10AgentGateError(f"M10 final evidence is invalid: {path.name}") from exc


def _atomic_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        raise M10AgentGateError(f"M10 final evidence already exists: {path.name}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o600)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


def assemble_m10_serving_lineage(
    *,
    artifact_root: Path,
    candidate_subject_id: str,
    platform_gate_path: Path,
    dev_summary_path: Path,
    release_summary_path: Path,
    bfcl_summary_path: Path,
    output_path: Path,
    now: datetime | None = None,
) -> M10ServingLineageEvidence:
    """Bind the exact Candidate to a qualified platform and completed service runs."""

    paths = (
        artifact_root,
        platform_gate_path,
        dev_summary_path,
        release_summary_path,
        bfcl_summary_path,
        output_path,
    )
    if any(not path.is_absolute() for path in paths):
        raise M10AgentGateError("M10 Serving lineage paths must be absolute")
    resolved = resolve_m10_lora_stage_evaluation_subject(artifact_root, candidate_subject_id)
    platform_gate = _load(platform_gate_path, M7ProductionGate)
    dev = _load(dev_summary_path, AgentEvalSummary)
    release = _load(release_summary_path, AgentEvalSummary)
    bfcl = _load(bfcl_summary_path, BFCLCoreProfileSummary)

    if (
        platform_gate.status != "accepted"
        or not platform_gate.production_eligible
        or not all(check.passed for check in platform_gate.checks)
    ):
        raise M10AgentGateError("M10 Serving lineage requires an accepted M7 platform gate")
    expected_agent_identity = (
        candidate_subject_id,
        resolved.model_artifact_sha256,
        resolved.evaluation_subject_sha256,
        True,
        False,
    )
    for summary in (dev, release):
        actual = (
            summary.model_id,
            summary.model_artifact_sha256,
            summary.evaluation_subject_sha256,
            summary.completed,
            summary.git_dirty,
        )
        if actual != expected_agent_identity:
            raise M10AgentGateError(
                "M10 Serving Agent evidence does not match the resolved Candidate"
            )
    if (
        not dev.suite_version.startswith("tinyllm-devops-agent-dev-")
        or dev.metrics.item_count != 80
        or not release.suite_version.startswith("tinyllm-devops-agent-release-")
        or release.metrics.item_count != 160
    ):
        raise M10AgentGateError("M10 Serving Agent evidence has the wrong evaluation scope")
    if (
        dev.environment_sha256 != release.environment_sha256
        or dev.hardware_sha256 != release.hardware_sha256
        or dev.gateway_version != release.gateway_version
        or dev.agent_runtime_version != release.agent_runtime_version
    ):
        raise M10AgentGateError("M10 Serving Dev and Release environments differ")
    if (
        not bfcl.completed
        or bfcl.total_items != 1840
        or bfcl.model_id != candidate_subject_id
        or bfcl.model_artifact_sha256 != resolved.model_artifact_sha256
    ):
        raise M10AgentGateError("M10 Serving BFCL evidence does not match the Candidate")

    evidence = M10ServingLineageEvidence(
        evaluated_at=now or datetime.now(UTC),
        candidate_subject_id=candidate_subject_id,
        candidate_evaluation_subject_sha256=resolved.evaluation_subject_sha256,
        candidate_model_artifact_sha256=resolved.model_artifact_sha256,
        platform_gate_id=platform_gate.gate_id,
        platform_gate_sha256=_sha256(platform_gate_path),
        dev_evaluation_id=dev.evaluation_id,
        dev_summary_sha256=_sha256(dev_summary_path),
        release_evaluation_id=release.evaluation_id,
        release_summary_sha256=_sha256(release_summary_path),
        bfcl_summary_sha256=_sha256(bfcl_summary_path),
        gateway_version=release.gateway_version,
        agent_runtime_version=release.agent_runtime_version,
    )
    _atomic_json(output_path, evidence.to_dict())
    return evidence


def assemble_m10_agent_model_gate(
    *,
    candidate_summary_path: Path,
    candidate_items_path: Path,
    parent_summary_path: Path,
    parent_items_path: Path,
    candidate_bfcl_path: Path,
    parent_bfcl_path: Path,
    m6_evidence_path: Path,
    serving_evidence_path: Path,
    output_path: Path,
) -> AgentGateResult:
    """Cross-check every evidence identity before applying the frozen M10 thresholds."""

    paths = (
        candidate_summary_path,
        candidate_items_path,
        parent_summary_path,
        parent_items_path,
        candidate_bfcl_path,
        parent_bfcl_path,
        m6_evidence_path,
        serving_evidence_path,
        output_path,
    )
    if any(not path.is_absolute() for path in paths):
        raise M10AgentGateError("M10 final gate paths must be absolute")
    candidate = _load(candidate_summary_path, AgentEvalSummary)
    parent = _load(parent_summary_path, AgentEvalSummary)
    candidate_bfcl = _load(candidate_bfcl_path, BFCLCoreProfileSummary)
    parent_bfcl = _load(parent_bfcl_path, BFCLCoreProfileSummary)
    m6 = _load(m6_evidence_path, M10LoRAM6RegressionEvidence)
    serving = _load(serving_evidence_path, M10ServingLineageEvidence)

    candidate_identity = (candidate.model_id, candidate.model_artifact_sha256)
    parent_identity = (parent.model_id, parent.model_artifact_sha256)
    if (
        candidate_identity != (m6.candidate_subject_id, m6.candidate_model_artifact_sha256)
        or candidate_identity
        != (serving.candidate_subject_id, serving.candidate_model_artifact_sha256)
        or candidate_identity != (candidate_bfcl.model_id, candidate_bfcl.model_artifact_sha256)
        or parent_identity != (m6.parent_subject_id, m6.parent_model_artifact_sha256)
        or parent_identity != (parent_bfcl.model_id, parent_bfcl.model_artifact_sha256)
        or candidate.evaluation_subject_sha256 != serving.candidate_evaluation_subject_sha256
        or candidate.evaluation_subject_sha256 != m6.candidate_evaluation_subject_sha256
        or parent.evaluation_subject_sha256 != m6.parent_evaluation_subject_sha256
        or _sha256(candidate_summary_path) != serving.release_summary_sha256
        or _sha256(candidate_bfcl_path) != serving.bfcl_summary_sha256
    ):
        raise M10AgentGateError("M10 final evidence belongs to different model subjects")

    try:
        gate = assemble_agent_gate(
            candidate_summary_path=candidate_summary_path,
            candidate_items_path=candidate_items_path,
            parent_summary_path=parent_summary_path,
            parent_items_path=parent_items_path,
            candidate_bfcl_path=candidate_bfcl_path,
            parent_bfcl_path=parent_bfcl_path,
            m6_regression_basis_points=m6.regression_basis_points,
            m6_evidence_sha256=_sha256(m6_evidence_path),
            serving_gate_valid=serving.status == "accepted",
            serving_evidence_sha256=_sha256(serving_evidence_path),
        )
    except AgentGateError as exc:
        raise M10AgentGateError(str(exc)) from exc
    _atomic_json(output_path, gate.to_dict())
    return gate


__all__ = [
    "M10AgentGateError",
    "assemble_m10_agent_model_gate",
    "assemble_m10_serving_lineage",
]
