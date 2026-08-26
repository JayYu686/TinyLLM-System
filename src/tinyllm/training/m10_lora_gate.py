"""Assemble lineage-bound M10 Agent LoRA continuation decisions."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ValidationError

from tinyllm.agent_eval import AgentEvalSummary
from tinyllm.deployment import resolve_m10_lora_stage_evaluation_subject
from tinyllm.training.m10_lora_schema import (
    M10_LORA_PARENT_MODEL_SHA256,
    M10_LORA_PARENT_RECORD_SHA256,
    M10_LORA_PARENT_SUBJECT,
    M10LoRAContinuationGate,
    M10LoRAGeneralPassSummary,
    M10LoRAM6RegressionEvidence,
    M10LoRARunResult,
)


class M10LoRAStageGateError(RuntimeError):
    """Raised when stage evidence cannot produce a trustworthy decision."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _load(path: Path, schema: type[BaseModel]) -> BaseModel:
    try:
        return schema.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise M10LoRAStageGateError(f"M10 Agent LoRA evidence is invalid: {path.name}") from exc


def _atomic_bundle(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    if path.exists():
        raise M10LoRAStageGateError("M10 Agent LoRA Gate output already exists")
    try:
        temporary.mkdir(mode=0o700)
        for name, value in values.items():
            with (temporary / name).open("xb") as handle:
                handle.write(_json_bytes(value))
                os.fchmod(handle.fileno(), 0o600)
                handle.flush()
                os.fsync(handle.fileno())
        os.rename(temporary, path)
    except Exception:
        if temporary.is_dir():
            for item in temporary.glob("*"):
                item.unlink(missing_ok=True)
            temporary.rmdir()
        raise


def assemble_m10_lora_stage_gate(
    *,
    artifact_root: Path,
    source_run: Path,
    candidate_subject_id: str,
    parent_agent_summary_path: Path,
    candidate_agent_summary_path: Path,
    output_directory: Path,
    parent_m6_summary_path: Path | None = None,
    candidate_m6_summary_path: Path | None = None,
) -> tuple[M10LoRAM6RegressionEvidence | None, M10LoRAContinuationGate]:
    """Validate paired evidence and atomically persist the 1M or 5M decision."""

    required = (
        artifact_root,
        source_run,
        parent_agent_summary_path,
        candidate_agent_summary_path,
        output_directory,
    )
    optional = tuple(
        path for path in (parent_m6_summary_path, candidate_m6_summary_path) if path is not None
    )
    if any(not path.is_absolute() for path in (*required, *optional)):
        raise M10LoRAStageGateError("M10 Agent LoRA Gate paths must be absolute")
    resolved = resolve_m10_lora_stage_evaluation_subject(artifact_root, candidate_subject_id)
    result = _load(source_run / "result.json", M10LoRARunResult)
    parent_agent = _load(parent_agent_summary_path, AgentEvalSummary)
    candidate_agent = _load(candidate_agent_summary_path, AgentEvalSummary)
    assert isinstance(result, M10LoRARunResult)
    assert isinstance(parent_agent, AgentEvalSummary)
    assert isinstance(candidate_agent, AgentEvalSummary)
    source_tokens = result.supervised_tokens
    if source_tokens not in {1_000_000, 5_000_000}:
        raise M10LoRAStageGateError("M10 Agent LoRA Gate accepts only 1M or 5M stages")

    parent_identities = (
        (parent_agent.completed, True),
        (parent_agent.suite_version, "tinyllm-devops-agent-dev-v1-f958bcc6"),
        (parent_agent.model_id, M10_LORA_PARENT_SUBJECT),
        (parent_agent.model_artifact_sha256, M10_LORA_PARENT_MODEL_SHA256),
        (parent_agent.evaluation_subject_sha256, M10_LORA_PARENT_RECORD_SHA256),
        (parent_agent.deployment_record_sha256, None),
    )
    candidate_identities = (
        (candidate_agent.completed, True),
        (candidate_agent.suite_version, parent_agent.suite_version),
        (candidate_agent.suite_content_sha256, parent_agent.suite_content_sha256),
        (candidate_agent.model_id, candidate_subject_id),
        (candidate_agent.model_artifact_sha256, resolved.model_artifact_sha256),
        (candidate_agent.evaluation_subject_sha256, resolved.evaluation_subject_sha256),
        (candidate_agent.deployment_record_sha256, None),
    )
    source_identities = (
        (result.run_id, source_run.name),
        (resolved.model.training_run_id, result.run_id),
        (resolved.model.training_config_sha256, result.config_sha256),
        (resolved.model.training_tokens, source_tokens),
        (resolved.adapter_artifact_sha256, result.stage_export.adapter_artifact_sha256),
    )
    if any(
        actual != expected
        for actual, expected in (*parent_identities, *candidate_identities, *source_identities)
    ):
        raise M10LoRAStageGateError("M10 Agent LoRA Gate lineage or protocol differs")
    expected_protocol = (
        "m10-agent-scoring-v2"
        if result.dataset_version.startswith("m10-agent-sft-v2-")
        else "m9-agent-scoring-v1"
    )
    if (
        parent_agent.scoring_protocol != expected_protocol
        or candidate_agent.scoring_protocol != expected_protocol
    ):
        raise M10LoRAStageGateError("M10 Agent LoRA scoring protocol differs from its Dataset")

    evaluated_at = datetime.now(UTC)
    m6_evidence: M10LoRAM6RegressionEvidence | None = None
    m6_regression: int | None = None
    m6_evidence_sha256: str | None = None
    bundle: dict[str, object] = {}
    if source_tokens == 1_000_000:
        if parent_m6_summary_path is not None or candidate_m6_summary_path is not None:
            raise M10LoRAStageGateError("M10 Agent LoRA 1M Gate must not consume M6 evidence")
    else:
        if parent_m6_summary_path is None or candidate_m6_summary_path is None:
            raise M10LoRAStageGateError("M10 Agent LoRA 5M Gate requires paired M6 evidence")
        parent_m6 = _load(parent_m6_summary_path, M10LoRAGeneralPassSummary)
        candidate_m6 = _load(candidate_m6_summary_path, M10LoRAGeneralPassSummary)
        assert isinstance(parent_m6, M10LoRAGeneralPassSummary)
        assert isinstance(candidate_m6, M10LoRAGeneralPassSummary)
        m6_identities = (
            (parent_m6.status, "succeeded"),
            (candidate_m6.status, "succeeded"),
            (parent_m6.protocol_version, "m6-release-v7"),
            (candidate_m6.protocol_version, parent_m6.protocol_version),
            (candidate_m6.config_sha256, parent_m6.config_sha256),
            (parent_m6.evaluation_subject_id, M10_LORA_PARENT_SUBJECT),
            (parent_m6.evaluation_subject_sha256, M10_LORA_PARENT_RECORD_SHA256),
            (candidate_m6.evaluation_subject_id, candidate_subject_id),
            (
                candidate_m6.evaluation_subject_sha256,
                resolved.evaluation_subject_sha256,
            ),
            (candidate_m6.model, resolved.model),
        )
        if any(actual != expected for actual, expected in m6_identities):
            raise M10LoRAStageGateError("M10 Agent LoRA M6 lineage or protocol differs")
        m6_regression = (
            parent_m6.general.aggregate_basis_points - candidate_m6.general.aggregate_basis_points
        )
        m6_evidence = M10LoRAM6RegressionEvidence(
            evaluated_at=evaluated_at,
            protocol_version="m6-release-v7",
            parent_subject_id=M10_LORA_PARENT_SUBJECT,
            parent_evaluation_subject_sha256=M10_LORA_PARENT_RECORD_SHA256,
            parent_model_artifact_sha256=M10_LORA_PARENT_MODEL_SHA256,
            parent_summary_sha256=_sha256_file(parent_m6_summary_path),
            parent_aggregate_basis_points=parent_m6.general.aggregate_basis_points,
            candidate_subject_id=candidate_subject_id,
            candidate_evaluation_subject_sha256=resolved.evaluation_subject_sha256,
            candidate_model_artifact_sha256=resolved.model_artifact_sha256,
            candidate_summary_sha256=_sha256_file(candidate_m6_summary_path),
            candidate_aggregate_basis_points=candidate_m6.general.aggregate_basis_points,
            regression_basis_points=m6_regression,
        )
        m6_payload = _json_bytes(m6_evidence.to_dict())
        m6_evidence_sha256 = hashlib.sha256(m6_payload).hexdigest()
        bundle["m6-general-regression.json"] = m6_evidence.to_dict()

    improvement = (
        candidate_agent.metrics.task_success_rate_basis_points
        - parent_agent.metrics.task_success_rate_basis_points
    )
    accepted = improvement >= 100 and (m6_regression is None or m6_regression <= 200)
    gate = M10LoRAContinuationGate(
        evaluated_at=evaluated_at,
        decision="accepted" if accepted else "rejected",
        scoring_protocol=expected_protocol,
        run_id=result.run_id,
        config_sha256=result.config_sha256,
        source_stage_tokens=cast(Literal[1_000_000, 5_000_000], source_tokens),
        target_stage_tokens=5_000_000 if source_tokens == 1_000_000 else 10_000_000,
        source_checkpoint_id=result.latest_checkpoint,
        source_adapter_artifact_sha256=result.stage_export.adapter_artifact_sha256,
        agent_dev_version="tinyllm-devops-agent-dev-v1-f958bcc6",
        parent_summary_sha256=_sha256_file(parent_agent_summary_path),
        candidate_summary_sha256=_sha256_file(candidate_agent_summary_path),
        parent_task_success_basis_points=(parent_agent.metrics.task_success_rate_basis_points),
        candidate_task_success_basis_points=(
            candidate_agent.metrics.task_success_rate_basis_points
        ),
        improvement_basis_points=improvement,
        m6_evidence_sha256=m6_evidence_sha256,
        m6_regression_basis_points=m6_regression,
    )
    bundle["continuation-gate.json"] = gate.to_dict()
    _atomic_bundle(output_directory, bundle)
    return m6_evidence, gate


__all__ = ["M10LoRAStageGateError", "assemble_m10_lora_stage_gate"]
