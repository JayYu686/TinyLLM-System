"""Assemble the lineage-bound M10 5M-to-10M continuation decision."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from tinyllm.agent_eval import AgentEvalSummary
from tinyllm.deployment import resolve_m10_stage_evaluation_subject
from tinyllm.evaluation.m6_schema import M6GeneralPassSummary
from tinyllm.training.m10_sft_schema import (
    M10ContinuationGate,
    M10FullSFTRunResult,
    M10M6RegressionEvidence,
)

AGENT_DEV_VERSION: Literal["tinyllm-devops-agent-dev-v1-f958bcc6"] = (
    "tinyllm-devops-agent-dev-v1-f958bcc6"
)
PARENT_PRODUCTION_VERSION: Literal["qwen3-0-6b-m7-fa678d92"] = "qwen3-0-6b-m7-fa678d92"
PARENT_PRODUCTION_RECORD_SHA256: Literal[
    "a83eaff5e9b91be403bddb7a4ac23926465e0a78872921429dad72b6ee7d0ca5"
] = "a83eaff5e9b91be403bddb7a4ac23926465e0a78872921429dad72b6ee7d0ca5"
PARENT_MODEL_SHA256: Literal["63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6"] = (
    "63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6"
)


class M10StageGateError(RuntimeError):
    """Raised when stage evidence cannot produce a trustworthy decision."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _atomic_gate_bundle(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    if path.exists():
        raise M10StageGateError("M10 stage gate output already exists")
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


def _load(path: Path, schema: type[BaseModel]) -> BaseModel:
    try:
        return schema.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise M10StageGateError(f"M10 stage evidence is invalid: {path.name}") from exc


def assemble_m10_stage_gate(
    *,
    artifact_root: Path,
    source_run: Path,
    candidate_subject_id: str,
    parent_agent_summary_path: Path,
    candidate_agent_summary_path: Path,
    parent_m6_summary_path: Path,
    candidate_m6_summary_path: Path,
    output_directory: Path,
) -> tuple[M10M6RegressionEvidence, M10ContinuationGate]:
    """Validate paired Dev/M6 summaries and atomically persist their gate decision."""

    paths = (
        artifact_root,
        source_run,
        parent_agent_summary_path,
        candidate_agent_summary_path,
        parent_m6_summary_path,
        candidate_m6_summary_path,
        output_directory,
    )
    if any(not path.is_absolute() for path in paths) or output_directory.is_symlink():
        raise M10StageGateError("M10 stage gate paths must be absolute")
    resolved = resolve_m10_stage_evaluation_subject(artifact_root, candidate_subject_id)
    result = _load(source_run / "result.json", M10FullSFTRunResult)
    parent_agent = _load(parent_agent_summary_path, AgentEvalSummary)
    candidate_agent = _load(candidate_agent_summary_path, AgentEvalSummary)
    parent_m6 = _load(parent_m6_summary_path, M6GeneralPassSummary)
    candidate_m6 = _load(candidate_m6_summary_path, M6GeneralPassSummary)
    assert isinstance(result, M10FullSFTRunResult)
    assert isinstance(parent_agent, AgentEvalSummary)
    assert isinstance(candidate_agent, AgentEvalSummary)
    assert isinstance(parent_m6, M6GeneralPassSummary)
    assert isinstance(candidate_m6, M6GeneralPassSummary)

    parent_agent_identities = (
        (parent_agent.completed, True),
        (parent_agent.suite_version, AGENT_DEV_VERSION),
        (parent_agent.model_id, PARENT_PRODUCTION_VERSION),
        (parent_agent.model_artifact_sha256, PARENT_MODEL_SHA256),
        (parent_agent.deployment_record_sha256, PARENT_PRODUCTION_RECORD_SHA256),
        (parent_agent.evaluation_subject_sha256, None),
    )
    candidate_agent_identities = (
        (candidate_agent.completed, True),
        (candidate_agent.suite_version, AGENT_DEV_VERSION),
        (candidate_agent.suite_content_sha256, parent_agent.suite_content_sha256),
        (candidate_agent.model_id, candidate_subject_id),
        (candidate_agent.model_artifact_sha256, resolved.model_artifact_sha256),
        (
            candidate_agent.evaluation_subject_sha256,
            resolved.evaluation_subject_sha256,
        ),
        (candidate_agent.deployment_record_sha256, None),
    )
    source_identities = (
        (result.run_id, source_run.name),
        (result.supervised_tokens, 5_000_000),
        (result.latest_checkpoint, "checkpoint-tokens-0005000000"),
        (result.stage_export.export_sha256, resolved.model_artifact_sha256),
        (resolved.model.training_run_id, result.run_id),
        (resolved.model.training_config_sha256, result.config_sha256),
        (resolved.model.training_tokens, result.supervised_tokens),
    )
    m6_identities = (
        (parent_m6.status, "succeeded"),
        (candidate_m6.status, "succeeded"),
        (parent_m6.protocol_version, "m6-release-v7"),
        (candidate_m6.protocol_version, parent_m6.protocol_version),
        (candidate_m6.config_sha256, parent_m6.config_sha256),
        (parent_m6.model.model_artifact_sha256, PARENT_MODEL_SHA256),
        (candidate_m6.model, resolved.model),
        (
            tuple(item.task for item in candidate_m6.general.tasks),
            tuple(item.task for item in parent_m6.general.tasks),
        ),
        (
            tuple(item.samples for item in candidate_m6.general.tasks),
            tuple(item.samples for item in parent_m6.general.tasks),
        ),
    )
    if any(
        actual != expected
        for actual, expected in (
            *parent_agent_identities,
            *candidate_agent_identities,
            *source_identities,
            *m6_identities,
        )
    ):
        raise M10StageGateError("M10 stage evidence lineage or protocol differs")

    evaluated_at = datetime.now(UTC)
    regression = (
        parent_m6.general.aggregate_basis_points - candidate_m6.general.aggregate_basis_points
    )
    m6_evidence = M10M6RegressionEvidence(
        evaluated_at=evaluated_at,
        protocol_version="m6-release-v7",
        parent_model_version=PARENT_PRODUCTION_VERSION,
        parent_model_artifact_sha256=PARENT_MODEL_SHA256,
        parent_summary_sha256=_sha256_file(parent_m6_summary_path),
        parent_aggregate_basis_points=parent_m6.general.aggregate_basis_points,
        candidate_subject_id=candidate_subject_id,
        candidate_evaluation_subject_sha256=resolved.evaluation_subject_sha256,
        candidate_model_artifact_sha256=resolved.model_artifact_sha256,
        candidate_summary_sha256=_sha256_file(candidate_m6_summary_path),
        candidate_aggregate_basis_points=candidate_m6.general.aggregate_basis_points,
        regression_basis_points=regression,
    )
    m6_payload = _json_bytes(m6_evidence.to_dict())
    improvement = (
        candidate_agent.metrics.task_success_rate_basis_points
        - parent_agent.metrics.task_success_rate_basis_points
    )
    gate = M10ContinuationGate(
        evaluated_at=evaluated_at,
        decision=("accepted" if improvement >= 100 and regression <= 200 else "rejected"),
        run_id=result.run_id,
        config_sha256=result.config_sha256,
        source_checkpoint_id="checkpoint-tokens-0005000000",
        source_stage_export_sha256=result.stage_export.export_sha256,
        agent_dev_version=AGENT_DEV_VERSION,
        parent_agent_dev_summary_sha256=_sha256_file(parent_agent_summary_path),
        candidate_agent_dev_summary_sha256=_sha256_file(candidate_agent_summary_path),
        parent_task_success_basis_points=parent_agent.metrics.task_success_rate_basis_points,
        candidate_task_success_basis_points=candidate_agent.metrics.task_success_rate_basis_points,
        agent_dev_improvement_basis_points=improvement,
        m6_evidence_sha256=hashlib.sha256(m6_payload).hexdigest(),
        m6_regression_basis_points=regression,
    )
    _atomic_gate_bundle(
        output_directory,
        {
            "continuation-gate.json": gate.to_dict(),
            "m6-general-regression.json": m6_evidence.to_dict(),
        },
    )
    return m6_evidence, gate
