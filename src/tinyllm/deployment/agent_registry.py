"""Immutable M10 Agent Production records and atomic Alias management."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator

from tinyllm.agent_eval.schema import AgentGateResult, M10ServingLineageEvidence
from tinyllm.deployment.evaluation_subject import (
    M10AdapterRoutingPolicy,
    ResolvedEvaluationSubject,
    resolve_m10_lora_stage_evaluation_subject,
)
from tinyllm.deployment.registry import (
    DeploymentError,
    DeploymentErrorCode,
    _atomic_json,
    _load_json,
    _require_absolute_root,
)
from tinyllm.evaluation.m6_schema import M6ModelIdentity
from tinyllm.schemas.base import StrictSchema

SHA256_PATTERN = r"^[0-9a-f]{64}$"
AGENT_PRODUCTION_VERSION_PATTERN = r"^qwen3-8b-m10-agent-production-[0-9a-f]{8}$"


class M10AgentProductionRecord(StrictSchema):
    """Immutable Agent Production identity backed by an accepted M10 gate."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["Production"] = "Production"
    production_version: str = Field(pattern=AGENT_PRODUCTION_VERSION_PATTERN)
    promoted_at: datetime
    source_subject_id: str = Field(pattern=r"^qwen3-8b-m10-agent-lora-(3m|4m|5m)-[0-9a-f]{8}$")
    source_evaluation_subject_sha256: str = Field(pattern=SHA256_PATTERN)
    model: M6ModelIdentity
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter_routing_policy: M10AdapterRoutingPolicy
    adapter_routing_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    agent_model_gate_sha256: str = Field(pattern=SHA256_PATTERN)
    serving_lineage_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_release_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_release_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_bfcl_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    bfcl_comparison_sha256: str = Field(pattern=SHA256_PATTERN)
    m6_regression_evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    platform_gate_sha256: str = Field(pattern=SHA256_PATTERN)
    gate_check_count: Literal[13] = 13
    production_eligible: Literal[True] = True

    @model_validator(mode="after")
    def validate_record(self) -> M10AgentProductionRecord:
        if self.promoted_at.tzinfo is None:
            raise ValueError("Agent Production timestamp must be timezone-aware")
        if self.model.model_artifact_sha256 != self.model_artifact_sha256:
            raise ValueError("Agent Production model hash differs from source identity")
        if self.model.repository != "Qwen/Qwen3-8B" or self.model.adaptation != "lora":
            raise ValueError("Agent Production is frozen to the Qwen3-8B LoRA route")
        if self.adapter_routing_policy.policy_sha256 != self.adapter_routing_policy_sha256:
            raise ValueError("Agent Production routing policy hash differs")
        if not self.production_version.endswith(self.agent_model_gate_sha256[:8]):
            raise ValueError("Agent Production version differs from the accepted gate")
        return self


class M10AgentProductionAlias(StrictSchema):
    """Atomically replaceable pointer to one immutable Agent Production record."""

    schema_version: Literal["1.0"] = "1.0"
    alias: Literal["agent-production"] = "agent-production"
    production_version: str = Field(pattern=AGENT_PRODUCTION_VERSION_PATTERN)
    production_record_sha256: str = Field(pattern=SHA256_PATTERN)
    previous_production_version: str | None = Field(
        default=None, pattern=AGENT_PRODUCTION_VERSION_PATTERN
    )
    updated_at: datetime

    @model_validator(mode="after")
    def validate_alias(self) -> M10AgentProductionAlias:
        if self.updated_at.tzinfo is None:
            raise ValueError("Agent Production Alias timestamp must be timezone-aware")
        if self.previous_production_version == self.production_version:
            raise ValueError("Agent Production Alias previous target must differ")
        return self


def _check(gate: AgentGateResult, name: str) -> str:
    matches = tuple(check for check in gate.checks if check.name == name)
    if len(matches) != 1 or not matches[0].passed:
        raise DeploymentError(
            DeploymentErrorCode.GATE_REJECTED,
            f"M10 Agent Gate check is missing or rejected: {name}",
        )
    return matches[0].evidence_sha256


def _load_record(
    artifact_root: Path, production_version: str
) -> tuple[M10AgentProductionRecord, str]:
    if re.fullmatch(AGENT_PRODUCTION_VERSION_PATTERN, production_version) is None:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "Agent Production version is invalid"
        )
    path = artifact_root / "registry" / "agent-production" / production_version / "model.json"
    record, digest = _load_json(path, M10AgentProductionRecord, "Agent Production record")
    return cast(M10AgentProductionRecord, record), digest


def _load_alias(artifact_root: Path) -> M10AgentProductionAlias:
    path = artifact_root / "registry" / "aliases" / "agent-production.json"
    alias, _ = _load_json(path, M10AgentProductionAlias, "Agent Production Alias")
    return cast(M10AgentProductionAlias, alias)


def promote_agent_production(
    artifact_root: Path,
    gate_path: Path,
    serving_lineage_path: Path,
    *,
    now: datetime | None = None,
) -> M10AgentProductionRecord:
    """Publish an accepted M10 Agent record and atomically update its Alias."""

    _require_absolute_root(artifact_root)
    if not gate_path.is_absolute() or not serving_lineage_path.is_absolute():
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT,
            "M10 promotion evidence paths must be absolute",
        )
    gate_value, gate_sha256 = _load_json(gate_path, AgentGateResult, "M10 Agent Model Gate")
    serving_value, serving_sha256 = _load_json(
        serving_lineage_path, M10ServingLineageEvidence, "M10 Serving lineage"
    )
    gate = cast(AgentGateResult, gate_value)
    serving = cast(M10ServingLineageEvidence, serving_value)
    if gate.decision != "accepted" or len(gate.checks) != 13:
        raise DeploymentError(
            DeploymentErrorCode.GATE_REJECTED, "M10 Agent Model Gate is not accepted"
        )
    if _check(gate, "serving_lineage_gate") != serving_sha256:
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH,
            "M10 Agent Gate and Serving lineage hashes differ",
        )
    if (
        gate.candidate_evaluation_id != serving.release_evaluation_id
        or gate.candidate_summary_sha256 != serving.release_summary_sha256
        or serving.status != "accepted"
    ):
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH,
            "M10 Agent Gate and Serving lineage identities differ",
        )

    resolved = resolve_m10_lora_stage_evaluation_subject(
        artifact_root, serving.candidate_subject_id, now=now
    )
    if (
        resolved.evaluation_subject_sha256 != serving.candidate_evaluation_subject_sha256
        or resolved.model_artifact_sha256 != serving.candidate_model_artifact_sha256
        or resolved.adapter_routing_policy is None
    ):
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH,
            "M10 Agent source subject differs from accepted evidence",
        )

    production_version = f"qwen3-8b-m10-agent-production-{gate_sha256[:8]}"
    record = M10AgentProductionRecord(
        production_version=production_version,
        promoted_at=now or datetime.now(UTC),
        source_subject_id=serving.candidate_subject_id,
        source_evaluation_subject_sha256=serving.candidate_evaluation_subject_sha256,
        model=resolved.model,
        model_artifact_sha256=serving.candidate_model_artifact_sha256,
        adapter_routing_policy=resolved.adapter_routing_policy,
        adapter_routing_policy_sha256=resolved.adapter_routing_policy.policy_sha256,
        agent_model_gate_sha256=gate_sha256,
        serving_lineage_sha256=serving_sha256,
        candidate_release_summary_sha256=gate.candidate_summary_sha256,
        parent_release_summary_sha256=gate.parent_summary_sha256,
        candidate_bfcl_summary_sha256=serving.bfcl_summary_sha256,
        bfcl_comparison_sha256=_check(gate, "bfcl_core_category_regression"),
        m6_regression_evidence_sha256=_check(gate, "m6_quality_regression"),
        platform_gate_sha256=serving.platform_gate_sha256,
    )

    target = artifact_root / "registry" / "agent-production" / production_version / "model.json"
    if target.exists():
        existing, existing_sha256 = _load_record(artifact_root, production_version)
        if existing.model_dump(mode="json", exclude={"promoted_at"}) != record.model_dump(
            mode="json", exclude={"promoted_at"}
        ):
            raise DeploymentError(
                DeploymentErrorCode.CONFLICT,
                "Agent Production record already exists with drift",
            )
        record = existing
        record_sha256 = existing_sha256
    else:
        _atomic_json(target, record.to_dict())
        record_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()

    alias_path = artifact_root / "registry" / "aliases" / "agent-production.json"
    previous: str | None = None
    if alias_path.exists():
        current = _load_alias(artifact_root)
        if (
            current.production_version == production_version
            and current.production_record_sha256 == record_sha256
        ):
            return record
        previous = current.production_version
    alias = M10AgentProductionAlias(
        production_version=production_version,
        production_record_sha256=record_sha256,
        previous_production_version=previous,
        updated_at=now or datetime.now(UTC),
    )
    _atomic_json(alias_path, alias.to_dict())
    return record


def resolve_agent_production(
    artifact_root: Path,
    model_ref: str = "agent-production",
    *,
    now: datetime | None = None,
) -> ResolvedEvaluationSubject:
    """Resolve the Agent Alias or immutable record and verify all deployed Artifacts."""

    _require_absolute_root(artifact_root)
    if model_ref == "agent-production":
        alias = _load_alias(artifact_root)
        production_version = alias.production_version
        record, record_sha256 = _load_record(artifact_root, production_version)
        if record_sha256 != alias.production_record_sha256:
            raise DeploymentError(
                DeploymentErrorCode.HASH_MISMATCH,
                "Agent Production Alias record hash differs",
            )
    else:
        production_version = model_ref
        record, record_sha256 = _load_record(artifact_root, production_version)
    if record.production_version != production_version:
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH, "Agent Production record identity differs"
        )
    resolved = resolve_m10_lora_stage_evaluation_subject(
        artifact_root, record.source_subject_id, now=now
    )
    if (
        resolved.evaluation_subject_sha256 != record.source_evaluation_subject_sha256
        or resolved.model != record.model
        or resolved.model_artifact_sha256 != record.model_artifact_sha256
        or resolved.adapter_routing_policy != record.adapter_routing_policy
    ):
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH,
            "Agent Production source subject has drifted",
        )
    return resolved.model_copy(
        update={
            "requested_ref": model_ref,
            "status": "Production",
            "production_record_sha256": record_sha256,
            "verified_at": now or datetime.now(UTC),
        }
    )


def rollback_agent_production(
    artifact_root: Path,
    target_version: str | None = None,
    *,
    now: datetime | None = None,
) -> M10AgentProductionAlias:
    """Atomically move the Agent Alias to a prior immutable Production record."""

    _require_absolute_root(artifact_root)
    current = _load_alias(artifact_root)
    target = target_version or current.previous_production_version
    if target is None:
        raise DeploymentError(
            DeploymentErrorCode.NOT_FOUND, "Agent Production Alias has no rollback target"
        )
    if target == current.production_version:
        raise DeploymentError(
            DeploymentErrorCode.CONFLICT, "Agent Production rollback target is already active"
        )
    _, target_sha256 = _load_record(artifact_root, target)
    alias = M10AgentProductionAlias(
        production_version=target,
        production_record_sha256=target_sha256,
        previous_production_version=current.production_version,
        updated_at=now or datetime.now(UTC),
    )
    _atomic_json(artifact_root / "registry" / "aliases" / "agent-production.json", alias.to_dict())
    return alias


__all__ = [
    "AGENT_PRODUCTION_VERSION_PATTERN",
    "M10AgentProductionAlias",
    "M10AgentProductionRecord",
    "promote_agent_production",
    "resolve_agent_production",
    "rollback_agent_production",
]
