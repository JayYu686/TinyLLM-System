"""Strict M7 deployment, gate, and alias schemas."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.evaluation.m6_schema import M6ModelIdentity
from tinyllm.schemas.base import StrictSchema

SHA256_PATTERN = r"^[0-9a-f]{64}$"
PRODUCTION_VERSION_PATTERN = r"^qwen3-(0-6b|8b)-m7-[0-9a-f]{8}$"
CANDIDATE_VERSION_PATTERN = r"^qwen3-(0-6b|8b)-m6-[0-9a-f]{8}$"


class ResolvedModel(StrictSchema):
    """A path-bearing private projection verified from immutable Registry evidence."""

    schema_version: Literal["1.0"] = "1.0"
    requested_ref: str = Field(min_length=1, max_length=180)
    status: Literal["Candidate", "Production"]
    model_version: str = Field(pattern=r"^qwen3-(0-6b|8b)-m[67]-[0-9a-f]{8}$")
    candidate_model_version: str = Field(pattern=CANDIDATE_VERSION_PATTERN)
    candidate_record_sha256: str = Field(pattern=SHA256_PATTERN)
    production_record_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    model: M6ModelIdentity
    model_dir: Path
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_dir: Path
    tokenizer_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    verified_at: datetime

    @field_validator("model_dir", "tokenizer_dir")
    @classmethod
    def require_absolute_paths(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("resolved model paths must be absolute")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> ResolvedModel:
        if self.verified_at.tzinfo is None:
            raise ValueError("model verification timestamp must be timezone-aware")
        if self.status == "Candidate":
            if self.model_version != self.candidate_model_version:
                raise ValueError("Candidate resolution must preserve the Candidate version")
            if self.production_record_sha256 is not None:
                raise ValueError("Candidate resolution cannot reference a Production record")
        elif self.production_record_sha256 is None:
            raise ValueError("Production resolution requires a Production record identity")
        if self.model.model_artifact_sha256 != self.model_artifact_sha256:
            raise ValueError("resolved model hash differs from M6 model identity")
        return self


class M7GateCheck(StrictSchema):
    """One deterministic M7 Production Gate decision."""

    schema_version: Literal["1.0"] = "1.0"
    name: Literal[
        "api_contract",
        "success_rate",
        "runtime_stability",
        "gateway_throughput",
        "p95_latency_overhead",
        "backend_recovery",
        "deployment_rollback",
        "m6_evidence",
        "security_audit",
    ]
    passed: bool
    actual: str = Field(min_length=1, max_length=200)
    requirement: str = Field(min_length=1, max_length=200)


class M7ProductionGate(StrictSchema):
    """Content-free evidence summary used to decide M7 Production eligibility."""

    schema_version: Literal["1.0"] = "1.0"
    gate_id: str = Field(pattern=r"^m7-production-gate-[0-9a-f]{8}$")
    evaluated_at: datetime
    candidate_model_version: str = Field(pattern=CANDIDATE_VERSION_PATTERN)
    candidate_record_sha256: str = Field(pattern=SHA256_PATTERN)
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    serving_config_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    benchmark_report_sha256: str = Field(pattern=SHA256_PATTERN)
    recovery_report_sha256: str = Field(pattern=SHA256_PATTERN)
    rollback_report_sha256: str = Field(pattern=SHA256_PATTERN)
    security_audit_sha256: str = Field(pattern=SHA256_PATTERN)
    total_requests: int = Field(ge=1)
    successful_requests: int = Field(ge=0)
    unexplained_5xx: int = Field(ge=0)
    oom_events: int = Field(ge=0)
    hung_processes: int = Field(ge=0)
    gateway_throughput_ratio_basis_points: int = Field(ge=0)
    p95_latency_overhead_median_basis_points: int
    readiness_failure_milliseconds: int = Field(ge=0)
    backend_recovery_milliseconds: int = Field(ge=0)
    rollback_recovery_milliseconds: int = Field(ge=0)
    api_contract_passed: bool
    auth_contract_passed: bool
    streaming_contract_passed: bool
    cancellation_contract_passed: bool
    error_mapping_contract_passed: bool
    thinking_content_hidden: bool
    backend_guard_passed: bool
    m6_quality_complete: bool
    lineage_complete: bool
    unmitigated_critical_vulnerabilities: int = Field(ge=0)
    unmitigated_high_vulnerabilities: int = Field(ge=0)
    checks: tuple[M7GateCheck, ...]
    status: Literal["accepted", "rejected"]
    production_eligible: bool

    @property
    def success_rate_basis_points(self) -> int:
        return self.successful_requests * 10_000 // self.total_requests

    @model_validator(mode="after")
    def validate_gate(self) -> M7ProductionGate:
        if self.evaluated_at.tzinfo is None:
            raise ValueError("M7 Gate timestamp must be timezone-aware")
        if self.successful_requests > self.total_requests:
            raise ValueError("successful requests cannot exceed total requests")
        expected_names = (
            "api_contract",
            "success_rate",
            "runtime_stability",
            "gateway_throughput",
            "p95_latency_overhead",
            "backend_recovery",
            "deployment_rollback",
            "m6_evidence",
            "security_audit",
        )
        if tuple(check.name for check in self.checks) != expected_names:
            raise ValueError("M7 Gate checks must use the frozen order")
        expected_passed = (
            all(
                (
                    self.api_contract_passed,
                    self.auth_contract_passed,
                    self.streaming_contract_passed,
                    self.cancellation_contract_passed,
                    self.error_mapping_contract_passed,
                    self.thinking_content_hidden,
                    self.backend_guard_passed,
                )
            ),
            self.success_rate_basis_points >= 9950,
            self.unexplained_5xx == 0 and self.oom_events == 0 and self.hung_processes == 0,
            self.gateway_throughput_ratio_basis_points >= 9000,
            self.p95_latency_overhead_median_basis_points <= 1000,
            self.readiness_failure_milliseconds <= 5000
            and self.backend_recovery_milliseconds <= 180_000,
            self.rollback_recovery_milliseconds <= 180_000,
            self.m6_quality_complete and self.lineage_complete,
            self.unmitigated_critical_vulnerabilities == 0
            and self.unmitigated_high_vulnerabilities == 0,
        )
        if tuple(check.passed for check in self.checks) != expected_passed:
            raise ValueError("M7 Gate checks differ from frozen thresholds")
        accepted = all(check.passed for check in self.checks)
        if self.production_eligible != accepted:
            raise ValueError("M7 Production eligibility differs from Gate checks")
        if self.status != ("accepted" if accepted else "rejected"):
            raise ValueError("M7 Gate status differs from Gate checks")
        return self


class M7ProductionRecord(StrictSchema):
    """Immutable Production record that references, but never mutates, M6 evidence."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["Production"] = "Production"
    production_version: str = Field(pattern=PRODUCTION_VERSION_PATTERN)
    promoted_at: datetime
    source_candidate_version: str = Field(pattern=CANDIDATE_VERSION_PATTERN)
    candidate_record_sha256: str = Field(pattern=SHA256_PATTERN)
    production_gate_id: str = Field(pattern=r"^m7-production-gate-[0-9a-f]{8}$")
    production_gate_sha256: str = Field(pattern=SHA256_PATTERN)
    model: M6ModelIdentity
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    serving_config_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    benchmark_report_sha256: str = Field(pattern=SHA256_PATTERN)
    recovery_report_sha256: str = Field(pattern=SHA256_PATTERN)
    rollback_report_sha256: str = Field(pattern=SHA256_PATTERN)
    security_audit_sha256: str = Field(pattern=SHA256_PATTERN)
    production_eligible: Literal[True] = True

    @model_validator(mode="after")
    def validate_record(self) -> M7ProductionRecord:
        if self.promoted_at.tzinfo is None:
            raise ValueError("M7 promotion timestamp must be timezone-aware")
        if self.model.role != "candidate":
            raise ValueError("Production must retain the immutable M6 Candidate identity")
        if self.model.model_artifact_sha256 != self.model_artifact_sha256:
            raise ValueError("Production model hash differs from M6 Candidate")
        expected_size = "0-6b" if self.model.repository.endswith("0.6B") else "8b"
        if not self.production_version.startswith(f"qwen3-{expected_size}-m7-"):
            raise ValueError("Production version and repository differ")
        return self


class M7ProductionAlias(StrictSchema):
    """Atomically replaceable pointer to one immutable Production record."""

    schema_version: Literal["1.0"] = "1.0"
    alias: Literal["production"] = "production"
    production_version: str = Field(pattern=PRODUCTION_VERSION_PATTERN)
    production_record_sha256: str = Field(pattern=SHA256_PATTERN)
    previous_production_version: str | None = Field(
        default=None, pattern=PRODUCTION_VERSION_PATTERN
    )
    updated_at: datetime

    @model_validator(mode="after")
    def validate_alias(self) -> M7ProductionAlias:
        if self.updated_at.tzinfo is None:
            raise ValueError("Production Alias timestamp must be timezone-aware")
        if self.previous_production_version == self.production_version:
            raise ValueError("Production Alias previous target must differ")
        return self
