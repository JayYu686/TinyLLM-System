"""Strict, content-free M7 service and security evidence schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.deployment.schema import CANDIDATE_VERSION_PATTERN, SHA256_PATTERN
from tinyllm.schemas.base import StrictSchema


class M7ContractEvidence(StrictSchema):
    """Real API contract checks collected against one running Gateway."""

    schema_version: Literal["1.0"] = "1.0"
    evidence_id: str = Field(pattern=r"^m7-contract-[0-9a-f]{8}$")
    evaluated_at: datetime
    candidate_model_version: str = Field(pattern=CANDIDATE_VERSION_PATTERN)
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    serving_config_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    api_contract_passed: bool
    auth_contract_passed: bool
    streaming_contract_passed: bool
    cancellation_contract_passed: bool
    error_mapping_contract_passed: bool
    thinking_content_hidden: bool
    backend_guard_passed: bool
    passed: bool

    @model_validator(mode="after")
    def validate_evidence(self) -> M7ContractEvidence:
        if self.evaluated_at.tzinfo is None:
            raise ValueError("M7 contract evidence timestamp must be timezone-aware")
        expected = all(
            (
                self.api_contract_passed,
                self.auth_contract_passed,
                self.streaming_contract_passed,
                self.cancellation_contract_passed,
                self.error_mapping_contract_passed,
                self.thinking_content_hidden,
                self.backend_guard_passed,
            )
        )
        if self.passed != expected:
            raise ValueError("M7 contract status differs from its checks")
        return self


class M7RecoveryEvidence(StrictSchema):
    """Managed-backend crash and restart evidence."""

    schema_version: Literal["1.0"] = "1.0"
    evidence_id: str = Field(pattern=r"^m7-recovery-[0-9a-f]{8}$")
    evaluated_at: datetime
    candidate_model_version: str = Field(pattern=CANDIDATE_VERSION_PATTERN)
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    serving_config_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    ready_before_failure: bool
    readiness_failure_milliseconds: int = Field(ge=0)
    backend_recovery_milliseconds: int = Field(ge=0)
    backend_restart_count: int = Field(ge=1)
    post_recovery_request_succeeded: bool
    passed: bool

    @model_validator(mode="after")
    def validate_evidence(self) -> M7RecoveryEvidence:
        if self.evaluated_at.tzinfo is None:
            raise ValueError("M7 recovery evidence timestamp must be timezone-aware")
        expected = (
            self.ready_before_failure
            and self.readiness_failure_milliseconds <= 5_000
            and self.backend_recovery_milliseconds <= 180_000
            and self.post_recovery_request_succeeded
        )
        if self.passed != expected:
            raise ValueError("M7 recovery status differs from its frozen thresholds")
        return self


class M7RollbackEvidence(StrictSchema):
    """Last Known Good preservation after a rejected configuration switch."""

    schema_version: Literal["1.0"] = "1.0"
    evidence_id: str = Field(pattern=r"^m7-rollback-[0-9a-f]{8}$")
    evaluated_at: datetime
    candidate_model_version: str = Field(pattern=CANDIDATE_VERSION_PATTERN)
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    serving_config_sha256: str = Field(pattern=SHA256_PATTERN)
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    failure_injected: Literal["model-hash-drift", "invalid-model-reference", "invalid-config"]
    switch_rejected_before_activation: bool
    last_known_good_identity_preserved: bool
    ready_after_rejection: bool
    post_rejection_request_succeeded: bool
    rollback_recovery_milliseconds: int = Field(ge=0)
    passed: bool

    @model_validator(mode="after")
    def validate_evidence(self) -> M7RollbackEvidence:
        if self.evaluated_at.tzinfo is None:
            raise ValueError("M7 rollback evidence timestamp must be timezone-aware")
        expected = (
            self.switch_rejected_before_activation
            and self.last_known_good_identity_preserved
            and self.ready_after_rejection
            and self.post_rejection_request_succeeded
            and self.rollback_recovery_milliseconds <= 180_000
        )
        if self.passed != expected:
            raise ValueError("M7 rollback status differs from its frozen thresholds")
        return self


class M7VulnerabilityAssessment(StrictSchema):
    """One canonical vulnerability and its reviewed applicability decision."""

    advisory_id: str = Field(pattern=r"^(GHSA|CVE)-[A-Za-z0-9-]{4,64}$")
    package: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    installed_version: str = Field(min_length=1, max_length=64)
    severity: Literal["critical", "high", "moderate", "low", "unknown"]
    disposition: Literal["fixed", "mitigated", "not_affected", "unmitigated"]
    rationale: str = Field(min_length=20, max_length=1000)
    controls: tuple[str, ...] = Field(default=(), max_length=16)

    @field_validator("controls", mode="before")
    @classmethod
    def freeze_controls(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_assessment(self) -> M7VulnerabilityAssessment:
        if self.disposition == "mitigated" and not self.controls:
            raise ValueError("mitigated vulnerability requires explicit controls")
        if self.disposition == "fixed" and self.controls:
            raise ValueError("fixed vulnerability cannot rely on compensating controls")
        return self


class M7SecurityAudit(StrictSchema):
    """Frozen package audit plus VEX-style applicability assessment."""

    schema_version: Literal["1.0"] = "1.0"
    audit_id: str = Field(pattern=r"^m7-security-[0-9a-f]{8}$")
    evaluated_at: datetime
    profile: Literal["loopback-qwen3-text-cu118-v1"]
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    pip_audit_sha256: str = Field(pattern=SHA256_PATTERN)
    osv_snapshot_sha256: str = Field(pattern=SHA256_PATTERN)
    control_evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    observed_advisories: int = Field(ge=0)
    reviewed_critical_high_advisories: int = Field(ge=0)
    assessments: tuple[M7VulnerabilityAssessment, ...]
    unmitigated_critical_vulnerabilities: int = Field(ge=0)
    unmitigated_high_vulnerabilities: int = Field(ge=0)
    status: Literal["accepted", "rejected"]

    @field_validator("assessments", mode="before")
    @classmethod
    def freeze_assessments(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_audit(self) -> M7SecurityAudit:
        if self.evaluated_at.tzinfo is None:
            raise ValueError("M7 security audit timestamp must be timezone-aware")
        identities = [item.advisory_id for item in self.assessments]
        if len(identities) != len(set(identities)):
            raise ValueError("M7 security audit contains duplicate canonical advisories")
        if self.reviewed_critical_high_advisories != len(self.assessments):
            raise ValueError("M7 security review count differs from its assessments")
        if self.observed_advisories < self.reviewed_critical_high_advisories:
            raise ValueError("M7 observed advisory count is smaller than its review set")
        critical = sum(
            item.severity == "critical" and item.disposition == "unmitigated"
            for item in self.assessments
        )
        high = sum(
            item.severity == "high" and item.disposition == "unmitigated"
            for item in self.assessments
        )
        if (critical, high) != (
            self.unmitigated_critical_vulnerabilities,
            self.unmitigated_high_vulnerabilities,
        ):
            raise ValueError("M7 unmitigated vulnerability counts do not match assessments")
        accepted = critical == 0 and high == 0
        if self.status != ("accepted" if accepted else "rejected"):
            raise ValueError("M7 security status differs from reviewed assessments")
        return self


class M7PackageVersion(StrictSchema):
    """One normalized installed distribution identity."""

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    version: str = Field(min_length=1, max_length=100)


class M7ServingEnvironment(StrictSchema):
    """Path-free software identity for a serving evidence campaign."""

    schema_version: Literal["1.0"] = "1.0"
    captured_at: datetime
    python_version: str = Field(min_length=1, max_length=100)
    platform: str = Field(min_length=1, max_length=300)
    packages: tuple[M7PackageVersion, ...]
    serving_constraints_sha256: str = Field(pattern=SHA256_PATTERN)
    vllm_wheel_sha256: str = Field(pattern=SHA256_PATTERN)
    vllm_wheel_source: Literal["vllm-github-release-cu118"]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool

    @field_validator("packages", mode="before")
    @classmethod
    def freeze_packages(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_environment(self) -> M7ServingEnvironment:
        if self.captured_at.tzinfo is None:
            raise ValueError("M7 environment timestamp must be timezone-aware")
        names = tuple(package.name for package in self.packages)
        if names != tuple(sorted(names, key=str.lower)) or len(names) != len(set(names)):
            raise ValueError("M7 packages must be uniquely sorted")
        required = {
            "fastapi",
            "protobuf",
            "ray",
            "starlette",
            "tokenizers",
            "torch",
            "transformers",
            "vllm",
            "xgrammar",
        }
        if not required.issubset(names):
            raise ValueError("M7 serving environment is missing a required package")
        return self


class M7ServingHardware(StrictSchema):
    """One selected physical GPU and driver identity without host identifiers."""

    schema_version: Literal["1.0"] = "1.0"
    captured_at: datetime
    physical_gpu_index: int = Field(ge=0, le=9)
    gpu_name: Literal["NVIDIA GeForce RTX 3090"]
    memory_total_mib: int = Field(ge=24_000, le=25_000)
    driver_version: str = Field(pattern=r"^[0-9.]{3,32}$")
    cuda_runtime_version: str = Field(min_length=1, max_length=32)
    bf16_supported: Literal[True]
    numa_node: int = Field(ge=0, le=15)
    cpu_affinity: str = Field(min_length=1, max_length=100)
    topology_sha256: str = Field(pattern=SHA256_PATTERN)
    memory_used_mib_at_preflight: int = Field(ge=0, le=25_000)
    utilization_percent_at_preflight: int = Field(ge=0, le=100)
    temperature_c_at_preflight: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_hardware(self) -> M7ServingHardware:
        if self.captured_at.tzinfo is None:
            raise ValueError("M7 hardware timestamp must be timezone-aware")
        return self
