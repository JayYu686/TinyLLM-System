from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tinyllm.deployment import (
    M7ContractEvidence,
    M7RecoveryEvidence,
    M7RollbackEvidence,
    M7SecurityAudit,
    M7ServingEnvironment,
    M7ServingHardware,
    M7VulnerabilityAssessment,
)

NOW = datetime(2026, 8, 13, tzinfo=UTC)
CANDIDATE = "qwen3-0-6b-m6-aaaaaaaa"
SHA = "a" * 64


def _contract(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "evidence_id": "m7-contract-1234abcd",
        "evaluated_at": NOW,
        "candidate_model_version": CANDIDATE,
        "model_artifact_sha256": SHA,
        "serving_config_sha256": SHA,
        "environment_sha256": SHA,
        "api_contract_passed": True,
        "auth_contract_passed": True,
        "streaming_contract_passed": True,
        "cancellation_contract_passed": True,
        "error_mapping_contract_passed": True,
        "thinking_content_hidden": True,
        "backend_guard_passed": True,
        "passed": True,
    }
    value.update(updates)
    return value


def test_contract_requires_aware_timestamp_and_self_consistent_status() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        M7ContractEvidence.model_validate(_contract(evaluated_at=datetime(2026, 8, 13)))
    with pytest.raises(ValidationError, match="differs"):
        M7ContractEvidence.model_validate(_contract(api_contract_passed=False))


def _recovery(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "evidence_id": "m7-recovery-1234abcd",
        "evaluated_at": NOW,
        "candidate_model_version": CANDIDATE,
        "model_artifact_sha256": SHA,
        "serving_config_sha256": SHA,
        "environment_sha256": SHA,
        "ready_before_failure": True,
        "readiness_failure_milliseconds": 5_000,
        "backend_recovery_milliseconds": 180_000,
        "backend_restart_count": 1,
        "post_recovery_request_succeeded": True,
        "passed": True,
    }
    value.update(updates)
    return value


def test_recovery_requires_aware_timestamp_and_frozen_thresholds() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        M7RecoveryEvidence.model_validate(_recovery(evaluated_at=datetime(2026, 8, 13)))
    with pytest.raises(ValidationError, match="frozen thresholds"):
        M7RecoveryEvidence.model_validate(_recovery(backend_recovery_milliseconds=180_001))


def _rollback(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "evidence_id": "m7-rollback-1234abcd",
        "evaluated_at": NOW,
        "candidate_model_version": CANDIDATE,
        "model_artifact_sha256": SHA,
        "serving_config_sha256": SHA,
        "environment_sha256": SHA,
        "failure_injected": "invalid-config",
        "switch_rejected_before_activation": True,
        "last_known_good_identity_preserved": True,
        "ready_after_rejection": True,
        "post_rejection_request_succeeded": True,
        "rollback_recovery_milliseconds": 180_000,
        "passed": True,
    }
    value.update(updates)
    return value


def test_rollback_requires_aware_timestamp_and_frozen_thresholds() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        M7RollbackEvidence.model_validate(_rollback(evaluated_at=datetime(2026, 8, 13)))
    with pytest.raises(ValidationError, match="frozen thresholds"):
        M7RollbackEvidence.model_validate(_rollback(last_known_good_identity_preserved=False))


def _assessment(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "advisory_id": "GHSA-aaaa-bbbb-cccc",
        "package": "vllm",
        "installed_version": "0.8.5.post1+cu118",
        "severity": "high",
        "disposition": "mitigated",
        "rationale": "The vulnerable feature is unreachable in the frozen serving profile.",
        "controls": ["loopback-only backend"],
    }
    value.update(updates)
    return value


def test_vulnerability_assessment_freezes_controls_and_enforces_disposition() -> None:
    assessment = M7VulnerabilityAssessment.model_validate(_assessment())
    assert assessment.controls == ("loopback-only backend",)
    with pytest.raises(ValidationError, match="requires explicit controls"):
        M7VulnerabilityAssessment.model_validate(_assessment(controls=[]))
    with pytest.raises(ValidationError, match="cannot rely"):
        M7VulnerabilityAssessment.model_validate(_assessment(disposition="fixed"))


def _audit(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "audit_id": "m7-security-1234abcd",
        "evaluated_at": NOW,
        "profile": "loopback-qwen3-text-cu118-v1",
        "environment_sha256": SHA,
        "pip_audit_sha256": SHA,
        "osv_snapshot_sha256": SHA,
        "control_evidence_sha256": SHA,
        "observed_advisories": 1,
        "reviewed_critical_high_advisories": 1,
        "assessments": [_assessment()],
        "unmitigated_critical_vulnerabilities": 0,
        "unmitigated_high_vulnerabilities": 0,
        "status": "accepted",
    }
    value.update(updates)
    return value


@pytest.mark.parametrize(
    "updates,error",
    [
        ({"evaluated_at": datetime(2026, 8, 13)}, "timezone-aware"),
        ({"assessments": [_assessment(), _assessment()]}, "duplicate"),
        ({"reviewed_critical_high_advisories": 0}, "review count"),
        ({"observed_advisories": 0}, "observed advisory count"),
        ({"unmitigated_high_vulnerabilities": 1}, "counts do not match"),
        ({"status": "rejected"}, "status differs"),
    ],
)
def test_security_audit_rejects_inconsistent_evidence(
    updates: dict[str, object], error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        M7SecurityAudit.model_validate(_audit(**updates))


def test_security_audit_counts_unmitigated_critical_and_high() -> None:
    critical = _assessment(
        advisory_id="GHSA-1111-2222-3333",
        severity="critical",
        disposition="unmitigated",
        controls=[],
    )
    high = _assessment(
        advisory_id="GHSA-4444-5555-6666",
        disposition="unmitigated",
        controls=[],
    )
    audit = M7SecurityAudit.model_validate(
        _audit(
            observed_advisories=2,
            reviewed_critical_high_advisories=2,
            assessments=[critical, high],
            unmitigated_critical_vulnerabilities=1,
            unmitigated_high_vulnerabilities=1,
            status="rejected",
        )
    )
    assert audit.status == "rejected"
    assert isinstance(audit.assessments, tuple)


def _packages() -> list[dict[str, str]]:
    return [
        {"name": name, "version": "1.0"}
        for name in sorted(
            (
                "fastapi",
                "protobuf",
                "ray",
                "starlette",
                "tokenizers",
                "torch",
                "transformers",
                "vllm",
                "xgrammar",
            )
        )
    ]


def _environment(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "captured_at": NOW,
        "python_version": "3.11.14",
        "platform": "Linux-unit",
        "packages": _packages(),
        "serving_constraints_sha256": SHA,
        "vllm_wheel_sha256": SHA,
        "vllm_wheel_source": "vllm-github-release-cu118",
        "git_commit": "a" * 40,
        "git_dirty": False,
    }
    value.update(updates)
    return value


def test_serving_environment_freezes_and_validates_package_inventory() -> None:
    environment = M7ServingEnvironment.model_validate(_environment())
    assert isinstance(environment.packages, tuple)
    with pytest.raises(ValidationError, match="timezone-aware"):
        M7ServingEnvironment.model_validate(_environment(captured_at=datetime(2026, 8, 13)))
    with pytest.raises(ValidationError, match="uniquely sorted"):
        M7ServingEnvironment.model_validate(_environment(packages=list(reversed(_packages()))))
    with pytest.raises(ValidationError, match="required package"):
        M7ServingEnvironment.model_validate(_environment(packages=_packages()[:-1]))


def test_serving_hardware_requires_aware_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        M7ServingHardware(
            captured_at=datetime(2026, 8, 13),
            physical_gpu_index=7,
            gpu_name="NVIDIA GeForce RTX 3090",
            memory_total_mib=24576,
            driver_version="535.261.03",
            cuda_runtime_version="11.8",
            bf16_supported=True,
            numa_node=1,
            cpu_affinity="16-31,48-63",
            topology_sha256=SHA,
            memory_used_mib_at_preflight=4,
            utilization_percent_at_preflight=0,
            temperature_c_at_preflight=28,
        )
