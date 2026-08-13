"""Deterministically assemble an M7 Production Gate from immutable evidence."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from tinyllm.benchmark.inference_schema import (
    InferenceBenchmarkSummary,
    load_inference_benchmark_config,
)
from tinyllm.deployment.evidence_schema import (
    M7ContractEvidence,
    M7RecoveryEvidence,
    M7RollbackEvidence,
    M7SecurityAudit,
    M7ServingEnvironment,
    M7ServingHardware,
)
from tinyllm.deployment.registry import DeploymentError, DeploymentErrorCode, resolve_model
from tinyllm.deployment.schema import M7GateCheck, M7ProductionGate
from tinyllm.evaluation.m6_schema import M6ComparisonResult, M6EvaluationResult
from tinyllm.schemas import canonical_config_hash
from tinyllm.schemas.base import StrictSchema
from tinyllm.serving.config import load_gateway_config

EvidenceT = TypeVar("EvidenceT", bound=StrictSchema)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, model: type[EvidenceT], label: str) -> tuple[EvidenceT, str]:
    if not path.is_absolute() or path.is_symlink():
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, f"{label} path must be absolute and non-symlink"
        )
    try:
        payload = path.read_bytes()
        return model.model_validate_json(payload), hashlib.sha256(payload).hexdigest()
    except (OSError, ValidationError, ValueError) as exc:
        raise DeploymentError(DeploymentErrorCode.INVALID_INPUT, f"{label} is invalid") from exc


def _atomic_write(path: Path, value: object) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "M7 Gate output must be a new absolute path"
        )
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise DeploymentError(DeploymentErrorCode.IO_ERROR, "cannot persist M7 Gate") from exc


def _check(name: str, passed: bool, actual: str, requirement: str) -> M7GateCheck:
    return M7GateCheck(name=name, passed=passed, actual=actual, requirement=requirement)  # type: ignore[arg-type]


def assemble_m7_production_gate(
    *,
    artifact_root: Path,
    candidate_model_version: str,
    benchmark_path: Path,
    contract_path: Path,
    recovery_path: Path,
    rollback_path: Path,
    security_path: Path,
    m6_comparison_path: Path,
    m6_candidate_evaluation_path: Path,
    serving_config_path: Path,
    benchmark_gateway_config_path: Path,
    benchmark_config_path: Path,
    environment_path: Path,
    hardware_path: Path,
    output_path: Path,
    now: datetime | None = None,
) -> M7ProductionGate:
    """Verify every referenced artifact, recompute every predicate, and persist the Gate."""

    resolved = resolve_model(artifact_root, candidate_model_version, now=now)
    benchmark, benchmark_sha256 = _load(
        benchmark_path, InferenceBenchmarkSummary, "M7 benchmark summary"
    )
    contract, contract_sha256 = _load(contract_path, M7ContractEvidence, "M7 contract evidence")
    recovery, recovery_sha256 = _load(recovery_path, M7RecoveryEvidence, "M7 recovery evidence")
    rollback, rollback_sha256 = _load(rollback_path, M7RollbackEvidence, "M7 rollback evidence")
    security, security_sha256 = _load(security_path, M7SecurityAudit, "M7 security audit")
    comparison, comparison_sha256 = _load(m6_comparison_path, M6ComparisonResult, "M6 comparison")
    candidate_evaluation, candidate_evaluation_sha256 = _load(
        m6_candidate_evaluation_path, M6EvaluationResult, "M6 Candidate evaluation"
    )
    try:
        serving_config = load_gateway_config(serving_config_path)
        benchmark_gateway_config = load_gateway_config(benchmark_gateway_config_path)
        benchmark_config = load_inference_benchmark_config(benchmark_config_path)
    except Exception as exc:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "M7 serving configuration is invalid"
        ) from exc
    if (
        not environment_path.is_absolute()
        or environment_path.is_symlink()
        or not hardware_path.is_absolute()
        or hardware_path.is_symlink()
    ):
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT,
            "M7 environment and hardware paths must be absolute and non-symlink",
        )
    try:
        serving_environment, environment_sha256 = _load(
            environment_path, M7ServingEnvironment, "M7 serving environment"
        )
        _serving_hardware, hardware_sha256 = _load(
            hardware_path, M7ServingHardware, "M7 serving hardware"
        )
    except OSError as exc:
        raise DeploymentError(
            DeploymentErrorCode.NOT_FOUND, "M7 environment evidence is missing"
        ) from exc

    candidate_path = (
        artifact_root / "registry" / "candidates" / candidate_model_version / "model.json"
    )
    try:
        candidate_record = json.loads(candidate_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError(
            DeploymentErrorCode.INVALID_INPUT, "M6 Candidate record is unreadable"
        ) from exc
    serving_config_sha256 = canonical_config_hash(serving_config)
    lineage_values = (
        benchmark.model_version == candidate_model_version,
        benchmark.model_artifact_sha256 == resolved.model_artifact_sha256,
        benchmark.tokenizer_artifact_sha256 == resolved.tokenizer_artifact_sha256,
        benchmark.environment_sha256 == environment_sha256,
        benchmark.hardware_sha256 == hardware_sha256,
        benchmark.config_sha256 == canonical_config_hash(benchmark_config),
        benchmark.gateway_config_sha256 == canonical_config_hash(benchmark_gateway_config),
        security.environment_sha256 == environment_sha256,
        not serving_environment.git_dirty,
        security.control_evidence_sha256 == contract_sha256,
        contract.candidate_model_version == candidate_model_version,
        recovery.candidate_model_version == candidate_model_version,
        rollback.candidate_model_version == candidate_model_version,
        contract.model_artifact_sha256 == resolved.model_artifact_sha256,
        recovery.model_artifact_sha256 == resolved.model_artifact_sha256,
        rollback.model_artifact_sha256 == resolved.model_artifact_sha256,
        contract.serving_config_sha256 == serving_config_sha256,
        recovery.serving_config_sha256 == serving_config_sha256,
        rollback.serving_config_sha256 == serving_config_sha256,
        contract.environment_sha256 == environment_sha256,
        recovery.environment_sha256 == environment_sha256,
        rollback.environment_sha256 == environment_sha256,
        comparison_sha256 == candidate_record["comparison_sha256"],
        candidate_evaluation_sha256 == candidate_record["candidate_evaluation_sha256"],
        comparison.config_sha256 == candidate_record["comparison_config_sha256"],
        comparison.candidate_evaluation_id == candidate_record["candidate_evaluation_id"],
        comparison.candidate_evaluation_sha256 == candidate_evaluation_sha256,
        comparison.candidate_model == resolved.model,
        candidate_evaluation.model == resolved.model,
    )
    lineage_complete = all(lineage_values)
    m6_quality_complete = comparison.candidate_eligible and comparison.status == "accepted"
    if not lineage_complete:
        raise DeploymentError(
            DeploymentErrorCode.HASH_MISMATCH, "M7 evidence lineage does not form one identity"
        )

    contract_passed = contract.passed
    checks = (
        _check(
            "api_contract",
            contract_passed,
            "passed" if contract_passed else "failed",
            "API/auth/stream/cancel/error contracts all pass",
        ),
        _check(
            "success_rate",
            benchmark.success_rate_basis_points >= 9950,
            f"{benchmark.success_rate_basis_points}bp",
            ">=9950bp",
        ),
        _check(
            "runtime_stability",
            benchmark.unexplained_5xx == benchmark.oom_events == benchmark.hung_processes == 0,
            (
                f"5xx={benchmark.unexplained_5xx},oom={benchmark.oom_events},"
                f"hung={benchmark.hung_processes}"
            ),
            "all zero",
        ),
        _check(
            "gateway_throughput",
            benchmark.gateway_throughput_ratio_basis_points >= 9000,
            f"{benchmark.gateway_throughput_ratio_basis_points}bp",
            ">=9000bp of Direct",
        ),
        _check(
            "p95_latency_overhead",
            benchmark.p95_latency_overhead_median_basis_points <= 1000,
            f"{benchmark.p95_latency_overhead_median_basis_points}bp",
            "<=1000bp",
        ),
        _check(
            "backend_recovery",
            recovery.passed,
            (
                f"failure={recovery.readiness_failure_milliseconds}ms,"
                f"recovery={recovery.backend_recovery_milliseconds}ms"
            ),
            "failure<=5000ms,recovery<=180000ms",
        ),
        _check(
            "deployment_rollback",
            rollback.passed,
            f"{rollback.rollback_recovery_milliseconds}ms",
            "<=180000ms and Last Known Good preserved",
        ),
        _check(
            "m6_evidence",
            m6_quality_complete and lineage_complete,
            "complete" if m6_quality_complete and lineage_complete else "incomplete",
            "accepted M6 quality and complete lineage",
        ),
        _check(
            "security_audit",
            security.status == "accepted",
            (
                f"critical={security.unmitigated_critical_vulnerabilities},"
                f"high={security.unmitigated_high_vulnerabilities}"
            ),
            "no unmitigated Critical/High",
        ),
    )
    accepted = all(check.passed for check in checks)
    identity = "|".join(
        (
            resolved.candidate_record_sha256,
            benchmark_sha256,
            contract_sha256,
            recovery_sha256,
            rollback_sha256,
            security_sha256,
        )
    )
    gate = M7ProductionGate(
        gate_id=f"m7-production-gate-{hashlib.sha256(identity.encode()).hexdigest()[:8]}",
        evaluated_at=now or datetime.now(UTC),
        candidate_model_version=candidate_model_version,
        candidate_record_sha256=resolved.candidate_record_sha256,
        model_artifact_sha256=resolved.model_artifact_sha256,
        tokenizer_artifact_sha256=resolved.tokenizer_artifact_sha256,
        serving_config_sha256=serving_config_sha256,
        environment_sha256=environment_sha256,
        benchmark_report_sha256=benchmark_sha256,
        recovery_report_sha256=recovery_sha256,
        rollback_report_sha256=rollback_sha256,
        security_audit_sha256=security_sha256,
        total_requests=benchmark.total_requests,
        successful_requests=benchmark.successful_requests,
        unexplained_5xx=benchmark.unexplained_5xx,
        oom_events=benchmark.oom_events,
        hung_processes=benchmark.hung_processes,
        gateway_throughput_ratio_basis_points=benchmark.gateway_throughput_ratio_basis_points,
        p95_latency_overhead_median_basis_points=(
            benchmark.p95_latency_overhead_median_basis_points
        ),
        readiness_failure_milliseconds=recovery.readiness_failure_milliseconds,
        backend_recovery_milliseconds=recovery.backend_recovery_milliseconds,
        rollback_recovery_milliseconds=rollback.rollback_recovery_milliseconds,
        api_contract_passed=contract.api_contract_passed,
        auth_contract_passed=contract.auth_contract_passed,
        streaming_contract_passed=contract.streaming_contract_passed,
        cancellation_contract_passed=contract.cancellation_contract_passed,
        error_mapping_contract_passed=contract.error_mapping_contract_passed,
        thinking_content_hidden=contract.thinking_content_hidden,
        backend_guard_passed=contract.backend_guard_passed,
        m6_quality_complete=m6_quality_complete,
        lineage_complete=lineage_complete,
        unmitigated_critical_vulnerabilities=(security.unmitigated_critical_vulnerabilities),
        unmitigated_high_vulnerabilities=security.unmitigated_high_vulnerabilities,
        checks=checks,
        status="accepted" if accepted else "rejected",
        production_eligible=accepted,
    )
    _atomic_write(output_path, gate.to_dict())
    return gate
