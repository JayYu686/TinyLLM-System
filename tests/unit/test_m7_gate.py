from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.unit.test_deployment_registry import CANDIDATE, _artifact_root
from tests.unit.test_m6 import _evaluation
from tinyllm.benchmark.inference_schema import (
    InferenceBenchmarkSummary,
    load_inference_benchmark_config,
)
from tinyllm.deployment import (
    DeploymentError,
    M7ContractEvidence,
    M7PackageVersion,
    M7RecoveryEvidence,
    M7RollbackEvidence,
    M7SecurityAudit,
    M7ServingEnvironment,
    M7ServingHardware,
)
from tinyllm.deployment.gate import assemble_m7_production_gate
from tinyllm.evaluation import compare_m6_evaluations, load_m6_release_config
from tinyllm.schemas import canonical_config_hash
from tinyllm.serving.config import load_gateway_config

NOW = datetime(2026, 8, 13, tzinfo=UTC)


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _evidence(tmp_path: Path) -> dict[str, object]:
    root, record = _artifact_root(tmp_path / "artifacts")
    evaluation = _evaluation("candidate", correct=True).model_copy(
        update={"model": record.model, "evaluation_id": "m6-candidate-unit"}
    )
    candidate_path = tmp_path / "candidate-evaluation.json"
    candidate_hash = _write(candidate_path, evaluation.to_dict())
    base = _evaluation("base", correct=False)
    release = load_m6_release_config(Path("configs/eval/m6_release.yaml"))
    comparison = compare_m6_evaluations(
        release,
        base,
        evaluation,
        base_evaluation_sha256="a" * 64,
        candidate_evaluation_sha256=candidate_hash,
    )
    comparison_path = tmp_path / "comparison.json"
    comparison_hash = _write(comparison_path, comparison.to_dict())
    record = record.model_copy(
        update={
            "comparison_sha256": comparison_hash,
            "comparison_config_sha256": comparison.config_sha256,
            "candidate_evaluation_id": evaluation.evaluation_id,
            "candidate_evaluation_sha256": candidate_hash,
        }
    )
    candidate_record_path = root / "registry" / "candidates" / CANDIDATE / "model.json"
    _write(candidate_record_path, record.to_dict())
    candidate_record_hash = hashlib.sha256(candidate_record_path.read_bytes()).hexdigest()

    environment = M7ServingEnvironment(
        captured_at=NOW,
        python_version="3.11.14",
        platform="Linux-unit",
        packages=tuple(
            M7PackageVersion(name=name, version="1.0")
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
        ),
        serving_constraints_sha256="7" * 64,
        vllm_wheel_sha256="8" * 64,
        vllm_wheel_source="vllm-github-release-cu118",
        git_commit="9" * 40,
        git_dirty=False,
    )
    environment_path = tmp_path / "environment.json"
    environment_hash = _write(environment_path, environment.to_dict())
    hardware = M7ServingHardware(
        captured_at=NOW,
        physical_gpu_index=7,
        gpu_name="NVIDIA GeForce RTX 3090",
        memory_total_mib=24576,
        driver_version="535.261.03",
        cuda_runtime_version="11.8",
        bf16_supported=True,
        numa_node=1,
        cpu_affinity="16-31,48-63",
        topology_sha256="6" * 64,
        memory_used_mib_at_preflight=4,
        utilization_percent_at_preflight=0,
        temperature_c_at_preflight=28,
    )
    hardware_path = tmp_path / "hardware.json"
    hardware_hash = _write(hardware_path, hardware.to_dict())
    serving_config = load_gateway_config(Path("configs/serving/m7_gateway.yaml"))
    serving_config_hash = canonical_config_hash(serving_config)
    benchmark_gateway_config = load_gateway_config(
        Path("configs/serving/m7_gateway_benchmark.yaml")
    )

    contract = M7ContractEvidence(
        evidence_id="m7-contract-1234abcd",
        evaluated_at=NOW,
        candidate_model_version=CANDIDATE,
        model_artifact_sha256=record.model.model_artifact_sha256,
        serving_config_sha256=serving_config_hash,
        environment_sha256=environment_hash,
        api_contract_passed=True,
        auth_contract_passed=True,
        streaming_contract_passed=True,
        cancellation_contract_passed=True,
        error_mapping_contract_passed=True,
        thinking_content_hidden=True,
        backend_guard_passed=True,
        passed=True,
    )
    contract_path = tmp_path / "contract.json"
    contract_hash = _write(contract_path, contract.to_dict())
    recovery = M7RecoveryEvidence(
        evidence_id="m7-recovery-1234abcd",
        evaluated_at=NOW,
        candidate_model_version=CANDIDATE,
        model_artifact_sha256=record.model.model_artifact_sha256,
        serving_config_sha256=serving_config_hash,
        environment_sha256=environment_hash,
        ready_before_failure=True,
        readiness_failure_milliseconds=100,
        backend_recovery_milliseconds=10_000,
        backend_restart_count=1,
        post_recovery_request_succeeded=True,
        passed=True,
    )
    recovery_path = tmp_path / "recovery.json"
    _write(recovery_path, recovery.to_dict())
    rollback = M7RollbackEvidence(
        evidence_id="m7-rollback-1234abcd",
        evaluated_at=NOW,
        candidate_model_version=CANDIDATE,
        model_artifact_sha256=record.model.model_artifact_sha256,
        serving_config_sha256=serving_config_hash,
        environment_sha256=environment_hash,
        failure_injected="invalid-model-reference",
        switch_rejected_before_activation=True,
        last_known_good_identity_preserved=True,
        ready_after_rejection=True,
        post_rejection_request_succeeded=True,
        rollback_recovery_milliseconds=1,
        passed=True,
    )
    rollback_path = tmp_path / "rollback.json"
    _write(rollback_path, rollback.to_dict())
    security = M7SecurityAudit(
        audit_id="m7-security-1234abcd",
        evaluated_at=NOW,
        profile="loopback-qwen3-text-cu118-v1",
        environment_sha256=environment_hash,
        pip_audit_sha256="1" * 64,
        osv_snapshot_sha256="2" * 64,
        control_evidence_sha256=contract_hash,
        observed_advisories=0,
        reviewed_critical_high_advisories=0,
        assessments=(),
        unmitigated_critical_vulnerabilities=0,
        unmitigated_high_vulnerabilities=0,
        status="accepted",
    )
    security_path = tmp_path / "security.json"
    _write(security_path, security.to_dict())
    inference_config = load_inference_benchmark_config(Path("configs/benchmark/m7_inference.yaml"))
    benchmark = InferenceBenchmarkSummary(
        benchmark_id=inference_config.benchmark_id,
        completed_at=NOW,
        model_version=CANDIDATE,
        model_artifact_sha256=record.model.model_artifact_sha256,
        tokenizer_artifact_sha256="e" * 64,
        config_sha256=canonical_config_hash(inference_config),
        gateway_config_sha256=canonical_config_hash(benchmark_gateway_config),
        environment_sha256=environment_hash,
        hardware_sha256=hardware_hash,
        request_results_sha256="f" * 64,
        total_requests=18_000,
        successful_requests=18_000,
        success_rate_basis_points=10_000,
        gateway_throughput_ratio_basis_points=9500,
        p95_latency_overhead_median_basis_points=500,
        oom_events=0,
        hung_processes=0,
        unexplained_5xx=0,
        status="succeeded",
    )
    from tinyllm.deployment import resolve_model

    resolved = resolve_model(root, CANDIDATE, now=NOW)
    benchmark = benchmark.model_copy(
        update={"tokenizer_artifact_sha256": resolved.tokenizer_artifact_sha256}
    )
    benchmark_path = tmp_path / "benchmark.json"
    _write(benchmark_path, benchmark.to_dict())
    return {
        "artifact_root": root,
        "candidate_model_version": CANDIDATE,
        "benchmark_path": benchmark_path,
        "contract_path": contract_path,
        "recovery_path": recovery_path,
        "rollback_path": rollback_path,
        "security_path": security_path,
        "m6_comparison_path": comparison_path,
        "m6_candidate_evaluation_path": candidate_path,
        "serving_config_path": Path("configs/serving/m7_gateway.yaml"),
        "benchmark_config_path": Path("configs/benchmark/m7_inference.yaml"),
        "benchmark_gateway_config_path": Path("configs/serving/m7_gateway_benchmark.yaml"),
        "environment_path": environment_path,
        "hardware_path": hardware_path,
        "output_path": tmp_path / "gate.json",
        "candidate_record_hash": candidate_record_hash,
    }


def test_m7_gate_recomputes_accepted_decision(tmp_path: Path) -> None:
    values = _evidence(tmp_path)
    expected_candidate_hash = values.pop("candidate_record_hash")

    gate = assemble_m7_production_gate(**values, now=NOW)  # type: ignore[arg-type]

    assert gate.status == "accepted"
    assert gate.production_eligible is True
    assert gate.candidate_record_sha256 == expected_candidate_hash
    assert all(check.passed for check in gate.checks)


def test_m7_gate_rejects_cross_run_evidence(tmp_path: Path) -> None:
    values = _evidence(tmp_path)
    values.pop("candidate_record_hash")
    benchmark_path = values["benchmark_path"]
    assert isinstance(benchmark_path, Path)
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    payload["environment_sha256"] = "0" * 64
    _write(benchmark_path, payload)

    with pytest.raises(DeploymentError, match="lineage"):
        assemble_m7_production_gate(**values, now=NOW)  # type: ignore[arg-type]


def test_m7_gate_can_persist_a_rejected_performance_decision(tmp_path: Path) -> None:
    values = _evidence(tmp_path)
    values.pop("candidate_record_hash")
    benchmark_path = values["benchmark_path"]
    assert isinstance(benchmark_path, Path)
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    payload["gateway_throughput_ratio_basis_points"] = 8999
    payload["p95_latency_overhead_median_basis_points"] = 1001
    _write(benchmark_path, payload)

    gate = assemble_m7_production_gate(**values, now=NOW)  # type: ignore[arg-type]

    assert gate.status == "rejected"
    assert gate.production_eligible is False
    assert {check.name for check in gate.checks if not check.passed} == {
        "gateway_throughput",
        "p95_latency_overhead",
    }


def test_m7_gate_requires_new_absolute_output(tmp_path: Path) -> None:
    values = _evidence(tmp_path)
    values.pop("candidate_record_hash")
    output_path = values["output_path"]
    assert isinstance(output_path, Path)
    output_path.write_text("occupied", encoding="utf-8")

    with pytest.raises(DeploymentError, match="new absolute path"):
        assemble_m7_production_gate(**values, now=NOW)  # type: ignore[arg-type]


def test_m7_gate_rejects_relative_or_invalid_evidence_paths(tmp_path: Path) -> None:
    values = _evidence(tmp_path)
    values.pop("candidate_record_hash")
    values["contract_path"] = Path("relative-contract.json")
    with pytest.raises(DeploymentError, match="absolute"):
        assemble_m7_production_gate(**values, now=NOW)  # type: ignore[arg-type]

    values = _evidence(tmp_path / "invalid")
    values.pop("candidate_record_hash")
    contract_path = values["contract_path"]
    assert isinstance(contract_path, Path)
    contract_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(DeploymentError, match="invalid"):
        assemble_m7_production_gate(**values, now=NOW)  # type: ignore[arg-type]
