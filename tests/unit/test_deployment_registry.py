from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from tinyllm.cli import main
from tinyllm.deployment import (
    DeploymentError,
    DeploymentErrorCode,
    M7GateCheck,
    M7ProductionGate,
    promote_production,
    resolve_model,
    rollback_production,
    show_deployment,
)
from tinyllm.evaluation import M6ModelIdentity, M6PromotionRecord

NOW = datetime(2026, 8, 13, tzinfo=UTC)
CANDIDATE = "qwen3-0-6b-m6-aaaaaaaa"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_root(tmp_path: Path) -> tuple[Path, M6PromotionRecord]:
    root = tmp_path.resolve()
    run_id = "20260813T000000Z-m7-unit-test-aaaaaaaa-beef"
    model_dir = root / "runs" / "campaign" / run_id / "exports" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        '{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}\n',
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"safe-model")
    model_hash = _artifact_hash(model_dir)
    tokenizer_dir = root / "cache" / "models" / "Qwen" / "Qwen3-0.6B" / REVISION
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (tokenizer_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    model = M6ModelIdentity(
        role="candidate",
        repository="Qwen/Qwen3-0.6B",
        base_revision=cast(Any, REVISION),
        attention_architecture="gqa",
        adaptation="full_sft",
        model_artifact_sha256=model_hash,
        model_parameters=596049920,
        training_run_id=run_id,
        training_checkpoint_id="checkpoint-tokens-0001000000",
        training_tokens=1_000_000,
        training_config_sha256="c" * 64,
        dataset_version="m7-unit-data-v1",
        dataset_manifest_sha256="d" * 64,
    )
    record = M6PromotionRecord(
        status="Candidate",
        model_version=CANDIDATE,
        promoted_at=NOW,
        comparison_sha256="a" * 64,
        comparison_config_sha256="b" * 64,
        candidate_evaluation_id="m6-candidate-unit",
        candidate_evaluation_sha256="e" * 64,
        model=model,
        production_eligible=False,
    )
    _write_json(root / "registry" / "candidates" / CANDIDATE / "model.json", record.to_dict())
    return root, record


def _gate(root: Path, record: M6PromotionRecord, *, accepted: bool = True) -> M7ProductionGate:
    resolved = resolve_model(root, CANDIDATE, now=NOW)
    passed_by_name = {
        "api_contract": accepted,
        "success_rate": accepted,
        "runtime_stability": True,
        "gateway_throughput": True,
        "p95_latency_overhead": True,
        "backend_recovery": True,
        "deployment_rollback": True,
        "m6_evidence": accepted,
        "security_audit": True,
    }
    checks = tuple(
        M7GateCheck(
            name=cast(Any, name),
            passed=passed_by_name[name],
            actual="pass",
            requirement="frozen",
        )
        for name in (
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
    )
    return M7ProductionGate(
        gate_id="m7-production-gate-1234abcd",
        evaluated_at=NOW,
        candidate_model_version=CANDIDATE,
        candidate_record_sha256=resolved.candidate_record_sha256,
        model_artifact_sha256=record.model.model_artifact_sha256,
        tokenizer_artifact_sha256=resolved.tokenizer_artifact_sha256,
        serving_config_sha256="1" * 64,
        environment_sha256="2" * 64,
        benchmark_report_sha256="3" * 64,
        recovery_report_sha256="4" * 64,
        rollback_report_sha256="5" * 64,
        security_audit_sha256="6" * 64,
        total_requests=100,
        successful_requests=100 if accepted else 99,
        unexplained_5xx=0,
        oom_events=0,
        hung_processes=0,
        gateway_throughput_ratio_basis_points=9500,
        p95_latency_overhead_median_basis_points=500,
        readiness_failure_milliseconds=1000,
        backend_recovery_milliseconds=10_000,
        rollback_recovery_milliseconds=10_000,
        api_contract_passed=accepted,
        auth_contract_passed=accepted,
        streaming_contract_passed=accepted,
        cancellation_contract_passed=accepted,
        error_mapping_contract_passed=accepted,
        thinking_content_hidden=accepted,
        backend_guard_passed=accepted,
        m6_quality_complete=accepted,
        lineage_complete=accepted,
        unmitigated_critical_vulnerabilities=0,
        unmitigated_high_vulnerabilities=0,
        checks=checks,
        status="accepted" if accepted else "rejected",
        production_eligible=accepted,
    )


def test_resolver_verifies_candidate_and_rejects_hash_drift(tmp_path: Path) -> None:
    root, _ = _artifact_root(tmp_path)
    resolved = resolve_model(root, CANDIDATE, now=NOW)

    assert resolved.status == "Candidate"
    assert resolved.candidate_record_sha256 == _sha256_file(
        root / "registry" / "candidates" / CANDIDATE / "model.json"
    )
    assert resolved.model_dir.is_absolute()

    (resolved.model_dir / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(DeploymentError) as failure:
        resolve_model(root, CANDIDATE)
    assert failure.value.code == DeploymentErrorCode.HASH_MISMATCH


def test_promote_and_rollback_publish_immutable_records(tmp_path: Path) -> None:
    root, record = _artifact_root(tmp_path)
    first_gate = _gate(root, record)
    first_path = root / "gate-1.json"
    _write_json(first_path, first_gate.to_dict())

    first = promote_production(root, first_path, now=NOW)
    assert resolve_model(root, "production", now=NOW).model_version == first.production_version
    assert (root / "registry" / "candidates" / CANDIDATE / "model.json").read_text(
        encoding="utf-8"
    ) == json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    second_gate = first_gate.model_copy(
        update={"gate_id": "m7-production-gate-fedcba98", "evaluated_at": NOW + timedelta(days=1)}
    )
    second_path = root / "gate-2.json"
    _write_json(second_path, second_gate.to_dict())
    second = promote_production(root, second_path, now=NOW + timedelta(days=1))
    assert second.production_version != first.production_version

    alias = rollback_production(root, now=NOW + timedelta(days=2))
    assert alias.production_version == first.production_version
    assert resolve_model(root, "production", now=NOW).model_version == first.production_version


def test_rejected_gate_and_cli_missing_production_use_frozen_exits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, record = _artifact_root(tmp_path)
    gate_path = root / "rejected.json"
    _write_json(gate_path, _gate(root, record, accepted=False).to_dict())

    assert (
        main(
            [
                "deploy",
                "promote",
                "--gate",
                str(gate_path),
                "--artifact-root",
                str(root),
                "--json",
            ]
        )
        == 6
    )
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "DEPLOYMENT_GATE_REJECTED"

    assert (
        main(
            [
                "deploy",
                "show",
                "--artifact-root",
                str(root),
                "--json",
            ]
        )
        == 7
    )
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "DEPLOYMENT_NOT_FOUND"


def test_gate_cannot_claim_pass_when_frozen_metrics_fail(tmp_path: Path) -> None:
    root, record = _artifact_root(tmp_path)
    payload = _gate(root, record).model_dump()
    payload["successful_requests"] = 99

    with pytest.raises(ValidationError, match="frozen thresholds"):
        M7ProductionGate.model_validate(payload)


def test_resolver_rejects_invalid_roots_records_and_refs(tmp_path: Path) -> None:
    root, _ = _artifact_root(tmp_path)
    with pytest.raises(DeploymentError) as relative:
        resolve_model(Path("relative"), CANDIDATE)
    assert relative.value.code == DeploymentErrorCode.INVALID_INPUT

    with pytest.raises(DeploymentError) as invalid_ref:
        resolve_model(root, "../../model")
    assert invalid_ref.value.code == DeploymentErrorCode.INVALID_INPUT

    with pytest.raises(DeploymentError) as invalid_candidate:
        resolve_model(root, "qwen3-0-6b-m6-nothex")
    assert invalid_candidate.value.code == DeploymentErrorCode.INVALID_INPUT

    candidate_path = root / "registry" / "candidates" / CANDIDATE / "model.json"
    candidate_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(DeploymentError) as corrupt:
        resolve_model(root, CANDIDATE)
    assert corrupt.value.code == DeploymentErrorCode.INVALID_INPUT


def test_resolver_rejects_unsafe_artifact_shapes(tmp_path: Path) -> None:
    root, _ = _artifact_root(tmp_path)
    candidate_path = root / "registry" / "candidates" / CANDIDATE / "model.json"
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload["model"]["repository"] = "unsafe/repo/name"
    _write_json(candidate_path, payload)
    with pytest.raises(DeploymentError) as unsafe_repo:
        resolve_model(root, CANDIDATE)
    assert unsafe_repo.value.code == DeploymentErrorCode.INVALID_INPUT

    root, _ = _artifact_root(tmp_path / "missing-tokenizer")
    tokenizer = root / "cache" / "models" / "Qwen" / "Qwen3-0.6B" / REVISION
    (tokenizer / "tokenizer.json").unlink()
    with pytest.raises(DeploymentError) as missing:
        resolve_model(root, CANDIDATE)
    assert missing.value.code == DeploymentErrorCode.NOT_FOUND

    root, _ = _artifact_root(tmp_path / "bad-model-set")
    model_dir = next((root / "runs").glob("*/*/exports/model"))
    (model_dir / "nested").mkdir()
    with pytest.raises(DeploymentError) as irregular:
        resolve_model(root, CANDIDATE)
    assert irregular.value.code == DeploymentErrorCode.UNSAFE_ARTIFACT


def test_production_version_resolution_validates_lineage(tmp_path: Path) -> None:
    root, record = _artifact_root(tmp_path)
    gate_path = root / "gate.json"
    _write_json(gate_path, _gate(root, record).to_dict())
    production = promote_production(root, gate_path, now=NOW)

    shown = show_deployment(root, production.production_version)
    assert shown["status"] == "Production"

    record_path = root / "registry" / "production" / production.production_version / "model.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["tokenizer_artifact_sha256"] = "0" * 64
    _write_json(record_path, payload)
    with pytest.raises(DeploymentError) as drift:
        resolve_model(root, production.production_version)
    assert drift.value.code == DeploymentErrorCode.HASH_MISMATCH


def test_gate_path_and_evidence_must_match_candidate(tmp_path: Path) -> None:
    root, record = _artifact_root(tmp_path)
    with pytest.raises(DeploymentError) as relative:
        promote_production(root, Path("gate.json"))
    assert relative.value.code == DeploymentErrorCode.INVALID_INPUT

    gate = _gate(root, record).model_copy(update={"model_artifact_sha256": "0" * 64})
    gate_path = root / "drifted-gate.json"
    _write_json(gate_path, gate.to_dict())
    with pytest.raises(DeploymentError) as drift:
        promote_production(root, gate_path)
    assert drift.value.code == DeploymentErrorCode.HASH_MISMATCH


def test_rollback_rejects_missing_or_active_target(tmp_path: Path) -> None:
    root, record = _artifact_root(tmp_path)
    gate_path = root / "gate.json"
    _write_json(gate_path, _gate(root, record).to_dict())
    production = promote_production(root, gate_path, now=NOW)

    with pytest.raises(DeploymentError, match="no previous"):
        rollback_production(root)
    with pytest.raises(DeploymentError, match="already active"):
        rollback_production(root, production.production_version)
