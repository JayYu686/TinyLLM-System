from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml

import tinyllm.training.m10_sft as m10_sft_module
from tinyllm.deployment.registry import DeploymentError, DeploymentErrorCode
from tinyllm.deployment.schema import ResolvedModel
from tinyllm.schemas import canonical_config_hash
from tinyllm.training.m10_sft import (
    M10CheckpointStore,
    M10FullSFTError,
    M10Progress,
    _batch,
    _export_stage,
    _json_bytes,
    _record_result,
    epoch_order,
    load_m10_continuation_gate,
    load_m10_full_sft_config,
    validate_m10_parent,
)
from tinyllm.training.m10_sft_schema import (
    M10_DATASET_MANIFEST_SHA256,
    M10_DATASET_VERSION,
    M10_PARENT_MODEL_SHA256,
    M10_PARENT_RECORD_SHA256,
    M10_PARENT_VERSION,
    M10ContinuationGate,
    M10FullSFTConfig,
    M10FullSFTRunResult,
)

CONFIG = Path("configs/sft/m10_agent_full_sft_qwen3_0_6b.yaml")


def test_m10_config_freezes_parent_data_stages_and_single_gpu() -> None:
    config = load_m10_full_sft_config(CONFIG)

    assert config.model.parent_production_version == M10_PARENT_VERSION
    assert config.data.dataset_version == M10_DATASET_VERSION
    assert config.optimization.stage_tokens == (1_000_000, 5_000_000, 10_000_000)
    assert config.parallel.world_size == 1
    assert config.optimization.micro_batch_size == 2
    assert len(canonical_config_hash(config)) == 64


def test_m10_config_rejects_stage_and_optimizer_drift() -> None:
    value: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["optimization"]["stage_tokens"] = [1_000_000, 4_000_000, 10_000_000]
    with pytest.raises(ValueError, match="5000000"):
        M10FullSFTConfig.model_validate(value)

    with pytest.raises(M10FullSFTError, match="config is invalid"):
        load_m10_full_sft_config(Path("missing-m10-config.yaml"))

    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["optimization"]["learning_rate"] = 2e-5
    with pytest.raises(ValueError, match="optimizer constants"):
        M10FullSFTConfig.model_validate(value)


def test_epoch_order_is_seeded_per_epoch_and_complete() -> None:
    first = epoch_order(20, seed=42, epoch=0)
    repeated = epoch_order(20, seed=42, epoch=0)
    second = epoch_order(20, seed=42, epoch=1)

    assert first == repeated
    assert first != second
    assert sorted(first) == list(range(20))
    with pytest.raises(ValueError, match="invalid"):
        epoch_order(0, seed=42, epoch=0)
    with pytest.raises(ValueError, match="invalid"):
        epoch_order(2, seed=42, epoch=-1)


def test_parent_validation_requires_the_frozen_m7_identity() -> None:
    config = load_m10_full_sft_config(CONFIG)
    resolved = cast(
        ResolvedModel,
        SimpleNamespace(
            status="Production",
            model_version=M10_PARENT_VERSION,
            production_record_sha256=M10_PARENT_RECORD_SHA256,
            model_artifact_sha256=M10_PARENT_MODEL_SHA256,
            model=SimpleNamespace(
                repository="Qwen/Qwen3-0.6B",
                base_revision="c1899de289a04d12100db370d81485cdf75e47ca",
            ),
        ),
    )

    validate_m10_parent(config, resolved)
    resolved.model_version = "qwen3-0-6b-m7-00000000"
    with pytest.raises(M10FullSFTError, match="parent differs"):
        validate_m10_parent(config, resolved)


def _progress(tokens: int) -> M10Progress:
    return M10Progress(
        global_step=tokens // 1000,
        completed_epochs=tokens // 1_000_000,
        sequence_cursor=0,
        supervised_tokens=tokens,
        initial_loss=2.0,
        final_loss=1.0,
    )


def test_checkpoint_round_trip_retention_and_corruption(tmp_path: Path) -> None:
    config = load_m10_full_sft_config(CONFIG)
    config_sha256 = canonical_config_hash(config)
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    store = M10CheckpointStore(tmp_path / "checkpoints")
    first = store.save(
        model=model,
        optimizer=optimizer,
        progress=_progress(1_000_000),
        config=config,
        config_sha256=config_sha256,
        run_id="m10-unit-run",
        git_commit="a" * 40,
        pin_reason="stage",
    )
    assert (
        store.save(
            model=model,
            optimizer=optimizer,
            progress=_progress(1_000_000),
            config=config,
            config_sha256=config_sha256,
            run_id="m10-unit-run",
            git_commit="a" * 40,
            pin_reason="stage",
        )
        == first
    )
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    loaded = store.load(
        first,
        model=model,
        optimizer=optimizer,
        config=config,
        config_sha256=config_sha256,
        git_commit="a" * 40,
        device=torch.device("cpu"),
    )
    assert loaded == _progress(1_000_000)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in original.items())

    for tokens in (2_000_000, 3_000_000, 4_000_000):
        store.save(
            model=model,
            optimizer=optimizer,
            progress=_progress(tokens),
            config=config,
            config_sha256=config_sha256,
            run_id="m10-unit-run",
            git_commit="a" * 40,
            pin_reason=None,
        )
    assert (store.root / first.checkpoint_id).is_dir()
    assert not (store.root / "checkpoint-tokens-0002000000").exists()
    assert store.latest_valid().supervised_tokens == 4_000_000

    payload = store.root / "checkpoint-tokens-0004000000" / "training_state.pt"
    with payload.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(M10FullSFTError, match="integrity"):
        store.validate("checkpoint-tokens-0004000000")
    assert store.latest_valid().supervised_tokens == 3_000_000


def test_checkpoint_store_fails_closed_on_invalid_policy_and_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixed to two"):
        M10CheckpointStore(tmp_path, keep_last=3)
    config = load_m10_full_sft_config(CONFIG)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    store = M10CheckpointStore(tmp_path / "checkpoints")
    with pytest.raises(M10FullSFTError, match="epoch boundaries"):
        store.save(
            model=model,
            optimizer=optimizer,
            progress=M10Progress(1, 0, 1, 10, 1.0, 1.0),
            config=config,
            config_sha256=canonical_config_hash(config),
            run_id="m10-unit-run",
            git_commit="a" * 40,
            pin_reason=None,
        )
    with pytest.raises(M10FullSFTError, match="no valid"):
        store.latest_valid()
    with pytest.raises(M10FullSFTError, match="metadata"):
        store.validate("checkpoint-tokens-0001000000")


def test_batch_export_and_result_artifacts_are_deterministic(tmp_path: Path) -> None:
    dataset = cast(
        Any,
        [
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([-100, 2]),
                "attention_mask": torch.tensor([1, 1]),
            },
            {
                "input_ids": torch.tensor([3, 4]),
                "labels": torch.tensor([-100, 4]),
                "attention_mask": torch.tensor([1, 1]),
            },
        ],
    )
    assert _batch(dataset, (0, 1))["input_ids"].tolist() == [[1, 2], [3, 4]]
    with pytest.raises(M10FullSFTError, match="cannot be empty"):
        _batch(dataset, ())

    class ExportableModel(torch.nn.Module):
        def save_pretrained(self, root: Path, *, safe_serialization: bool) -> None:
            assert safe_serialization is True
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")

    export = _export_stage(ExportableModel(), tmp_path / "exports", "checkpoint-tokens-0001000000")
    repeated = _export_stage(
        ExportableModel(), tmp_path / "exports", "checkpoint-tokens-0001000000"
    )
    assert export == repeated
    with pytest.raises(M10FullSFTError, match="save_pretrained"):
        _export_stage(torch.nn.Linear(1, 1), tmp_path / "bad", "checkpoint-tokens-0001000000")
    incomplete = tmp_path / "incomplete" / "checkpoint-tokens-0001000000"
    incomplete.mkdir(parents=True)
    with pytest.raises(M10FullSFTError, match="incomplete or corrupt"):
        _export_stage(ExportableModel(), tmp_path / "incomplete", incomplete.name)

    result = M10FullSFTRunResult.model_validate(_result_mapping())
    _record_result(tmp_path / "run", result)
    assert (tmp_path / "run" / "result.json").is_file()
    events = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8")
    assert '"event": "m10_stage_completed"' in events


def test_preflight_binds_dataset_and_production_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_m10_full_sft_config(CONFIG)
    manifest = SimpleNamespace(
        dataset_version=config.data.dataset_version,
        target_supervised_tokens=config.data.target_supervised_tokens_per_epoch,
        sequence_length=config.data.sequence_length,
    )
    resolved = cast(
        ResolvedModel,
        SimpleNamespace(
            status="Production",
            model_version=M10_PARENT_VERSION,
            production_record_sha256=M10_PARENT_RECORD_SHA256,
            model_artifact_sha256=M10_PARENT_MODEL_SHA256,
            model=SimpleNamespace(
                repository="Qwen/Qwen3-0.6B",
                base_revision="c1899de289a04d12100db370d81485cdf75e47ca",
            ),
        ),
    )
    monkeypatch.setattr(m10_sft_module, "open_frozen_mixture", lambda _: manifest)
    monkeypatch.setattr(m10_sft_module, "_sha256_file", lambda _: M10_DATASET_MANIFEST_SHA256)
    monkeypatch.setattr(m10_sft_module, "resolve_model", lambda *_: resolved)
    loaded, actual, digest = m10_sft_module.preflight_m10_full_sft(
        config_path=CONFIG,
        mixture_root=Path("frozen"),
        artifact_root=Path("artifacts"),
    )
    assert loaded == config and actual == resolved and digest == M10_DATASET_MANIFEST_SHA256

    monkeypatch.setattr(m10_sft_module, "_sha256_file", lambda _: "f" * 64)
    with pytest.raises(M10FullSFTError, match="mixture identity"):
        m10_sft_module.preflight_m10_full_sft(
            config_path=CONFIG,
            mixture_root=Path("frozen"),
            artifact_root=Path("artifacts"),
        )
    monkeypatch.setattr(m10_sft_module, "_sha256_file", lambda _: M10_DATASET_MANIFEST_SHA256)

    def fail_resolution(*_: object) -> ResolvedModel:
        raise DeploymentError(DeploymentErrorCode.NOT_FOUND, "missing")

    monkeypatch.setattr(m10_sft_module, "resolve_model", fail_resolution)
    with pytest.raises(M10FullSFTError, match="cannot be resolved"):
        m10_sft_module.preflight_m10_full_sft(
            config_path=CONFIG,
            mixture_root=Path("frozen"),
            artifact_root=Path("artifacts"),
        )


def _result_mapping() -> dict[str, object]:
    return {
        "status": "stage_completed",
        "mode": "fresh",
        "run_id": "m10-unit-run",
        "config_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "dataset_version": M10_DATASET_VERSION,
        "dataset_manifest_sha256": M10_DATASET_MANIFEST_SHA256,
        "parent_production_version": M10_PARENT_VERSION,
        "parent_production_record_sha256": M10_PARENT_RECORD_SHA256,
        "parent_model_artifact_sha256": M10_PARENT_MODEL_SHA256,
        "attention_architecture": "gqa",
        "seed": 42,
        "physical_gpu_index": 7,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "global_step": 100,
        "completed_epochs": 1,
        "supervised_tokens": 1_000_000,
        "initial_loss": 2.0,
        "final_loss": 1.0,
        "duration_seconds": 10.0,
        "peak_allocated_bytes": 10,
        "peak_reserved_bytes": 20,
        "latest_checkpoint": "checkpoint-tokens-0001000000",
        "resumed_from_tokens": None,
        "continuation_gate_sha256": None,
        "stage_export": {
            "checkpoint_id": "checkpoint-tokens-0001000000",
            "supervised_tokens": 1_000_000,
            "export_sha256": "c" * 64,
        },
    }


def test_result_requires_stage_status_resume_and_export_alignment() -> None:
    assert M10FullSFTRunResult.model_validate(_result_mapping()).status == "stage_completed"

    with pytest.raises(ValueError, match="status differs"):
        M10FullSFTRunResult.model_validate(_result_mapping() | {"status": "succeeded"})
    resumed = _result_mapping()
    resumed.update({"mode": "exact_resume", "resumed_from_tokens": 1_000_000})
    assert M10FullSFTRunResult.model_validate(resumed).mode == "exact_resume"
    with pytest.raises(ValueError, match="Checkpoint and export differ"):
        M10FullSFTRunResult.model_validate(
            _result_mapping()
            | {
                "stage_export": {
                    "checkpoint_id": "checkpoint-tokens-0005000000",
                    "supervised_tokens": 5_000_000,
                    "export_sha256": "c" * 64,
                }
            }
        )

    final = _result_mapping() | {
        "status": "succeeded",
        "mode": "exact_resume",
        "completed_epochs": 10,
        "supervised_tokens": 10_000_000,
        "latest_checkpoint": "checkpoint-tokens-0010000000",
        "resumed_from_tokens": 5_000_000,
        "continuation_gate_sha256": "d" * 64,
        "stage_export": {
            "checkpoint_id": "checkpoint-tokens-0010000000",
            "supervised_tokens": 10_000_000,
            "export_sha256": "c" * 64,
        },
    }
    assert M10FullSFTRunResult.model_validate(final).status == "succeeded"
    with pytest.raises(ValueError, match="continuation gate"):
        M10FullSFTRunResult.model_validate(final | {"continuation_gate_sha256": None})


def _continuation_gate_mapping() -> dict[str, object]:
    return {
        "evaluated_at": datetime.now(UTC),
        "decision": "accepted",
        "run_id": "m10-unit-run",
        "config_sha256": "a" * 64,
        "source_checkpoint_id": "checkpoint-tokens-0005000000",
        "source_stage_export_sha256": "b" * 64,
        "agent_dev_version": "tinyllm-devops-agent-dev-v1-f958bcc6",
        "parent_agent_dev_summary_sha256": "c" * 64,
        "candidate_agent_dev_summary_sha256": "d" * 64,
        "parent_task_success_basis_points": 5000,
        "candidate_task_success_basis_points": 5100,
        "agent_dev_improvement_basis_points": 100,
        "m6_evidence_sha256": "e" * 64,
        "m6_regression_basis_points": 200,
    }


def test_continuation_gate_freezes_thresholds_and_lineage(tmp_path: Path) -> None:
    accepted = M10ContinuationGate.model_validate(_continuation_gate_mapping())
    assert accepted.decision == "accepted"
    with pytest.raises(ValueError, match="improvement differs"):
        M10ContinuationGate.model_validate(
            _continuation_gate_mapping() | {"agent_dev_improvement_basis_points": 101}
        )
    with pytest.raises(ValueError, match="decision differs"):
        M10ContinuationGate.model_validate(
            _continuation_gate_mapping()
            | {
                "decision": "rejected",
                "candidate_task_success_basis_points": 5100,
            }
        )
    rejected = M10ContinuationGate.model_validate(
        _continuation_gate_mapping()
        | {
            "decision": "rejected",
            "candidate_task_success_basis_points": 5099,
            "agent_dev_improvement_basis_points": 99,
        }
    )
    path = tmp_path / "gate.json"
    path.write_text(rejected.model_dump_json(), encoding="utf-8")
    with pytest.raises(M10FullSFTError, match="was rejected"):
        load_m10_continuation_gate(
            path,
            run_id="m10-unit-run",
            config_sha256="a" * 64,
            source_stage_export_sha256="b" * 64,
        )

    payload = accepted.model_dump_json().encode()
    path.write_bytes(payload)
    loaded, digest = load_m10_continuation_gate(
        path,
        run_id="m10-unit-run",
        config_sha256="a" * 64,
        source_stage_export_sha256="b" * 64,
    )
    assert loaded == accepted
    assert digest == hashlib.sha256(_json_bytes(accepted.to_dict())).hexdigest()
    with pytest.raises(M10FullSFTError, match="lineage differs"):
        load_m10_continuation_gate(
            path,
            run_id="wrong-run",
            config_sha256="a" * 64,
            source_stage_export_sha256="b" * 64,
        )
    path.write_text(json.dumps({"decision": "accepted"}), encoding="utf-8")
    with pytest.raises(M10FullSFTError, match="missing or invalid"):
        load_m10_continuation_gate(
            path,
            run_id="m10-unit-run",
            config_sha256="a" * 64,
            source_stage_export_sha256="b" * 64,
        )
