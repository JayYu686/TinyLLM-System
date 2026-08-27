from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
import torch
import yaml

import tinyllm.training.m10_lora as m10_lora_module
from tinyllm.deployment.evaluation_subject import ResolvedEvaluationSubject
from tinyllm.deployment.registry import DeploymentError, DeploymentErrorCode
from tinyllm.schemas import canonical_config_hash
from tinyllm.training.m10_lora import (
    M10LoRACheckpointStore,
    M10LoRAError,
    M10LoRAProgress,
    collect_m10_lora_environment,
    collect_m10_lora_hardware,
    export_m10_lora_stage,
    load_m10_lora_config,
    load_m10_lora_continuation_gate,
    load_m10_lora_memory_probe,
    preflight_m10_lora,
    require_m10_lora_storage,
)
from tinyllm.training.m10_lora_schema import (
    M10_DATASET_MANIFEST_SHA256,
    M10_DATASET_VERSION,
    M10_LORA_PARENT_MODEL_SHA256,
    M10_LORA_PARENT_RECORD_SHA256,
    M10_LORA_PARENT_SUBJECT,
    M10_LORA_PARENT_TOKENIZER_SHA256,
    M10LoRACheckpointManifest,
    M10LoRAConfig,
    M10LoRAContinuationGate,
    M10LoRAMemoryProbeResult,
    M10LoRARunResult,
)

CONFIG = Path("configs/sft/m10_agent_lora_qwen3_8b.yaml")
REPAIR_V3_CONFIG = Path("configs/sft/m10_5_agent_repair_v3_lora_qwen3_8b.yaml")


def test_m10_lora_config_freezes_parent_data_adapter_and_stages() -> None:
    config = load_m10_lora_config(CONFIG)

    assert config.model.parent_evaluation_subject == M10_LORA_PARENT_SUBJECT
    assert config.data.dataset_version == M10_DATASET_VERSION
    assert config.model.lora.rank == 16
    assert config.model.lora.target_modules == (
        "down_proj",
        "gate_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    )
    assert config.optimization.stage_tokens == (1_000_000, 5_000_000, 10_000_000)
    assert config.optimization.micro_batch_size == 1
    assert config.optimization.gradient_accumulation_steps == 8
    assert len(canonical_config_hash(config)) == 64


def test_m10_lora_config_rejects_optimizer_adapter_and_stage_drift() -> None:
    value: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["model"]["lora"]["dropout"] = 0.1
    with pytest.raises(ValueError, match="Adapter constants"):
        M10LoRAConfig.model_validate(value)

    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["optimization"]["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="optimizer constants"):
        M10LoRAConfig.model_validate(value)

    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["optimization"]["stage_tokens"] = [1_000_000, 4_000_000, 10_000_000]
    with pytest.raises(ValueError, match="5000000"):
        M10LoRAConfig.model_validate(value)

    with pytest.raises(M10LoRAError, match="config is invalid"):
        load_m10_lora_config(Path("missing-m10-lora.yaml"))


def test_m10_lora_repair_campaign_requires_v2_data_lr_and_scoring() -> None:
    value: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["run"]["name"] = "m10-5-agent-repair-lora-qwen3-8b-seed42"
    value["data"]["dataset_version"] = "m10-agent-sft-v2-12345678"
    value["data"]["manifest_sha256"] = "a" * 64
    value["optimization"]["learning_rate"] = 5e-5
    value["evaluation"]["parent_task_success_basis_points"] = 4750
    value["evaluation"]["scoring_protocol"] = "m10-agent-scoring-v2"

    config = M10LoRAConfig.model_validate(value)

    assert config.run.name.startswith("m10-5-agent-repair-")
    assert config.optimization.learning_rate == 5e-5
    assert config.to_dict()["evaluation"]["scoring_protocol"] == "m10-agent-scoring-v2"

    value["optimization"]["learning_rate"] = 2e-4
    with pytest.raises(ValueError, match="M10.5 repair"):
        M10LoRAConfig.model_validate(value)


def test_m10_lora_v3_repair_freezes_low_lr_and_migrated_scoring() -> None:
    value: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["run"]["name"] = "m10-5-agent-repair-v3-lora-qwen3-8b-seed42"
    value["data"]["dataset_version"] = "m10-agent-sft-v2-12345678"
    value["data"]["manifest_sha256"] = "a" * 64
    value["optimization"]["learning_rate"] = 1e-5
    value["evaluation"]["parent_task_success_basis_points"] = 4875
    value["evaluation"]["scoring_protocol"] = "m10-agent-scoring-v3"

    config = M10LoRAConfig.model_validate(value)

    assert config.optimization.learning_rate == 1e-5
    assert config.evaluation.scoring_protocol == "m10-agent-scoring-v3"

    value["optimization"]["learning_rate"] = 5e-5
    with pytest.raises(ValueError, match="M10.5 v3 repair"):
        M10LoRAConfig.model_validate(value)


def test_m10_lora_v3_repair_file_binds_frozen_mixture() -> None:
    config = load_m10_lora_config(REPAIR_V3_CONFIG)

    assert config.data.dataset_version == "m10-agent-sft-v2-435b9fbc"
    assert config.data.manifest_sha256 == (
        "e7b7943e3dddee9cad3403e22e26de4c65c92d063efa15b2d4c168d29afe21d2"
    )
    assert config.optimization.learning_rate == 1e-5
    assert config.evaluation.parent_task_success_basis_points == 4875


def _resolved_parent() -> ResolvedEvaluationSubject:
    return cast(
        ResolvedEvaluationSubject,
        SimpleNamespace(
            status="Evaluation",
            model_version=M10_LORA_PARENT_SUBJECT,
            evaluation_subject_sha256=M10_LORA_PARENT_RECORD_SHA256,
            model_artifact_sha256=M10_LORA_PARENT_MODEL_SHA256,
            tokenizer_artifact_sha256=M10_LORA_PARENT_TOKENIZER_SHA256,
            adapter_dir=None,
            model_dir=Path("/artifacts/model"),
            model=SimpleNamespace(
                repository="Qwen/Qwen3-8B",
                base_revision="b968826d9c46dd6066d109eabc6255188de91218",
                role="base",
                adaptation="base",
            ),
        ),
    )


def test_preflight_binds_frozen_dataset_and_qwen3_8b_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_m10_lora_config(CONFIG)
    manifest = SimpleNamespace(
        dataset_version=config.data.dataset_version,
        target_supervised_tokens=config.data.target_supervised_tokens_per_epoch,
        sequence_length=config.data.sequence_length,
    )
    parent = _resolved_parent()
    monkeypatch.setattr(m10_lora_module, "open_frozen_mixture", lambda _: manifest)
    monkeypatch.setattr(m10_lora_module, "_sha256_file", lambda _: M10_DATASET_MANIFEST_SHA256)
    monkeypatch.setattr(m10_lora_module, "resolve_evaluation_subject", lambda *_: parent)

    loaded, actual, digest = preflight_m10_lora(
        config_path=CONFIG,
        mixture_root=Path("frozen"),
        artifact_root=Path("/artifacts"),
    )
    assert loaded == config
    assert actual == parent
    assert digest == M10_DATASET_MANIFEST_SHA256

    drifted = _resolved_parent()
    drifted.model_artifact_sha256 = "f" * 64
    monkeypatch.setattr(m10_lora_module, "resolve_evaluation_subject", lambda *_: drifted)
    with pytest.raises(M10LoRAError, match="differs from the frozen"):
        preflight_m10_lora(
            config_path=CONFIG,
            mixture_root=Path("frozen"),
            artifact_root=Path("/artifacts"),
        )

    def fail(*_: object) -> ResolvedEvaluationSubject:
        raise DeploymentError(DeploymentErrorCode.NOT_FOUND, "missing")

    monkeypatch.setattr(m10_lora_module, "resolve_evaluation_subject", fail)
    with pytest.raises(M10LoRAError, match="cannot be resolved"):
        preflight_m10_lora(
            config_path=CONFIG,
            mixture_root=Path("frozen"),
            artifact_root=Path("/artifacts"),
        )


def _progress(tokens: int) -> M10LoRAProgress:
    return M10LoRAProgress(
        global_step=tokens // 1000,
        completed_epochs=tokens // 1_000_000,
        sequence_cursor=0,
        supervised_tokens=tokens,
        initial_loss=2.0,
        final_loss=1.0,
    )


def _save_checkpoint(
    store: M10LoRACheckpointStore,
    optimizer: torch.optim.Optimizer,
    config: M10LoRAConfig,
    tokens: int,
    *,
    pin_reason: Literal["stage", "final"] | None,
) -> M10LoRACheckpointManifest:
    return store.save(
        adapter_state={"adapter.weight": torch.tensor([2.0])},
        optimizer=optimizer,
        progress=_progress(tokens),
        config=config,
        config_sha256=canonical_config_hash(config),
        run_id="m10-lora-unit-run",
        git_commit="a" * 40,
        environment_sha256="b" * 64,
        hardware_sha256="c" * 64,
        memory_probe_sha256="d" * 64,
        pin_reason=pin_reason,
    )


def test_adapter_checkpoint_round_trip_retention_and_corruption(tmp_path: Path) -> None:
    config = load_m10_lora_config(CONFIG)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    store = M10LoRACheckpointStore(tmp_path / "checkpoints")
    first = _save_checkpoint(store, optimizer, config, 1_000_000, pin_reason="stage")
    payload, progress = store.load_payload(
        first,
        config=config,
        config_sha256=canonical_config_hash(config),
        git_commit="a" * 40,
        environment_sha256="b" * 64,
        hardware_sha256="c" * 64,
        memory_probe_sha256="d" * 64,
        device=torch.device("cpu"),
    )
    assert progress == _progress(1_000_000)
    assert torch.equal(payload["adapter"]["adapter.weight"], torch.tensor([2.0]))

    for tokens in (2_000_000, 3_000_000, 4_000_000):
        _save_checkpoint(store, optimizer, config, tokens, pin_reason=None)
    assert (store.root / first.checkpoint_id).is_dir()
    assert not (store.root / "checkpoint-tokens-0002000000").exists()
    assert store.latest_valid().supervised_tokens == 4_000_000

    state = store.root / "checkpoint-tokens-0004000000" / "training_state.pt"
    with state.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(M10LoRAError, match="integrity"):
        store.validate("checkpoint-tokens-0004000000")
    assert store.latest_valid().supervised_tokens == 3_000_000


def test_checkpoint_rejects_cursor_and_resume_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixed to two"):
        M10LoRACheckpointStore(tmp_path, keep_last=3)
    config = load_m10_lora_config(CONFIG)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter])
    store = M10LoRACheckpointStore(tmp_path / "checkpoints")
    with pytest.raises(M10LoRAError, match="epoch boundary"):
        store.save(
            adapter_state={"adapter.weight": torch.tensor([2.0])},
            optimizer=optimizer,
            progress=M10LoRAProgress(1, 0, 1, 10, 2.0, 1.0),
            config=config,
            config_sha256=canonical_config_hash(config),
            run_id="m10-lora-unit-run",
            git_commit="a" * 40,
            environment_sha256="b" * 64,
            hardware_sha256="c" * 64,
            memory_probe_sha256="d" * 64,
            pin_reason=None,
        )
    manifest = _save_checkpoint(store, optimizer, config, 1_000_000, pin_reason="stage")
    with pytest.raises(M10LoRAError, match="lineage changed"):
        store.load_payload(
            manifest,
            config=config,
            config_sha256=canonical_config_hash(config),
            git_commit="a" * 40,
            environment_sha256="b" * 64,
            hardware_sha256="e" * 64,
            memory_probe_sha256="d" * 64,
            device=torch.device("cpu"),
        )


class _ExportableAdapter:
    peft_config: dict[str, Any] = {}

    def save_pretrained(
        self,
        root: Path,
        *,
        safe_serialization: bool,
        save_embedding_layers: bool,
    ) -> None:
        assert safe_serialization is True
        assert save_embedding_layers is False
        (root / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (root / "adapter_model.safetensors").write_bytes(b"adapter")


def test_adapter_stage_export_is_atomic_and_hash_verified(tmp_path: Path) -> None:
    export = export_m10_lora_stage(
        _ExportableAdapter(), tmp_path / "exports", "checkpoint-tokens-0001000000"
    )
    repeated = export_m10_lora_stage(
        _ExportableAdapter(), tmp_path / "exports", "checkpoint-tokens-0001000000"
    )
    assert repeated == export
    assert export.supervised_tokens == 1_000_000
    payload = (
        tmp_path
        / "exports"
        / "checkpoint-tokens-0001000000"
        / "adapter"
        / "adapter_model.safetensors"
    )
    payload.write_bytes(b"drift")
    with pytest.raises(M10LoRAError, match="integrity"):
        export_m10_lora_stage(
            _ExportableAdapter(), tmp_path / "exports", "checkpoint-tokens-0001000000"
        )


def _gate_mapping(source_tokens: int = 1_000_000) -> dict[str, object]:
    target = 5_000_000 if source_tokens == 1_000_000 else 10_000_000
    value: dict[str, object] = {
        "evaluated_at": datetime.now(UTC),
        "decision": "accepted",
        "run_id": "m10-lora-unit-run",
        "config_sha256": "a" * 64,
        "source_stage_tokens": source_tokens,
        "target_stage_tokens": target,
        "source_checkpoint_id": f"checkpoint-tokens-{source_tokens:010d}",
        "source_adapter_artifact_sha256": "b" * 64,
        "agent_dev_version": "tinyllm-devops-agent-dev-v1-f958bcc6",
        "parent_summary_sha256": "c" * 64,
        "candidate_summary_sha256": "d" * 64,
        "parent_task_success_basis_points": 4500,
        "candidate_task_success_basis_points": 4600,
        "improvement_basis_points": 100,
    }
    if source_tokens == 5_000_000:
        value.update(
            {
                "m6_evidence_sha256": "e" * 64,
                "m6_regression_basis_points": 200,
            }
        )
    return value


def test_continuation_gate_freezes_both_stage_policies(tmp_path: Path) -> None:
    one_million = M10LoRAContinuationGate.model_validate(_gate_mapping())
    five_million = M10LoRAContinuationGate.model_validate(_gate_mapping(5_000_000))
    assert one_million.decision == five_million.decision == "accepted"

    repair = _gate_mapping()
    repair.update(
        {
            "scoring_protocol": "m10-agent-scoring-v2",
            "parent_task_success_basis_points": 4750,
            "candidate_task_success_basis_points": 4850,
        }
    )
    assert M10LoRAContinuationGate.model_validate(repair).decision == "accepted"
    repair["parent_task_success_basis_points"] = 4500
    repair["improvement_basis_points"] = 350
    with pytest.raises(ValueError, match="parent score"):
        M10LoRAContinuationGate.model_validate(repair)

    migrated = _gate_mapping()
    migrated.update(
        {
            "scoring_protocol": "m10-agent-scoring-v3",
            "parent_task_success_basis_points": 4875,
            "candidate_task_success_basis_points": 6250,
            "improvement_basis_points": 1375,
        }
    )
    assert M10LoRAContinuationGate.model_validate(migrated).decision == "accepted"

    with pytest.raises(ValueError, match="must not consume M6"):
        M10LoRAContinuationGate.model_validate(
            _gate_mapping() | {"m6_evidence_sha256": "e" * 64, "m6_regression_basis_points": 0}
        )
    with pytest.raises(ValueError, match="requires M6"):
        M10LoRAContinuationGate.model_validate(
            _gate_mapping(5_000_000)
            | {"m6_evidence_sha256": None, "m6_regression_basis_points": None}
        )
    rejected = _gate_mapping() | {
        "decision": "rejected",
        "candidate_task_success_basis_points": 4599,
        "improvement_basis_points": 99,
    }
    gate = M10LoRAContinuationGate.model_validate(rejected)
    path = tmp_path / "gate.json"
    path.write_text(gate.model_dump_json(), encoding="utf-8")
    with pytest.raises(M10LoRAError, match="was rejected"):
        load_m10_lora_continuation_gate(
            path,
            run_id="m10-lora-unit-run",
            config_sha256="a" * 64,
            source_stage_tokens=1_000_000,
            source_adapter_artifact_sha256="b" * 64,
        )

    path.write_text(one_million.model_dump_json(), encoding="utf-8")
    loaded, digest = load_m10_lora_continuation_gate(
        path,
        run_id="m10-lora-unit-run",
        config_sha256="a" * 64,
        source_stage_tokens=1_000_000,
        source_adapter_artifact_sha256="b" * 64,
    )
    assert loaded == one_million
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_memory_probe_and_run_result_require_lineage(tmp_path: Path) -> None:
    probe = M10LoRAMemoryProbeResult(
        config_sha256="a" * 64,
        git_commit="b" * 40,
        git_dirty=False,
        dataset_version=M10_DATASET_VERSION,
        parent_evaluation_subject=M10_LORA_PARENT_SUBJECT,
        environment_sha256="c" * 64,
        hardware_compatibility_sha256="d" * 64,
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        optimizer_steps=10,
        supervised_tokens=10_000,
        peak_allocated_bytes=10,
        peak_reserved_bytes=20,
        duration_seconds=1.0,
    )
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(probe.model_dump_json(), encoding="utf-8")
    loaded, digest = load_m10_lora_memory_probe(
        probe_path, config_sha256="a" * 64, git_commit="b" * 40
    )
    assert loaded == probe
    assert digest == hashlib.sha256(probe_path.read_bytes()).hexdigest()
    with pytest.raises(M10LoRAError, match="lineage differs"):
        load_m10_lora_memory_probe(probe_path, config_sha256="f" * 64, git_commit="b" * 40)

    result: dict[str, object] = {
        "status": "stage_completed",
        "mode": "fresh",
        "run_id": "m10-lora-unit-run",
        "config_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "dataset_version": M10_DATASET_VERSION,
        "dataset_manifest_sha256": M10_DATASET_MANIFEST_SHA256,
        "parent_evaluation_subject": M10_LORA_PARENT_SUBJECT,
        "parent_evaluation_subject_sha256": M10_LORA_PARENT_RECORD_SHA256,
        "parent_model_artifact_sha256": M10_LORA_PARENT_MODEL_SHA256,
        "attention_architecture": "gqa",
        "adaptation": "lora",
        "peft_version": "0.19.1",
        "seed": 42,
        "physical_gpu_index": 4,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "trainable_parameters": 10,
        "total_parameters": 100,
        "global_step": 100,
        "completed_epochs": 1,
        "supervised_tokens": 1_000_000,
        "initial_loss": 2.0,
        "final_loss": 1.0,
        "duration_seconds": 10.0,
        "peak_allocated_bytes": 10,
        "peak_reserved_bytes": 20,
        "memory_probe_sha256": "c" * 64,
        "latest_checkpoint": "checkpoint-tokens-0001000000",
        "stage_export": {
            "checkpoint_id": "checkpoint-tokens-0001000000",
            "supervised_tokens": 1_000_000,
            "adapter_artifact_sha256": "d" * 64,
            "adapter_files": ["adapter_config.json", "adapter_model.safetensors"],
        },
    }
    assert M10LoRARunResult.model_validate(result).status == "stage_completed"
    with pytest.raises(ValueError, match="accepted continuation"):
        M10LoRARunResult.model_validate(result | {"mode": "exact_resume"})


def test_environment_hardware_and_storage_preflights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    versions = {"transformers": "4.57.6", "peft": "0.19.1"}
    monkeypatch.setattr(
        "tinyllm.training.m10_lora.importlib.metadata.version",
        lambda package: versions[package],
    )
    monkeypatch.setattr(torch, "__version__", "2.7.1+cu118")
    monkeypatch.setattr(torch.version, "cuda", "11.8")
    environment, environment_sha256 = collect_m10_lora_environment()
    assert environment["peft_version"] == "0.19.1"
    assert len(environment_sha256) == 64
    versions["peft"] = "0.18.0"
    with pytest.raises(M10LoRAError, match="software identity differs"):
        collect_m10_lora_environment()

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        if command[1] == "topo":
            return SimpleNamespace(stdout="GPU0 GPU4 CPU Affinity\nGPU4 X 1 0-31\n")
        return SimpleNamespace(
            stdout=(
                "4, GPU-unit-test, NVIDIA GeForce RTX 3090, 24576, 00000000:41:00.0, 535.261.03\n"
            )
        )

    monkeypatch.setattr("tinyllm.training.m10_lora.subprocess.run", fake_run)
    hardware, compatibility_sha256 = collect_m10_lora_hardware(4)
    selected_gpu = hardware["selected_gpu"]
    assert isinstance(selected_gpu, dict)
    assert selected_gpu["physical_gpu_index"] == 4
    assert hardware["exact_resume_compatibility_sha256"] == compatibility_sha256
    with pytest.raises(M10LoRAError, match="selected physical RTX 3090"):
        collect_m10_lora_hardware(5)

    assert require_m10_lora_storage(tmp_path, minimum_free_bytes=1) > 0
    with pytest.raises(ValueError, match="must be positive"):
        require_m10_lora_storage(tmp_path, minimum_free_bytes=0)
    monkeypatch.setattr(
        "tinyllm.training.m10_lora.shutil.disk_usage",
        lambda _: SimpleNamespace(total=10, used=9, free=1),
    )
    with pytest.raises(M10LoRAError, match="storage preflight failed"):
        require_m10_lora_storage(tmp_path / "future" / "run", minimum_free_bytes=2)
