from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest
import torch

from tinyllm.schemas import CheckpointManifest, canonical_config_hash, generate_run_id
from tinyllm.training import (
    CheckpointContext,
    CheckpointError,
    CheckpointErrorCode,
    CheckpointStore,
    SingleDeviceTrainer,
    build_m1_cpu_trainer,
)
from tinyllm.training.config import M1TrainingConfig, training_config_from_mapping


def checkpoint_config() -> M1TrainingConfig:
    return training_config_from_mapping(
        {
            "schema_version": "1.0",
            "run": {"name": "checkpoint-unit", "seed": 19},
            "model": {
                "vocab_size": 16,
                "hidden_size": 16,
                "num_layers": 1,
                "num_heads": 2,
                "intermediate_size": 32,
                "max_sequence_length": 8,
                "rope_theta": 10_000.0,
                "rms_norm_epsilon": 1.0e-6,
                "dropout": 0.0,
                "tie_word_embeddings": True,
            },
            "data": {
                "kind": "toy",
                "vocab_size": 16,
                "sequence_length": 8,
                "num_samples": 8,
            },
            "training": {
                "max_steps": 4,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "warmup_steps": 1,
            },
            "precision": {
                "dtype": "fp32",
                "allow_tf32": False,
                "use_grad_scaler": False,
            },
            "checkpoint": {
                "output_dir": "runs/checkpoint-unit",
                "save_steps": 1,
                "keep_last": 2,
                "resume": "none",
            },
        }
    )


def checkpoint_context(config: M1TrainingConfig) -> CheckpointContext:
    config_hash = canonical_config_hash(config)
    return CheckpointContext(
        run_id=generate_run_id(
            config.run.name,
            config_hash,
            now=datetime(2026, 7, 14, tzinfo=UTC),
            nonce="cafe",
        ),
        dataset_version="toy-checkpoint-v1",
        git_commit="a" * 40,
        environment={"python": "3.11", "torch": str(torch.__version__), "device": "cpu"},
    )


def save_current_checkpoint(
    store: CheckpointStore,
    trainer: SingleDeviceTrainer,
    config: M1TrainingConfig,
    *,
    pin_reason: Literal["interruption", "best", "final"] | None = None,
) -> CheckpointManifest:
    assert trainer.sampler is not None
    return store.save(
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=None,
        sampler=trainer.sampler,
        trainer_state=trainer.state,
        config=config,
        context=checkpoint_context(config),
        pin_reason=pin_reason,
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
    )


def resign_changed_file(checkpoint_dir: Path, filename: str) -> None:
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_path = checkpoint_dir / filename
    changed_bytes = changed_path.read_bytes()
    for entry in manifest["files"]:
        if entry["path"] == filename:
            entry["size_bytes"] = len(changed_bytes)
            entry["sha256"] = hashlib.sha256(changed_bytes).hexdigest()
            break
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    marker = {
        "checkpoint_id": manifest["checkpoint_id"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "schema_version": "1.0",
    }
    (checkpoint_dir / "COMMITTED").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_checkpoint_save_validates_full_state_and_latest(tmp_path: Path) -> None:
    config = checkpoint_config()
    trainer = build_m1_cpu_trainer(config)
    trainer.train(target_global_step=1)
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)

    manifest = save_current_checkpoint(store, trainer, config)

    assert manifest.checkpoint_id == "checkpoint-step-00000001"
    assert manifest.state.supports_exact_resume is True
    assert store.latest() == manifest.checkpoint_id
    assert {entry.path for entry in manifest.files} == {
        "training_state.pt",
        "config.resolved.json",
        "environment.json",
    }
    payload = store.load_training_state(manifest.checkpoint_id)
    assert payload["trainer_state"] == trainer.state.to_dict()
    assert trainer.sampler is not None
    assert payload["sampler"] == trainer.sampler.state_dict()
    assert payload["grad_scaler"] == {"not_applicable": True}
    assert not list(store.root.glob(".*.tmp-*"))


def test_checkpoint_corruption_and_incomplete_marker_fail_closed(tmp_path: Path) -> None:
    config = checkpoint_config()
    trainer = build_m1_cpu_trainer(config)
    trainer.train(target_global_step=1)
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)
    manifest = save_current_checkpoint(store, trainer, config)
    checkpoint_dir = store.root / manifest.checkpoint_id

    with (checkpoint_dir / "training_state.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(CheckpointError) as caught:
        store.validate(manifest.checkpoint_id)
    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_CORRUPT

    (checkpoint_dir / "COMMITTED").unlink()
    with pytest.raises(CheckpointError) as caught:
        store.validate(manifest.checkpoint_id)
    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_INCOMPLETE


def test_semantic_config_and_payload_drift_fail_after_hashes_are_resigned(
    tmp_path: Path,
) -> None:
    config = checkpoint_config()
    trainer = build_m1_cpu_trainer(config)
    trainer.train(target_global_step=1)
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)
    manifest = save_current_checkpoint(store, trainer, config)
    checkpoint_dir = store.root / manifest.checkpoint_id

    config_path = checkpoint_dir / "config.resolved.json"
    resolved_config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved_config["training"]["learning_rate"] = 0.02
    config_path.write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resign_changed_file(checkpoint_dir, "config.resolved.json")
    with pytest.raises(CheckpointError) as caught:
        store.validate(manifest.checkpoint_id)
    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_CORRUPT

    config_path.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resign_changed_file(checkpoint_dir, "config.resolved.json")
    state_path = checkpoint_dir / "training_state.pt"
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    payload["dataset_version"] = "different-data-version"
    torch.save(payload, state_path)
    resign_changed_file(checkpoint_dir, "training_state.pt")
    assert store.validate(manifest.checkpoint_id).checkpoint_id == manifest.checkpoint_id
    with pytest.raises(CheckpointError) as caught:
        store.load_training_state(manifest.checkpoint_id)
    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_CORRUPT


def test_write_failure_cleans_temporary_state_and_does_not_publish_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = checkpoint_config()
    trainer = build_m1_cpu_trainer(config)
    trainer.train(target_global_step=1)
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)

    def fail_save(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("simulated disk failure")

    monkeypatch.setattr("tinyllm.training.checkpoint.torch.save", fail_save)
    with pytest.raises(CheckpointError) as caught:
        save_current_checkpoint(store, trainer, config)

    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_WRITE_FAILED
    assert list(store.root.iterdir()) == []


def test_retention_preserves_pins_and_two_latest_ordinary_points(tmp_path: Path) -> None:
    config = checkpoint_config()
    trainer = build_m1_cpu_trainer(config)
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)

    for step in range(1, 5):
        trainer.train(target_global_step=step)
        pin_reason: Literal["interruption", "best", "final"] | None = (
            "interruption" if step == 2 else None
        )
        save_current_checkpoint(store, trainer, config, pin_reason=pin_reason)

    checkpoint_ids = sorted(path.name for path in store.root.glob("checkpoint-step-*"))
    assert checkpoint_ids == [
        "checkpoint-step-00000002",
        "checkpoint-step-00000003",
        "checkpoint-step-00000004",
    ]
    assert store.validate("checkpoint-step-00000002").pin_reason == "interruption"
    assert store.latest() == "checkpoint-step-00000004"


def test_checkpoint_identifier_cannot_escape_store_root(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints", keep_last=2)

    with pytest.raises(CheckpointError) as caught:
        store.validate("../../outside")

    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_NOT_FOUND
