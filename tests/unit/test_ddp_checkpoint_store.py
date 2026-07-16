from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
import torch

from tinyllm.data import StatefulDistributedSampler, ToyTokenDataset
from tinyllm.schemas import (
    CheckpointFile,
    CheckpointManifest,
    CheckpointStateCoverage,
    canonical_config_hash,
    generate_run_id,
)
from tinyllm.training import (
    CheckpointContext,
    CheckpointError,
    CheckpointErrorCode,
    DDPCheckpointStore,
    build_m1_cpu_trainer,
    build_rank_state,
)
from tinyllm.training.config import M1TrainingConfig, training_config_from_mapping


def ddp_checkpoint_config() -> M1TrainingConfig:
    return training_config_from_mapping(
        {
            "schema_version": "1.0",
            "run": {"name": "ddp-checkpoint-unit", "seed": 23},
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
                "num_samples": 16,
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
                "output_dir": "runs/ddp-checkpoint-unit",
                "save_steps": 1,
                "keep_last": 2,
                "resume": "auto",
            },
            "distributed": {
                "strategy": "ddp",
                "backend": "gloo",
                "world_size": 2,
                "timeout_seconds": 30,
                "broadcast_buffers": False,
                "find_unused_parameters": False,
            },
        }
    )


def checkpoint_context(config: M1TrainingConfig) -> CheckpointContext:
    config_hash = canonical_config_hash(config)
    return CheckpointContext(
        run_id=generate_run_id(
            config.run.name,
            config_hash,
            now=datetime(2026, 7, 15, tzinfo=UTC),
            nonce="cafe",
        ),
        dataset_version="toy-ddp-checkpoint-v1",
        git_commit="a" * 40,
        environment={"python": "3.11", "torch": str(torch.__version__), "device": "cpu"},
        strategy="ddp",
        world_size=2,
    )


def _rank_states(config: M1TrainingConfig, *, step: int) -> list[dict[str, object]]:
    dataset = ToyTokenDataset(
        vocab_size=config.data.vocab_size,
        sequence_length=config.data.sequence_length,
        num_samples=config.data.num_samples,
        seed=config.run.seed,
    )
    states: list[dict[str, object]] = []
    consumed = step * config.training.micro_batch_size
    trainer = build_m1_cpu_trainer(config)
    trainer.train(target_global_step=step)
    for rank in range(config.distributed.world_size):
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=config.distributed.world_size,
            rank=rank,
            seed=config.run.seed,
        )
        iterator = iter(sampler)
        for _ in range(consumed):
            next(iterator)
        states.append(
            build_rank_state(
                rank=rank,
                world_size=config.distributed.world_size,
                trainer_state=trainer.state,
                sampler_state=sampler.state_dict(),
                device=torch.device("cpu"),
            )
        )
    return states


def _save(store: DDPCheckpointStore, config: M1TrainingConfig, *, step: int = 1) -> str:
    trainer = build_m1_cpu_trainer(config)
    trainer.train(target_global_step=step)
    manifest = store.save(
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        trainer_state=trainer.state,
        config=config,
        context=checkpoint_context(config),
        rank_states=_rank_states(config, step=step),
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )
    return manifest.checkpoint_id


def _resign(checkpoint_dir: Path, filename: str) -> None:
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = (checkpoint_dir / filename).read_bytes()
    for entry in manifest["files"]:
        if entry["path"] == filename:
            entry["size_bytes"] = len(changed)
            entry["sha256"] = hashlib.sha256(changed).hexdigest()
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


def test_ddp_checkpoint_round_trip_contains_every_rank(tmp_path: Path) -> None:
    config = ddp_checkpoint_config()
    store = DDPCheckpointStore(tmp_path / "checkpoints", keep_last=2)

    checkpoint_id = _save(store, config)
    loaded = store.load(
        checkpoint_id,
        rank=1,
        expected_world_size=2,
        map_location="cpu",
    )

    assert loaded.manifest.state.supports_exact_resume is True
    assert loaded.rank["rank"] == 1
    assert loaded.rank["sampler"]["cursor"] == 2
    assert {entry.path for entry in loaded.manifest.files} == {
        "training_state.pt",
        "rank-00000.pt",
        "rank-00001.pt",
        "config.resolved.json",
        "environment.json",
    }
    assert (store.root / "LATEST").read_text(encoding="utf-8").strip() == checkpoint_id


def test_ddp_checkpoint_rejects_wrong_world_size(tmp_path: Path) -> None:
    config = ddp_checkpoint_config()
    store = DDPCheckpointStore(tmp_path / "checkpoints", keep_last=2)
    checkpoint_id = _save(store, config)

    with pytest.raises(CheckpointError) as caught:
        store.validate(checkpoint_id, expected_world_size=1)

    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE


@pytest.mark.parametrize("mutation", ["missing_rank", "corrupt_rank"])
def test_ddp_checkpoint_rejects_partial_or_corrupt_rank_state(
    tmp_path: Path,
    mutation: str,
) -> None:
    config = ddp_checkpoint_config()
    store = DDPCheckpointStore(tmp_path / "checkpoints", keep_last=2)
    checkpoint_id = _save(store, config)
    rank_path = store.root / checkpoint_id / "rank-00001.pt"
    if mutation == "missing_rank":
        rank_path.unlink()
    else:
        with rank_path.open("ab") as stream:
            stream.write(b"corrupt")

    with pytest.raises(CheckpointError) as caught:
        store.validate(checkpoint_id, expected_world_size=2)

    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_CORRUPT


def test_ddp_checkpoint_environment_payload_must_match_snapshot(tmp_path: Path) -> None:
    config = ddp_checkpoint_config()
    store = DDPCheckpointStore(tmp_path / "checkpoints", keep_last=2)
    checkpoint_id = _save(store, config)
    checkpoint_dir = store.root / checkpoint_id
    state_path = checkpoint_dir / "training_state.pt"
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    payload["environment"] = {"python": "different"}
    torch.save(payload, state_path)
    _resign(checkpoint_dir, "training_state.pt")

    with pytest.raises(CheckpointError) as caught:
        store.validate(checkpoint_id, expected_world_size=2)

    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_CORRUPT


def test_ddp_manifest_requires_contiguous_rank_state_files() -> None:
    config = ddp_checkpoint_config()

    with pytest.raises(ValueError, match="contiguous state file per Rank"):
        CheckpointManifest(
            checkpoint_id="checkpoint-step-00000001",
            run_id=checkpoint_context(config).run_id,
            created_at=datetime(2026, 7, 15, tzinfo=UTC),
            strategy="ddp",
            resume_capability="exact",
            world_size=2,
            global_step=1,
            micro_step=1,
            epoch=0,
            config_hash=canonical_config_hash(config),
            dataset_version="toy-ddp-checkpoint-v1",
            git_commit="a" * 40,
            state=CheckpointStateCoverage(
                model=True,
                optimizer=True,
                scheduler=True,
                grad_scaler=True,
                python_rng=True,
                numpy_rng=True,
                torch_rng=True,
                cuda_rng=True,
                sampler=True,
                config_snapshot=True,
                environment=True,
            ),
            files=(
                CheckpointFile(
                    path="training_state.pt",
                    role="training_state",
                    size_bytes=1,
                    sha256="a" * 64,
                ),
            ),
        )


def test_ddp_checkpoint_rejects_invalid_context_and_incomplete_rank_set(
    tmp_path: Path,
) -> None:
    config = ddp_checkpoint_config()
    trainer = build_m1_cpu_trainer(config)
    trainer.train(target_global_step=1)
    store = DDPCheckpointStore(tmp_path / "checkpoints", keep_last=2)
    rank_states = _rank_states(config, step=1)
    valid_context = checkpoint_context(config)
    invalid_context = CheckpointContext(
        run_id=valid_context.run_id,
        dataset_version=valid_context.dataset_version,
        git_commit=valid_context.git_commit,
        environment=valid_context.environment,
        strategy="single",
        world_size=1,
    )

    with pytest.raises(ValueError, match="strategy=ddp"):
        store.save(
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            trainer_state=trainer.state,
            config=config,
            context=invalid_context,
            rank_states=rank_states,
        )
    with pytest.raises(CheckpointError) as caught:
        store.save(
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            trainer_state=trainer.state,
            config=config,
            context=checkpoint_context(config),
            rank_states=rank_states[:1],
        )
    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_INCOMPLETE


def test_ddp_checkpoint_rejects_rank_progress_disagreement(tmp_path: Path) -> None:
    config = ddp_checkpoint_config()
    trainer = build_m1_cpu_trainer(config)
    trainer.train(target_global_step=1)
    rank_states = _rank_states(config, step=1)
    saved_progress = cast(dict[str, object], rank_states[1]["trainer_state"])
    rank_states[1]["trainer_state"] = {
        **saved_progress,
        "global_step": 0,
    }
    store = DDPCheckpointStore(tmp_path / "checkpoints", keep_last=2)

    with pytest.raises(CheckpointError) as caught:
        store.save(
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            trainer_state=trainer.state,
            config=config,
            context=checkpoint_context(config),
            rank_states=rank_states,
        )

    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE


def test_ddp_checkpoint_rejects_semantically_invalid_resigned_payloads(tmp_path: Path) -> None:
    config = ddp_checkpoint_config()
    store = DDPCheckpointStore(tmp_path / "checkpoints", keep_last=2)
    checkpoint_id = _save(store, config)
    checkpoint_dir = store.root / checkpoint_id
    state_path = checkpoint_dir / "training_state.pt"
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    payload["model"] = {}
    torch.save(payload, state_path)
    _resign(checkpoint_dir, "training_state.pt")

    with pytest.raises(CheckpointError) as caught:
        store.validate(checkpoint_id)
    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_CORRUPT

    payload["model"] = build_m1_cpu_trainer(config).model.state_dict()
    torch.save(payload, state_path)
    _resign(checkpoint_dir, "training_state.pt")
    (checkpoint_dir / "environment.json").write_text("not-json\n", encoding="utf-8")
    _resign(checkpoint_dir, "environment.json")
    with pytest.raises(CheckpointError) as caught:
        store.validate(checkpoint_id)
    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_CORRUPT


def test_ddp_latest_valid_skips_newer_corruption_and_empty_store_fails(tmp_path: Path) -> None:
    config = ddp_checkpoint_config()
    empty = DDPCheckpointStore(tmp_path / "empty", keep_last=2)
    with pytest.raises(CheckpointError) as caught:
        empty.latest_valid(expected_world_size=2)
    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_NO_VALID

    store = DDPCheckpointStore(tmp_path / "checkpoints", keep_last=2)
    first = _save(store, config, step=1)
    second = _save(store, config, step=2)
    with (store.root / second / "rank-00001.pt").open("ab") as stream:
        stream.write(b"corrupt")

    selection = store.latest_valid(expected_world_size=2)

    assert selection.checkpoint_id == first
    assert selection.skipped_invalid_checkpoints == (second,)


def test_ddp_checkpoint_duplicate_destination_fails_without_replacing_latest(
    tmp_path: Path,
) -> None:
    config = ddp_checkpoint_config()
    store = DDPCheckpointStore(tmp_path / "checkpoints", keep_last=2)
    checkpoint_id = _save(store, config)

    with pytest.raises(CheckpointError) as caught:
        _save(store, config)

    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_EXISTS
    assert (store.root / "LATEST").read_text(encoding="utf-8").strip() == checkpoint_id
