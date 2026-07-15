from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from tinyllm.data import StatefulDistributedSampler, ToyTokenDataset
from tinyllm.models.tinygpt import TinyGPT
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training import (
    CheckpointContext,
    CheckpointError,
    CheckpointErrorCode,
    DDPCheckpointStore,
    SingleDeviceTrainer,
    build_rank_state,
    restore_ddp_trainer,
    seed_everything,
)
from tinyllm.training.config import M1TrainingConfig, training_config_from_mapping
from tinyllm.training.scheduler import build_adamw, build_warmup_cosine_scheduler


def ddp_resume_config() -> M1TrainingConfig:
    return training_config_from_mapping(
        {
            "schema_version": "1.0",
            "run": {"name": "ddp-resume-unit", "seed": 29},
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
                "output_dir": "runs/ddp-resume-unit",
                "save_steps": 2,
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


def build_rank_trainer(
    config: M1TrainingConfig,
    *,
    rank: int,
) -> tuple[SingleDeviceTrainer, TinyGPT, StatefulDistributedSampler]:
    seed_everything(config.run.seed, deterministic_algorithms=True)
    dataset = ToyTokenDataset(
        vocab_size=config.data.vocab_size,
        sequence_length=config.data.sequence_length,
        num_samples=config.data.num_samples,
        seed=config.run.seed,
    )
    sampler = StatefulDistributedSampler(
        dataset,
        num_replicas=config.distributed.world_size,
        rank=rank,
        seed=config.run.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.micro_batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
    )
    model = TinyGPT(config.model)
    optimizer = build_adamw(model, config.training)
    scheduler = build_warmup_cosine_scheduler(optimizer, config.training)
    return (
        SingleDeviceTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=torch.device("cpu"),
            sampler=sampler,
        ),
        model,
        sampler,
    )


def context(config: M1TrainingConfig) -> CheckpointContext:
    config_hash = canonical_config_hash(config)
    return CheckpointContext(
        run_id=generate_run_id(
            config.run.name,
            config_hash,
            now=datetime(2026, 7, 15, tzinfo=UTC),
            nonce="feed",
        ),
        dataset_version=f"toy-ddp-{config_hash[:8]}",
        git_commit="a" * 40,
        environment={"backend": "gloo", "world_size": 2, "device": "cpu"},
        strategy="ddp",
        world_size=2,
    )


def save_step_two(
    tmp_path: Path,
) -> tuple[
    M1TrainingConfig,
    DDPCheckpointStore,
    CheckpointContext,
    SingleDeviceTrainer,
    TinyGPT,
]:
    config = ddp_resume_config()
    source, model, sampler = build_rank_trainer(config, rank=0)
    source.train(target_global_step=2)
    rank_zero = build_rank_state(
        rank=0,
        world_size=2,
        trainer_state=source.state,
        sampler_state=sampler.state_dict(),
        device=torch.device("cpu"),
    )
    _, _, rank_one_sampler = build_rank_trainer(config, rank=1)
    rank_one_iterator = iter(rank_one_sampler)
    for _ in range(sampler.cursor):
        next(rank_one_iterator)
    rank_one = build_rank_state(
        rank=1,
        world_size=2,
        trainer_state=source.state,
        sampler_state=rank_one_sampler.state_dict(),
        device=torch.device("cpu"),
    )
    store = DDPCheckpointStore(tmp_path / "checkpoints", keep_last=2)
    checkpoint_context = context(config)
    store.save(
        model=model,
        optimizer=source.optimizer,
        scheduler=source.scheduler,
        trainer_state=source.state,
        config=config,
        context=checkpoint_context,
        rank_states=(rank_zero, rank_one),
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )
    return config, store, checkpoint_context, source, model


def resign_training_state(checkpoint_dir: Path) -> None:
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = (checkpoint_dir / "training_state.pt").read_bytes()
    for entry in manifest["files"]:
        if entry["path"] == "training_state.pt":
            entry["size_bytes"] = len(state)
            entry["sha256"] = hashlib.sha256(state).hexdigest()
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    (checkpoint_dir / "COMMITTED").write_text(
        json.dumps(
            {
                "checkpoint_id": manifest["checkpoint_id"],
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "schema_version": "1.0",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_ddp_exact_resume_continues_without_repeating_optimizer_steps(tmp_path: Path) -> None:
    config, store, checkpoint_context, source, source_model = save_step_two(tmp_path)
    uninterrupted = source.train(target_global_step=4)
    expected_parameters = {
        name: value.detach().clone() for name, value in source_model.state_dict().items()
    }

    target, target_model, target_sampler = build_rank_trainer(config, rank=0)
    synchronized = False

    def synchronize() -> None:
        nonlocal synchronized
        synchronized = True

    result = restore_ddp_trainer(
        store=store,
        checkpoint_id="checkpoint-step-00000002",
        trainer=target,
        unwrapped_model=target_model,
        sampler=target_sampler,
        context=checkpoint_context,
        rank=0,
        synchronize=synchronize,
    )
    resumed = target.train(target_global_step=4)

    assert result.target_global_step == 2
    assert synchronized is True
    assert [metric.global_step for metric in resumed.metrics] == [3, 4]
    assert [metric.to_dict() for metric in resumed.metrics] == [
        metric.to_dict() for metric in uninterrupted.metrics
    ]
    for name, value in target_model.state_dict().items():
        assert torch.equal(value, expected_parameters[name]), name


def test_ddp_exact_resume_rejects_lineage_drift_before_mutation(tmp_path: Path) -> None:
    config, store, checkpoint_context, _, _ = save_step_two(tmp_path)
    target, target_model, target_sampler = build_rank_trainer(config, rank=0)
    changed_context = CheckpointContext(
        run_id=checkpoint_context.run_id,
        dataset_version=checkpoint_context.dataset_version,
        git_commit=checkpoint_context.git_commit,
        environment={"backend": "gloo", "world_size": 2, "device": "different"},
        strategy="ddp",
        world_size=2,
    )

    with pytest.raises(CheckpointError) as caught:
        restore_ddp_trainer(
            store=store,
            checkpoint_id="checkpoint-step-00000002",
            trainer=target,
            unwrapped_model=target_model,
            sampler=target_sampler,
            context=changed_context,
            rank=0,
        )

    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE
    assert caught.value.context["reason"] == "lineage:environment"
    assert target.is_pristine


def test_ddp_exact_resume_rejects_wrong_world_size_before_mutation(tmp_path: Path) -> None:
    config, store, checkpoint_context, _, _ = save_step_two(tmp_path)
    target, target_model, target_sampler = build_rank_trainer(config, rank=0)
    changed_context = CheckpointContext(
        run_id=checkpoint_context.run_id,
        dataset_version=checkpoint_context.dataset_version,
        git_commit=checkpoint_context.git_commit,
        environment=checkpoint_context.environment,
        strategy="ddp",
        world_size=1,
    )

    with pytest.raises(CheckpointError) as caught:
        restore_ddp_trainer(
            store=store,
            checkpoint_id="checkpoint-step-00000002",
            trainer=target,
            unwrapped_model=target_model,
            sampler=target_sampler,
            context=changed_context,
            rank=0,
        )

    assert caught.value.code == CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE
    assert target.is_pristine


def test_ddp_exact_resume_rejects_nonpristine_target(tmp_path: Path) -> None:
    config, store, checkpoint_context, _, _ = save_step_two(tmp_path)
    target, target_model, target_sampler = build_rank_trainer(config, rank=0)
    target.train(target_global_step=1)

    with pytest.raises(CheckpointError) as caught:
        restore_ddp_trainer(
            store=store,
            checkpoint_id="checkpoint-step-00000002",
            trainer=target,
            unwrapped_model=target_model,
            sampler=target_sampler,
            context=checkpoint_context,
            rank=0,
        )

    assert caught.value.context["reason"] == "target_not_pristine"


def test_ddp_exact_resume_rejects_training_config_drift(tmp_path: Path) -> None:
    config, store, checkpoint_context, _, _ = save_step_two(tmp_path)
    changed_training = config.training.model_copy(update={"learning_rate": 0.02})
    changed_config = config.model_copy(update={"training": changed_training})
    target, target_model, target_sampler = build_rank_trainer(changed_config, rank=0)

    with pytest.raises(CheckpointError) as caught:
        restore_ddp_trainer(
            store=store,
            checkpoint_id="checkpoint-step-00000002",
            trainer=target,
            unwrapped_model=target_model,
            sampler=target_sampler,
            context=checkpoint_context,
            rank=0,
        )

    assert str(caught.value.context["reason"]).startswith("config:training.learning_rate")
    assert target.is_pristine


def test_ddp_exact_resume_rejects_model_key_drift_before_mutation(tmp_path: Path) -> None:
    config, store, checkpoint_context, _, _ = save_step_two(tmp_path)
    checkpoint_dir = store.root / "checkpoint-step-00000002"
    state_path = checkpoint_dir / "training_state.pt"
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    payload["model"].pop(next(iter(payload["model"])))
    torch.save(payload, state_path)
    resign_training_state(checkpoint_dir)
    target, target_model, target_sampler = build_rank_trainer(config, rank=0)

    with pytest.raises(CheckpointError) as caught:
        restore_ddp_trainer(
            store=store,
            checkpoint_id="checkpoint-step-00000002",
            trainer=target,
            unwrapped_model=target_model,
            sampler=target_sampler,
            context=checkpoint_context,
            rank=0,
        )

    assert caught.value.context["reason"] == "model_state"
    assert target.is_pristine


def test_ddp_exact_resume_rolls_back_failed_optimizer_application(tmp_path: Path) -> None:
    config, store, checkpoint_context, _, _ = save_step_two(tmp_path)
    checkpoint_dir = store.root / "checkpoint-step-00000002"
    state_path = checkpoint_dir / "training_state.pt"
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    payload["optimizer"] = {"state": {}, "param_groups": []}
    torch.save(payload, state_path)
    resign_training_state(checkpoint_dir)
    target, target_model, target_sampler = build_rank_trainer(config, rank=0)
    initial = {name: value.clone() for name, value in target_model.state_dict().items()}

    with pytest.raises(CheckpointError) as caught:
        restore_ddp_trainer(
            store=store,
            checkpoint_id="checkpoint-step-00000002",
            trainer=target,
            unwrapped_model=target_model,
            sampler=target_sampler,
            context=checkpoint_context,
            rank=0,
        )

    assert caught.value.context["reason"] == "state_application"
    assert target.is_pristine
    for name, value in target_model.state_dict().items():
        assert torch.equal(value, initial[name])
