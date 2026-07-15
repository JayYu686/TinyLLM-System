"""Fail-closed same-World-Size Exact Resume for native DDP training."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any, cast

from torch import Tensor, nn

from tinyllm.data import DistributedSamplerState, StatefulDistributedSampler
from tinyllm.schemas import ResumeResult
from tinyllm.training.checkpoint import (
    CheckpointContext,
    CheckpointError,
    CheckpointErrorCode,
)
from tinyllm.training.config import M1TrainingConfig
from tinyllm.training.ddp_checkpoint import (
    DDPCheckpointStore,
    restore_local_rng_state,
    validate_local_rng_state,
)
from tinyllm.training.metrics import TrainerState
from tinyllm.training.trainer import SingleDeviceTrainer


def _incompatible(checkpoint_id: str, message: str, *, reason: str) -> CheckpointError:
    return CheckpointError(
        CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
        message,
        context={"checkpoint_id": checkpoint_id, "reason": reason},
    )


def _normalized_exact_config(config: M1TrainingConfig) -> dict[str, Any]:
    value = copy.deepcopy(config.to_dict())
    checkpoint = cast(dict[str, Any], value["checkpoint"])
    checkpoint.pop("output_dir", None)
    checkpoint.pop("resume", None)
    return value


def _changed_paths(left: object, right: object, *, prefix: str = "") -> tuple[str, ...]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_changed_paths(left[key], right[key], prefix=path))
        return tuple(paths)
    return () if left == right else (prefix or "<root>",)


def _exact_model_state(
    model: nn.Module,
    raw: object,
    *,
    checkpoint_id: str,
) -> dict[str, Tensor]:
    current = model.state_dict()
    if not isinstance(raw, Mapping):
        raise _incompatible(
            checkpoint_id,
            "DDP Checkpoint model state is not a mapping",
            reason="model_schema",
        )
    loaded: dict[str, Tensor] = {}
    invalid: list[str] = []
    for key, value in raw.items():
        if (
            not isinstance(key, str)
            or key not in current
            or not isinstance(value, Tensor)
            or value.shape != current[key].shape
            or value.dtype != current[key].dtype
        ):
            invalid.append(str(key))
        else:
            loaded[key] = value
    if invalid or set(loaded) != set(current):
        raise _incompatible(
            checkpoint_id,
            "DDP Checkpoint model tensors differ from the target model",
            reason="model_state",
        )
    return loaded


def restore_ddp_trainer(
    *,
    store: DDPCheckpointStore,
    checkpoint_id: str,
    trainer: SingleDeviceTrainer,
    unwrapped_model: nn.Module,
    sampler: StatefulDistributedSampler,
    context: CheckpointContext,
    rank: int,
    synchronize: Callable[[], None] | None = None,
    skipped_invalid_checkpoints: tuple[str, ...] = (),
) -> ResumeResult:
    """Apply one validated Rank state, synchronizing before RNG restoration."""

    if not trainer.is_pristine:
        raise _incompatible(
            checkpoint_id,
            "DDP Exact Resume requires a pristine target Trainer",
            reason="target_not_pristine",
        )
    loaded = store.load(
        checkpoint_id,
        rank=rank,
        expected_world_size=context.world_size,
        map_location=trainer.device,
    )
    manifest = loaded.manifest
    shared = loaded.shared
    rank_state = loaded.rank
    if manifest.resume_capability != "exact":
        raise _incompatible(
            checkpoint_id,
            "DDP Checkpoint does not declare complete Exact Resume state",
            reason="resume_capability",
        )
    saved_config = M1TrainingConfig.model_validate(shared["config"])
    saved_normalized = _normalized_exact_config(saved_config)
    current_normalized = _normalized_exact_config(trainer.config)
    if saved_normalized != current_normalized:
        changed = ",".join(_changed_paths(saved_normalized, current_normalized))
        raise _incompatible(
            checkpoint_id,
            "DDP training configuration changed since the Checkpoint",
            reason=f"config:{changed}",
        )
    comparisons = {
        "run_id": (context.run_id, manifest.run_id),
        "dataset_version": (context.dataset_version, manifest.dataset_version),
        "git_commit": (context.git_commit, manifest.git_commit),
        "environment": (context.environment, shared["environment"]),
        "strategy": (context.strategy, manifest.strategy),
        "world_size": (context.world_size, manifest.world_size),
    }
    mismatches = [name for name, values in comparisons.items() if values[0] != values[1]]
    if mismatches:
        raise _incompatible(
            checkpoint_id,
            "DDP runtime lineage changed since the Checkpoint",
            reason="lineage:" + ",".join(mismatches),
        )

    model_state = _exact_model_state(
        unwrapped_model,
        shared["model"],
        checkpoint_id=checkpoint_id,
    )
    progress = TrainerState.model_validate(shared["trainer_state"])
    sampler_state = DistributedSamplerState.model_validate(rank_state["sampler"])
    if sampler_state.rank != rank or sampler_state.num_replicas != context.world_size:
        raise _incompatible(
            checkpoint_id,
            "DDP Sampler state belongs to a different Rank or World Size",
            reason="sampler_identity",
        )
    validate_local_rng_state(rank_state["rng"], device=trainer.device)

    rollback_model = copy.deepcopy(unwrapped_model.state_dict())
    rollback_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    rollback_scheduler = copy.deepcopy(trainer.scheduler.state_dict())  # type: ignore[no-untyped-call]
    rollback_sampler = sampler.state_dict()
    rollback_progress = trainer.state
    try:
        unwrapped_model.load_state_dict(model_state, strict=True)
        trainer.optimizer.load_state_dict(shared["optimizer"])
        trainer.scheduler.load_state_dict(shared["scheduler"])
        sampler.load_state_dict(sampler_state.to_dict())
        trainer.restore_progress(progress)
        if synchronize is not None:
            synchronize()
        restore_local_rng_state(rank_state["rng"], device=trainer.device)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        unwrapped_model.load_state_dict(rollback_model, strict=True)
        trainer.optimizer.load_state_dict(rollback_optimizer)
        trainer.scheduler.load_state_dict(rollback_scheduler)
        sampler.load_state_dict(rollback_sampler)
        trainer.restore_progress(rollback_progress)
        raise _incompatible(
            checkpoint_id,
            "DDP Checkpoint state could not be applied atomically",
            reason="state_application",
        ) from exc

    return ResumeResult(
        mode="exact",
        checkpoint_id=checkpoint_id,
        source_run_id=manifest.run_id,
        source_global_step=manifest.global_step,
        target_global_step=trainer.state.global_step,
        optimizer_restored=True,
        scheduler_restored=True,
        scaler_restored=True,
        sampler_restored=True,
        rng_restored=True,
        loaded_model_keys=tuple(sorted(model_state)),
        skipped_invalid_checkpoints=skipped_invalid_checkpoints,
    )
