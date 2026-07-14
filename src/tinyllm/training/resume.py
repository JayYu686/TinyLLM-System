"""Explicit Exact, Warm, and Transfer restore semantics for M1."""

from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, cast

import numpy
import torch
from torch import Tensor

from tinyllm.schemas import CheckpointManifest, ResumeResult
from tinyllm.training.checkpoint import (
    CheckpointContext,
    CheckpointError,
    CheckpointErrorCode,
    CheckpointStore,
    StatefulScaler,
)
from tinyllm.training.config import M1TrainingConfig
from tinyllm.training.metrics import TrainerState
from tinyllm.training.trainer import SingleDeviceTrainer


class ResumeMode(StrEnum):
    """Mutually exclusive checkpoint application policies."""

    EXACT = "exact"
    WARM = "warm"
    TRANSFER = "transfer"


def _incompatible(
    message: str,
    *,
    checkpoint_id: str,
    reason: str,
) -> CheckpointError:
    return CheckpointError(
        CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
        message,
        context={"checkpoint_id": checkpoint_id, "reason": reason},
    )


def _require_pristine(trainer: SingleDeviceTrainer, *, checkpoint_id: str) -> None:
    if not trainer.is_pristine:
        raise _incompatible(
            "checkpoint restore requires a pristine target trainer",
            checkpoint_id=checkpoint_id,
            reason="target_not_pristine",
        )


def _normalized_exact_config(config: M1TrainingConfig) -> dict[str, Any]:
    normalized = copy.deepcopy(config.to_dict())
    checkpoint = cast(dict[str, Any], normalized["checkpoint"])
    checkpoint.pop("output_dir", None)
    checkpoint.pop("resume", None)
    return normalized


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


def _model_partitions(
    trainer: SingleDeviceTrainer,
    raw_checkpoint_state: object,
) -> tuple[dict[str, Tensor], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not isinstance(raw_checkpoint_state, Mapping):
        return {}, tuple(trainer.model.state_dict()), (), ("<invalid-model-state>",)

    current = trainer.model.state_dict()
    loaded: dict[str, Tensor] = {}
    unexpected: list[str] = []
    incompatible: list[str] = []
    for raw_key, value in raw_checkpoint_state.items():
        if not isinstance(raw_key, str):
            incompatible.append("<non-string-key>")
            continue
        if raw_key not in current:
            unexpected.append(raw_key)
            continue
        if not isinstance(value, Tensor) or value.shape != current[raw_key].shape:
            incompatible.append(raw_key)
            continue
        loaded[raw_key] = value
    missing = tuple(sorted(set(current) - set(loaded)))
    return loaded, missing, tuple(sorted(unexpected)), tuple(sorted(incompatible))


def _validate_rng_state(raw: object, *, device: torch.device, checkpoint_id: str) -> None:
    if not isinstance(raw, dict) or set(raw) != {"python", "numpy", "torch", "cuda"}:
        raise _incompatible(
            "checkpoint RNG state is invalid",
            checkpoint_id=checkpoint_id,
            reason="rng_schema",
        )
    try:
        local_python = random.Random()
        local_python.setstate(raw["python"])
        local_numpy = numpy.random.RandomState()
        local_numpy.set_state(raw["numpy"])
        torch_state = raw["torch"]
        if not isinstance(torch_state, Tensor):
            raise TypeError("PyTorch RNG state is not a tensor")
        torch.Generator().set_state(torch_state.cpu())
        cuda_state = raw["cuda"]
        if isinstance(cuda_state, dict):
            if cuda_state != {"not_applicable": True} or device.type == "cuda":
                raise ValueError("CUDA RNG state is unavailable for a CUDA target")
        elif not (
            isinstance(cuda_state, list)
            and cuda_state
            and all(isinstance(state, Tensor) for state in cuda_state)
            and torch.cuda.is_available()
            and len(cuda_state) == torch.cuda.device_count()
        ):
            raise ValueError("CUDA RNG state does not match visible devices")
    except (RuntimeError, TypeError, ValueError) as exc:
        raise _incompatible(
            "checkpoint RNG state is incompatible with the target environment",
            checkpoint_id=checkpoint_id,
            reason="rng_environment",
        ) from exc


def _restore_rng_state(raw: dict[str, Any]) -> None:
    random.setstate(raw["python"])
    numpy.random.set_state(raw["numpy"])
    torch.set_rng_state(cast(Tensor, raw["torch"]).cpu())
    cuda_state = raw["cuda"]
    if isinstance(cuda_state, list):
        torch.cuda.set_rng_state_all(cuda_state)


def _require_exact_compatibility(
    *,
    trainer: SingleDeviceTrainer,
    manifest: CheckpointManifest,
    payload: dict[str, Any],
    context: CheckpointContext,
) -> None:
    checkpoint_id = manifest.checkpoint_id
    if manifest.resume_capability != "exact":
        raise _incompatible(
            "checkpoint does not declare complete Exact Resume state",
            checkpoint_id=checkpoint_id,
            reason="resume_capability",
        )
    saved_config = M1TrainingConfig.model_validate(payload["config"])
    saved_normalized = _normalized_exact_config(saved_config)
    current_normalized = _normalized_exact_config(trainer.config)
    if saved_normalized != current_normalized:
        changed = ",".join(_changed_paths(saved_normalized, current_normalized))
        raise _incompatible(
            "training configuration is incompatible with Exact Resume",
            checkpoint_id=checkpoint_id,
            reason=f"config:{changed}",
        )

    comparisons = {
        "run_id": (context.run_id, manifest.run_id),
        "dataset_version": (context.dataset_version, manifest.dataset_version),
        "git_commit": (context.git_commit, manifest.git_commit),
        "environment": (context.environment, payload["environment"]),
        "strategy": (context.strategy, manifest.strategy),
        "world_size": (context.world_size, manifest.world_size),
    }
    mismatches = [name for name, (current, saved) in comparisons.items() if current != saved]
    if mismatches:
        raise _incompatible(
            "runtime lineage is incompatible with Exact Resume",
            checkpoint_id=checkpoint_id,
            reason="lineage:" + ",".join(mismatches),
        )


def _restore_exact(
    *,
    trainer: SingleDeviceTrainer,
    manifest: CheckpointManifest,
    payload: dict[str, Any],
    context: CheckpointContext,
    scaler: StatefulScaler | None,
    skipped: tuple[str, ...],
) -> ResumeResult:
    checkpoint_id = manifest.checkpoint_id
    _require_pristine(trainer, checkpoint_id=checkpoint_id)
    _require_exact_compatibility(
        trainer=trainer,
        manifest=manifest,
        payload=payload,
        context=context,
    )
    if trainer.sampler is None:
        raise _incompatible(
            "Exact Resume requires a stateful sampler",
            checkpoint_id=checkpoint_id,
            reason="sampler_missing",
        )

    loaded, missing, unexpected, incompatible = _model_partitions(trainer, payload["model"])
    if missing or unexpected or incompatible:
        raise _incompatible(
            "model state is incompatible with Exact Resume",
            checkpoint_id=checkpoint_id,
            reason="model_state",
        )
    rng = payload["rng"]
    _validate_rng_state(rng, device=trainer.device, checkpoint_id=checkpoint_id)
    scaler_state = payload["grad_scaler"]
    if scaler is None and scaler_state != {"not_applicable": True}:
        raise _incompatible(
            "checkpoint requires a GradScaler but none was provided",
            checkpoint_id=checkpoint_id,
            reason="scaler_missing",
        )
    if scaler is not None and (
        not isinstance(scaler_state, dict) or scaler_state == {"not_applicable": True}
    ):
        raise _incompatible(
            "checkpoint GradScaler state is incompatible",
            checkpoint_id=checkpoint_id,
            reason="scaler_state",
        )

    try:
        trainer.model.load_state_dict(loaded, strict=True)
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(scaler_state)
        trainer.sampler.load_state_dict(payload["sampler"])
        trainer.restore_progress(TrainerState.model_validate(payload["trainer_state"]))
        _restore_rng_state(rng)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise _incompatible(
            "checkpoint state could not be applied to the target trainer",
            checkpoint_id=checkpoint_id,
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
        loaded_model_keys=tuple(sorted(loaded)),
        skipped_invalid_checkpoints=skipped,
    )


def _restore_weights_only(
    *,
    trainer: SingleDeviceTrainer,
    manifest: CheckpointManifest,
    payload: dict[str, Any],
    mode: ResumeMode,
    skipped: tuple[str, ...],
) -> ResumeResult:
    checkpoint_id = manifest.checkpoint_id
    _require_pristine(trainer, checkpoint_id=checkpoint_id)
    loaded, missing, unexpected, incompatible = _model_partitions(trainer, payload["model"])
    if mode is ResumeMode.WARM and (missing or unexpected or incompatible):
        raise _incompatible(
            "model state is incompatible with Warm Resume",
            checkpoint_id=checkpoint_id,
            reason="model_state",
        )
    if not loaded:
        raise _incompatible(
            "checkpoint has no compatible model tensors",
            checkpoint_id=checkpoint_id,
            reason="no_compatible_weights",
        )
    try:
        trainer.model.load_state_dict(loaded, strict=mode is ResumeMode.WARM)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise _incompatible(
            "checkpoint model weights could not be applied",
            checkpoint_id=checkpoint_id,
            reason="model_application",
        ) from exc
    return ResumeResult(
        mode=mode.value,
        checkpoint_id=checkpoint_id,
        source_run_id=manifest.run_id,
        source_global_step=manifest.global_step,
        target_global_step=0,
        optimizer_restored=False,
        scheduler_restored=False,
        scaler_restored=False,
        sampler_restored=False,
        rng_restored=False,
        loaded_model_keys=tuple(sorted(loaded)),
        missing_model_keys=missing,
        unexpected_checkpoint_keys=unexpected,
        incompatible_checkpoint_keys=incompatible,
        skipped_invalid_checkpoints=skipped,
    )


def restore_trainer(
    *,
    store: CheckpointStore,
    trainer: SingleDeviceTrainer,
    mode: ResumeMode,
    context: CheckpointContext | None = None,
    checkpoint_id: str | None = None,
    scaler: StatefulScaler | None = None,
) -> ResumeResult:
    """Validate and apply one explicit restore mode to a pristine Trainer."""

    if checkpoint_id is None:
        selection = store.latest_valid()
        checkpoint_id = selection.checkpoint_id
        skipped = selection.skipped_invalid_checkpoints
    else:
        skipped = ()
    manifest = store.validate(checkpoint_id)
    payload = store.load_training_state(checkpoint_id, map_location=trainer.device)
    if mode is ResumeMode.EXACT:
        if context is None:
            raise _incompatible(
                "Exact Resume requires current lineage context",
                checkpoint_id=checkpoint_id,
                reason="context_missing",
            )
        return _restore_exact(
            trainer=trainer,
            manifest=manifest,
            payload=payload,
            context=context,
            scaler=scaler,
            skipped=skipped,
        )
    return _restore_weights_only(
        trainer=trainer,
        manifest=manifest,
        payload=payload,
        mode=mode,
        skipped=skipped,
    )


def restore_from_config(
    *,
    store: CheckpointStore,
    trainer: SingleDeviceTrainer,
    context: CheckpointContext | None = None,
    checkpoint_id: str | None = None,
    scaler: StatefulScaler | None = None,
) -> ResumeResult | None:
    """Map the YAML resume policy to an explicit restore operation."""

    configured = trainer.config.checkpoint.resume
    if configured == "none":
        return None
    mode = ResumeMode.EXACT if configured == "auto" else ResumeMode(configured)
    return restore_trainer(
        store=store,
        trainer=trainer,
        mode=mode,
        context=context,
        checkpoint_id=checkpoint_id,
        scaler=scaler,
    )
