"""Single-device YAML training entry used by the public CLI."""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch

from tinyllm.lineage import read_git_identity
from tinyllm.schemas import TrainingRunResult, canonical_config_hash, generate_run_id
from tinyllm.training.checkpoint import CheckpointContext, CheckpointStore
from tinyllm.training.config import M1TrainingConfig, load_training_config
from tinyllm.training.resume import ResumeMode, restore_trainer
from tinyllm.training.trainer import (
    SingleDeviceTrainer,
    build_m1_cpu_trainer,
    build_m1_cuda_trainer,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _runtime_environment(device: torch.device) -> dict[str, object]:
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment.update(
            {
                "visible_device_count": torch.cuda.device_count(),
                "gpu_name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return environment


def _build_trainer(
    config: M1TrainingConfig,
    *,
    device: Literal["auto", "cpu", "cuda"],
) -> SingleDeviceTrainer:
    selected = "cuda" if device == "auto" and config.precision.dtype == "bf16" else device
    if selected == "auto":
        selected = "cpu"
    if selected == "cpu":
        return build_m1_cpu_trainer(config)
    return build_m1_cuda_trainer(config)


def _save_checkpoint(
    *,
    store: CheckpointStore,
    trainer: SingleDeviceTrainer,
    config: M1TrainingConfig,
    context: CheckpointContext,
    pin_reason: Literal["interruption", "final"] | None,
) -> str:
    if trainer.sampler is None:
        raise RuntimeError("single-device training requires a stateful sampler")
    manifest = store.save(
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=None,
        sampler=trainer.sampler,
        trainer_state=trainer.state,
        config=config,
        context=context,
        pin_reason=pin_reason,
    )
    return manifest.checkpoint_id


def _new_run(
    *,
    config_path: Path,
    config: M1TrainingConfig,
    output_root: Path,
    context: CheckpointContext,
    environment: dict[str, object],
) -> Path:
    artifact_dir = output_root / context.run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "evaluations").mkdir()
    (artifact_dir / "exports").mkdir()
    shutil.copyfile(config_path, artifact_dir / "config.original.yaml")
    _atomic_json(artifact_dir / "config.resolved.json", config.to_dict())
    _atomic_json(artifact_dir / "environment.json", environment)
    _atomic_json(
        artifact_dir / "hardware.json",
        {
            "schema_version": "1.0",
            "device": environment["device"],
            "gpu_name": environment.get("gpu_name"),
            "compute_capability": environment.get("compute_capability"),
        },
    )
    return artifact_dir


def run_single_device_training(
    *,
    config_path: Path,
    output_root: Path | None = None,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    resume_run: Path | None = None,
    resume_mode: Literal["exact", "warm", "transfer"] = "exact",
) -> TrainingRunResult:
    """Execute one YAML run, optionally importing or exactly continuing a prior Run."""

    config_path = config_path.resolve()
    config = load_training_config(config_path)
    project_root = config_path.parent
    git_commit, git_dirty = read_git_identity(project_root)
    config_hash = canonical_config_hash(config)
    trainer = _build_trainer(config, device=device)
    environment = _runtime_environment(trainer.device)
    output_root = (output_root or Path(config.checkpoint.output_dir)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_store: CheckpointStore | None = None
    source_payload: dict[str, object] | None = None
    resumed_from_step: int | None = None
    skipped_invalid_checkpoints: tuple[str, ...] = ()
    if resume_run is not None:
        resume_run = resume_run.resolve()
        source_store = CheckpointStore(
            resume_run / "checkpoints", keep_last=config.checkpoint.keep_last
        )
        selection = source_store.latest_valid()
        source_manifest = source_store.validate(selection.checkpoint_id)
        source_payload = source_store.load_training_state(selection.checkpoint_id)
    else:
        source_manifest = None

    if source_manifest is not None and resume_mode == "exact":
        run_id = source_manifest.run_id
        artifact_dir = cast_path(resume_run)
        dataset_version = source_manifest.dataset_version
        context = CheckpointContext(
            run_id=run_id,
            dataset_version=dataset_version,
            git_commit=git_commit,
            environment=environment,
        )
        if source_store is None:
            raise RuntimeError("Exact Resume source store is missing")
        resume_result = restore_trainer(
            store=source_store,
            trainer=trainer,
            mode=ResumeMode.EXACT,
            context=context,
        )
        resumed_from_step = resume_result.source_global_step
        skipped_invalid_checkpoints = resume_result.skipped_invalid_checkpoints
        store = source_store
        if source_payload is None:
            raise RuntimeError("Exact Resume source payload is missing")
        checkpoint_config = M1TrainingConfig.model_validate(source_payload["config"])
    else:
        run_id = generate_run_id(config.run.name, config_hash, now=datetime.now(UTC))
        dataset_version = f"toy-{config_hash[:8]}"
        context = CheckpointContext(
            run_id=run_id,
            dataset_version=dataset_version,
            git_commit=git_commit,
            environment=environment,
        )
        artifact_dir = _new_run(
            config_path=config_path,
            config=config,
            output_root=output_root,
            context=context,
            environment=environment,
        )
        store = CheckpointStore(artifact_dir / "checkpoints", keep_last=config.checkpoint.keep_last)
        checkpoint_config = config
        if source_store is not None:
            resume_result = restore_trainer(
                store=source_store,
                trainer=trainer,
                mode=ResumeMode(resume_mode),
            )
            resumed_from_step = resume_result.source_global_step
            skipped_invalid_checkpoints = resume_result.skipped_invalid_checkpoints

    run_config_hash = canonical_config_hash(checkpoint_config)

    _append_jsonl(
        artifact_dir / "events.jsonl",
        {
            "event": "run_started",
            "run_id": run_id,
            "resume_mode": resume_mode if resume_run is not None else "none",
            "resumed_from_step": resumed_from_step,
        },
    )
    _atomic_json(
        artifact_dir / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "running",
            "config_hash": run_config_hash,
            "dataset_version": dataset_version,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "environment": environment,
        },
    )

    termination_requested = False

    def request_termination(signum: int, frame: object) -> None:
        nonlocal termination_requested
        del signum, frame
        termination_requested = True

    previous_handler = signal.signal(signal.SIGTERM, request_termination)
    checkpoint_id: str | None = (
        store.latest() if trainer.state.global_step == config.training.max_steps else None
    )
    status: Literal["succeeded", "terminated"] = "succeeded"
    try:
        while trainer.state.global_step < config.training.max_steps:
            step_result = trainer.train(target_global_step=trainer.state.global_step + 1)
            _append_jsonl(artifact_dir / "metrics.jsonl", step_result.metrics[0].to_dict())
            if (
                trainer.state.global_step % config.checkpoint.save_steps == 0
                and trainer.state.global_step < config.training.max_steps
            ):
                checkpoint_id = _save_checkpoint(
                    store=store,
                    trainer=trainer,
                    config=checkpoint_config,
                    context=context,
                    pin_reason=None,
                )
            if termination_requested:
                checkpoint_id = _save_checkpoint(
                    store=store,
                    trainer=trainer,
                    config=checkpoint_config,
                    context=context,
                    pin_reason="interruption",
                )
                status = "terminated"
                break
        if status == "succeeded":
            if trainer.state.global_step == config.training.max_steps and checkpoint_id is None:
                checkpoint_id = _save_checkpoint(
                    store=store,
                    trainer=trainer,
                    config=checkpoint_config,
                    context=context,
                    pin_reason="final",
                )
            elif trainer.state.global_step == config.training.max_steps:
                latest = store.latest()
                if store.validate(latest).global_step == trainer.state.global_step:
                    checkpoint_id = latest
                else:
                    checkpoint_id = _save_checkpoint(
                        store=store,
                        trainer=trainer,
                        config=checkpoint_config,
                        context=context,
                        pin_reason="final",
                    )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    if checkpoint_id is None:
        raise RuntimeError("training ended without a committed checkpoint")

    _atomic_json(
        artifact_dir / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": status,
            "global_step": trainer.state.global_step,
            "checkpoint_id": checkpoint_id,
            "config_hash": run_config_hash,
            "dataset_version": dataset_version,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "environment": environment,
        },
    )
    _append_jsonl(
        artifact_dir / "events.jsonl",
        {
            "event": "run_terminated" if status == "terminated" else "run_succeeded",
            "global_step": trainer.state.global_step,
            "checkpoint_id": checkpoint_id,
        },
    )
    return TrainingRunResult(
        status=status,
        run_id=run_id,
        artifact_dir=artifact_dir,
        device=str(trainer.device),
        global_step=trainer.state.global_step,
        checkpoint_id=checkpoint_id,
        resume_mode=resume_mode if resume_run is not None else "none",
        resumed_from_step=resumed_from_step,
        skipped_invalid_checkpoints=skipped_invalid_checkpoints,
    )


def cast_path(value: Path | None) -> Path:
    """Narrow an already-validated optional resume path for MyPy."""

    if value is None:
        raise RuntimeError("resume path is missing")
    return value
