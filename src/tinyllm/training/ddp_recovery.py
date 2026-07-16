"""M3.2 DDP checkpoint, Exact Resume, interruption, and Rank-failure runtime."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from tinyllm.data import StatefulDistributedSampler, ToyTokenDataset
from tinyllm.lineage import read_git_identity
from tinyllm.models.tinygpt import TinyGPT
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.checkpoint import CheckpointContext
from tinyllm.training.config import M1TrainingConfig, load_training_config
from tinyllm.training.ddp_checkpoint import DDPCheckpointStore, build_rank_state
from tinyllm.training.ddp_recovery_schema import DDPRecoveryResult
from tinyllm.training.ddp_resume import restore_ddp_trainer
from tinyllm.training.distributed import (
    TorchrunEnvironment,
    all_gather_objects,
    distributed_barrier,
    initialize_process_group,
    model_state_sha256,
    rank_environment,
    reduced_mean,
    select_ddp_device,
    torchrun_environment,
)
from tinyllm.training.errors import TrainingError, TrainingErrorCode
from tinyllm.training.metrics import TrainingStepMetrics
from tinyllm.training.scheduler import build_adamw, build_warmup_cosine_scheduler
from tinyllm.training.seed import seed_everything
from tinyllm.training.trainer import SingleDeviceTrainer


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _environment_snapshot(
    config: M1TrainingConfig,
    rank_environments: list[object],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "backend": config.distributed.backend,
        "world_size": config.distributed.world_size,
        "ranks": rank_environments,
    }


def _hardware_snapshot(config: M1TrainingConfig, ranks: list[object]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "world_size": config.distributed.world_size,
        "devices": [
            {
                key: cast(dict[str, object], item).get(key)
                for key in (
                    "rank",
                    "local_rank",
                    "physical_gpu_index",
                    "gpu_name",
                    "memory_total_bytes",
                    "compute_capability",
                )
            }
            for item in ranks
        ],
    }


def _new_run(
    *,
    config_path: Path,
    config: M1TrainingConfig,
    output_root: Path,
    run_id: str,
    config_hash: str,
    git_commit: str,
    git_dirty: bool,
    environment: dict[str, object],
    rank_environments: list[object],
) -> Path:
    artifact_dir = output_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "evaluations", "exports", "failures"):
        (artifact_dir / name).mkdir()
    shutil.copyfile(config_path, artifact_dir / "config.original.yaml")
    _atomic_json(artifact_dir / "config.resolved.json", config.to_dict())
    _atomic_json(artifact_dir / "environment.json", environment)
    _atomic_json(artifact_dir / "hardware.json", _hardware_snapshot(config, rank_environments))
    dataset_version = f"toy-ddp-{config_hash[:8]}"
    _atomic_json(
        artifact_dir / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "running",
            "strategy": "ddp",
            "world_size": config.distributed.world_size,
            "config_hash": config_hash,
            "dataset_version": dataset_version,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "checkpoint_status": "enabled_m3_2",
        },
    )
    _append_jsonl(
        artifact_dir / "events.jsonl",
        {
            "event": "ddp_recovery_run_started",
            "run_id": run_id,
            "world_size": config.distributed.world_size,
            "backend": config.distributed.backend,
        },
    )
    (artifact_dir / "metrics.jsonl").touch()
    return artifact_dir


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return cast(dict[str, object], value)


def _validate_resume_run(
    *,
    artifact_dir: Path,
    config: M1TrainingConfig,
    config_hash: str,
    git_commit: str,
    current_environment: dict[str, object],
) -> tuple[str, str, dict[str, object], bool]:
    run = _read_object(artifact_dir / "run.json", label="resume Run manifest")
    environment = _read_object(
        artifact_dir / "environment.json",
        label="resume environment snapshot",
    )
    required = {
        "run_id": str,
        "dataset_version": str,
        "config_hash": str,
        "git_commit": str,
        "git_dirty": bool,
        "world_size": int,
        "strategy": str,
    }
    if any(not isinstance(run.get(key), expected) for key, expected in required.items()):
        raise RuntimeError("resume Run manifest is incomplete")
    mismatches = (
        run["config_hash"] != config_hash
        or run["git_commit"] != git_commit
        or run["world_size"] != config.distributed.world_size
        or run["strategy"] != "ddp"
        or environment != current_environment
    )
    if mismatches:
        raise RuntimeError("resume Run lineage, World Size, or environment changed")
    return (
        cast(str, run["run_id"]),
        cast(str, run["dataset_version"]),
        environment,
        cast(bool, run["git_dirty"]),
    )


def _truncate_metrics(path: Path, *, checkpoint_step: int) -> tuple[int, int]:
    metrics: list[TrainingStepMetrics] = []
    total_rows = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total_rows += 1
                metric = TrainingStepMetrics.model_validate_json(line)
                if metric.global_step <= checkpoint_step:
                    metrics.append(metric)
    except (OSError, ValueError) as exc:
        raise RuntimeError("Run metrics cannot be reconciled with the Checkpoint") from exc
    steps = [metric.global_step for metric in metrics]
    if steps != list(range(1, checkpoint_step + 1)):
        raise RuntimeError("Run metrics do not contain one canonical row per committed step")
    _atomic_bytes(
        path,
        b"".join(
            (json.dumps(metric.to_dict(), sort_keys=True) + "\n").encode("utf-8")
            for metric in metrics
        ),
    )
    return len(metrics), total_rows - len(metrics)


def _validate_future_injection_steps(
    *,
    current_step: int,
    stop_after_step: int | None,
    fail_after_step: int | None,
) -> None:
    for name, value in (
        ("stop-after-step", stop_after_step),
        ("fail-after-step", fail_after_step),
    ):
        if value is not None and value <= current_step:
            raise ValueError(f"{name} must be after the selected resume Checkpoint")


def _broadcast_rank_zero(value: object, *, launch: TorchrunEnvironment) -> object:
    values = [value if launch.rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _save_checkpoint(
    *,
    store: DDPCheckpointStore,
    model: TinyGPT,
    trainer: SingleDeviceTrainer,
    sampler: StatefulDistributedSampler,
    config: M1TrainingConfig,
    context: CheckpointContext,
    launch: TorchrunEnvironment,
    device: torch.device,
    pin_reason: Literal["interruption", "final"] | None,
) -> tuple[str, str]:
    local_state = build_rank_state(
        rank=launch.rank,
        world_size=launch.world_size,
        trainer_state=trainer.state,
        sampler_state=sampler.state_dict(),
        device=device,
    )
    gathered = all_gather_objects(local_state, world_size=launch.world_size)
    model_hashes = all_gather_objects(
        model_state_sha256(model),
        world_size=launch.world_size,
    )
    if len({str(value) for value in model_hashes}) != 1:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
            "model parameters differ across Ranks at the Checkpoint boundary",
        )
    status: object = None
    if launch.rank == 0:
        try:
            manifest = store.save(
                model=model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                trainer_state=trainer.state,
                config=config,
                context=context,
                rank_states=[cast(dict[str, object], item) for item in gathered],
                pin_reason=pin_reason,
            )
            status = {"ok": True, "checkpoint_id": manifest.checkpoint_id}
        except Exception as exc:
            status = {"ok": False, "error_type": type(exc).__name__}
    status = _broadcast_rank_zero(status, launch=launch)
    if not isinstance(status, dict) or not status.get("ok"):
        raise RuntimeError("Rank zero failed to publish the DDP Checkpoint")
    distributed_barrier(device, launch)
    return str(status["checkpoint_id"]), str(model_hashes[0])


def _durable_metric(
    *,
    local: TrainingStepMetrics,
    device: torch.device,
    launch: TorchrunEnvironment,
) -> TrainingStepMetrics | None:
    losses = [
        cast(float, item) for item in all_gather_objects(local.loss, world_size=launch.world_size)
    ]
    reduced_loss = reduced_mean(local.loss, device=device, world_size=launch.world_size)
    if abs(reduced_loss - sum(losses) / launch.world_size) > 1e-12:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
            "DDP reduced loss differs from the Rank mean",
        )
    norms = [
        cast(float, item)
        for item in all_gather_objects(local.gradient_norm, world_size=launch.world_size)
    ]
    clipped = all_gather_objects(local.gradient_clipped, world_size=launch.world_size)
    if launch.rank != 0:
        return None
    return TrainingStepMetrics(
        global_step=local.global_step,
        micro_step=local.micro_step,
        epoch=local.epoch,
        loss=reduced_loss,
        learning_rate=local.learning_rate,
        gradient_norm=sum(norms) / launch.world_size,
        gradient_clipped=any(bool(item) for item in clipped),
        tokens_seen=local.tokens_seen * launch.world_size,
    )


def _run_status(
    *,
    run_id: str,
    status: str,
    config_hash: str,
    dataset_version: str,
    git_commit: str,
    git_dirty: bool,
    world_size: int,
    global_step: int,
    checkpoint_id: str,
    reason: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": status,
        "strategy": "ddp",
        "world_size": world_size,
        "global_step": global_step,
        "config_hash": config_hash,
        "dataset_version": dataset_version,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "checkpoint_status": "valid",
        "latest_checkpoint": checkpoint_id,
    }
    if reason is not None:
        value["reason"] = reason
    return value


def run_ddp_recovery(
    *,
    config_path: Path,
    output_root: Path,
    resume_run: Path | None = None,
    stop_after_step: int | None = None,
    fail_rank: int | None = None,
    fail_after_step: int | None = None,
) -> DDPRecoveryResult | None:
    """Run one fresh/resumed phase; Rank zero owns every durable shared artifact."""

    config_path = config_path.resolve()
    output_root = output_root.resolve()
    resume_run = resume_run.resolve() if resume_run is not None else None
    config = load_training_config(config_path)
    if config.distributed.strategy != "ddp" or config.distributed.backend is None:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_LAUNCH_REQUIRED,
            "M3.2 recovery requires an explicit DDP backend",
        )
    launch = torchrun_environment(os.environ)
    if launch.world_size != config.distributed.world_size:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID,
            "torchrun WORLD_SIZE does not match the recovery config",
        )
    if (fail_rank is None) != (fail_after_step is None):
        raise ValueError("fail-rank and fail-after-step must be provided together")
    if fail_rank is not None and not 0 <= fail_rank < launch.world_size:
        raise ValueError("fail-rank is outside the launched World Size")
    for name, value in (("stop-after-step", stop_after_step), ("fail-after-step", fail_after_step)):
        if value is not None and not 1 <= value < config.training.max_steps:
            raise ValueError(f"{name} must be before training.max_steps")
    if stop_after_step is not None and fail_after_step is not None:
        raise ValueError("coordinated interruption and Rank failure are mutually exclusive")

    device = select_ddp_device(config, launch)
    initialize_process_group(config, device=device)
    artifact_dir: Path | None = resume_run
    failure_armed = False
    try:
        config_hash = canonical_config_hash(config)
        git_commit, current_git_dirty = read_git_identity(config_path.parent)
        local_environment = rank_environment(launch, device)
        rank_environments = all_gather_objects(local_environment, world_size=launch.world_size)
        current_environment = _environment_snapshot(config, rank_environments)

        if resume_run is None:
            run_id_value = (
                generate_run_id(config.run.name, config_hash, now=datetime.now(UTC))
                if launch.rank == 0
                else None
            )
            run_id = str(_broadcast_rank_zero(run_id_value, launch=launch))
            dataset_version = f"toy-ddp-{config_hash[:8]}"
            git_dirty = current_git_dirty
            environment = current_environment
            if launch.rank == 0:
                output_root.mkdir(parents=True, exist_ok=True)
                artifact_dir = _new_run(
                    config_path=config_path,
                    config=config,
                    output_root=output_root,
                    run_id=run_id,
                    config_hash=config_hash,
                    git_commit=git_commit,
                    git_dirty=git_dirty,
                    environment=environment,
                    rank_environments=rank_environments,
                )
            artifact_value = _broadcast_rank_zero(
                str(artifact_dir) if artifact_dir is not None else None,
                launch=launch,
            )
            artifact_dir = Path(str(artifact_value))
            mode: Literal["fresh", "exact_resume"] = "fresh"
        else:
            resume_identity: object = None
            if launch.rank == 0:
                run_id, dataset_version, environment, git_dirty = _validate_resume_run(
                    artifact_dir=resume_run,
                    config=config,
                    config_hash=config_hash,
                    git_commit=git_commit,
                    current_environment=current_environment,
                )
                resume_identity = {
                    "run_id": run_id,
                    "dataset_version": dataset_version,
                    "environment": environment,
                    "git_dirty": git_dirty,
                }
            resume_identity = _broadcast_rank_zero(resume_identity, launch=launch)
            if not isinstance(resume_identity, dict):
                raise RuntimeError("Rank zero did not publish the resume identity")
            run_id = str(resume_identity["run_id"])
            dataset_version = str(resume_identity["dataset_version"])
            environment = cast(dict[str, object], resume_identity["environment"])
            git_dirty = bool(resume_identity["git_dirty"])
            artifact_dir = resume_run
            mode = "exact_resume"
        distributed_barrier(device, launch)

        context = CheckpointContext(
            run_id=run_id,
            dataset_version=dataset_version,
            git_commit=git_commit,
            environment=environment,
            strategy="ddp",
            world_size=launch.world_size,
        )
        store = DDPCheckpointStore(
            artifact_dir / "checkpoints",
            keep_last=config.checkpoint.keep_last,
        )
        seed_everything(config.run.seed, deterministic_algorithms=device.type == "cpu")
        dataset = ToyTokenDataset(
            vocab_size=config.data.vocab_size,
            sequence_length=config.data.sequence_length,
            num_samples=config.data.num_samples,
            seed=config.run.seed,
        )
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=launch.world_size,
            rank=launch.rank,
            seed=config.run.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.training.micro_batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        model = TinyGPT(config.model).to(device)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[launch.local_rank] if device.type == "cuda" else None,
            output_device=launch.local_rank if device.type == "cuda" else None,
            broadcast_buffers=config.distributed.broadcast_buffers,
            find_unused_parameters=config.distributed.find_unused_parameters,
        )
        optimizer = build_adamw(ddp_model, config.training)
        scheduler = build_warmup_cosine_scheduler(optimizer, config.training)
        trainer = SingleDeviceTrainer(
            model=ddp_model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            sampler=sampler,
            autocast_dtype=torch.bfloat16 if config.precision.dtype == "bf16" else None,
        )

        resumed_from_step: int | None = None
        if mode == "exact_resume":
            selection_value: object = None
            if launch.rank == 0:
                selection = store.latest_valid(expected_world_size=launch.world_size)
                selection_value = {
                    "checkpoint_id": selection.checkpoint_id,
                    "skipped": selection.skipped_invalid_checkpoints,
                }
            selection_value = _broadcast_rank_zero(selection_value, launch=launch)
            if not isinstance(selection_value, dict):
                raise RuntimeError("Rank zero did not select a resume Checkpoint")
            checkpoint_id = str(selection_value["checkpoint_id"])
            skipped = tuple(str(item) for item in selection_value["skipped"])
            resume_progress: object = None
            if launch.rank == 0:
                resumed_from_step = store.validate(
                    checkpoint_id,
                    expected_world_size=launch.world_size,
                ).global_step
                retained_metrics, discarded_metrics = _truncate_metrics(
                    artifact_dir / "metrics.jsonl",
                    checkpoint_step=resumed_from_step,
                )
                resume_progress = {
                    "step": resumed_from_step,
                    "retained_metrics": retained_metrics,
                    "discarded_metrics": discarded_metrics,
                }
            resume_progress = _broadcast_rank_zero(resume_progress, launch=launch)
            if not isinstance(resume_progress, dict):
                raise RuntimeError("Rank zero did not reconcile the resume progress")
            resumed_from_step = int(resume_progress["step"])
            retained_metrics = int(resume_progress["retained_metrics"])
            discarded_metrics = int(resume_progress["discarded_metrics"])
            restore_ddp_trainer(
                store=store,
                checkpoint_id=checkpoint_id,
                trainer=trainer,
                unwrapped_model=model,
                sampler=sampler,
                context=context,
                rank=launch.rank,
                synchronize=lambda: distributed_barrier(device, launch),
                skipped_invalid_checkpoints=skipped,
            )
            if launch.rank == 0:
                _append_jsonl(
                    artifact_dir / "events.jsonl",
                    {
                        "event": "ddp_exact_resume_applied",
                        "checkpoint_id": checkpoint_id,
                        "global_step": resumed_from_step,
                        "skipped_invalid_checkpoints": skipped,
                        "retained_metric_records": retained_metrics,
                        "discarded_metric_records": discarded_metrics,
                    },
                )

        _validate_future_injection_steps(
            current_step=trainer.state.global_step,
            stop_after_step=stop_after_step,
            fail_after_step=fail_after_step,
        )

        latest_checkpoint = (
            f"checkpoint-step-{trainer.state.global_step:08d}"
            if trainer.state.global_step > 0
            else ""
        )
        latest_parameter_hash = ""
        while trainer.state.global_step < config.training.max_steps:
            result = trainer.train(target_global_step=trainer.state.global_step + 1)
            durable = _durable_metric(local=result.metrics[0], device=device, launch=launch)
            if launch.rank == 0:
                assert durable is not None
                _append_jsonl(artifact_dir / "metrics.jsonl", durable.to_dict())

            step = trainer.state.global_step
            interruption = stop_after_step == step
            rank_failure = fail_after_step == step
            final = step == config.training.max_steps
            if step % config.checkpoint.save_steps == 0 or interruption or rank_failure or final:
                pin_reason: Literal["interruption", "final"] | None = None
                if interruption or rank_failure:
                    pin_reason = "interruption"
                elif final:
                    pin_reason = "final"
                latest_checkpoint, latest_parameter_hash = _save_checkpoint(
                    store=store,
                    model=model,
                    trainer=trainer,
                    sampler=sampler,
                    config=config,
                    context=context,
                    launch=launch,
                    device=device,
                    pin_reason=pin_reason,
                )
                if launch.rank == 0:
                    _append_jsonl(
                        artifact_dir / "events.jsonl",
                        {
                            "event": "ddp_checkpoint_committed",
                            "checkpoint_id": latest_checkpoint,
                            "global_step": step,
                            "pin_reason": pin_reason,
                        },
                    )

            if interruption:
                if launch.rank == 0:
                    _atomic_json(
                        artifact_dir / "run.json",
                        _run_status(
                            run_id=run_id,
                            status="interrupted",
                            config_hash=config_hash,
                            dataset_version=dataset_version,
                            git_commit=git_commit,
                            git_dirty=git_dirty,
                            world_size=launch.world_size,
                            global_step=step,
                            checkpoint_id=latest_checkpoint,
                            reason="coordinated_stop",
                        ),
                    )
                    _append_jsonl(
                        artifact_dir / "events.jsonl",
                        {
                            "event": "ddp_coordinated_interruption",
                            "global_step": step,
                            "checkpoint_id": latest_checkpoint,
                        },
                    )
                distributed_barrier(device, launch)
                return (
                    DDPRecoveryResult(
                        status="interrupted",
                        mode=mode,
                        run_id=run_id,
                        artifact_dir=artifact_dir,
                        config_sha256=config_hash,
                        git_commit=git_commit,
                        git_dirty=git_dirty,
                        backend=config.distributed.backend,
                        world_size=launch.world_size,
                        global_step=step,
                        checkpoint_id=latest_checkpoint,
                        model_parameter_sha256=latest_parameter_hash,
                        resumed_from_step=resumed_from_step,
                        durable_metric_records=step,
                    )
                    if launch.rank == 0
                    else None
                )

            if rank_failure:
                failure_armed = True
                if launch.rank == 0:
                    failure = {
                        "schema_version": "1.0",
                        "event": "forced_rank_exit",
                        "rank": fail_rank,
                        "exit_code": 17,
                        "global_step": step,
                        "checkpoint_id": latest_checkpoint,
                        "resumable": True,
                    }
                    _atomic_json(
                        artifact_dir / "failures" / f"rank-{fail_rank}-step-{step:08d}.json",
                        failure,
                    )
                    _append_jsonl(artifact_dir / "events.jsonl", failure)
                    _atomic_json(
                        artifact_dir / "run.json",
                        _run_status(
                            run_id=run_id,
                            status="failure_injected",
                            config_hash=config_hash,
                            dataset_version=dataset_version,
                            git_commit=git_commit,
                            git_dirty=git_dirty,
                            world_size=launch.world_size,
                            global_step=step,
                            checkpoint_id=latest_checkpoint,
                            reason="forced_rank_exit",
                        ),
                    )
                distributed_barrier(device, launch)
                if launch.rank == fail_rank:
                    os._exit(17)
                distributed_barrier(device, launch)

        if launch.rank == 0:
            _atomic_json(
                artifact_dir / "run.json",
                _run_status(
                    run_id=run_id,
                    status="succeeded",
                    config_hash=config_hash,
                    dataset_version=dataset_version,
                    git_commit=git_commit,
                    git_dirty=git_dirty,
                    world_size=launch.world_size,
                    global_step=trainer.state.global_step,
                    checkpoint_id=latest_checkpoint,
                ),
            )
            _append_jsonl(
                artifact_dir / "events.jsonl",
                {
                    "event": "ddp_recovery_run_succeeded",
                    "global_step": trainer.state.global_step,
                    "checkpoint_id": latest_checkpoint,
                },
            )
        distributed_barrier(device, launch)
        return (
            DDPRecoveryResult(
                status="succeeded",
                mode=mode,
                run_id=run_id,
                artifact_dir=artifact_dir,
                config_sha256=config_hash,
                git_commit=git_commit,
                git_dirty=git_dirty,
                backend=config.distributed.backend,
                world_size=launch.world_size,
                global_step=trainer.state.global_step,
                checkpoint_id=latest_checkpoint,
                model_parameter_sha256=latest_parameter_hash,
                resumed_from_step=resumed_from_step,
                durable_metric_records=trainer.state.global_step,
            )
            if launch.rank == 0
            else None
        )
    except Exception as exc:
        if launch.rank == 0 and artifact_dir is not None:
            _append_jsonl(
                artifact_dir / "events.jsonl",
                {
                    "event": "ddp_recovery_phase_failed",
                    "error_type": type(exc).__name__,
                    "failure_injection_armed": failure_armed,
                },
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
