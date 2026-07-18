"""M4.2 FSDP2 DCP Checkpoint, interruption, and Exact Resume runtime."""

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
from torch.utils.data import DataLoader

from tinyllm.data import StatefulDistributedSampler, ToyTokenDataset
from tinyllm.lineage import read_git_identity
from tinyllm.models.tinygpt import TinyGPT
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.checkpoint import (
    CheckpointContext,
    CheckpointError,
    CheckpointErrorCode,
)
from tinyllm.training.distributed import (
    all_gather_objects,
    distributed_barrier,
    rank_environment,
    reduced_mean,
    torchrun_environment,
)
from tinyllm.training.errors import TrainingError, TrainingErrorCode
from tinyllm.training.fsdp2 import (
    _apply_activation_checkpointing,
    _apply_fully_shard,
    _gradient_norm,
    _initialize_process_group,
    _require_finite_scalar_loss,
    _select_device,
    full_fsdp2_state_sha256,
)
from tinyllm.training.fsdp2_checkpoint import FSDP2CheckpointStore
from tinyllm.training.fsdp2_config import FSDP2RecoveryConfig, load_fsdp2_recovery_config
from tinyllm.training.fsdp2_recovery_schema import FSDP2RecoveryResult
from tinyllm.training.metrics import TrainerState, TrainingStepMetrics
from tinyllm.training.scheduler import build_adamw, build_warmup_cosine_scheduler
from tinyllm.training.seed import seed_everything


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


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


def _broadcast_rank_zero(value: object, *, rank: int) -> object:
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _environment_snapshot(
    config: FSDP2RecoveryConfig,
    ranks: list[object],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "strategy": "fsdp2",
        "backend": config.distributed.backend,
        "device_type": config.distributed.device_type,
        "world_size": config.distributed.world_size,
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "ranks": ranks,
    }


def _new_run(
    *,
    config_path: Path,
    config: FSDP2RecoveryConfig,
    output_root: Path,
    run_id: str,
    config_hash: str,
    dataset_version: str,
    git_commit: str,
    git_dirty: bool,
    environment: dict[str, object],
) -> Path:
    artifact_dir = (output_root / run_id).resolve()
    artifact_dir.mkdir(parents=False, exist_ok=False)
    for name in ("checkpoints", "failures", "evaluations", "exports"):
        (artifact_dir / name).mkdir()
    shutil.copyfile(config_path, artifact_dir / "config.original.yaml")
    _atomic_json(artifact_dir / "config.resolved.json", config.to_dict())
    _atomic_json(artifact_dir / "environment.json", environment)
    _atomic_json(
        artifact_dir / "hardware.json",
        {
            "schema_version": "1.0",
            "device_type": config.distributed.device_type,
            "world_size": config.distributed.world_size,
            "ranks": environment["ranks"],
        },
    )
    _atomic_json(
        artifact_dir / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "running",
            "strategy": "fsdp2",
            "world_size": config.distributed.world_size,
            "config_hash": config_hash,
            "dataset_version": dataset_version,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "checkpoint_status": "enabled_m4_2_dcp",
        },
    )
    _append_jsonl(
        artifact_dir / "events.jsonl",
        {"event": "fsdp2_recovery_run_started", "writer_rank": 0},
    )
    (artifact_dir / "metrics.jsonl").touch()
    return artifact_dir


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(
            CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
            f"cannot read {label}",
        ) from exc
    if not isinstance(raw, dict):
        raise CheckpointError(
            CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
            f"{label} must be a JSON object",
        )
    return cast(dict[str, object], raw)


def _validate_resume_run(
    *,
    artifact_dir: Path,
    config: FSDP2RecoveryConfig,
    config_hash: str,
    git_commit: str,
    environment: dict[str, object],
) -> tuple[str, str, bool]:
    run = _read_json_object(artifact_dir / "run.json", label="resume Run manifest")
    saved_environment = _read_json_object(
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
        raise CheckpointError(
            CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
            "resume Run manifest is incomplete",
        )
    if (
        run["config_hash"] != config_hash
        or run["git_commit"] != git_commit
        or run["world_size"] != config.distributed.world_size
        or run["strategy"] != "fsdp2"
        or saved_environment != environment
    ):
        raise CheckpointError(
            CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
            "resume Run lineage, World Size, or environment changed",
        )
    return str(run["run_id"]), str(run["dataset_version"]), bool(run["git_dirty"])


def _truncate_metrics(path: Path, *, checkpoint_step: int) -> tuple[int, int]:
    metrics: list[TrainingStepMetrics] = []
    total = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            metric = TrainingStepMetrics.model_validate_json(line)
            if metric.global_step <= checkpoint_step:
                metrics.append(metric)
    except (OSError, ValueError) as exc:
        raise CheckpointError(
            CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
            "Run metrics cannot be reconciled with the FSDP2 Checkpoint",
        ) from exc
    if [item.global_step for item in metrics] != list(range(1, checkpoint_step + 1)):
        raise CheckpointError(
            CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
            "Run metrics do not contain one canonical row per committed step",
        )
    _atomic_bytes(
        path,
        b"".join((json.dumps(item.to_dict(), sort_keys=True) + "\n").encode() for item in metrics),
    )
    return len(metrics), total - len(metrics)


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
        "strategy": "fsdp2",
        "world_size": world_size,
        "global_step": global_step,
        "config_hash": config_hash,
        "dataset_version": dataset_version,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "checkpoint_status": "valid_dcp",
        "latest_checkpoint": checkpoint_id,
    }
    if reason is not None:
        value["reason"] = reason
    return value


def _validate_stop(*, stop_after_step: int | None, current_step: int, max_steps: int) -> None:
    if stop_after_step is None:
        return
    if not current_step < stop_after_step < max_steps:
        raise ValueError("stop-after-step must be after the current point and before max_steps")


def run_fsdp2_recovery(
    *,
    config_path: Path,
    output_root: Path,
    resume_run: Path | None = None,
    stop_after_step: int | None = None,
) -> FSDP2RecoveryResult | None:
    """Run a fresh/interrupted/resumed FSDP2 phase under torchrun."""

    config_path = config_path.resolve()
    if not output_root.is_absolute():
        raise ValueError("FSDP2 output_root must be absolute")
    output_root = output_root.resolve()
    resume_run = resume_run.resolve() if resume_run is not None else None
    config = load_fsdp2_recovery_config(config_path)
    launch = torchrun_environment(os.environ)
    if launch.world_size != config.distributed.world_size:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID,
            "torchrun WORLD_SIZE does not match the FSDP2 recovery config",
        )
    device = _select_device(config, local_rank=launch.local_rank)
    _initialize_process_group(config, device=device)
    artifact_dir: Path | None = resume_run
    try:
        config_hash = canonical_config_hash(config)
        project_root = Path(__file__).resolve().parents[3]
        git_commit, current_git_dirty = read_git_identity(project_root)
        local_environment = rank_environment(launch, device)
        if device.type == "cpu":
            local_environment["physical_gpu_index"] = None
        rank_environments = all_gather_objects(local_environment, world_size=launch.world_size)
        environment = _environment_snapshot(config, rank_environments)

        identity: object = None
        if launch.rank == 0:
            try:
                if resume_run is None:
                    run_id = generate_run_id(config.run.name, config_hash, now=datetime.now(UTC))
                    dataset_version = f"toy-fsdp2-{config_hash[:8]}"
                    git_dirty = current_git_dirty
                    output_root.mkdir(parents=True, exist_ok=True)
                    artifact_dir = _new_run(
                        config_path=config_path,
                        config=config,
                        output_root=output_root,
                        run_id=run_id,
                        config_hash=config_hash,
                        dataset_version=dataset_version,
                        git_commit=git_commit,
                        git_dirty=git_dirty,
                        environment=environment,
                    )
                    mode: Literal["fresh", "exact_resume"] = "fresh"
                else:
                    run_id, dataset_version, git_dirty = _validate_resume_run(
                        artifact_dir=resume_run,
                        config=config,
                        config_hash=config_hash,
                        git_commit=git_commit,
                        environment=environment,
                    )
                    artifact_dir = resume_run
                    mode = "exact_resume"
                identity = {
                    "ok": True,
                    "run_id": run_id,
                    "dataset_version": dataset_version,
                    "git_dirty": git_dirty,
                    "artifact_dir": str(artifact_dir),
                    "mode": mode,
                }
            except Exception as exc:
                identity = {"ok": False, "error_type": type(exc).__name__}
        identity = _broadcast_rank_zero(identity, rank=launch.rank)
        if not isinstance(identity, dict) or not identity.get("ok"):
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
                "Rank zero rejected the FSDP2 Run identity",
            )
        run_id = str(identity["run_id"])
        dataset_version = str(identity["dataset_version"])
        git_dirty = bool(identity["git_dirty"])
        artifact_dir = Path(str(identity["artifact_dir"]))
        mode = cast(Literal["fresh", "exact_resume"], identity["mode"])
        distributed_barrier(device, launch)

        context = CheckpointContext(
            run_id=run_id,
            dataset_version=dataset_version,
            git_commit=git_commit,
            environment=environment,
            strategy="fsdp2",
            world_size=launch.world_size,
        )
        store = FSDP2CheckpointStore(
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
        _apply_activation_checkpointing(model, config=config)
        _apply_fully_shard(model, config=config)
        optimizer = build_adamw(model, config.training)
        scheduler = build_warmup_cosine_scheduler(optimizer, config.training)
        state = TrainerState()
        resumed_from_step: int | None = None

        if mode == "exact_resume":
            selection_status: object = None
            if launch.rank == 0:
                try:
                    selection = store.latest_valid(expected_world_size=launch.world_size)
                    progress = store.validate(
                        selection.checkpoint_id,
                        expected_world_size=launch.world_size,
                    )
                    retained, discarded = _truncate_metrics(
                        artifact_dir / "metrics.jsonl",
                        checkpoint_step=progress.global_step,
                    )
                    selection_status = {
                        "ok": True,
                        "checkpoint_id": selection.checkpoint_id,
                        "step": progress.global_step,
                        "skipped": selection.skipped_invalid_checkpoints,
                        "retained": retained,
                        "discarded": discarded,
                    }
                except Exception as exc:
                    selection_status = {"ok": False, "error_type": type(exc).__name__}
            selection_status = _broadcast_rank_zero(selection_status, rank=launch.rank)
            if not isinstance(selection_status, dict) or not selection_status.get("ok"):
                raise CheckpointError(
                    CheckpointErrorCode.CHECKPOINT_NO_VALID,
                    "Rank zero could not select a compatible FSDP2 Checkpoint",
                )
            checkpoint_id = str(selection_status["checkpoint_id"])
            resumed_from_step = int(selection_status["step"])
            state = store.restore(
                checkpoint_id=checkpoint_id,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                device=device,
                config=config,
                context=context,
                rank=launch.rank,
            )
            if launch.rank == 0:
                _append_jsonl(
                    artifact_dir / "events.jsonl",
                    {
                        "event": "fsdp2_exact_resume_applied",
                        "checkpoint_id": checkpoint_id,
                        "global_step": resumed_from_step,
                        "skipped_invalid_checkpoints": selection_status["skipped"],
                        "retained_metric_records": selection_status["retained"],
                        "discarded_metric_records": selection_status["discarded"],
                    },
                )

        _validate_stop(
            stop_after_step=stop_after_step,
            current_step=state.global_step,
            max_steps=config.training.max_steps,
        )
        iterator = iter(dataloader)
        autocast_dtype = torch.bfloat16 if config.precision.dtype == "bf16" else None
        latest_checkpoint = f"checkpoint-step-{state.global_step:08d}" if state.global_step else ""
        latest_hash = ""

        while state.global_step < config.training.max_steps:
            batch = next(iterator).to(device)
            optimizer.zero_grad(set_to_none=True)
            next_step = state.global_step + 1
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                output = model(batch, labels=batch)
            loss = _require_finite_scalar_loss(output.loss, global_step=next_step)
            torch.autograd.backward(loss)
            gradient_norm = _gradient_norm(model, max_norm=config.training.max_grad_norm)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            state = TrainerState(
                global_step=next_step,
                micro_step=next_step,
                epoch=sampler.epoch,
                tokens_seen=(
                    next_step * config.global_batch_size * (config.data.sequence_length - 1)
                ),
            )
            local_loss = float(loss.detach().float().item())
            losses = [
                cast(float, item)
                for item in all_gather_objects(local_loss, world_size=launch.world_size)
            ]
            reduced_loss = reduced_mean(
                local_loss,
                device=device,
                world_size=launch.world_size,
            )
            if abs(reduced_loss - sum(losses) / launch.world_size) > 1e-12:
                raise TrainingError(
                    TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
                    "FSDP2 reduced loss differs from the Rank mean",
                )
            norms = [
                cast(float, item)
                for item in all_gather_objects(gradient_norm, world_size=launch.world_size)
            ]
            if launch.rank == 0:
                _append_jsonl(
                    artifact_dir / "metrics.jsonl",
                    TrainingStepMetrics(
                        global_step=next_step,
                        micro_step=next_step,
                        epoch=sampler.epoch,
                        loss=reduced_loss,
                        learning_rate=learning_rate,
                        gradient_norm=sum(norms) / launch.world_size,
                        gradient_clipped=any(
                            item > config.training.max_grad_norm for item in norms
                        ),
                        tokens_seen=state.tokens_seen,
                    ).to_dict(),
                )

            interruption = stop_after_step == next_step
            final = next_step == config.training.max_steps
            if next_step % config.checkpoint.save_steps == 0 or interruption or final:
                pin_reason: Literal["interruption", "final"] | None = None
                if interruption:
                    pin_reason = "interruption"
                elif final:
                    pin_reason = "final"
                manifest = store.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    trainer_state=state,
                    sampler=sampler,
                    device=device,
                    config=config,
                    context=context,
                    rank=launch.rank,
                    pin_reason=pin_reason,
                )
                latest_checkpoint = manifest.checkpoint_id
                latest_hash = full_fsdp2_state_sha256(model)
                if launch.rank == 0:
                    _append_jsonl(
                        artifact_dir / "events.jsonl",
                        {
                            "event": "fsdp2_dcp_checkpoint_committed",
                            "checkpoint_id": latest_checkpoint,
                            "global_step": next_step,
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
                            global_step=next_step,
                            checkpoint_id=latest_checkpoint,
                            reason="coordinated_stop",
                        ),
                    )
                distributed_barrier(device, launch)
                return (
                    FSDP2RecoveryResult(
                        status="interrupted",
                        mode=mode,
                        run_id=run_id,
                        artifact_dir=artifact_dir,
                        config_sha256=config_hash,
                        git_commit=git_commit,
                        git_dirty=git_dirty,
                        backend=config.distributed.backend,
                        device_type=config.distributed.device_type,
                        world_size=launch.world_size,
                        global_step=next_step,
                        checkpoint_id=latest_checkpoint,
                        model_parameter_sha256=latest_hash,
                        resumed_from_step=resumed_from_step,
                        durable_metric_records=next_step,
                    )
                    if launch.rank == 0
                    else None
                )

        if not latest_checkpoint or not latest_hash:
            raise RuntimeError("final FSDP2 Checkpoint was not committed")
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
                    global_step=state.global_step,
                    checkpoint_id=latest_checkpoint,
                ),
            )
            _append_jsonl(
                artifact_dir / "events.jsonl",
                {
                    "event": "fsdp2_recovery_run_succeeded",
                    "global_step": state.global_step,
                    "checkpoint_id": latest_checkpoint,
                },
            )
        distributed_barrier(device, launch)
        return (
            FSDP2RecoveryResult(
                status="succeeded",
                mode=mode,
                run_id=run_id,
                artifact_dir=artifact_dir,
                config_sha256=config_hash,
                git_commit=git_commit,
                git_dirty=git_dirty,
                backend=config.distributed.backend,
                device_type=config.distributed.device_type,
                world_size=launch.world_size,
                global_step=state.global_step,
                checkpoint_id=latest_checkpoint,
                model_parameter_sha256=latest_hash,
                resumed_from_step=resumed_from_step,
                durable_metric_records=state.global_step,
            )
            if launch.rank == 0
            else None
        )
    except Exception as exc:
        if launch.rank == 0 and artifact_dir is not None:
            _append_jsonl(
                artifact_dir / "events.jsonl",
                {"event": "fsdp2_recovery_phase_failed", "error_type": type(exc).__name__},
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
