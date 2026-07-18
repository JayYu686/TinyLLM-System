"""Formal four-GPU Qwen3-8B FSDP2 Probe, training, DCP Resume, and export."""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader

from tinyllm.data import StatefulDistributedSampler
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.checkpoint import (
    CheckpointContext,
    CheckpointError,
    CheckpointErrorCode,
)
from tinyllm.training.distributed import (
    TorchrunEnvironment,
    all_gather_objects,
    distributed_barrier,
    rank_environment,
    reduced_mean,
    torchrun_environment,
)
from tinyllm.training.errors import TrainingError, TrainingErrorCode
from tinyllm.training.fsdp2 import (
    _gradient_norm,
    _require_finite_scalar_loss,
    full_fsdp2_state_sha256,
)
from tinyllm.training.fsdp2_checkpoint import FSDP2CheckpointStore
from tinyllm.training.fsdp2_recovery import (
    _append_jsonl,
    _atomic_json,
    _broadcast_rank_zero,
    _read_json_object,
    _truncate_metrics,
)
from tinyllm.training.m4_dataset import M4DatasetViewManifest, M4RegisteredDatasetView
from tinyllm.training.m4_export import export_full_safetensors
from tinyllm.training.m4_model import (
    apply_qwen_activation_checkpointing,
    apply_qwen_fully_shard,
    inspect_qwen3_8b_artifact,
    load_qwen3_8b,
)
from tinyllm.training.m4_model_schema import M4ModelArtifactManifest
from tinyllm.training.m4_qwen_config import M4QwenFSDP2Config, load_m4_qwen_config
from tinyllm.training.m4_qwen_schema import M4QwenRankMemory, M4QwenRunResult
from tinyllm.training.metrics import TrainerState, TrainingStepMetrics
from tinyllm.training.scheduler import build_adamw, build_warmup_cosine_scheduler
from tinyllm.training.seed import seed_everything


def _initialize_cuda_process_group(
    config: M4QwenFSDP2Config,
) -> tuple[TorchrunEnvironment, torch.device]:
    launch = torchrun_environment(os.environ)
    if launch.world_size != 4 or launch.world_size != config.distributed.world_size:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID,
            "formal M4 Qwen run requires torchrun World Size 4",
        )
    if not torch.cuda.is_available() or launch.local_rank >= torch.cuda.device_count():
        raise TrainingError(
            TrainingErrorCode.ACCELERATOR_UNAVAILABLE,
            "formal M4 Qwen Rank cannot select its local CUDA device",
        )
    torch.cuda.set_device(launch.local_rank)
    if not torch.cuda.is_bf16_supported():
        raise TrainingError(
            TrainingErrorCode.UNSUPPORTED_PRECISION,
            "formal M4 Qwen run requires BF16-capable GPUs",
        )
    torch.backends.cuda.matmul.allow_tf32 = config.precision.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.precision.allow_tf32
    device = torch.device("cuda", launch.local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=config.distributed.timeout_seconds),
        device_id=device,
    )
    return launch, device


def _new_run(
    *,
    config_path: Path,
    config: M4QwenFSDP2Config,
    output_root: Path,
    run_id: str,
    config_hash: str,
    git_commit: str,
    git_dirty: bool,
    environment: dict[str, object],
    model_artifact: M4ModelArtifactManifest,
    data_view: M4DatasetViewManifest,
    probe_only: bool,
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
            "world_size": 4,
            "device_type": "cuda",
            "ranks": environment["ranks"],
        },
    )
    _atomic_json(artifact_dir / "model_artifact.json", model_artifact.to_dict())
    _atomic_json(artifact_dir / "data_view.json", data_view.to_dict())
    _atomic_json(
        artifact_dir / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "running",
            "mode": "probe" if probe_only else "formal",
            "strategy": "fsdp2",
            "world_size": 4,
            "config_hash": config_hash,
            "dataset_version": data_view.view_version,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "checkpoint_status": "not_applicable_probe" if probe_only else "enabled_dcp",
        },
    )
    _append_jsonl(
        artifact_dir / "events.jsonl",
        {"event": "m4_qwen_run_started", "probe_only": probe_only, "writer_rank": 0},
    )
    (artifact_dir / "metrics.jsonl").touch()
    return artifact_dir


def _validate_resume_identity(
    *,
    artifact_dir: Path,
    config_hash: str,
    git_commit: str,
    environment: dict[str, object],
    dataset_version: str,
) -> tuple[str, bool]:
    run = _read_json_object(artifact_dir / "run.json", label="M4 resume Run manifest")
    saved_environment = _read_json_object(
        artifact_dir / "environment.json",
        label="M4 resume environment snapshot",
    )
    if (
        run.get("config_hash") != config_hash
        or run.get("git_commit") != git_commit
        or run.get("world_size") != 4
        or run.get("strategy") != "fsdp2"
        or run.get("dataset_version") != dataset_version
        or saved_environment != environment
        or not isinstance(run.get("run_id"), str)
        or not isinstance(run.get("git_dirty"), bool)
    ):
        raise CheckpointError(
            CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
            "M4 Run lineage, data view, World Size, physical GPUs, or environment changed",
        )
    return str(run["run_id"]), bool(run["git_dirty"])


def _memory_evidence(
    *,
    launch: TorchrunEnvironment,
    device: torch.device,
    physical_gpu_index: int,
) -> tuple[M4QwenRankMemory, M4QwenRankMemory, M4QwenRankMemory, M4QwenRankMemory]:
    rank = launch.rank
    world_size = launch.world_size
    local = M4QwenRankMemory(
        rank=rank,
        physical_gpu_index=physical_gpu_index,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        final_allocated_bytes=torch.cuda.memory_allocated(device),
        final_reserved_bytes=torch.cuda.memory_reserved(device),
    )
    gathered = tuple(
        M4QwenRankMemory.model_validate(item)
        for item in all_gather_objects(local.to_dict(), world_size=world_size)
    )
    if len(gathered) != 4:
        raise RuntimeError("formal M4 memory evidence requires four Ranks")
    return gathered


def _run_status(
    *,
    run_id: str,
    status: str,
    config_hash: str,
    dataset_version: str,
    git_commit: str,
    git_dirty: bool,
    global_step: int,
    checkpoint_id: str | None,
    reason: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": status,
        "strategy": "fsdp2",
        "world_size": 4,
        "global_step": global_step,
        "config_hash": config_hash,
        "dataset_version": dataset_version,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "checkpoint_status": "valid_dcp" if checkpoint_id else "not_applicable_probe",
        "latest_checkpoint": checkpoint_id,
    }
    if reason is not None:
        value["reason"] = reason
    return value


def run_m4_qwen(
    *,
    config_path: Path,
    artifact_root: Path,
    model_dir: Path,
    output_root: Path,
    resume_run: Path | None = None,
    stop_after_step: int | None = None,
    probe_only: bool = False,
    export_final: bool = False,
) -> M4QwenRunResult | None:
    """Run the formal one-Step Probe or Step-25-to-50 acceptance workflow."""

    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    model_dir = model_dir.resolve()
    output_root = output_root.resolve()
    resume_run = resume_run.resolve() if resume_run is not None else None
    config = load_m4_qwen_config(config_path)
    if probe_only and (resume_run is not None or stop_after_step is not None or export_final):
        raise ValueError("Probe cannot Resume, stop at a Checkpoint, or export")
    if not probe_only and stop_after_step not in {None, 25}:
        raise ValueError("formal M4 coordinated stop is fixed at Step 25")
    launch, device = _initialize_cuda_process_group(config)
    rank = launch.rank
    world_size = launch.world_size
    torch.cuda.reset_peak_memory_stats(device)
    artifact_dir: Path | None = resume_run
    try:
        config_hash = canonical_config_hash(config)
        project_root = Path(__file__).resolve().parents[3]
        git_commit, current_git_dirty = read_git_identity(project_root)
        model_status: object = None
        if rank == 0:
            try:
                model_artifact = inspect_qwen3_8b_artifact(model_dir=model_dir, config=config)
                model_status = {"ok": True, "value": model_artifact.model_dump_json()}
            except Exception as exc:
                model_status = {"ok": False, "error_type": type(exc).__name__}
        model_status = _broadcast_rank_zero(model_status, rank=rank)
        if not isinstance(model_status, dict) or not model_status.get("ok"):
            raise TrainingError(
                TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID,
                "Rank zero rejected the pinned Qwen model artifact",
            )
        model_artifact = M4ModelArtifactManifest.model_validate_json(str(model_status["value"]))

        data_view = M4RegisteredDatasetView(artifact_root=artifact_root, config=config.data)
        data_hashes = all_gather_objects(
            data_view.manifest.content_sha256,
            world_size=world_size,
        )
        if len({str(value) for value in data_hashes}) != 1:
            raise TrainingError(
                TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
                "M4 registered-data view differs across Ranks",
            )
        local_environment = rank_environment(launch, device)
        rank_environments = all_gather_objects(local_environment, world_size=world_size)
        physical_gpu_index = cast(int, local_environment["physical_gpu_index"])
        environment: dict[str, object] = {
            "schema_version": "1.0",
            "strategy": "fsdp2",
            "backend": "nccl",
            "device_type": "cuda",
            "world_size": 4,
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "model_artifact_sha256": model_artifact.content_sha256,
            "data_view_sha256": data_view.manifest.content_sha256,
            "ranks": rank_environments,
        }

        identity: object = None
        if rank == 0:
            try:
                if resume_run is None:
                    run_id = generate_run_id(config.run.name, config_hash, now=datetime.now(UTC))
                    git_dirty = current_git_dirty
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
                        model_artifact=model_artifact,
                        data_view=data_view.manifest,
                        probe_only=probe_only,
                    )
                    mode: Literal["probe", "fresh", "exact_resume"] = (
                        "probe" if probe_only else "fresh"
                    )
                else:
                    run_id, git_dirty = _validate_resume_identity(
                        artifact_dir=resume_run,
                        config_hash=config_hash,
                        git_commit=git_commit,
                        environment=environment,
                        dataset_version=data_view.manifest.view_version,
                    )
                    artifact_dir = resume_run
                    mode = "exact_resume"
                identity = {
                    "ok": True,
                    "run_id": run_id,
                    "git_dirty": git_dirty,
                    "artifact_dir": str(artifact_dir),
                    "mode": mode,
                }
            except Exception as exc:
                identity = {"ok": False, "error_type": type(exc).__name__}
        identity = _broadcast_rank_zero(identity, rank=rank)
        if not isinstance(identity, dict) or not identity.get("ok"):
            raise CheckpointError(
                CheckpointErrorCode.CHECKPOINT_INCOMPATIBLE,
                "Rank zero rejected the M4 Run identity",
            )
        run_id = str(identity["run_id"])
        git_dirty = bool(identity["git_dirty"])
        artifact_dir = Path(str(identity["artifact_dir"]))
        mode = cast(Literal["probe", "fresh", "exact_resume"], identity["mode"])
        distributed_barrier(device, launch)

        seed_everything(config.run.seed, deterministic_algorithms=False)
        sampler = StatefulDistributedSampler(
            data_view,
            num_replicas=world_size,
            rank=rank,
            seed=config.run.seed,
        )
        dataloader = DataLoader(
            data_view,
            batch_size=1,
            sampler=sampler,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )
        model = load_qwen3_8b(model_dir=model_dir, config=config, device=device)
        activation_layers = apply_qwen_activation_checkpointing(model, expected_layers=36)
        activation_layers_literal = cast(Literal[36], activation_layers)
        apply_qwen_fully_shard(model, config=config)
        optimizer = build_adamw(model, config.training)
        scheduler = build_warmup_cosine_scheduler(optimizer, config.training)
        context = CheckpointContext(
            run_id=run_id,
            dataset_version=data_view.manifest.view_version,
            git_commit=git_commit,
            environment=environment,
            strategy="fsdp2",
            world_size=4,
        )
        store = FSDP2CheckpointStore(
            artifact_dir / "checkpoints",
            keep_last=config.checkpoint.keep_last,
        )
        state = TrainerState()
        resumed_from_step: int | None = None

        if mode == "exact_resume":
            selection_status: object = None
            if rank == 0:
                try:
                    selection = store.latest_valid(expected_world_size=4)
                    progress = store.validate(selection.checkpoint_id, expected_world_size=4)
                    retained, discarded = _truncate_metrics(
                        artifact_dir / "metrics.jsonl",
                        checkpoint_step=progress.global_step,
                    )
                    selection_status = {
                        "ok": True,
                        "checkpoint_id": selection.checkpoint_id,
                        "step": progress.global_step,
                        "retained": retained,
                        "discarded": discarded,
                        "skipped": selection.skipped_invalid_checkpoints,
                    }
                except Exception as exc:
                    selection_status = {"ok": False, "error_type": type(exc).__name__}
            selection_status = _broadcast_rank_zero(selection_status, rank=rank)
            if not isinstance(selection_status, dict) or not selection_status.get("ok"):
                raise CheckpointError(
                    CheckpointErrorCode.CHECKPOINT_NO_VALID,
                    "Rank zero could not select the M4 DCP Checkpoint",
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
                rank=rank,
            )
            if rank == 0:
                _append_jsonl(
                    artifact_dir / "events.jsonl",
                    {
                        "event": "m4_qwen_exact_resume_applied",
                        "checkpoint_id": checkpoint_id,
                        "global_step": resumed_from_step,
                        "retained_metric_records": selection_status["retained"],
                        "discarded_metric_records": selection_status["discarded"],
                        "skipped_invalid_checkpoints": selection_status["skipped"],
                    },
                )

        if stop_after_step is not None and state.global_step >= stop_after_step:
            raise ValueError("formal stop Step must be after the selected Resume Checkpoint")
        iterator = iter(dataloader)
        target_step = 1 if probe_only else config.training.max_steps
        latest_checkpoint: str | None = None
        latest_hash: str | None = None

        while state.global_step < target_step:
            batch = {
                key: value.to(device, non_blocking=True) for key, value in next(iterator).items()
            }
            optimizer.zero_grad(set_to_none=True)
            next_step = state.global_step + 1
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch, use_cache=False)
            loss = _require_finite_scalar_loss(getattr(output, "loss", None), global_step=next_step)
            torch.autograd.backward(loss)
            gradient_norm = _gradient_norm(model, max_norm=config.training.max_grad_norm)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            state = TrainerState(
                global_step=next_step,
                micro_step=next_step,
                epoch=sampler.epoch,
                tokens_seen=next_step * config.global_batch_size * config.data.sequence_length,
            )
            local_loss = float(loss.detach().float().item())
            losses = [
                cast(float, item) for item in all_gather_objects(local_loss, world_size=world_size)
            ]
            reduced_loss = reduced_mean(local_loss, device=device, world_size=world_size)
            if abs(reduced_loss - sum(losses) / world_size) > 1e-12:
                raise TrainingError(
                    TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
                    "M4 Qwen reduced loss differs from the Rank mean",
                )
            norms = [
                cast(float, item)
                for item in all_gather_objects(gradient_norm, world_size=world_size)
            ]
            if rank == 0:
                _append_jsonl(
                    artifact_dir / "metrics.jsonl",
                    TrainingStepMetrics(
                        global_step=next_step,
                        micro_step=next_step,
                        epoch=sampler.epoch,
                        loss=reduced_loss,
                        learning_rate=learning_rate,
                        gradient_norm=sum(norms) / world_size,
                        gradient_clipped=any(
                            value > config.training.max_grad_norm for value in norms
                        ),
                        tokens_seen=state.tokens_seen,
                    ).to_dict(),
                )

            interruption = stop_after_step == next_step
            final = not probe_only and next_step == config.training.max_steps
            if not probe_only and (next_step == 25 or final):
                manifest = store.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    trainer_state=state,
                    sampler=sampler,
                    device=device,
                    config=config,
                    context=context,
                    rank=rank,
                    pin_reason="interruption" if interruption else "final" if final else None,
                )
                latest_checkpoint = manifest.checkpoint_id
                latest_hash = full_fsdp2_state_sha256(model)
                if rank == 0:
                    _append_jsonl(
                        artifact_dir / "events.jsonl",
                        {
                            "event": "m4_qwen_dcp_checkpoint_committed",
                            "checkpoint_id": latest_checkpoint,
                            "global_step": next_step,
                        },
                    )

            if interruption:
                if latest_checkpoint is None or latest_hash is None:
                    raise RuntimeError("Step-25 interruption has no valid DCP Checkpoint")
                memory = _memory_evidence(
                    launch=launch,
                    device=device,
                    physical_gpu_index=physical_gpu_index,
                )
                if rank == 0:
                    _atomic_json(
                        artifact_dir / "run.json",
                        _run_status(
                            run_id=run_id,
                            status="interrupted",
                            config_hash=config_hash,
                            dataset_version=data_view.manifest.view_version,
                            git_commit=git_commit,
                            git_dirty=git_dirty,
                            global_step=next_step,
                            checkpoint_id=latest_checkpoint,
                            reason="coordinated_stop_at_step_25",
                        ),
                    )
                distributed_barrier(device, launch)
                return (
                    M4QwenRunResult(
                        status="interrupted",
                        mode=mode,
                        run_id=run_id,
                        artifact_dir=artifact_dir,
                        config_sha256=config_hash,
                        git_commit=git_commit,
                        git_dirty=git_dirty,
                        model_artifact_sha256=model_artifact.content_sha256,
                        data_view_sha256=data_view.manifest.content_sha256,
                        world_size=4,
                        global_step=next_step,
                        checkpoint_id=latest_checkpoint,
                        model_parameter_sha256=latest_hash,
                        resumed_from_step=resumed_from_step,
                        durable_metric_records=next_step,
                        activation_checkpointed_layers=activation_layers_literal,
                        rank_memory=memory,
                    )
                    if rank == 0
                    else None
                )

        memory = _memory_evidence(
            launch=launch,
            device=device,
            physical_gpu_index=physical_gpu_index,
        )
        if probe_only:
            if rank == 0:
                _atomic_json(
                    artifact_dir / "memory_probe.json",
                    {
                        "schema_version": "1.0",
                        "status": "pass",
                        "ranks": [item.to_dict() for item in memory],
                    },
                )
                _atomic_json(
                    artifact_dir / "run.json",
                    _run_status(
                        run_id=run_id,
                        status="probe_succeeded",
                        config_hash=config_hash,
                        dataset_version=data_view.manifest.view_version,
                        git_commit=git_commit,
                        git_dirty=git_dirty,
                        global_step=1,
                        checkpoint_id=None,
                    ),
                )
            distributed_barrier(device, launch)
            return (
                M4QwenRunResult(
                    status="probe_succeeded",
                    mode="probe",
                    run_id=run_id,
                    artifact_dir=artifact_dir,
                    config_sha256=config_hash,
                    git_commit=git_commit,
                    git_dirty=git_dirty,
                    model_artifact_sha256=model_artifact.content_sha256,
                    data_view_sha256=data_view.manifest.content_sha256,
                    world_size=4,
                    global_step=1,
                    durable_metric_records=1,
                    activation_checkpointed_layers=activation_layers_literal,
                    rank_memory=memory,
                )
                if rank == 0
                else None
            )

        if latest_checkpoint is None or latest_hash is None:
            raise RuntimeError("formal M4 final Checkpoint was not committed")
        export_sha256 = None
        if export_final:
            export_sha256 = export_full_safetensors(
                model=model,
                export_dir=artifact_dir / "exports" / "safetensors",
                source_model_dir=model_dir,
                rank=rank,
            )
        if rank == 0:
            _atomic_json(
                artifact_dir / "run.json",
                _run_status(
                    run_id=run_id,
                    status="succeeded",
                    config_hash=config_hash,
                    dataset_version=data_view.manifest.view_version,
                    git_commit=git_commit,
                    git_dirty=git_dirty,
                    global_step=state.global_step,
                    checkpoint_id=latest_checkpoint,
                ),
            )
            _append_jsonl(
                artifact_dir / "events.jsonl",
                {
                    "event": "m4_qwen_run_succeeded",
                    "global_step": state.global_step,
                    "checkpoint_id": latest_checkpoint,
                    "export_sha256": export_sha256,
                },
            )
        distributed_barrier(device, launch)
        return (
            M4QwenRunResult(
                status="succeeded",
                mode=mode,
                run_id=run_id,
                artifact_dir=artifact_dir,
                config_sha256=config_hash,
                git_commit=git_commit,
                git_dirty=git_dirty,
                model_artifact_sha256=model_artifact.content_sha256,
                data_view_sha256=data_view.manifest.content_sha256,
                world_size=4,
                global_step=state.global_step,
                checkpoint_id=latest_checkpoint,
                model_parameter_sha256=latest_hash,
                resumed_from_step=resumed_from_step,
                durable_metric_records=state.global_step,
                activation_checkpointed_layers=activation_layers_literal,
                rank_memory=memory,
                export_sha256=export_sha256,
            )
            if rank == 0
            else None
        )
    except Exception as exc:
        if rank == 0 and artifact_dir is not None:
            _append_jsonl(
                artifact_dir / "events.jsonl",
                {"event": "m4_qwen_phase_failed", "error_type": type(exc).__name__},
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
