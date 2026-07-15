"""Bounded DDP correctness run used by the M3.1 torchrun gate."""

from __future__ import annotations

import json
import os
import platform
import shutil
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from tinyllm.data import ToyTokenDataset
from tinyllm.lineage import read_git_identity
from tinyllm.models.tinygpt import TinyGPT
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.config import M1TrainingConfig, load_training_config
from tinyllm.training.ddp_schema import DDPCorrectnessSummary, DDPTrainingResult
from tinyllm.training.distributed import (
    TorchrunEnvironment,
    all_gather_objects,
    model_state_sha256,
    reduced_mean,
    torchrun_environment,
    validate_sampler_partitions,
)
from tinyllm.training.errors import TrainingError, TrainingErrorCode
from tinyllm.training.metrics import TrainingStepMetrics
from tinyllm.training.scheduler import build_adamw, build_warmup_cosine_scheduler
from tinyllm.training.seed import seed_everything
from tinyllm.training.trainer import SingleDeviceTrainer


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


def _physical_gpu_index(launch: TorchrunEnvironment) -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return launch.local_rank
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if launch.local_rank >= len(entries) or not entries[launch.local_rank].isdigit():
        return None
    return int(entries[launch.local_rank])


def _select_device(config: M1TrainingConfig, launch: TorchrunEnvironment) -> torch.device:
    backend = config.distributed.backend
    if backend == "gloo":
        if config.precision.dtype != "fp32":
            raise TrainingError(
                TrainingErrorCode.UNSUPPORTED_PRECISION,
                "gloo DDP correctness requires fp32",
            )
        return torch.device("cpu")
    if backend != "nccl":
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID,
            "DDP backend must be gloo or nccl",
        )
    if not torch.cuda.is_available() or launch.local_rank >= torch.cuda.device_count():
        raise TrainingError(
            TrainingErrorCode.ACCELERATOR_UNAVAILABLE,
            "local CUDA rank is unavailable",
            context={"local_rank": launch.local_rank},
        )
    torch.cuda.set_device(launch.local_rank)
    if config.precision.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise TrainingError(
            TrainingErrorCode.UNSUPPORTED_PRECISION,
            "NCCL BF16 correctness requires BF16-capable visible GPUs",
        )
    torch.backends.cuda.matmul.allow_tf32 = config.precision.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.precision.allow_tf32
    return torch.device("cuda", launch.local_rank)


def _rank_environment(launch: TorchrunEnvironment, device: torch.device) -> dict[str, object]:
    result: dict[str, object] = {
        "rank": launch.rank,
        "local_rank": launch.local_rank,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "physical_gpu_index": _physical_gpu_index(launch),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu_name": properties.name,
                "memory_total_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return result


def _barrier(device: torch.device, launch: TorchrunEnvironment) -> None:
    """Bind NCCL barriers to the already selected local CUDA device."""

    if device.type == "cuda":
        dist.barrier(device_ids=[launch.local_rank])
    else:
        dist.barrier()


def _new_rank_zero_run(
    *,
    config_path: Path,
    config: M1TrainingConfig,
    output_root: Path,
    run_id: str,
    config_hash: str,
    git_commit: str,
    git_dirty: bool,
    rank_environments: list[object],
) -> Path:
    artifact_dir = output_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "evaluations", "exports"):
        (artifact_dir / name).mkdir()
    shutil.copyfile(config_path, artifact_dir / "config.original.yaml")
    _atomic_json(artifact_dir / "config.resolved.json", config.to_dict())
    _atomic_json(
        artifact_dir / "environment.json",
        {
            "schema_version": "1.0",
            "backend": config.distributed.backend,
            "world_size": config.distributed.world_size,
            "ranks": rank_environments,
        },
    )
    _atomic_json(
        artifact_dir / "hardware.json",
        {
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
                for item in rank_environments
            ],
        },
    )
    _atomic_json(
        artifact_dir / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "running",
            "strategy": "ddp",
            "world_size": config.distributed.world_size,
            "config_hash": config_hash,
            "dataset_version": f"toy-ddp-{config_hash[:8]}",
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "checkpoint_status": "not_evaluated_m3_1",
        },
    )
    _append_jsonl(
        artifact_dir / "events.jsonl",
        {
            "event": "ddp_run_started",
            "run_id": run_id,
            "world_size": config.distributed.world_size,
            "backend": config.distributed.backend,
        },
    )
    (artifact_dir / "metrics.jsonl").touch()
    return artifact_dir


def _require_identical_hashes(hashes: list[object], *, stage: str) -> str:
    values = [str(value) for value in hashes]
    if len(set(values)) != 1:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
            f"model parameters differ across ranks at {stage}",
            context={"stage": stage, "unique_hashes": len(set(values))},
        )
    return values[0]


def run_ddp_correctness(
    *,
    config_path: Path,
    output_root: Path,
) -> DDPTrainingResult | None:
    """Execute a bounded DDP run; only rank zero returns and writes durable artifacts."""

    config_path = config_path.resolve()
    output_root = output_root.resolve()
    config = load_training_config(config_path)
    if config.distributed.strategy != "ddp" or config.distributed.backend is None:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_LAUNCH_REQUIRED,
            "M3 DDP worker requires distributed.strategy=ddp and an explicit backend",
        )
    launch = torchrun_environment(os.environ)
    if launch.world_size != config.distributed.world_size:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID,
            "torchrun WORLD_SIZE does not match the resolved config",
            context={
                "actual_world_size": launch.world_size,
                "expected_world_size": config.distributed.world_size,
            },
        )
    device = _select_device(config, launch)
    if device.type == "cuda":
        dist.init_process_group(
            backend=config.distributed.backend,
            init_method="env://",
            timeout=timedelta(seconds=config.distributed.timeout_seconds),
            device_id=device,
        )
    else:
        dist.init_process_group(
            backend=config.distributed.backend,
            init_method="env://",
            timeout=timedelta(seconds=config.distributed.timeout_seconds),
        )
    artifact_dir: Path | None = None
    try:
        config_hash = canonical_config_hash(config)
        git_commit, git_dirty = read_git_identity(config_path.parent)
        run_id_values: list[object] = [
            generate_run_id(config.run.name, config_hash, now=datetime.now(UTC))
            if launch.rank == 0
            else None
        ]
        dist.broadcast_object_list(run_id_values, src=0)
        run_id = str(run_id_values[0])

        rank_environment = _rank_environment(launch, device)
        gathered_environments = all_gather_objects(
            rank_environment,
            world_size=launch.world_size,
        )
        if launch.rank == 0:
            output_root.mkdir(parents=True, exist_ok=True)
            artifact_dir = _new_rank_zero_run(
                config_path=config_path,
                config=config,
                output_root=output_root,
                run_id=run_id,
                config_hash=config_hash,
                git_commit=git_commit,
                git_dirty=git_dirty,
                rank_environments=gathered_environments,
            )
        _barrier(device, launch)

        seed_everything(
            config.run.seed,
            deterministic_algorithms=device.type == "cpu",
        )
        dataset = ToyTokenDataset(
            vocab_size=config.data.vocab_size,
            sequence_length=config.data.sequence_length,
            num_samples=config.data.num_samples,
            seed=config.run.seed,
        )
        sampler: DistributedSampler[int] = DistributedSampler(
            dataset,
            num_replicas=launch.world_size,
            rank=launch.rank,
            shuffle=True,
            seed=config.run.seed,
            drop_last=True,
        )
        sampler.set_epoch(0)
        local_indices = tuple(iter(sampler))
        raw_partitions = all_gather_objects(local_indices, world_size=launch.world_size)
        partitions = validate_sampler_partitions(
            [cast(tuple[int, ...], item) for item in raw_partitions],
            num_samples=len(dataset),
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
        initial_hash = _require_identical_hashes(
            all_gather_objects(model_state_sha256(model), world_size=launch.world_size),
            stage="initialization",
        )
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
            autocast_dtype=torch.bfloat16 if config.precision.dtype == "bf16" else None,
        )

        max_loss_diff = 0.0
        max_gradient_norm_diff = 0.0
        durable_metrics = 0
        while trainer.state.global_step < config.training.max_steps:
            result = trainer.train(target_global_step=trainer.state.global_step + 1)
            local_metric = result.metrics[0]
            local_losses = [
                cast(float, value)
                for value in all_gather_objects(local_metric.loss, world_size=launch.world_size)
            ]
            reduced_loss = reduced_mean(
                local_metric.loss,
                device=device,
                world_size=launch.world_size,
            )
            expected_loss = sum(local_losses) / launch.world_size
            max_loss_diff = max(max_loss_diff, abs(reduced_loss - expected_loss))
            gradient_norms = [
                cast(float, value)
                for value in all_gather_objects(
                    local_metric.gradient_norm,
                    world_size=launch.world_size,
                )
            ]
            max_gradient_norm_diff = max(
                max_gradient_norm_diff,
                max(gradient_norms) - min(gradient_norms),
            )
            if launch.rank == 0:
                if artifact_dir is None:
                    raise RuntimeError("rank-zero Artifact directory is missing")
                metric = TrainingStepMetrics(
                    global_step=local_metric.global_step,
                    micro_step=local_metric.micro_step,
                    epoch=local_metric.epoch,
                    loss=reduced_loss,
                    learning_rate=local_metric.learning_rate,
                    gradient_norm=sum(gradient_norms) / launch.world_size,
                    gradient_clipped=any(
                        bool(value)
                        for value in all_gather_objects(
                            local_metric.gradient_clipped,
                            world_size=launch.world_size,
                        )
                    ),
                    tokens_seen=local_metric.tokens_seen * launch.world_size,
                )
                _append_jsonl(artifact_dir / "metrics.jsonl", metric.to_dict())
                durable_metrics += 1
            else:
                all_gather_objects(
                    local_metric.gradient_clipped,
                    world_size=launch.world_size,
                )

        final_hash = _require_identical_hashes(
            all_gather_objects(model_state_sha256(model), world_size=launch.world_size),
            stage="final optimizer step",
        )
        step_counts = [
            cast(int, value)
            for value in all_gather_objects(
                trainer.state.global_step,
                world_size=launch.world_size,
            )
        ]
        if len(set(step_counts)) != 1 or step_counts[0] != config.training.max_steps:
            raise TrainingError(
                TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
                "optimizer-step progress differs across ranks",
                context={"unique_step_counts": len(set(step_counts))},
            )

        result_value: DDPTrainingResult | None = None
        if launch.rank == 0:
            if artifact_dir is None:
                raise RuntimeError("rank-zero Artifact directory is missing")
            summary = DDPCorrectnessSummary(
                backend=config.distributed.backend,
                world_size=launch.world_size,
                global_batch_size=config.global_batch_size,
                optimizer_steps=config.training.max_steps,
                durable_metric_records=durable_metrics,
                model_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                initial_parameter_sha256=initial_hash,
                final_parameter_sha256=final_hash,
                sampler_num_samples=len(dataset),
                sampler_union_samples=sum(item.sample_count for item in partitions),
                sampler_no_overlap=True,
                partitions=partitions,
                max_loss_reduction_abs_diff=max_loss_diff,
                max_gradient_norm_abs_diff=max_gradient_norm_diff,
            )
            _atomic_json(artifact_dir / "correctness.json", summary.to_dict())
            _atomic_json(
                artifact_dir / "run.json",
                {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "status": "succeeded",
                    "strategy": "ddp",
                    "world_size": launch.world_size,
                    "global_step": config.training.max_steps,
                    "config_hash": config_hash,
                    "dataset_version": f"toy-ddp-{config_hash[:8]}",
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "checkpoint_status": "not_evaluated_m3_1",
                    "correctness_status": "pass",
                },
            )
            _append_jsonl(
                artifact_dir / "events.jsonl",
                {
                    "event": "ddp_run_succeeded",
                    "global_step": config.training.max_steps,
                    "metrics": durable_metrics,
                    "writer_rank": 0,
                },
            )
            result_value = DDPTrainingResult(
                run_id=run_id,
                artifact_dir=artifact_dir,
                config_sha256=config_hash,
                git_commit=git_commit,
                git_dirty=git_dirty,
                summary=summary,
            )
        _barrier(device, launch)
        return result_value
    except Exception as exc:
        if launch.rank == 0 and artifact_dir is not None:
            _atomic_json(
                artifact_dir / "run.json",
                {
                    "schema_version": "1.0",
                    "status": "failed",
                    "strategy": "ddp",
                    "world_size": launch.world_size,
                    "error_type": type(exc).__name__,
                },
            )
            _append_jsonl(
                artifact_dir / "events.jsonl",
                {"event": "ddp_run_failed", "error_type": type(exc).__name__},
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
