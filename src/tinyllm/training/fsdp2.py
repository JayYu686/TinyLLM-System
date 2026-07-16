"""Bounded FSDP2 correctness runtime used by the M4.1 gate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, DistributedSampler

from tinyllm.data import ToyTokenDataset
from tinyllm.lineage import read_git_identity
from tinyllm.models.tinygpt import TinyGPT
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.distributed import (
    all_gather_objects,
    distributed_barrier,
    model_state_sha256,
    rank_environment,
    reduced_mean,
    torchrun_environment,
    validate_sampler_partitions,
)
from tinyllm.training.errors import TrainingError, TrainingErrorCode
from tinyllm.training.fsdp2_config import FSDP2CorrectnessConfig, load_fsdp2_config
from tinyllm.training.fsdp2_schema import (
    FSDP2CorrectnessSummary,
    FSDP2RankEvidence,
    FSDP2TrainingResult,
)
from tinyllm.training.metrics import TrainingStepMetrics
from tinyllm.training.scheduler import build_adamw, build_warmup_cosine_scheduler
from tinyllm.training.seed import seed_everything


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


def _tensor_digest_update(digest: hashlib._Hash, name: str, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())


def full_fsdp2_state_sha256(model: nn.Module) -> str:
    """Collect every DTensor and hash the complete logical model identically on each Rank."""

    digest = hashlib.sha256()
    for name, raw_tensor in sorted(model.state_dict().items()):
        tensor = raw_tensor.full_tensor() if isinstance(raw_tensor, DTensor) else raw_tensor
        if not isinstance(tensor, Tensor):
            raise TrainingError(
                TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
                "FSDP2 state_dict contains a non-tensor value",
            )
        _tensor_digest_update(digest, name, tensor)
    return digest.hexdigest()


def local_fsdp2_shard_evidence(
    model: nn.Module,
    *,
    rank: int,
    device_type: Literal["cpu", "cuda"],
) -> FSDP2RankEvidence:
    """Hash one Rank's local DTensor shards and count their physical elements."""

    digest = hashlib.sha256()
    local_numel = 0
    parameters = tuple(model.named_parameters())
    if not parameters or any(not isinstance(parameter, DTensor) for _, parameter in parameters):
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
            "fully_shard did not expose DTensor parameters on every Rank",
        )
    for name, parameter in parameters:
        local = cast(DTensor, parameter).to_local()
        local_numel += local.numel()
        _tensor_digest_update(digest, name, local)
    return FSDP2RankEvidence(
        rank=rank,
        device_type=device_type,
        local_shard_numel=local_numel,
        local_shard_sha256=digest.hexdigest(),
    )


def _require_identical_hashes(values: list[object], *, stage: str) -> str:
    hashes = [str(value) for value in values]
    if len(set(hashes)) != 1:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
            f"FSDP2 full model state differs across Ranks during {stage}",
            context={"unique_hashes": len(set(hashes))},
        )
    return hashes[0]


def _select_device(config: FSDP2CorrectnessConfig, *, local_rank: int) -> torch.device:
    if config.distributed.device_type == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise TrainingError(
            TrainingErrorCode.ACCELERATOR_UNAVAILABLE,
            "local CUDA Rank is unavailable for FSDP2",
            context={"local_rank": local_rank},
        )
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise TrainingError(
            TrainingErrorCode.UNSUPPORTED_PRECISION,
            "FSDP2 BF16 correctness requires BF16-capable visible GPUs",
        )
    torch.backends.cuda.matmul.allow_tf32 = config.precision.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.precision.allow_tf32
    return torch.device("cuda", local_rank)


def _initialize_process_group(config: FSDP2CorrectnessConfig, *, device: torch.device) -> None:
    arguments: dict[str, object] = {
        "backend": config.distributed.backend,
        "init_method": "env://",
        "timeout": timedelta(seconds=config.distributed.timeout_seconds),
    }
    if device.type == "cuda":
        arguments["device_id"] = device
    dist.init_process_group(**arguments)  # type: ignore[arg-type]


def _mixed_precision_policy(config: FSDP2CorrectnessConfig) -> MixedPrecisionPolicy:
    if config.precision.dtype == "bf16":
        return MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.float32,
        )
    return MixedPrecisionPolicy()


def _apply_fully_shard(
    model: TinyGPT,
    *,
    config: FSDP2CorrectnessConfig,
) -> None:
    mesh = init_device_mesh(
        config.distributed.device_type,
        (config.distributed.world_size,),
        mesh_dim_names=("fsdp",),
    )
    policy = _mixed_precision_policy(config)
    for block in model.blocks:
        fully_shard(
            block,
            mesh=mesh,
            reshard_after_forward=config.distributed.reshard_after_forward,
            mp_policy=policy,
        )
    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=config.distributed.reshard_after_forward,
        mp_policy=policy,
    )


def _new_rank_zero_run(
    *,
    config_path: Path,
    config: FSDP2CorrectnessConfig,
    output_root: Path,
    run_id: str,
    config_hash: str,
    git_commit: str,
    git_dirty: bool,
    rank_environments: list[object],
) -> Path:
    artifact_dir = (output_root / run_id).resolve()
    artifact_dir.mkdir(parents=False, exist_ok=False)
    (artifact_dir / "checkpoints").mkdir()
    shutil.copyfile(config_path, artifact_dir / "config.original.yaml")
    _atomic_json(artifact_dir / "config.resolved.json", config.to_dict())
    _atomic_json(
        artifact_dir / "environment.json",
        {
            "schema_version": "1.0",
            "strategy": "fsdp2",
            "backend": config.distributed.backend,
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "ranks": rank_environments,
        },
    )
    _atomic_json(
        artifact_dir / "hardware.json",
        {
            "schema_version": "1.0",
            "device_type": config.distributed.device_type,
            "world_size": config.distributed.world_size,
            "ranks": rank_environments,
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
            "dataset_version": f"toy-fsdp2-{config_hash[:8]}",
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "checkpoint_status": "not_evaluated_m4_1",
        },
    )
    _append_jsonl(
        artifact_dir / "events.jsonl",
        {"event": "fsdp2_run_started", "writer_rank": 0},
    )
    return artifact_dir


def _gradient_norm(model: nn.Module, *, max_norm: float) -> float:
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not parameters:
        raise TrainingError(
            TrainingErrorCode.NON_FINITE_GRADIENT,
            "FSDP2 produced no gradients before the optimizer step",
        )
    norm = nn.utils.clip_grad_norm_(parameters, max_norm)
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    if not bool(torch.isfinite(norm).item()):
        raise TrainingError(
            TrainingErrorCode.NON_FINITE_GRADIENT,
            "FSDP2 produced a non-finite gradient norm",
        )
    return float(norm.item())


def _require_finite_scalar_loss(loss: Tensor | None, *, global_step: int) -> Tensor:
    if loss is None or loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise TrainingError(
            TrainingErrorCode.NON_FINITE_LOSS,
            "FSDP2 model output must contain a finite scalar loss",
            context={"global_step": global_step},
        )
    return loss


def run_fsdp2_correctness(
    *,
    config_path: Path,
    output_root: Path,
) -> FSDP2TrainingResult | None:
    """Run one bounded torchrun FSDP2 gate and return the Rank-zero result."""

    config_path = config_path.resolve()
    if not output_root.is_absolute():
        raise ValueError("FSDP2 output_root must be absolute")
    output_root = output_root.resolve()
    config = load_fsdp2_config(config_path)
    launch = torchrun_environment(os.environ)
    if launch.world_size != config.distributed.world_size:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID,
            "torchrun WORLD_SIZE does not match the FSDP2 config",
            context={
                "actual_world_size": launch.world_size,
                "expected_world_size": config.distributed.world_size,
            },
        )
    device = _select_device(config, local_rank=launch.local_rank)
    _initialize_process_group(config, device=device)
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

        local_environment = rank_environment(launch, device)
        if device.type == "cpu":
            local_environment["physical_gpu_index"] = None
        rank_environments = all_gather_objects(
            local_environment,
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
                rank_environments=rank_environments,
            )
        distributed_barrier(device, launch)

        seed_everything(config.run.seed, deterministic_algorithms=device.type == "cpu")
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
        validate_sampler_partitions(
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
        logical_parameter_count = model.parameter_count()
        initial_hash = _require_identical_hashes(
            all_gather_objects(model_state_sha256(model), world_size=launch.world_size),
            stage="initialization",
        )
        _apply_fully_shard(model, config=config)
        optimizer = build_adamw(model, config.training)
        scheduler = build_warmup_cosine_scheduler(optimizer, config.training)
        iterator = iter(dataloader)
        autocast_dtype = torch.bfloat16 if config.precision.dtype == "bf16" else None
        max_loss_diff = 0.0
        max_gradient_norm_diff = 0.0
        durable_metrics = 0

        for global_step in range(1, config.training.max_steps + 1):
            batch = next(iterator).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                output = model(batch, labels=batch)
            loss = _require_finite_scalar_loss(output.loss, global_step=global_step)
            torch.autograd.backward(loss)
            gradient_norm = _gradient_norm(model, max_norm=config.training.max_grad_norm)
            optimizer.step()
            scheduler.step()

            local_loss = float(loss.detach().float().item())
            local_losses = [
                cast(float, value)
                for value in all_gather_objects(local_loss, world_size=launch.world_size)
            ]
            reduced_loss = reduced_mean(
                local_loss,
                device=device,
                world_size=launch.world_size,
            )
            expected_loss = sum(local_losses) / launch.world_size
            max_loss_diff = max(max_loss_diff, abs(reduced_loss - expected_loss))
            gradient_norms = [
                cast(float, value)
                for value in all_gather_objects(gradient_norm, world_size=launch.world_size)
            ]
            max_gradient_norm_diff = max(
                max_gradient_norm_diff,
                max(gradient_norms) - min(gradient_norms),
            )
            if launch.rank == 0:
                if artifact_dir is None:
                    raise RuntimeError("rank-zero FSDP2 Artifact directory is missing")
                metric = TrainingStepMetrics(
                    global_step=global_step,
                    micro_step=global_step,
                    epoch=0,
                    loss=reduced_loss,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    gradient_norm=sum(gradient_norms) / launch.world_size,
                    gradient_clipped=any(
                        value > config.training.max_grad_norm for value in gradient_norms
                    ),
                    tokens_seen=(
                        global_step * config.global_batch_size * (config.data.sequence_length - 1)
                    ),
                )
                _append_jsonl(artifact_dir / "metrics.jsonl", metric.to_dict())
                durable_metrics += 1

        final_hash = _require_identical_hashes(
            all_gather_objects(
                full_fsdp2_state_sha256(model),
                world_size=launch.world_size,
            ),
            stage="final optimizer step",
        )
        local_evidence = local_fsdp2_shard_evidence(
            model,
            rank=launch.rank,
            device_type=cast(Literal["cpu", "cuda"], device.type),
        )
        rank_evidence = tuple(
            FSDP2RankEvidence.model_validate(value)
            for value in all_gather_objects(
                local_evidence.to_dict(),
                world_size=launch.world_size,
            )
        )

        result_value: FSDP2TrainingResult | None = None
        if launch.rank == 0:
            if artifact_dir is None:
                raise RuntimeError("rank-zero FSDP2 Artifact directory is missing")
            summary = FSDP2CorrectnessSummary(
                backend=config.distributed.backend,
                device_type=config.distributed.device_type,
                world_size=launch.world_size,
                global_batch_size=config.global_batch_size,
                optimizer_steps=config.training.max_steps,
                durable_metric_records=durable_metrics,
                logical_parameter_count=logical_parameter_count,
                local_shard_parameter_sum=sum(item.local_shard_numel for item in rank_evidence),
                initial_full_parameter_sha256=initial_hash,
                final_full_parameter_sha256=final_hash,
                rank_evidence=rank_evidence,
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
                    "strategy": "fsdp2",
                    "world_size": launch.world_size,
                    "global_step": config.training.max_steps,
                    "config_hash": config_hash,
                    "dataset_version": f"toy-fsdp2-{config_hash[:8]}",
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "checkpoint_status": "not_evaluated_m4_1",
                    "correctness_status": "pass",
                },
            )
            _append_jsonl(
                artifact_dir / "events.jsonl",
                {
                    "event": "fsdp2_run_succeeded",
                    "global_step": config.training.max_steps,
                    "metrics": durable_metrics,
                    "writer_rank": 0,
                },
            )
            result_value = FSDP2TrainingResult(
                run_id=run_id,
                artifact_dir=artifact_dir,
                config_sha256=config_hash,
                git_commit=git_commit,
                git_dirty=git_dirty,
                summary=summary,
            )
        distributed_barrier(device, launch)
        return result_value
    except Exception as exc:
        if launch.rank == 0 and artifact_dir is not None:
            _atomic_json(
                artifact_dir / "run.json",
                {
                    "schema_version": "1.0",
                    "status": "failed",
                    "strategy": "fsdp2",
                    "world_size": launch.world_size,
                    "error_type": type(exc).__name__,
                },
            )
            _append_jsonl(
                artifact_dir / "events.jsonl",
                {"event": "fsdp2_run_failed", "error_type": type(exc).__name__},
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
