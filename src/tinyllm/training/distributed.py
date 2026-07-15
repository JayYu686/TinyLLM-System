"""Pure and torch.distributed helpers for M3 DDP correctness."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Mapping, Sequence
from datetime import timedelta

import torch
from pydantic import Field, model_validator
from torch import distributed as dist
from torch import nn

from tinyllm.schemas.base import StrictSchema
from tinyllm.training.config import M1TrainingConfig
from tinyllm.training.ddp_schema import DDPPartitionEvidence
from tinyllm.training.errors import TrainingError, TrainingErrorCode


class TorchrunEnvironment(StrictSchema):
    """Validated rank coordinates provided by torchrun."""

    rank: int = Field(ge=0)
    local_rank: int = Field(ge=0)
    world_size: int = Field(ge=1, le=10)
    local_world_size: int = Field(ge=1, le=10)

    @model_validator(mode="after")
    def validate_coordinates(self) -> TorchrunEnvironment:
        """Reject rank coordinates outside their declared groups."""

        if self.rank >= self.world_size:
            raise ValueError("RANK must be smaller than WORLD_SIZE")
        if self.local_rank >= self.local_world_size:
            raise ValueError("LOCAL_RANK must be smaller than LOCAL_WORLD_SIZE")
        if self.local_world_size > self.world_size:
            raise ValueError("LOCAL_WORLD_SIZE cannot exceed WORLD_SIZE")
        return self


def torchrun_environment(environ: Mapping[str, str]) -> TorchrunEnvironment:
    """Parse required torchrun rank variables without accepting implicit defaults."""

    values: dict[str, int] = {}
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        raw = environ.get(key)
        if raw is None:
            raise TrainingError(
                TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID,
                f"torchrun environment is missing {key}",
                context={"variable": key},
            )
        try:
            values[key] = int(raw)
        except ValueError as exc:
            raise TrainingError(
                TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID,
                f"torchrun environment variable {key} must be an integer",
                context={"variable": key},
            ) from exc
    try:
        return TorchrunEnvironment(
            rank=values["RANK"],
            local_rank=values["LOCAL_RANK"],
            world_size=values["WORLD_SIZE"],
            local_world_size=values["LOCAL_WORLD_SIZE"],
        )
    except ValueError as exc:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID,
            str(exc),
        ) from exc


def physical_gpu_index(
    launch: TorchrunEnvironment,
    *,
    environ: Mapping[str, str] | None = None,
) -> int | None:
    """Map one local Rank to the physical index selected by CUDA_VISIBLE_DEVICES."""

    visible = (environ or os.environ).get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return launch.local_rank
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if launch.local_rank >= len(entries) or not entries[launch.local_rank].isdigit():
        return None
    return int(entries[launch.local_rank])


def select_ddp_device(config: M1TrainingConfig, launch: TorchrunEnvironment) -> torch.device:
    """Select and validate the Rank-local CPU or CUDA device for a DDP run."""

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


def initialize_process_group(
    config: M1TrainingConfig,
    *,
    device: torch.device,
) -> None:
    """Initialize one bounded env:// process group on the selected device."""

    if config.distributed.backend is None:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID,
            "DDP backend is missing",
        )
    arguments: dict[str, object] = {
        "backend": config.distributed.backend,
        "init_method": "env://",
        "timeout": timedelta(seconds=config.distributed.timeout_seconds),
    }
    if device.type == "cuda":
        arguments["device_id"] = device
    dist.init_process_group(**arguments)  # type: ignore[arg-type]


def rank_environment(
    launch: TorchrunEnvironment,
    device: torch.device,
) -> dict[str, object]:
    """Capture stable runtime and hardware identity for one Rank."""

    result: dict[str, object] = {
        "rank": launch.rank,
        "local_rank": launch.local_rank,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "physical_gpu_index": physical_gpu_index(launch),
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


def distributed_barrier(device: torch.device, launch: TorchrunEnvironment) -> None:
    """Bind NCCL barriers to the selected local CUDA device."""

    if device.type == "cuda":
        dist.barrier(device_ids=[launch.local_rank])
    else:
        dist.barrier()


def model_state_sha256(model: nn.Module) -> str:
    """Hash model state without relying on dtype-specific NumPy conversion."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def sample_ids_sha256(indices: Sequence[int]) -> str:
    """Hash one ordered sampler partition using canonical JSON."""

    payload = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_sampler_partitions(
    partitions: Sequence[Sequence[int]],
    *,
    num_samples: int,
) -> tuple[DDPPartitionEvidence, ...]:
    """Require non-overlapping, complete, rank-ordered sampler partitions."""

    if not partitions:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
            "DistributedSampler produced no rank partitions",
        )
    seen: set[int] = set()
    evidence: list[DDPPartitionEvidence] = []
    for rank, raw_indices in enumerate(partitions):
        indices = tuple(int(index) for index in raw_indices)
        if not indices:
            raise TrainingError(
                TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
                "DistributedSampler produced an empty partition",
                context={"rank": rank},
            )
        if len(indices) != len(set(indices)) or seen.intersection(indices):
            raise TrainingError(
                TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
                "DistributedSampler partitions overlap or contain duplicates",
                context={"rank": rank},
            )
        if any(index < 0 or index >= num_samples for index in indices):
            raise TrainingError(
                TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
                "DistributedSampler produced an out-of-range sample ID",
                context={"rank": rank},
            )
        seen.update(indices)
        evidence.append(
            DDPPartitionEvidence(
                rank=rank,
                sample_count=len(indices),
                sample_ids_sha256=sample_ids_sha256(indices),
            )
        )
    if seen != set(range(num_samples)):
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
            "DistributedSampler partitions do not cover the complete dataset",
            context={"covered": len(seen), "expected": num_samples},
        )
    return tuple(evidence)


def all_gather_objects(value: object, *, world_size: int) -> list[object]:
    """Gather one small correctness object from every initialized rank."""

    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def reduced_mean(value: float, *, device: torch.device, world_size: int) -> float:
    """All-reduce one scalar and return the data-parallel mean on every rank."""

    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return float(tensor.item())
