"""torchrun worker for one bounded M3 DDP benchmark repetition."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.cuda import Event
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, DistributedSampler

from tinyllm.benchmark.config import (
    BenchmarkProfile,
    DDPBenchmarkConfig,
    ResolvedBenchmarkProfile,
    load_ddp_benchmark_config,
    resolve_benchmark_profile,
)
from tinyllm.benchmark.schema import (
    BenchmarkGroup,
    BenchmarkTimingSummary,
    CommunicationMeasurement,
    DDPBenchmarkRunResult,
    RankBenchmarkMetrics,
)
from tinyllm.data import ToyTokenDataset
from tinyllm.lineage import read_git_identity
from tinyllm.models.tinygpt import TinyGPT
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.config import TrainingLoopConfig
from tinyllm.training.distributed import physical_gpu_index, torchrun_environment
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty timing window")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_timings(values: Sequence[float]) -> BenchmarkTimingSummary:
    """Summarize a non-empty finite millisecond timing window."""

    if not values or any(
        not value > 0 or not torch.isfinite(torch.tensor(value)) for value in values
    ):
        raise ValueError("timings must contain finite positive values")
    return BenchmarkTimingSummary(
        count=len(values),
        total_ms=sum(values),
        min_ms=min(values),
        median_ms=_percentile(values, 0.5),
        p95_ms=_percentile(values, 0.95),
        max_ms=max(values),
    )


def _communication_from_profiler(
    profiler: profile | None,
    *,
    world_size: int,
    profiled_steps: int,
) -> CommunicationMeasurement:
    if world_size == 1:
        return CommunicationMeasurement(
            status="not_applicable",
            profiled_steps=profiled_steps,
        )
    if profiler is None or profiled_steps == 0:
        return CommunicationMeasurement(status="not_collected", profiled_steps=0)
    event_times: dict[str, float] = {}
    for event in profiler.key_averages():
        key = str(event.key)
        normalized = key.casefold().replace("_", "")
        if not any(
            marker in normalized for marker in ("nccl", "allreduce", "reducescatter", "allgather")
        ):
            continue
        device_time_us = float(getattr(event, "device_time_total", 0.0))
        if device_time_us > 0:
            event_times[key] = device_time_us / 1000.0
    if not event_times:
        return CommunicationMeasurement(status="unavailable", profiled_steps=profiled_steps)
    return CommunicationMeasurement(
        status="measured",
        profiled_steps=profiled_steps,
        device_time_ms=sum(event_times.values()),
        event_keys=tuple(sorted(event_times)),
    )


def _select_device(config: DDPBenchmarkConfig, *, local_rank: int) -> torch.device:
    if config.distributed.backend == "gloo":
        return torch.device("cpu")
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError("benchmark CUDA local Rank is unavailable")
    torch.cuda.set_device(local_rank)
    if config.precision.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected GPU does not support BF16")
    torch.backends.cuda.matmul.allow_tf32 = config.precision.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.precision.allow_tf32
    return torch.device("cuda", local_rank)


def _initialize_process_group(config: DDPBenchmarkConfig, device: torch.device) -> None:
    arguments: dict[str, object] = {
        "backend": config.distributed.backend,
        "init_method": "env://",
        "timeout": timedelta(seconds=config.distributed.timeout_seconds),
    }
    if device.type == "cuda":
        arguments["device_id"] = device
    dist.init_process_group(**arguments)  # type: ignore[arg-type]


def _barrier(device: torch.device, local_rank: int) -> None:
    if device.type == "cuda":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _create_artifact_dir(
    *,
    output_root: Path,
    run_id: str,
    config_path: Path,
    config: DDPBenchmarkConfig,
    resolved: ResolvedBenchmarkProfile,
    group: BenchmarkGroup,
    base_hash: str,
    resolved_hash: str,
    git_commit: str,
    git_dirty: bool,
) -> Path:
    artifact_dir = output_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "profiles").mkdir()
    shutil.copyfile(config_path, artifact_dir / "config.original.yaml")
    _atomic_json(artifact_dir / "config.resolved.json", config.to_dict())
    _atomic_json(artifact_dir / "profile.resolved.json", resolved.to_dict())
    _atomic_json(
        artifact_dir / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "running",
            "strategy": "ddp",
            "group": group,
            "profile": resolved.profile,
            "world_size": resolved.world_size,
            "repeat": resolved.repeat,
            "base_config_sha256": base_hash,
            "resolved_config_sha256": resolved_hash,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
    )
    return artifact_dir


def _make_dataloader(
    config: DDPBenchmarkConfig,
    resolved: ResolvedBenchmarkProfile,
    *,
    rank: int,
) -> DataLoader[Tensor]:
    dataset = ToyTokenDataset(
        vocab_size=config.data.vocab_size,
        sequence_length=config.data.sequence_length,
        num_samples=resolved.dataset_samples,
        seed=resolved.seed,
    )
    sampler: DistributedSampler[int] = DistributedSampler(
        dataset,
        num_replicas=resolved.world_size,
        rank=rank,
        shuffle=True,
        seed=resolved.seed,
        drop_last=True,
    )
    sampler.set_epoch(0)
    return DataLoader(
        dataset,
        batch_size=resolved.micro_batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
        pin_memory=config.distributed.backend == "nccl",
    )


def _next_batches(
    iterator: Iterator[Tensor],
    *,
    accumulation_steps: int,
) -> tuple[list[Tensor], float]:
    started = time.perf_counter()
    batches: list[Tensor] = []
    for _ in range(accumulation_steps):
        try:
            batches.append(next(iterator))
        except StopIteration as exc:
            raise RuntimeError("benchmark DataLoader exhausted before the fixed window") from exc
    elapsed_ms = max((time.perf_counter() - started) * 1000.0, 1.0e-9)
    return batches, elapsed_ms


def _optimizer_step(
    *,
    ddp_model: DistributedDataParallel,
    batches: Sequence[Tensor],
    device: torch.device,
    config: DDPBenchmarkConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    accumulation_steps = len(batches)
    for index, cpu_batch in enumerate(batches):
        sync_context = (
            contextlib.nullcontext() if index == accumulation_steps - 1 else ddp_model.no_sync()
        )
        with sync_context:
            batch = cpu_batch.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if config.precision.dtype == "bf16" else None,
                enabled=config.precision.dtype == "bf16",
            ):
                output = ddp_model(batch, labels=batch)
                loss = getattr(output, "loss", None)
                if not isinstance(loss, Tensor) or loss.ndim != 0:
                    raise RuntimeError("benchmark model returned an invalid loss")
                torch.autograd.backward(loss / accumulation_steps)
    norm = nn.utils.clip_grad_norm_(ddp_model.parameters(), config.training.max_grad_norm)
    if not bool(torch.isfinite(norm).item()):
        raise RuntimeError("benchmark produced a non-finite gradient norm")
    optimizer.step()
    scheduler.step()


def _run_steps(
    *,
    count: int,
    iterator: Iterator[Tensor],
    resolved: ResolvedBenchmarkProfile,
    ddp_model: DistributedDataParallel,
    device: torch.device,
    config: DDPBenchmarkConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    profiler: profile | None = None,
) -> tuple[list[float], list[float]]:
    data_times: list[float] = []
    cpu_step_times: list[float] = []
    cuda_events: list[tuple[Event, Event]] = []
    for _ in range(count):
        batches, data_time = _next_batches(
            iterator,
            accumulation_steps=resolved.gradient_accumulation_steps,
        )
        data_times.append(data_time)
        if device.type == "cuda":
            start = Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end = Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start.record()  # type: ignore[no-untyped-call]
        else:
            cpu_started = time.perf_counter()
        _optimizer_step(
            ddp_model=ddp_model,
            batches=batches,
            device=device,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if device.type == "cuda":
            end.record()  # type: ignore[no-untyped-call]
            cuda_events.append((start, end))
        else:
            cpu_step_times.append(max((time.perf_counter() - cpu_started) * 1000.0, 1.0e-9))
        if profiler is not None:
            profiler.step()  # type: ignore[no-untyped-call]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return [
            start.elapsed_time(end)  # type: ignore[no-untyped-call]
            for start, end in cuda_events
        ], data_times
    return cpu_step_times, data_times


def _rank_environment(
    *,
    rank: int,
    local_rank: int,
    device: torch.device,
) -> dict[str, object]:
    value: dict[str, object] = {
        "rank": rank,
        "local_rank": local_rank,
        "physical_gpu_index": physical_gpu_index(torchrun_environment(os.environ)),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        value.update(
            {
                "gpu_name": properties.name,
                "memory_total_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return value


def _gather_rank_metrics(
    metric: RankBenchmarkMetrics,
    *,
    world_size: int,
) -> tuple[RankBenchmarkMetrics, ...]:
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, metric)
    if any(not isinstance(item, RankBenchmarkMetrics) for item in gathered):
        raise RuntimeError("distributed benchmark gathered an invalid Rank metric")
    return tuple(cast(RankBenchmarkMetrics, item) for item in gathered)


def run_ddp_benchmark(
    *,
    config_path: Path,
    output_root: Path,
    profile_name: BenchmarkProfile,
    group: BenchmarkGroup,
    repeat: int,
) -> DDPBenchmarkRunResult | None:
    """Execute one benchmark repeat; only Rank zero returns the strict result."""

    config_path = config_path.resolve()
    output_root = output_root.resolve()
    config = load_ddp_benchmark_config(config_path)
    launch = torchrun_environment(os.environ)
    resolved = resolve_benchmark_profile(
        config,
        profile=profile_name,
        world_size=launch.world_size,
        repeat=repeat,
    )
    device = _select_device(config, local_rank=launch.local_rank)
    _initialize_process_group(config, device)
    artifact_dir: Path | None = None
    started_at = datetime.now(UTC)
    try:
        base_hash = canonical_config_hash(config)
        resolved_hash = canonical_config_hash(
            {"config": config.to_dict(), "group": group, "resolved": resolved.to_dict()}
        )
        git_commit, git_dirty = read_git_identity(Path.cwd())
        run_ids: list[object] = [
            generate_run_id(config.run.name, resolved_hash, now=started_at)
            if launch.rank == 0
            else None
        ]
        dist.broadcast_object_list(run_ids, src=0)
        run_id = str(run_ids[0])
        if launch.rank == 0:
            output_root.mkdir(parents=True, exist_ok=True)
            artifact_dir = _create_artifact_dir(
                output_root=output_root,
                run_id=run_id,
                config_path=config_path,
                config=config,
                resolved=resolved,
                group=group,
                base_hash=base_hash,
                resolved_hash=resolved_hash,
                git_commit=git_commit,
                git_dirty=git_dirty,
            )
        artifact_paths: list[object] = [str(artifact_dir) if artifact_dir else None]
        dist.broadcast_object_list(artifact_paths, src=0)
        artifact_dir = Path(str(artifact_paths[0]))
        _barrier(device, launch.local_rank)

        seed_everything(resolved.seed, deterministic_algorithms=device.type == "cpu")
        dataloader = _make_dataloader(config, resolved, rank=launch.rank)
        iterator = iter(dataloader)
        model = TinyGPT(config.model).to(device)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[launch.local_rank] if device.type == "cuda" else None,
            output_device=launch.local_rank if device.type == "cuda" else None,
            broadcast_buffers=config.distributed.broadcast_buffers,
            find_unused_parameters=config.distributed.find_unused_parameters,
        )
        loop_config = TrainingLoopConfig(
            max_steps=resolved.warmup_steps + resolved.measurement_steps,
            micro_batch_size=resolved.micro_batch_size,
            gradient_accumulation_steps=resolved.gradient_accumulation_steps,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            max_grad_norm=config.training.max_grad_norm,
            warmup_steps=config.training.learning_rate_warmup_steps,
        )
        optimizer = build_adamw(ddp_model, loop_config)
        scheduler = build_warmup_cosine_scheduler(optimizer, loop_config)
        cast(nn.Module, ddp_model).train()
        _run_steps(
            count=resolved.warmup_steps,
            iterator=iterator,
            resolved=resolved,
            ddp_model=ddp_model,
            device=device,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        _barrier(device, launch.local_rank)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        profiler_value: profile | None = None
        step_times: list[float] = []
        data_times: list[float] = []
        trace_hash: str | None = None
        if resolved.profiler_steps:
            activities = [ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)
            with profile(activities=activities, record_shapes=False, with_stack=False) as active:
                profiled_step_times, profiled_data_times = _run_steps(
                    count=resolved.profiler_steps,
                    iterator=iterator,
                    resolved=resolved,
                    ddp_model=ddp_model,
                    device=device,
                    config=config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    profiler=active,
                )
            profiler_value = active
            step_times.extend(profiled_step_times)
            data_times.extend(profiled_data_times)
            trace_path = artifact_dir / "profiles" / f"rank-{launch.rank:05d}.trace.json"
            active.export_chrome_trace(str(trace_path))
            trace_hash = _sha256_file(trace_path)
        remaining_steps = resolved.measurement_steps - resolved.profiler_steps
        remaining_step_times, remaining_data_times = _run_steps(
            count=remaining_steps,
            iterator=iterator,
            resolved=resolved,
            ddp_model=ddp_model,
            device=device,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        step_times.extend(remaining_step_times)
        data_times.extend(remaining_data_times)
        _barrier(device, launch.local_rank)

        rank_env = _rank_environment(
            rank=launch.rank,
            local_rank=launch.local_rank,
            device=device,
        )
        peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        rank_metric = RankBenchmarkMetrics(
            rank=launch.rank,
            local_rank=launch.local_rank,
            physical_gpu_index=cast(int | None, rank_env["physical_gpu_index"]),
            gpu_name=cast(str | None, rank_env.get("gpu_name")),
            step_time_ms=tuple(step_times),
            data_wait_ms=tuple(data_times),
            peak_memory_allocated_bytes=peak_memory,
            communication=_communication_from_profiler(
                profiler_value,
                world_size=launch.world_size,
                profiled_steps=resolved.profiler_steps,
            ),
            profiler_trace_sha256=trace_hash,
        )
        rank_metrics = _gather_rank_metrics(rank_metric, world_size=launch.world_size)
        environments: list[object] = [None] * launch.world_size
        dist.all_gather_object(environments, rank_env)

        result: DDPBenchmarkRunResult | None = None
        if launch.rank == 0:
            effective_steps = [
                max(item.step_time_ms[index] for item in rank_metrics)
                for index in range(resolved.measurement_steps)
            ]
            effective_data = [
                max(max(item.data_wait_ms[index], 1.0e-9) for item in rank_metrics)
                for index in range(resolved.measurement_steps)
            ]
            step_summary = summarize_timings(effective_steps)
            data_summary = summarize_timings(effective_data)
            predicted_tokens = resolved.global_batch_size * (config.data.sequence_length - 1)
            measured_seconds = step_summary.total_ms / 1000.0
            finished_at = datetime.now(UTC)
            result = DDPBenchmarkRunResult(
                run_id=run_id,
                artifact_dir=artifact_dir,
                group=group,
                profile=resolved.profile,
                world_size=launch.world_size,
                repeat=resolved.repeat,
                seed=resolved.seed,
                base_config_sha256=base_hash,
                resolved_config_sha256=resolved_hash,
                git_commit=git_commit,
                git_dirty=git_dirty,
                started_at=started_at,
                finished_at=finished_at,
                backend=config.distributed.backend,
                precision=config.precision.dtype,
                model_parameter_count=model.parameter_count(),
                sequence_length=config.data.sequence_length,
                warmup_steps=resolved.warmup_steps,
                measurement_steps=resolved.measurement_steps,
                micro_batch_size=resolved.micro_batch_size,
                gradient_accumulation_steps=resolved.gradient_accumulation_steps,
                global_batch_size=resolved.global_batch_size,
                predicted_tokens_per_step=predicted_tokens,
                tokens_per_second=(predicted_tokens * resolved.measurement_steps)
                / measured_seconds,
                samples_per_second=(resolved.global_batch_size * resolved.measurement_steps)
                / measured_seconds,
                effective_step_time=step_summary,
                effective_data_wait=data_summary,
                data_wait_percent=100.0 * data_summary.total_ms / step_summary.total_ms,
                peak_memory_allocated_bytes=max(
                    item.peak_memory_allocated_bytes for item in rank_metrics
                ),
                rank_metrics=rank_metrics,
            )
            _atomic_json(artifact_dir / "environment.json", {"ranks": environments})
            _atomic_json(artifact_dir / "benchmark.json", result.to_dict())
            _atomic_json(
                artifact_dir / "run.json",
                {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "status": "succeeded",
                    "strategy": "ddp",
                    "group": group,
                    "profile": resolved.profile,
                    "world_size": resolved.world_size,
                    "repeat": resolved.repeat,
                    "base_config_sha256": base_hash,
                    "resolved_config_sha256": resolved_hash,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                },
            )
        _barrier(device, launch.local_rank)
        return result
    except Exception as exc:
        if launch.rank == 0 and artifact_dir is not None:
            _atomic_json(
                artifact_dir / "run.json",
                {
                    "schema_version": "1.0",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    """Build the internal torchrun worker interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", choices=("strong", "weak"), required=True)
    parser.add_argument(
        "--group",
        choices=("standard", "same_numa", "cross_numa"),
        required=True,
    )
    parser.add_argument("--repeat", type=int, required=True)
    return parser


def main() -> int:
    """Emit one Rank-zero JSON result and a bounded nonzero failure."""

    args = build_parser().parse_args()
    try:
        result = run_ddp_benchmark(
            config_path=args.config,
            output_root=args.output_root,
            profile_name=args.profile,
            group=args.group,
            repeat=args.repeat,
        )
    except Exception as exc:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(
                json.dumps(
                    {"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}},
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        return 4
    if result is not None:
        print(result.model_dump_json(indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
