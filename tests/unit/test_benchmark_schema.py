from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tinyllm.benchmark import (
    BenchmarkProfileAggregate,
    BenchmarkTimingSummary,
    CommunicationMeasurement,
    DDPBenchmarkRunResult,
    RankBenchmarkMetrics,
)
from tinyllm.schemas import generate_run_id


def _timing(count: int = 2) -> BenchmarkTimingSummary:
    return BenchmarkTimingSummary(
        count=count,
        total_ms=10.0 * count,
        min_ms=10.0,
        median_ms=10.0,
        p95_ms=10.0,
        max_ms=10.0,
    )


def _run() -> DDPBenchmarkRunResult:
    resolved_hash = "a" * 64
    now = datetime(2026, 7, 16, tzinfo=UTC)
    ranks = tuple(
        RankBenchmarkMetrics(
            rank=rank,
            local_rank=rank,
            physical_gpu_index=rank + 4,
            gpu_name="RTX 3090",
            step_time_ms=(10.0, 10.0),
            data_wait_ms=(1.0, 1.0),
            peak_memory_allocated_bytes=1024,
            communication=CommunicationMeasurement(
                status="not_collected",
                profiled_steps=0,
            ),
        )
        for rank in range(2)
    )
    return DDPBenchmarkRunResult(
        run_id=generate_run_id("schema", resolved_hash, now=now, nonce="1234"),
        artifact_dir=Path("/tmp/private"),
        group="standard",
        profile="weak",
        world_size=2,
        repeat=1,
        seed=1,
        base_config_sha256="b" * 64,
        resolved_config_sha256=resolved_hash,
        git_commit="c" * 40,
        git_dirty=False,
        started_at=now,
        finished_at=now + timedelta(seconds=1),
        backend="nccl",
        precision="bf16",
        model_parameter_count=100,
        sequence_length=16,
        warmup_steps=1,
        measurement_steps=2,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        global_batch_size=2,
        predicted_tokens_per_step=30,
        tokens_per_second=3000.0,
        samples_per_second=200.0,
        effective_step_time=_timing(),
        effective_data_wait=BenchmarkTimingSummary(
            count=2,
            total_ms=2.0,
            min_ms=1.0,
            median_ms=1.0,
            p95_ms=1.0,
            max_ms=1.0,
        ),
        data_wait_percent=10.0,
        peak_memory_allocated_bytes=1024,
        rank_metrics=ranks,
    )


def _reject_run(field: str, value: object, message: str) -> None:
    raw = _run().to_dict()
    raw[field] = value
    with pytest.raises(ValueError, match=message):
        DDPBenchmarkRunResult.model_validate_json(json.dumps(raw))


def test_timing_and_communication_status_invariants() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        BenchmarkTimingSummary(
            count=1,
            total_ms=1.0,
            min_ms=2.0,
            median_ms=1.0,
            p95_ms=1.0,
            max_ms=3.0,
        )
    with pytest.raises(ValueError, match="requires"):
        CommunicationMeasurement(status="measured", profiled_steps=1)
    with pytest.raises(ValueError, match="cannot contain"):
        CommunicationMeasurement(
            status="unavailable",
            profiled_steps=1,
            device_time_ms=1.0,
        )


def test_rank_metric_rejects_bad_windows_and_missing_trace() -> None:
    base = _run().rank_metrics[0].to_dict()
    base["step_time_ms"] = []
    with pytest.raises(ValueError, match="non-empty"):
        RankBenchmarkMetrics.model_validate_json(json.dumps(base))
    base = _run().rank_metrics[0].to_dict()
    base["step_time_ms"] = [10.0, -1.0]
    with pytest.raises(ValueError, match="positive"):
        RankBenchmarkMetrics.model_validate_json(json.dumps(base))
    base = _run().rank_metrics[0].to_dict()
    base["data_wait_ms"] = [1.0, -1.0]
    with pytest.raises(ValueError, match="non-negative"):
        RankBenchmarkMetrics.model_validate_json(json.dumps(base))
    base = _run().rank_metrics[0].to_dict()
    base["communication"] = {
        "status": "unavailable",
        "profiled_steps": 1,
        "device_time_ms": None,
        "event_keys": [],
    }
    with pytest.raises(ValueError, match="trace"):
        RankBenchmarkMetrics.model_validate_json(json.dumps(base))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_dir", "relative", "absolute"),
        ("resolved_config_sha256", "d" * 64, "run_id"),
        ("finished_at", "2026-07-15T00:00:00Z", "follow"),
        ("world_size", 1, "rank_metrics count"),
        ("measurement_steps", 3, "retain all"),
        ("global_batch_size", 4, "arithmetic"),
        ("predicted_tokens_per_step", 31, "predicted_tokens"),
    ],
)
def test_run_result_rejects_identity_and_arithmetic_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    _reject_run(field, value, message)


def test_profile_aggregate_rejects_incomplete_repeats_gpu_count_and_range() -> None:
    kwargs = {
        "group": "standard",
        "profile": "weak",
        "world_size": 2,
        "gpu_indices": (4, 5),
        "repeats": (1, 2, 3),
        "run_ids": ("a", "b", "c"),
        "tokens_per_second_by_repeat": (1.0, 2.0, 3.0),
        "step_time_ms_by_repeat": (1.0, 1.0, 1.0),
        "peak_memory_bytes_by_repeat": (1, 1, 1),
        "data_wait_percent_by_repeat": (1.0, 1.0, 1.0),
        "tokens_per_second_median": 2.0,
        "tokens_per_second_min": 1.0,
        "tokens_per_second_max": 3.0,
        "step_time_ms_median": 1.0,
        "peak_memory_bytes_median": 1.0,
        "data_wait_percent_median": 1.0,
    }
    BenchmarkProfileAggregate(**kwargs)  # type: ignore[arg-type]
    for update, message in (
        ({"repeats": (1, 2), "run_ids": ("a", "b")}, "repeats"),
        ({"gpu_indices": (4,)}, "GPU index"),
        ({"tokens_per_second_min": 4.0}, "range"),
    ):
        with pytest.raises(ValueError, match=message):
            BenchmarkProfileAggregate(**(kwargs | update))  # type: ignore[arg-type]
