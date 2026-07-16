from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tinyllm.benchmark import (
    BenchmarkTimingSummary,
    CommunicationMeasurement,
    DDPBenchmarkRunResult,
    RankBenchmarkMetrics,
    build_m3_matrix_summary,
    load_benchmark_evidence,
)
from tinyllm.benchmark.schema import (
    BenchmarkGroup,
    BenchmarkProfile,
    DDPBenchmarkMatrixSummary,
)
from tinyllm.schemas import generate_run_id

BASE_HASH = "a" * 64
GIT_COMMIT = "b" * 40


def _timing(value: float, *, count: int = 2) -> BenchmarkTimingSummary:
    return BenchmarkTimingSummary(
        count=count,
        total_ms=value * count,
        min_ms=value,
        median_ms=value,
        p95_ms=value,
        max_ms=value,
    )


def _result(
    *,
    group: BenchmarkGroup,
    profile: BenchmarkProfile,
    world_size: int,
    repeat: int,
    throughput: float,
) -> DDPBenchmarkRunResult:
    resolved_hash = f"{repeat:x}" * 64
    now = datetime(2026, 7, 16, tzinfo=UTC) + timedelta(seconds=repeat)
    ranks = tuple(
        RankBenchmarkMetrics(
            rank=rank,
            local_rank=rank,
            physical_gpu_index=rank,
            gpu_name="RTX 3090",
            step_time_ms=(10.0, 10.0),
            data_wait_ms=(1.0, 1.0),
            peak_memory_allocated_bytes=1024,
            communication=CommunicationMeasurement(
                status="not_applicable" if world_size == 1 else "not_collected",
                profiled_steps=0,
            ),
        )
        for rank in range(world_size)
    )
    global_batch = 8 if profile == "strong" else world_size
    return DDPBenchmarkRunResult(
        run_id=generate_run_id("test", resolved_hash, now=now, nonce=f"{repeat:04x}"),
        artifact_dir=Path("/tmp/private") / str(repeat),
        group=group,
        profile=profile,
        world_size=world_size,
        repeat=repeat,
        seed=repeat,
        base_config_sha256=BASE_HASH,
        resolved_config_sha256=resolved_hash,
        git_commit=GIT_COMMIT,
        git_dirty=False,
        started_at=now,
        finished_at=now + timedelta(seconds=1),
        backend="nccl",
        precision="bf16",
        model_parameter_count=117_197_568,
        sequence_length=1024,
        warmup_steps=20,
        measurement_steps=2,
        micro_batch_size=1,
        gradient_accumulation_steps=8 // world_size if profile == "strong" else 1,
        global_batch_size=global_batch,
        predicted_tokens_per_step=global_batch * 1023,
        tokens_per_second=throughput,
        samples_per_second=throughput / 1023,
        effective_step_time=_timing(10.0),
        effective_data_wait=_timing(1.0),
        data_wait_percent=10.0,
        peak_memory_allocated_bytes=1024,
        rank_metrics=ranks,
    )


def _matrix_runs() -> list[DDPBenchmarkRunResult]:
    runs: list[DDPBenchmarkRunResult] = []
    for profile in ("strong", "weak"):
        for world_size in (1, 2, 4):
            for repeat in (1, 2, 3):
                runs.append(
                    _result(
                        group="standard",
                        profile=profile,
                        world_size=world_size,
                        repeat=repeat,
                        throughput=100.0 * world_size + repeat,
                    )
                )
    for repeat in (1, 2, 3):
        runs.append(
            _result(
                group="same_numa",
                profile="weak",
                world_size=4,
                repeat=repeat,
                throughput=400.0 + repeat,
            )
        )
    return runs


def test_complete_matrix_uses_repeat_medians_and_scaling_formulas() -> None:
    summary = build_m3_matrix_summary(_matrix_runs())

    assert summary.status == "pass"
    assert summary.schema_version == "1.1"
    assert summary.acceptance_world_sizes == (1, 2, 4)
    assert summary.eight_gpu_status == "not_collected"
    assert summary.numa_comparison_status == "partial"
    assert len(summary.standard) == 6
    assert len(summary.numa) == 1
    strong_two = next(
        item for item in summary.standard if item.profile == "strong" and item.world_size == 2
    )
    assert strong_two.tokens_per_second_median == 202.0
    assert strong_two.tokens_per_second_by_repeat == (201.0, 202.0, 203.0)
    assert strong_two.step_time_ms_by_repeat == (10.0, 10.0, 10.0)
    assert strong_two.peak_memory_bytes_by_repeat == (1024, 1024, 1024)
    assert strong_two.data_wait_percent_by_repeat == (10.0, 10.0, 10.0)
    assert strong_two.scaling_efficiency == pytest.approx(202.0 / (2 * 102.0))
    weak_four = next(
        item for item in summary.standard if item.profile == "weak" and item.world_size == 4
    )
    assert weak_four.scaling_efficiency == pytest.approx((402.0 / 4) / 102.0)


def test_complete_optional_eight_gpu_and_numa_evidence_is_retained() -> None:
    runs = _matrix_runs()
    for profile in ("strong", "weak"):
        for repeat in (1, 2, 3):
            runs.append(
                _result(
                    group="standard",
                    profile=profile,
                    world_size=8,
                    repeat=repeat,
                    throughput=800.0 + repeat,
                )
            )
    for repeat in (1, 2, 3):
        runs.append(
            _result(
                group="cross_numa",
                profile="weak",
                world_size=4,
                repeat=repeat,
                throughput=390.0 + repeat,
            )
        )

    summary = build_m3_matrix_summary(runs)

    assert summary.eight_gpu_status == "complete"
    assert summary.numa_comparison_status == "complete"
    assert len(summary.standard) == 8
    assert len(summary.numa) == 2


def test_matrix_rejects_partial_optional_eight_gpu_evidence() -> None:
    runs = _matrix_runs()
    for repeat in (1, 2, 3):
        runs.append(
            _result(
                group="standard",
                profile="weak",
                world_size=8,
                repeat=repeat,
                throughput=800.0 + repeat,
            )
        )

    with pytest.raises(ValueError, match="optional eight-GPU"):
        build_m3_matrix_summary(runs)


def test_matrix_rejects_efficiency_that_disagrees_with_raw_throughput() -> None:
    summary = build_m3_matrix_summary(_matrix_runs()).model_dump(mode="python")
    summary["standard"][1]["scaling_efficiency"] = 1.0

    with pytest.raises(ValueError, match="scaling efficiency"):
        DDPBenchmarkMatrixSummary.model_validate(summary)


def test_public_m3_summary_is_a_valid_strict_snapshot() -> None:
    root = Path(__file__).resolve().parents[2]
    summary = DDPBenchmarkMatrixSummary.model_validate_json(
        (root / "reports/m3/ddp_scaling_summary.json").read_text(encoding="utf-8")
    )

    assert summary.status == "pass"
    assert summary.git_commit == "a373b4623e22ab14360100793e9af396cfc88d98"
    assert summary.acceptance_world_sizes == (1, 2, 4)


def test_matrix_rejects_missing_repeat_and_dirty_runs() -> None:
    runs = _matrix_runs()
    with pytest.raises(ValueError, match="repeats"):
        build_m3_matrix_summary(runs[:-1])

    dirty = runs[0].model_copy(update={"git_dirty": True})
    with pytest.raises(ValueError, match="dirty"):
        build_m3_matrix_summary([dirty, *runs[1:]])


def test_evidence_loader_ignores_retained_failures_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    run = _matrix_runs()[0]
    passed = tmp_path / "pass" / "summary.json"
    passed.parent.mkdir()
    passed.write_text(
        json.dumps({"status": "pass", "result": run.to_dict()}),
        encoding="utf-8",
    )
    failed = tmp_path / "fail" / "summary.json"
    failed.parent.mkdir()
    failed.write_text(
        json.dumps({"status": "fail", "reason": "timeout"}),
        encoding="utf-8",
    )

    assert load_benchmark_evidence(tmp_path) == (run,)

    duplicate = tmp_path / "duplicate" / "summary.json"
    duplicate.parent.mkdir()
    duplicate.write_text(passed.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_benchmark_evidence(tmp_path)
