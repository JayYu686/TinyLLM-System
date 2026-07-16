"""Pure aggregation for the formal M3 DDP scaling matrix."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from tinyllm.benchmark.schema import (
    BenchmarkGroup,
    BenchmarkProfile,
    BenchmarkProfileAggregate,
    DDPBenchmarkMatrixSummary,
    DDPBenchmarkRunResult,
)


def load_benchmark_evidence(root: Path) -> tuple[DDPBenchmarkRunResult, ...]:
    """Load every successful private supervisor result without deleting failures."""

    if not root.is_dir():
        raise ValueError(f"benchmark evidence root does not exist: {root}")
    results: list[DDPBenchmarkRunResult] = []
    keys: set[tuple[BenchmarkGroup, BenchmarkProfile, int, int]] = set()
    run_ids: set[str] = set()
    for path in sorted(root.rglob("summary.json")):
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid benchmark supervisor summary: {path}") from exc
        if not isinstance(decoded, dict) or decoded.get("status") != "pass":
            continue
        if "result" not in decoded:
            raise ValueError(f"passing benchmark summary has no result: {path}")
        try:
            result = DDPBenchmarkRunResult.model_validate_json(json.dumps(decoded["result"]))
        except ValueError as exc:
            raise ValueError(f"invalid benchmark result in: {path}") from exc
        key = (result.group, result.profile, result.world_size, result.repeat)
        if key in keys or result.run_id in run_ids:
            raise ValueError(f"duplicate benchmark evidence for {key}")
        keys.add(key)
        run_ids.add(result.run_id)
        results.append(result)
    return tuple(results)


def _gpu_indices(result: DDPBenchmarkRunResult) -> tuple[int, ...]:
    indices = tuple(item.physical_gpu_index for item in result.rank_metrics)
    if any(index is None for index in indices):
        raise ValueError("formal M3 runs require physical GPU indices")
    return tuple(cast(int, index) for index in indices)


def _aggregate_cell(
    results: list[DDPBenchmarkRunResult],
) -> BenchmarkProfileAggregate:
    ordered = sorted(results, key=lambda item: item.repeat)
    if [item.repeat for item in ordered] != [1, 2, 3]:
        raise ValueError("each formal matrix cell requires repeats 1, 2, and 3")
    gpu_sets = {_gpu_indices(item) for item in ordered}
    if len(gpu_sets) != 1:
        raise ValueError("all repeats in one cell must use the same ordered GPU set")
    throughput = [item.tokens_per_second for item in ordered]
    return BenchmarkProfileAggregate(
        group=ordered[0].group,
        profile=ordered[0].profile,
        world_size=ordered[0].world_size,
        gpu_indices=next(iter(gpu_sets)),
        repeats=(1, 2, 3),
        run_ids=tuple(item.run_id for item in ordered),
        tokens_per_second_median=statistics.median(throughput),
        tokens_per_second_min=min(throughput),
        tokens_per_second_max=max(throughput),
        step_time_ms_median=statistics.median(
            item.effective_step_time.median_ms for item in ordered
        ),
        peak_memory_bytes_median=statistics.median(
            item.peak_memory_allocated_bytes for item in ordered
        ),
        data_wait_percent_median=statistics.median(item.data_wait_percent for item in ordered),
    )


def build_m3_matrix_summary(
    runs: Iterable[DDPBenchmarkRunResult],
) -> DDPBenchmarkMatrixSummary:
    """Validate and aggregate all required M3 benchmark cells."""

    values = list(runs)
    if not values:
        raise ValueError("M3 benchmark matrix is empty")
    if any(item.git_dirty for item in values):
        raise ValueError("formal M3 matrix cannot include dirty Git runs")
    identities = {
        (item.base_config_sha256, item.git_commit, item.model_parameter_count) for item in values
    }
    if len(identities) != 1:
        raise ValueError("matrix cannot mix config, Git, or model identities")
    grouped: dict[
        tuple[BenchmarkGroup, BenchmarkProfile, int],
        list[DDPBenchmarkRunResult],
    ] = defaultdict(list)
    for item in values:
        grouped[(item.group, item.profile, item.world_size)].append(item)
    expected = {
        ("standard", profile, world_size)
        for profile in cast(tuple[BenchmarkProfile, ...], ("strong", "weak"))
        for world_size in (1, 2, 4, 8)
    }
    expected.update(
        {
            ("same_numa", "weak", 4),
            ("cross_numa", "weak", 4),
        }
    )
    if set(grouped) != expected:
        missing = sorted(expected - set(grouped))
        unexpected = sorted(set(grouped) - expected)
        raise ValueError(f"incomplete M3 matrix: missing={missing}, unexpected={unexpected}")
    aggregates = {key: _aggregate_cell(items) for key, items in grouped.items()}
    strong_one = aggregates[("standard", "strong", 1)].tokens_per_second_median
    weak_one_per_gpu = aggregates[("standard", "weak", 1)].tokens_per_second_median
    standard: list[BenchmarkProfileAggregate] = []
    for profile in cast(tuple[BenchmarkProfile, ...], ("strong", "weak")):
        for world_size in (1, 2, 4, 8):
            cell = aggregates[("standard", profile, world_size)]
            if profile == "strong":
                efficiency = cell.tokens_per_second_median / (world_size * strong_one)
            else:
                efficiency = cell.tokens_per_second_median / world_size / weak_one_per_gpu
            standard.append(cell.model_copy(update={"scaling_efficiency": efficiency}))
    numa = [
        aggregates[("same_numa", "weak", 4)],
        aggregates[("cross_numa", "weak", 4)],
    ]
    base_config_sha256, git_commit, model_parameter_count = next(iter(identities))
    return DDPBenchmarkMatrixSummary(
        base_config_sha256=base_config_sha256,
        git_commit=git_commit,
        model_parameter_count=model_parameter_count,
        standard=tuple(standard),
        numa=tuple(numa),
    )
