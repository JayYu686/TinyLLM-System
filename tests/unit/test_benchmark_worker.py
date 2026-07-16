from __future__ import annotations

from dataclasses import dataclass

import pytest

from tinyllm.benchmark.ddp_worker import (
    _communication_from_profiler,
    summarize_timings,
)


@dataclass
class _Event:
    key: str
    device_time_total: float


class _Profiler:
    def key_averages(self) -> list[_Event]:
        return [
            _Event("aten::mm", 500.0),
            _Event("ncclDevKernel_AllReduce", 2500.0),
        ]


def test_timing_summary_uses_interpolated_p95_and_raw_count() -> None:
    summary = summarize_timings([1.0, 2.0, 3.0, 4.0])
    assert summary.count == 4
    assert summary.total_ms == 10.0
    assert summary.median_ms == 2.5
    assert summary.p95_ms == pytest.approx(3.85)


def test_profiler_communication_distinguishes_measured_and_missing() -> None:
    measured = _communication_from_profiler(
        _Profiler(),  # type: ignore[arg-type]
        world_size=2,
        profiled_steps=5,
    )
    assert measured.status == "measured"
    assert measured.device_time_ms == 2.5
    assert measured.event_keys == ("ncclDevKernel_AllReduce",)

    unavailable = _communication_from_profiler(
        None,
        world_size=2,
        profiled_steps=5,
    )
    assert unavailable.status == "not_collected"
    single = _communication_from_profiler(
        None,
        world_size=1,
        profiled_steps=5,
    )
    assert single.status == "not_applicable"
