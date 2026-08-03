from __future__ import annotations

from scripts.wait_for_gpus import select_gpus
from tinyllm.training.smoke_preflight import GpuPreflight


def _gpu(index: int, *, memory: int = 0, utilization: int = 0) -> GpuPreflight:
    return {
        "index": index,
        "name": "NVIDIA GeForce RTX 3090",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "temperature_c": 30,
        "driver_version": "535.183.01",
    }


def test_wait_selector_preserves_priority_and_rejects_busy_rows() -> None:
    rows = (
        _gpu(4, memory=2048),
        _gpu(5),
        _gpu(6, utilization=90),
        _gpu(7),
        _gpu(8),
        _gpu(9),
    )

    assert select_gpus(rows, count=4) == (5, 7, 8, 9)
    assert select_gpus(rows, count=5) is None
