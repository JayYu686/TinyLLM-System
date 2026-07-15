from __future__ import annotations

import argparse

import pytest

from scripts.run_m3_ddp_smoke import (
    MAX_MEMORY_USED_MIB,
    GpuPreflight,
    parse_gpu_indices,
    validate_gpu_preflight,
)


def _gpu(*, index: int, memory_used_mib: int = 0) -> GpuPreflight:
    return {
        "index": index,
        "name": "NVIDIA GeForce RTX 3090",
        "memory_used_mib": memory_used_mib,
        "utilization_percent": 0,
        "temperature_c": 30,
        "driver_version": "test",
    }


def test_parse_gpu_indices_preserves_explicit_rank_order() -> None:
    assert parse_gpu_indices("6,9,4") == (6, 9, 4)

    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        parse_gpu_indices("4,4")


def test_gpu_preflight_has_no_busy_override() -> None:
    validate_gpu_preflight((_gpu(index=4), _gpu(index=5)))

    with pytest.raises(RuntimeError, match=r"\[5\]"):
        validate_gpu_preflight(
            (_gpu(index=4), _gpu(index=5, memory_used_mib=MAX_MEMORY_USED_MIB + 1))
        )
