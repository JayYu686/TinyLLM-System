from __future__ import annotations

import argparse
import subprocess

import pytest

from scripts.run_m3_ddp_smoke import (
    MAX_MEMORY_USED_MIB,
    GpuPreflight,
    inspect_gpus,
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


def test_gpu_inspection_parses_requested_indices_in_rank_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "\n".join(
        [
            "4, NVIDIA GeForce RTX 3090, 10, 0, 31, 570.00",
            "6, NVIDIA GeForce RTX 3090, 20, 1, 32, 570.00",
        ]
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )

    rows = inspect_gpus((6, 4))

    assert [row["index"] for row in rows] == [6, 4]
    assert rows[0]["memory_used_mib"] == 20


def test_gpu_inspection_rejects_missing_or_malformed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "4, RTX 3090, invalid, 0, 31, 570.00\n", ""
        ),
    )
    with pytest.raises(RuntimeError, match="unexpected"):
        inspect_gpus((4,))

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "4, RTX 3090, 0, 0, 31, 570.00\n", ""
        ),
    )
    with pytest.raises(RuntimeError, match="not discovered"):
        inspect_gpus((5,))
