"""Shared fail-closed GPU selection preflight for bounded M3 Smoke supervisors."""

from __future__ import annotations

import argparse
import csv
import subprocess
from typing import TypedDict

MAX_MEMORY_USED_MIB = 1024
MAX_UTILIZATION_PERCENT = 10
MAX_TEMPERATURE_C = 79


class GpuPreflight(TypedDict):
    """GPU state captured immediately before a formal distributed Smoke."""

    index: int
    name: str
    memory_used_mib: int
    utilization_percent: int
    temperature_c: int
    driver_version: str


def parse_gpu_indices(value: str) -> tuple[int, ...]:
    """Parse a non-empty comma-separated list of unique physical GPU indices."""

    try:
        indices = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU indices must be comma-separated integers") from exc
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("GPU indices must be non-negative")
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("GPU indices must not contain duplicates")
    return indices


def inspect_gpus(indices: tuple[int, ...]) -> tuple[GpuPreflight, ...]:
    """Capture the selected GPUs with one bounded read-only nvidia-smi query."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("nvidia-smi GPU preflight failed") from exc
    selected = set(indices)
    rows: list[GpuPreflight] = []
    try:
        for raw in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
            if len(raw) != 6:
                raise RuntimeError("unexpected nvidia-smi GPU preflight output")
            index = int(raw[0].strip())
            if index in selected:
                rows.append(
                    {
                        "index": index,
                        "name": raw[1].strip(),
                        "memory_used_mib": int(raw[2].strip()),
                        "utilization_percent": int(raw[3].strip()),
                        "temperature_c": int(raw[4].strip()),
                        "driver_version": raw[5].strip(),
                    }
                )
    except ValueError as exc:
        raise RuntimeError("unexpected nvidia-smi GPU preflight output") from exc
    by_index = {row["index"]: row for row in rows}
    if set(by_index) != selected:
        raise RuntimeError("one or more requested GPUs were not discovered")
    return tuple(by_index[index] for index in indices)


def validate_gpu_preflight(rows: tuple[GpuPreflight, ...]) -> None:
    """Reject a busy or hot selected GPU; formal evidence has no override."""

    rejected = [
        row["index"]
        for row in rows
        if row["memory_used_mib"] > MAX_MEMORY_USED_MIB
        or row["utilization_percent"] > MAX_UTILIZATION_PERCENT
        or row["temperature_c"] > MAX_TEMPERATURE_C
    ]
    if rejected:
        raise RuntimeError(f"GPU preflight rejected physical indices: {rejected}")
