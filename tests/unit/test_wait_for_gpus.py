from __future__ import annotations

import os
import sys
import time

import pytest

from scripts import wait_for_gpus
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
    assert select_gpus(rows, count=1, max_memory_used_mib=3072) == (4,)


def test_waiter_retries_transient_preflight_failure_before_exec(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attempts = 0

    def inspect(_: tuple[int, ...]) -> tuple[GpuPreflight, ...]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient nvidia-smi failure")
        return (_gpu(0),)

    class ExecCalled(RuntimeError):
        pass

    def execvpe(executable: str, command: list[str], env: dict[str, str]) -> None:
        assert executable.endswith("/bin/true")
        assert command == ["/bin/true", "0"]
        assert env
        raise ExecCalled

    monkeypatch.setattr(wait_for_gpus, "inspect_gpus", inspect)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(os, "execvpe", execvpe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wait_for_gpus.py",
            "--candidate-gpus",
            "0",
            "--count",
            "1",
            "--poll-seconds",
            "5",
            "--",
            "/bin/true",
            "{gpu_index}",
        ],
    )

    with pytest.raises(ExecCalled):
        wait_for_gpus.main()

    captured = capsys.readouterr()
    assert attempts == 2
    assert '"status": "preflight_retry"' in captured.err
    assert '"consecutive_errors": 1' in captured.err
    assert '"status": "selected"' in captured.out
