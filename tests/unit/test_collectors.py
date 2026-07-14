from collections.abc import Sequence

import pytest

from tinyllm.doctor.collector import collect_gpus, collect_torch, parse_gpu_csv
from tinyllm.doctor.runner import CommandResult


class FakeRunner:
    def __init__(self, result: CommandResult) -> None:
        self.result = result

    def run(self, args: Sequence[str], *, timeout: float = 10.0) -> CommandResult:
        del timeout
        return CommandResult(
            tuple(args),
            self.result.returncode,
            self.result.stdout,
            self.result.stderr,
            available=self.result.available,
            timed_out=self.result.timed_out,
        )


def test_parse_gpu_csv() -> None:
    raw = "0, NVIDIA GeForce RTX 3090, 24576, 1, 0000:01:00.0, 350.00, 535.1, 8.6, 31, 0"
    rows = parse_gpu_csv(raw)
    assert rows[0]["index"] == 0
    assert rows[0]["memory_total_mib"] == 24576
    assert rows[0]["compute_capability"] == "8.6"


def test_parse_gpu_csv_rejects_changed_shape() -> None:
    with pytest.raises(ValueError, match="expected 10 GPU fields"):
        parse_gpu_csv("0, RTX 3090")


def test_missing_nvidia_smi_is_a_required_failure() -> None:
    runner = FakeRunner(CommandResult(("nvidia-smi",), None, "", "missing", available=False))
    gpus, checks = collect_gpus(runner)
    assert gpus == []
    assert checks[0].required is True
    assert checks[0].status == "fail"


def test_malformed_nvidia_smi_is_a_required_failure() -> None:
    runner = FakeRunner(CommandResult(("nvidia-smi",), 0, "bad,row", ""))
    _, checks = collect_gpus(runner)
    assert checks[0].status == "fail"
    assert "parsed" in checks[0].summary


def test_broken_torch_is_a_required_failure() -> None:
    runner = FakeRunner(CommandResult(("python",), 1, "", "RuntimeError"))
    inventory, check = collect_torch(runner)
    assert inventory == {}
    assert check.required is True
    assert check.status == "fail"


def test_torch_without_cuda_is_a_required_failure() -> None:
    runner = FakeRunner(
        CommandResult(
            ("python",),
            0,
            '{"version": "2.5.0", "cuda_available": false, "gpu_count": 0}',
            "",
        )
    )
    inventory, check = collect_torch(runner)
    assert inventory["version"] == "2.5.0"
    assert check.status == "fail"


def test_empty_gpu_inventory_is_a_required_failure() -> None:
    runner = FakeRunner(CommandResult(("nvidia-smi",), 0, "", ""))
    gpus, checks = collect_gpus(runner)
    assert gpus == []
    assert checks[0].status == "fail"
