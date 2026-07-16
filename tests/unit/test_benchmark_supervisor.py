from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TextIO

import pytest

from tinyllm.benchmark import (
    BenchmarkTimingSummary,
    CommunicationMeasurement,
    DDPBenchmarkRunResult,
    RankBenchmarkMetrics,
    load_ddp_benchmark_config,
    resolve_benchmark_profile,
)
from tinyllm.benchmark.supervisor import (
    BenchmarkPreflightError,
    BenchmarkRunError,
    _inspect_telemetry,
    _inspect_topology,
    _terminate_process_group,
    _validate_group,
    _validate_numa_topology,
    exit_code_for_error,
    run_formal_benchmark,
)
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.smoke_preflight import GpuPreflight

GIT_COMMIT = "b" * 40


def _gpu(*, index: int, memory: int = 1) -> GpuPreflight:
    return {
        "index": index,
        "name": "NVIDIA GeForce RTX 3090",
        "memory_used_mib": memory,
        "utilization_percent": 0,
        "temperature_c": 30,
        "driver_version": "535.261.03",
    }


def _timing(value: float) -> BenchmarkTimingSummary:
    return BenchmarkTimingSummary(
        count=100,
        total_ms=value * 100,
        min_ms=value,
        median_ms=value,
        p95_ms=value,
        max_ms=value,
    )


def _formal_result(tmp_path: Path) -> DDPBenchmarkRunResult:
    config = load_ddp_benchmark_config(Path("configs/benchmark/m3_tinygpt_120m_ddp.yaml"))
    resolved = resolve_benchmark_profile(
        config,
        profile="weak",
        world_size=1,
        repeat=2,
    )
    base_hash = canonical_config_hash(config)
    resolved_hash = canonical_config_hash(
        {"config": config.to_dict(), "group": "standard", "resolved": resolved.to_dict()}
    )
    now = datetime(2026, 7, 16, tzinfo=UTC)
    rank = RankBenchmarkMetrics(
        rank=0,
        local_rank=0,
        physical_gpu_index=5,
        gpu_name="NVIDIA GeForce RTX 3090",
        step_time_ms=tuple([10.0] * 100),
        data_wait_ms=tuple([1.0] * 100),
        peak_memory_allocated_bytes=1024,
        communication=CommunicationMeasurement(
            status="not_applicable",
            profiled_steps=0,
        ),
    )
    return DDPBenchmarkRunResult(
        run_id=generate_run_id("benchmark", resolved_hash, now=now, nonce="1234"),
        artifact_dir=(tmp_path / "run").resolve(),
        group="standard",
        profile="weak",
        world_size=1,
        repeat=2,
        seed=resolved.seed,
        base_config_sha256=base_hash,
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
        measurement_steps=100,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        global_batch_size=1,
        predicted_tokens_per_step=1023,
        tokens_per_second=102_300.0,
        samples_per_second=100.0,
        effective_step_time=_timing(10.0),
        effective_data_wait=_timing(1.0),
        data_wait_percent=10.0,
        peak_memory_allocated_bytes=1024,
        rank_metrics=(rank,),
    )


class _CompletedProcess:
    def __init__(self, *, stdout: TextIO, result: DDPBenchmarkRunResult) -> None:
        stdout.write(result.model_dump_json(indent=2))
        stdout.flush()
        self.pid = 12345

    def poll(self) -> int:
        return 0

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        return 0


class _ImmediateProcess:
    def __init__(self, *, stdout: TextIO, text: str, return_code: int) -> None:
        stdout.write(text)
        stdout.flush()
        self.pid = 12345
        self.return_code = return_code

    def poll(self) -> int:
        return self.return_code

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        return self.return_code


def _patch_clean_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor.read_git_identity",
        lambda _root: (GIT_COMMIT, False),
    )
    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor.inspect_gpus",
        lambda _indices: (_gpu(index=5),),
    )
    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor._inspect_topology",
        lambda: "GPU0 X\n",
    )
    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor._inspect_telemetry",
        lambda _indices: (
            {
                "captured_at": datetime.now(UTC).isoformat(),
                "index": 5,
                "memory_used_mib": 1024,
                "utilization_percent": 99,
                "temperature_c": 60,
                "sm_clock_mhz": 1800,
                "power_draw_watts": 300.0,
            },
        ),
    )


def test_formal_supervisor_retains_preflight_telemetry_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _formal_result(tmp_path)
    _patch_clean_preflight(monkeypatch)
    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor.subprocess.Popen",
        lambda *_args, **kwargs: _CompletedProcess(stdout=kwargs["stdout"], result=result),
    )

    actual = run_formal_benchmark(
        config_path=Path("configs/benchmark/m3_tinygpt_120m_ddp.yaml"),
        output_root=(tmp_path / "runs").resolve(),
        evidence_dir=(tmp_path / "evidence").resolve(),
        profile="weak",
        group="standard",
        repeat=2,
        gpu_indices=(5,),
        timeout_seconds=60,
    )

    assert actual == result
    summary = json.loads((tmp_path / "evidence" / "summary.json").read_text())
    assert summary["status"] == "pass"
    telemetry = json.loads((tmp_path / "evidence" / "telemetry.json").read_text())
    assert telemetry["samples"][0]["sm_clock_mhz"] == 1800


def test_formal_supervisor_retains_busy_preflight_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor.read_git_identity",
        lambda _root: (GIT_COMMIT, False),
    )
    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor.inspect_gpus",
        lambda _indices: (_gpu(index=5, memory=2048),),
    )
    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor._inspect_topology",
        lambda: "GPU0 X\n",
    )
    evidence = tmp_path / "busy"
    with pytest.raises(BenchmarkPreflightError, match="rejected"):
        run_formal_benchmark(
            config_path=Path("configs/benchmark/m3_tinygpt_120m_ddp.yaml"),
            output_root=(tmp_path / "runs").resolve(),
            evidence_dir=evidence.resolve(),
            profile="weak",
            group="standard",
            repeat=1,
            gpu_indices=(5,),
            timeout_seconds=60,
        )
    summary = json.loads((evidence / "summary.json").read_text())
    assert summary == {"schema_version": "1.0", "status": "fail", "reason": "gpu_preflight"}


def test_group_contract_rejects_wrong_numa_indices() -> None:
    _validate_group("same_numa", (6, 7, 8, 9))
    _validate_group("cross_numa", (4, 5, 6, 7))
    with pytest.raises(BenchmarkPreflightError, match="6,7,8,9"):
        _validate_group("same_numa", (5, 6, 7, 8))
    with pytest.raises(BenchmarkPreflightError, match="4,5,6,7"):
        _validate_group("cross_numa", (4, 5, 6, 8))
    with pytest.raises(BenchmarkPreflightError, match="four GPUs"):
        _validate_group("same_numa", (6, 7, 8))


def test_numa_label_is_verified_against_topology() -> None:
    topology = "\n".join(
        [
            "GPU4 X SYS 0-15,32-47 0 N/A",
            "GPU5 SYS X 16-31,48-63 1 N/A",
            "GPU6 SYS PXB 16-31,48-63 1 N/A",
            "GPU7 SYS PIX 16-31,48-63 1 N/A",
            "GPU8 SYS PXB 16-31,48-63 1 N/A",
            "GPU9 SYS PXB 16-31,48-63 1 N/A",
        ]
    )
    _validate_numa_topology(
        topology,
        group="same_numa",
        indices=(6, 7, 8, 9),
    )
    _validate_numa_topology(
        topology,
        group="cross_numa",
        indices=(4, 5, 6, 7),
    )
    _validate_numa_topology(topology, group="standard", indices=(4,))
    with pytest.raises(BenchmarkPreflightError, match="disagrees"):
        _validate_numa_topology(
            topology,
            group="same_numa",
            indices=(4, 5, 6, 7),
        )
    with pytest.raises(BenchmarkPreflightError, match="lacks"):
        _validate_numa_topology(
            "GPU6 malformed",
            group="same_numa",
            indices=(6, 7, 8, 9),
        )


def test_telemetry_parser_preserves_selected_order(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "\n".join(
        [
            "5, 100, 90, 60, 1800, 300.5",
            "6, 101, 91, 61, 1815, 301.5",
        ]
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )

    rows = _inspect_telemetry((6, 5))

    assert [row["index"] for row in rows] == [6, 5]
    assert rows[0]["power_draw_watts"] == 301.5


@pytest.mark.parametrize(
    "output",
    [
        "5, too, short",
        "5, bad, 0, 30, 1800, 300",
        "6, 1, 0, 30, 1800, 300",
    ],
)
def test_telemetry_parser_rejects_malformed_or_missing_rows(
    output: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )
    with pytest.raises(BenchmarkRunError):
        _inspect_telemetry((5,))


def test_telemetry_and_topology_wrap_command_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["nvidia-smi"])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(BenchmarkRunError, match="telemetry"):
        _inspect_telemetry((5,))
    with pytest.raises(BenchmarkPreflightError, match="topology"):
        _inspect_topology()


def test_formal_supervisor_retains_nonzero_and_invalid_worker_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_clean_preflight(monkeypatch)
    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor.subprocess.Popen",
        lambda *_args, **kwargs: _ImmediateProcess(
            stdout=kwargs["stdout"], text="", return_code=17
        ),
    )
    evidence = tmp_path / "nonzero"
    with pytest.raises(BenchmarkRunError, match="exit code 17"):
        run_formal_benchmark(
            config_path=Path("configs/benchmark/m3_tinygpt_120m_ddp.yaml"),
            output_root=(tmp_path / "runs").resolve(),
            evidence_dir=evidence.resolve(),
            profile="weak",
            group="standard",
            repeat=2,
            gpu_indices=(5,),
            timeout_seconds=60,
        )
    assert json.loads((evidence / "summary.json").read_text())["reason"] == "torchrun_nonzero_exit"

    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor.subprocess.Popen",
        lambda *_args, **kwargs: _ImmediateProcess(
            stdout=kwargs["stdout"], text="not-json", return_code=0
        ),
    )
    with pytest.raises(BenchmarkRunError, match="invalid Rank-zero"):
        run_formal_benchmark(
            config_path=Path("configs/benchmark/m3_tinygpt_120m_ddp.yaml"),
            output_root=(tmp_path / "runs").resolve(),
            evidence_dir=(tmp_path / "invalid").resolve(),
            profile="weak",
            group="standard",
            repeat=2,
            gpu_indices=(5,),
            timeout_seconds=60,
        )
    invalid_summary = json.loads((tmp_path / "invalid" / "summary.json").read_text())
    assert invalid_summary["reason"] == "invalid_worker_result"


def test_formal_supervisor_rejects_dirty_existing_and_wrong_numa_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor.read_git_identity",
        lambda _root: (GIT_COMMIT, True),
    )
    with pytest.raises(BenchmarkPreflightError, match="clean Git"):
        run_formal_benchmark(
            config_path=Path("configs/benchmark/m3_tinygpt_120m_ddp.yaml"),
            output_root=(tmp_path / "runs").resolve(),
            evidence_dir=(tmp_path / "dirty").resolve(),
            profile="weak",
            group="standard",
            repeat=1,
            gpu_indices=(5,),
            timeout_seconds=60,
        )

    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor.read_git_identity",
        lambda _root: (GIT_COMMIT, False),
    )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(BenchmarkPreflightError, match="already exists"):
        run_formal_benchmark(
            config_path=Path("configs/benchmark/m3_tinygpt_120m_ddp.yaml"),
            output_root=(tmp_path / "runs").resolve(),
            evidence_dir=existing.resolve(),
            profile="weak",
            group="standard",
            repeat=1,
            gpu_indices=(5,),
            timeout_seconds=60,
        )

    with pytest.raises(BenchmarkPreflightError, match="Weak"):
        run_formal_benchmark(
            config_path=Path("configs/benchmark/m3_tinygpt_120m_ddp.yaml"),
            output_root=(tmp_path / "runs").resolve(),
            evidence_dir=(tmp_path / "numa").resolve(),
            profile="strong",
            group="same_numa",
            repeat=1,
            gpu_indices=(6, 7, 8, 9),
            timeout_seconds=60,
        )


def test_process_group_termination_escalates_and_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []

    class Process:
        pid = 123
        waits = 0

        def poll(self) -> None:
            return None

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("torchrun", 10)
            return -9

    monkeypatch.setattr(
        "tinyllm.benchmark.supervisor.os.killpg",
        lambda _pid, signal_value: signals.append(signal_value),
    )
    _terminate_process_group(Process())  # type: ignore[arg-type]
    assert len(signals) == 2
    assert exit_code_for_error(BenchmarkPreflightError("x")) == 3
    assert exit_code_for_error(BenchmarkRunError("x")) == 4
    assert exit_code_for_error(ValueError("x")) == 2
