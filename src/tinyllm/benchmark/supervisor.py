"""Fail-closed supervisor for one formal M3 DDP benchmark repetition."""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

from tinyllm.benchmark.config import (
    BenchmarkProfile,
    load_ddp_benchmark_config,
    resolve_benchmark_profile,
    validate_formal_m3_config,
)
from tinyllm.benchmark.schema import BenchmarkGroup, DDPBenchmarkRunResult
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash
from tinyllm.training.smoke_preflight import (
    MAX_MEMORY_USED_MIB,
    MAX_TEMPERATURE_C,
    MAX_UTILIZATION_PERCENT,
    inspect_gpus,
    validate_gpu_preflight,
)


class BenchmarkPreflightError(RuntimeError):
    """A formal run was rejected before torchrun started."""


class BenchmarkRunError(RuntimeError):
    """A launched benchmark failed or emitted invalid evidence."""


class GpuTelemetry(TypedDict):
    """One selected-GPU telemetry sample captured during a benchmark."""

    captured_at: str
    index: int
    memory_used_mib: int
    utilization_percent: int
    temperature_c: int
    sm_clock_mhz: int
    power_draw_watts: float


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torchrun_command(
    *,
    config_path: Path,
    output_root: Path,
    profile: BenchmarkProfile,
    group: BenchmarkGroup,
    repeat: int,
    world_size: int,
) -> list[str]:
    executable = Path(sys.executable).with_name("torchrun")
    if not executable.is_file():
        raise BenchmarkPreflightError(
            "torchrun is unavailable beside the active Python interpreter"
        )
    return [
        str(executable),
        "--standalone",
        f"--nproc-per-node={world_size}",
        "-m",
        "tinyllm.benchmark.ddp_worker",
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--profile",
        profile,
        "--group",
        group,
        "--repeat",
        str(repeat),
    ]


def _inspect_telemetry(indices: tuple[int, ...]) -> tuple[GpuTelemetry, ...]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,"
                "clocks.sm,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise BenchmarkRunError("nvidia-smi telemetry collection failed") from exc
    captured_at = datetime.now(UTC).isoformat()
    selected = set(indices)
    rows: dict[int, GpuTelemetry] = {}
    try:
        for raw in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
            if len(raw) != 6:
                raise BenchmarkRunError("unexpected nvidia-smi telemetry output")
            index = int(raw[0].strip())
            if index in selected:
                rows[index] = {
                    "captured_at": captured_at,
                    "index": index,
                    "memory_used_mib": int(raw[1].strip()),
                    "utilization_percent": int(raw[2].strip()),
                    "temperature_c": int(raw[3].strip()),
                    "sm_clock_mhz": int(raw[4].strip()),
                    "power_draw_watts": float(raw[5].strip()),
                }
    except ValueError as exc:
        raise BenchmarkRunError("unexpected nvidia-smi telemetry output") from exc
    if set(rows) != selected:
        raise BenchmarkRunError("telemetry did not discover every selected GPU")
    return tuple(rows[index] for index in indices)


def _inspect_topology() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise BenchmarkPreflightError("nvidia-smi topology inspection failed") from exc
    return completed.stdout


def _validate_group(group: BenchmarkGroup, indices: tuple[int, ...]) -> None:
    if group != "standard" and len(indices) != 4:
        raise BenchmarkPreflightError("NUMA comparison groups require four GPUs")
    if group == "same_numa" and indices != (6, 7, 8, 9):
        raise BenchmarkPreflightError("same_numa group must use ordered GPUs 6,7,8,9")
    if group == "cross_numa" and indices != (4, 5, 6, 7):
        raise BenchmarkPreflightError("cross_numa group must use ordered GPUs 4,5,6,7")


def _validate_numa_topology(
    topology: str,
    *,
    group: BenchmarkGroup,
    indices: tuple[int, ...],
) -> None:
    """Verify that NUMA labels agree with the controlled group name."""

    if group == "standard":
        return
    affinities: dict[int, int] = {}
    for line in topology.splitlines():
        columns = line.split()
        if not columns or not columns[0].startswith("GPU"):
            continue
        try:
            index = int(columns[0][3:])
        except ValueError:
            continue
        if index not in indices or len(columns) < 4:
            continue
        try:
            affinities[index] = int(columns[-2])
        except ValueError:
            continue
    if set(affinities) != set(indices):
        raise BenchmarkPreflightError("topology lacks NUMA affinity for selected GPUs")
    unique = set(affinities.values())
    if group == "same_numa" and len(unique) != 1:
        raise BenchmarkPreflightError("same_numa label disagrees with actual topology")
    if group == "cross_numa" and len(unique) < 2:
        raise BenchmarkPreflightError("cross_numa label disagrees with actual topology")


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _parse_worker_result(stdout_path: Path) -> DDPBenchmarkRunResult:
    try:
        return DDPBenchmarkRunResult.model_validate_json(stdout_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchmarkRunError("benchmark worker emitted an invalid Rank-zero result") from exc


def run_formal_benchmark(
    *,
    config_path: Path,
    output_root: Path,
    evidence_dir: Path,
    profile: BenchmarkProfile,
    group: BenchmarkGroup,
    repeat: int,
    gpu_indices: tuple[int, ...],
    timeout_seconds: int,
) -> DDPBenchmarkRunResult:
    """Preflight, supervise, retain telemetry, and validate one formal repetition."""

    project_root = Path(__file__).resolve().parents[3]
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    evidence_dir = evidence_dir.resolve()
    config = load_ddp_benchmark_config(config_path)
    validate_formal_m3_config(config)
    resolved = resolve_benchmark_profile(
        config,
        profile=profile,
        world_size=len(gpu_indices),
        repeat=repeat,
    )
    _validate_group(group, gpu_indices)
    if group != "standard" and profile != "weak":
        raise BenchmarkPreflightError("NUMA comparison uses only the Weak profile")
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise BenchmarkPreflightError("formal benchmark requires a clean Git worktree")
    if evidence_dir.exists():
        raise BenchmarkPreflightError("evidence directory already exists")
    evidence_dir.mkdir(parents=True)
    started_at = datetime.now(UTC)
    preflight = inspect_gpus(gpu_indices)
    preflight_passed = all(
        row["memory_used_mib"] <= MAX_MEMORY_USED_MIB
        and row["utilization_percent"] <= MAX_UTILIZATION_PERCENT
        and row["temperature_c"] <= MAX_TEMPERATURE_C
        for row in preflight
    )
    _write_json(
        evidence_dir / "preflight.json",
        {
            "schema_version": "1.0",
            "status": "pass" if preflight_passed else "fail",
            "captured_at": started_at.isoformat(),
            "thresholds": {
                "memory_used_mib_lte": MAX_MEMORY_USED_MIB,
                "utilization_percent_lte": MAX_UTILIZATION_PERCENT,
                "temperature_c_lte": MAX_TEMPERATURE_C,
            },
            "gpus": preflight,
        },
    )
    try:
        topology = _inspect_topology()
        _validate_numa_topology(topology, group=group, indices=gpu_indices)
    except BenchmarkPreflightError:
        _write_json(
            evidence_dir / "summary.json",
            {"schema_version": "1.0", "status": "fail", "reason": "topology"},
        )
        raise
    (evidence_dir / "topology.txt").write_text(topology, encoding="utf-8")
    try:
        validate_gpu_preflight(preflight)
    except RuntimeError as exc:
        _write_json(
            evidence_dir / "summary.json",
            {"schema_version": "1.0", "status": "fail", "reason": "gpu_preflight"},
        )
        raise BenchmarkPreflightError(str(exc)) from exc

    command = _torchrun_command(
        config_path=config_path,
        output_root=output_root,
        profile=profile,
        group=group,
        repeat=repeat,
        world_size=len(gpu_indices),
    )
    _write_json(
        evidence_dir / "command.json",
        {
            "schema_version": "1.0",
            "git_commit": git_commit,
            "base_config_sha256": canonical_config_hash(config),
            "profile": profile,
            "group": group,
            "repeat": repeat,
            "gpu_indices": gpu_indices,
            "world_size": len(gpu_indices),
            "resolved_profile": resolved.to_dict(),
            "argv": [Path(command[0]).name, *command[1:]],
        },
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
    environment["OMP_NUM_THREADS"] = "1"
    environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    stdout_path = evidence_dir / "stdout.log"
    stderr_path = evidence_dir / "stderr.log"
    telemetry: list[GpuTelemetry] = []
    timed_out = False
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_stream,
        stderr_path.open("w", encoding="utf-8") as stderr_stream,
    ):
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        try:
            while process.poll() is None:
                telemetry.extend(_inspect_telemetry(gpu_indices))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    _terminate_process_group(process)
                    break
                time.sleep(min(config.benchmark.telemetry_interval_seconds, remaining))
        except Exception as exc:
            _terminate_process_group(process)
            _write_json(
                evidence_dir / "summary.json",
                {"schema_version": "1.0", "status": "fail", "reason": "telemetry"},
            )
            if isinstance(exc, BenchmarkRunError):
                raise
            raise BenchmarkRunError("benchmark supervision failed") from exc
        return_code = process.wait()
    telemetry.extend(_inspect_telemetry(gpu_indices))
    _write_json(
        evidence_dir / "telemetry.json",
        {"schema_version": "1.0", "samples": telemetry},
    )
    if timed_out:
        _write_json(
            evidence_dir / "summary.json",
            {"schema_version": "1.0", "status": "fail", "reason": "timeout"},
        )
        raise BenchmarkRunError("benchmark timed out")
    if return_code != 0:
        _write_json(
            evidence_dir / "summary.json",
            {
                "schema_version": "1.0",
                "status": "fail",
                "reason": "torchrun_nonzero_exit",
                "exit_code": return_code,
            },
        )
        raise BenchmarkRunError(f"benchmark failed with exit code {return_code}")
    try:
        result = _parse_worker_result(stdout_path)
    except BenchmarkRunError:
        _write_json(
            evidence_dir / "summary.json",
            {
                "schema_version": "1.0",
                "status": "fail",
                "reason": "invalid_worker_result",
            },
        )
        raise
    expected_indices = tuple(item.physical_gpu_index for item in result.rank_metrics)
    if expected_indices != gpu_indices:
        raise BenchmarkRunError("worker hardware lineage differs from selected GPU indices")
    if (
        result.git_commit != git_commit
        or result.git_dirty
        or result.base_config_sha256 != canonical_config_hash(config)
        or result.profile != profile
        or result.group != group
        or result.repeat != repeat
        or result.world_size != len(gpu_indices)
    ):
        _write_json(
            evidence_dir / "summary.json",
            {
                "schema_version": "1.0",
                "status": "fail",
                "reason": "lineage_mismatch",
            },
        )
        raise BenchmarkRunError("worker lineage differs from the supervising command")
    finished_at = datetime.now(UTC)
    _write_json(
        evidence_dir / "summary.json",
        {
            "schema_version": "1.0",
            "status": "pass",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "result": result.to_dict(),
        },
    )
    return result


def exit_code_for_error(error: Exception) -> Literal[2, 3, 4]:
    """Map benchmark supervisor failures to the public CLI classes."""

    if isinstance(error, BenchmarkPreflightError):
        return 3
    if isinstance(error, BenchmarkRunError):
        return 4
    return 2
