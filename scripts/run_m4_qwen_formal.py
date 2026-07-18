#!/usr/bin/env python3
"""Strict four-GPU supervisor for M4.3 Probe, Step-25 stop, and Resume/export."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from tinyllm.lineage import read_git_identity
from tinyllm.training.m4_qwen_config import load_m4_qwen_config
from tinyllm.training.m4_qwen_schema import M4QwenRunResult
from tinyllm.training.smoke_preflight import (
    MAX_MEMORY_USED_MIB,
    MAX_TEMPERATURE_C,
    MAX_UTILIZATION_PERCENT,
    inspect_gpus,
    parse_gpu_indices,
    validate_gpu_preflight,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selected_numa_nodes(gpu_indices: tuple[int, ...]) -> tuple[int, ...]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        buses = {
            int(row[0].strip()): row[1].strip().lower()
            for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True)
        }
        nodes: list[int] = []
        for index in gpu_indices:
            bus = buses[index]
            if bus.startswith("00000000:"):
                bus = "0000:" + bus.removeprefix("00000000:")
            node_path = Path("/sys/bus/pci/devices") / bus / "numa_node"
            nodes.append(int(node_path.read_text(encoding="utf-8").strip()))
    except (KeyError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("cannot resolve selected GPU NUMA identity") from exc
    if any(node < 0 for node in nodes) or len(set(nodes)) != 1:
        raise RuntimeError("formal M4 requires four GPUs on one NUMA node")
    return tuple(nodes)


def _command(
    *,
    config_path: Path,
    artifact_root: Path,
    model_dir: Path,
    output_root: Path,
    phase: Literal["probe", "train", "resume"],
    resume_run: Path | None,
) -> list[str]:
    executable = Path(sys.executable).with_name("torchrun")
    if not executable.is_file():
        raise RuntimeError("torchrun is unavailable beside the active Python interpreter")
    command = [
        str(executable),
        "--standalone",
        "--nproc-per-node=4",
        "-m",
        "tinyllm.training.m4_qwen_worker",
        "--config",
        str(config_path),
        "--artifact-root",
        str(artifact_root),
        "--model-dir",
        str(model_dir),
        "--output-root",
        str(output_root),
    ]
    if phase == "probe":
        command.append("--probe-only")
    elif phase == "train":
        command.extend(("--stop-after-step", "25"))
    else:
        if resume_run is None:
            raise RuntimeError("resume phase requires --resume-run")
        command.extend(("--resume-run", str(resume_run), "--export-final"))
    return command


def run_phase(
    *,
    config_path: Path,
    artifact_root: Path,
    model_dir: Path,
    output_root: Path,
    evidence_dir: Path,
    gpu_indices: tuple[int, ...],
    phase: Literal["probe", "train", "resume"],
    resume_run: Path | None,
    timeout_seconds: int,
) -> M4QwenRunResult:
    """Preflight and supervise exactly one immutable M4.3 phase."""

    project_root = Path(__file__).resolve().parents[1]
    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    model_dir = model_dir.resolve()
    output_root = output_root.resolve()
    evidence_dir = evidence_dir.resolve()
    resume_run = resume_run.resolve() if resume_run is not None else None
    config = load_m4_qwen_config(config_path)
    if len(gpu_indices) != config.distributed.world_size:
        raise RuntimeError("formal M4 requires exactly four explicit physical GPU indices")
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise RuntimeError("formal M4 Qwen phase requires a clean Git worktree")
    if evidence_dir.exists():
        raise RuntimeError("evidence directory already exists")
    evidence_dir.mkdir(parents=True)
    started_at = datetime.now(UTC)
    preflight = inspect_gpus(gpu_indices)
    numa_nodes = _selected_numa_nodes(gpu_indices)
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
            "numa_nodes": numa_nodes,
            "gpus": preflight,
        },
    )
    try:
        validate_gpu_preflight(preflight)
    except RuntimeError:
        _write_json(
            evidence_dir / "summary.json",
            {"schema_version": "1.0", "status": "fail", "reason": "gpu_preflight"},
        )
        raise

    command = _command(
        config_path=config_path,
        artifact_root=artifact_root,
        model_dir=model_dir,
        output_root=output_root,
        phase=phase,
        resume_run=resume_run,
    )
    _write_json(
        evidence_dir / "command.json",
        {
            "schema_version": "1.0",
            "git_commit": git_commit,
            "phase": phase,
            "config": config_path.name,
            "gpu_indices": gpu_indices,
            "world_size": 4,
            "argv": [Path(command[0]).name, *command[1:]],
        },
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
    environment["OMP_NUM_THREADS"] = "1"
    environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
        (evidence_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (evidence_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        _write_json(
            evidence_dir / "summary.json",
            {"schema_version": "1.0", "status": "fail", "reason": "timeout"},
        )
        raise RuntimeError(f"M4 Qwen {phase} phase timed out") from exc
    (evidence_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (evidence_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    expected_nonzero = phase == "train"
    if (completed.returncode != 0) != expected_nonzero:
        _write_json(
            evidence_dir / "summary.json",
            {
                "schema_version": "1.0",
                "status": "fail",
                "reason": "unexpected_torchrun_exit",
                "exit_code": completed.returncode,
            },
        )
        raise RuntimeError(f"M4 Qwen {phase} phase returned an unexpected exit code")
    try:
        result = M4QwenRunResult.model_validate_json(completed.stdout)
    except ValueError as exc:
        raise RuntimeError("M4 Qwen worker emitted an invalid Rank-zero result") from exc
    expected_status = {
        "probe": "probe_succeeded",
        "train": "interrupted",
        "resume": "succeeded",
    }[phase]
    if result.status != expected_status:
        raise RuntimeError("M4 Qwen phase result does not match the requested phase")
    hardware = cast(
        dict[str, object],
        json.loads((result.artifact_dir / "hardware.json").read_text(encoding="utf-8")),
    )
    ranks = cast(list[dict[str, object]], hardware["ranks"])
    recorded_indices = tuple(cast(int, item["physical_gpu_index"]) for item in ranks)
    if recorded_indices != gpu_indices:
        raise RuntimeError("Run hardware lineage does not match requested physical GPUs")
    if result.git_commit != git_commit or result.git_dirty:
        raise RuntimeError("formal M4 Run Git lineage differs from the clean supervisor")
    finished_at = datetime.now(UTC)
    _write_json(
        evidence_dir / "summary.json",
        {
            "schema_version": "1.0",
            "status": "pass",
            "phase": phase,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "result": result.to_dict(),
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the formal M4.3 supervisor interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--gpu-indices", type=parse_gpu_indices, required=True)
    parser.add_argument("--phase", choices=("probe", "train", "resume"), required=True)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser


def main() -> int:
    """Run one strict phase and print its validated result."""

    args = build_parser().parse_args()
    try:
        result = run_phase(
            config_path=args.config,
            artifact_root=args.artifact_root,
            model_dir=args.model_dir,
            output_root=args.output_root,
            evidence_dir=args.evidence_dir,
            gpu_indices=args.gpu_indices,
            phase=args.phase,
            resume_run=args.resume_run,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"M4 Qwen formal phase failed: {exc}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
