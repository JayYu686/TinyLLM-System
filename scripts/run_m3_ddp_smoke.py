#!/usr/bin/env python3
"""Run one fail-closed M3.1 NCCL/BF16 correctness Smoke on explicit idle GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from tinyllm.lineage import read_git_identity
from tinyllm.training import DDPTrainingResult, load_training_config
from tinyllm.training.smoke_preflight import (
    MAX_MEMORY_USED_MIB,
    MAX_TEMPERATURE_C,
    MAX_UTILIZATION_PERCENT,
    GpuPreflight,
    inspect_gpus,
    parse_gpu_indices,
    validate_gpu_preflight,
)

__all__ = [
    "MAX_MEMORY_USED_MIB",
    "MAX_TEMPERATURE_C",
    "MAX_UTILIZATION_PERCENT",
    "GpuPreflight",
    "inspect_gpus",
    "parse_gpu_indices",
    "validate_gpu_preflight",
]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torchrun_command(config_path: Path, output_root: Path, world_size: int) -> list[str]:
    executable = Path(sys.executable).with_name("torchrun")
    if not executable.is_file():
        raise RuntimeError("torchrun is unavailable beside the active Python interpreter")
    return [
        str(executable),
        "--standalone",
        f"--nproc-per-node={world_size}",
        "-m",
        "tinyllm.training.ddp_worker",
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
    ]


def run_smoke(
    *,
    config_path: Path,
    output_root: Path,
    evidence_dir: Path,
    gpu_indices: tuple[int, ...],
    timeout_seconds: int,
) -> DDPTrainingResult:
    """Validate inputs, retain private launch logs, and return a strict rank-zero result."""

    project_root = Path(__file__).resolve().parents[1]
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    evidence_dir = evidence_dir.resolve()
    config = load_training_config(config_path)
    if config.distributed.strategy != "ddp" or config.distributed.backend != "nccl":
        raise RuntimeError("formal GPU Smoke requires distributed.strategy=ddp and backend=nccl")
    if config.precision.dtype != "bf16":
        raise RuntimeError("formal RTX 3090 DDP Smoke requires BF16")
    if config.distributed.world_size != len(gpu_indices):
        raise RuntimeError("GPU index count must equal the resolved DDP world_size")
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise RuntimeError("formal DDP Smoke requires a clean Git worktree")
    if evidence_dir.exists():
        raise RuntimeError("evidence directory already exists")

    preflight = inspect_gpus(gpu_indices)
    evidence_dir.mkdir(parents=True)
    started_at = datetime.now(UTC)
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
        validate_gpu_preflight(preflight)
    except RuntimeError:
        _write_json(
            evidence_dir / "summary.json",
            {"schema_version": "1.0", "status": "fail", "reason": "gpu_preflight"},
        )
        raise

    command = _torchrun_command(config_path, output_root, len(gpu_indices))
    _write_json(
        evidence_dir / "command.json",
        {
            "schema_version": "1.0",
            "git_commit": git_commit,
            "config": config_path.name,
            "gpu_indices": gpu_indices,
            "world_size": len(gpu_indices),
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
        raise RuntimeError("DDP Smoke timed out") from exc

    (evidence_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (evidence_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        _write_json(
            evidence_dir / "summary.json",
            {
                "schema_version": "1.0",
                "status": "fail",
                "reason": "torchrun_nonzero_exit",
                "exit_code": completed.returncode,
            },
        )
        raise RuntimeError(f"DDP Smoke failed with exit code {completed.returncode}")
    try:
        result = DDPTrainingResult.model_validate_json(completed.stdout)
    except ValueError as exc:
        _write_json(
            evidence_dir / "summary.json",
            {"schema_version": "1.0", "status": "fail", "reason": "invalid_worker_result"},
        )
        raise RuntimeError("DDP worker emitted an invalid rank-zero result") from exc

    hardware = cast(
        dict[str, object],
        json.loads((result.artifact_dir / "hardware.json").read_text(encoding="utf-8")),
    )
    devices = cast(list[dict[str, object]], hardware["devices"])
    recorded_indices = tuple(int(cast(int, item["physical_gpu_index"])) for item in devices)
    if recorded_indices != gpu_indices:
        raise RuntimeError("Run hardware lineage does not match requested physical GPU indices")
    if result.git_commit != git_commit or result.git_dirty:
        raise RuntimeError("Run Git lineage does not match the clean supervising commit")

    finished_at = datetime.now(UTC)
    _write_json(
        evidence_dir / "summary.json",
        {
            "schema_version": "1.0",
            "status": "pass",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "run_id": result.run_id,
            "git_commit": git_commit,
            "config_sha256": result.config_sha256,
            "gpu_indices": gpu_indices,
            "correctness": result.summary.to_dict(),
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded M3.1 GPU Smoke interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--gpu-indices", type=parse_gpu_indices, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser


def main() -> int:
    """Run the Smoke and print its strict result."""

    args = build_parser().parse_args()
    try:
        result = run_smoke(
            config_path=args.config,
            output_root=args.output_root,
            evidence_dir=args.evidence_dir,
            gpu_indices=args.gpu_indices,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"M3 DDP Smoke failed: {exc}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
