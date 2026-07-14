#!/usr/bin/env python3
"""Run a bounded, auditable nccl-tests matrix on explicitly selected GPUs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

COLLECTIVES = {
    "all_reduce": "all_reduce_perf",
    "all_gather": "all_gather_perf",
    "reduce_scatter": "reduce_scatter_perf",
}


class GpuStatus(TypedDict):
    """GPU state captured immediately before an NCCL test."""

    index: int
    memory_used_mib: int
    utilization_percent: int
    temperature_c: int


def parse_gpu_group(value: str) -> tuple[int, ...]:
    """Parse a comma-separated GPU group with no duplicate indices."""

    try:
        indices = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU groups must contain integer indices") from exc
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("GPU groups must contain non-negative indices")
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("GPU groups must not contain duplicate indices")
    return indices


def _run(
    command: list[str], *, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
        return subprocess.CompletedProcess(
            command,
            124,
            stdout,
            f"{stderr}\ncommand timed out after 300 seconds".strip(),
        )


def inspect_gpus(group: tuple[int, ...]) -> list[GpuStatus]:
    """Read utilization, memory, and temperature before launching a collective."""

    completed = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi GPU preflight failed")
    selected = set(group)
    rows: list[GpuStatus] = []
    for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 4:
            raise RuntimeError("unexpected nvidia-smi preflight output")
        index, memory, utilization, temperature = (int(value.strip()) for value in row)
        if index in selected:
            rows.append(
                {
                    "index": index,
                    "memory_used_mib": memory,
                    "utilization_percent": utilization,
                    "temperature_c": temperature,
                }
            )
    if {row["index"] for row in rows} != selected:
        raise RuntimeError("one or more requested GPUs were not discovered")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nccl-tests-dir", type=Path, required=True)
    parser.add_argument(
        "--nccl-tests-revision",
        help="Pinned tag/commit for an archive without Git metadata.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-group", type=parse_gpu_group, action="append", required=True)
    parser.add_argument(
        "--collective",
        choices=sorted(COLLECTIVES),
        action="append",
        help="Collective to run; repeat as needed (default: all three).",
    )
    parser.add_argument("--allow-busy", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir
    if not output_dir.is_dir():
        raise SystemExit(f"output directory does not exist: {output_dir}")
    nccl_tests_dir: Path = args.nccl_tests_dir
    collectives = args.collective or list(COLLECTIVES)
    revision = _run(["git", "-C", str(nccl_tests_dir), "rev-parse", "HEAD"])
    revision_name = (
        revision.stdout.strip() if revision.returncode == 0 else args.nccl_tests_revision
    )
    if not revision_name:
        raise SystemExit("provide a Git checkout or --nccl-tests-revision")

    results: list[dict[str, object]] = []
    failed = False
    for group in args.gpu_group:
        preflight = inspect_gpus(group)
        busy = [
            row["index"]
            for row in preflight
            if row["memory_used_mib"] > 1024 or row["utilization_percent"] >= 10
        ]
        if busy and not args.allow_busy:
            results.append(
                {
                    "gpu_group": list(group),
                    "status": "not_run",
                    "reason": "busy_gpus",
                    "busy_gpu_indices": busy,
                    "preflight": preflight,
                }
            )
            continue
        for collective in collectives:
            binary = nccl_tests_dir / "build" / COLLECTIVES[collective]
            if not binary.is_file():
                raise SystemExit(f"missing nccl-tests binary: {binary}")
            command = [
                str(binary),
                "-b",
                "8",
                "-e",
                "512M",
                "-f",
                "2",
                "-g",
                str(len(group)),
                "-w",
                "5",
                "-n",
                "20",
                "-c",
                "1",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in group)
            started_at = datetime.now(UTC)
            completed = _run(command, environment=environment)
            finished_at = datetime.now(UTC)
            slug = "-".join(str(index) for index in group)
            log_path = output_dir / f"{collective}_gpus_{slug}.log"
            log_path.write_text(
                "\n".join(
                    [
                        f"# CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']}",
                        f"# command={' '.join(command)}",
                        f"# started_at={started_at.isoformat()}",
                        f"# finished_at={finished_at.isoformat()}",
                        f"# exit_code={completed.returncode}",
                        completed.stdout,
                        completed.stderr,
                    ]
                ),
                encoding="utf-8",
            )
            status = "pass" if completed.returncode == 0 else "fail"
            failed = failed or status == "fail"
            results.append(
                {
                    "collective": collective,
                    "gpu_group": list(group),
                    "status": status,
                    "exit_code": completed.returncode,
                    "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
                    "preflight": preflight,
                    "log_path": str(log_path),
                }
            )

    summary = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "nccl_tests_revision": revision_name,
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
