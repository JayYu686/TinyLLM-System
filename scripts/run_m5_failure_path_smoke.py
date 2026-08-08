#!/usr/bin/env python3
"""Run safe CPU fault injection for every M5 acceptance failure path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, cast

import torch

from tinyllm.lineage import read_git_identity
from tinyllm.training.m5_failure import (
    M5FailurePathError,
    normalize_cuda_oom,
    require_child_success,
    require_dataset_identity,
    require_file_sha256,
    require_finite_metric,
    require_storage_capacity,
    require_world_size,
)
from tinyllm.training.m5_failure_schema import (
    M5FailurePathCase,
    M5FailurePathEvidence,
    M5FailurePathName,
)


def _capture(name: str, action: Callable[[], None]) -> M5FailurePathCase:
    try:
        action()
    except M5FailurePathError as exc:
        return M5FailurePathCase(
            name=cast(M5FailurePathName, name),
            status="rejected_as_expected",
            injection_kind="safe_cpu_fault_injection",
            observed_error=str(exc),
        )
    raise RuntimeError(f"M5 failure injection was not rejected: {name}")


def _raise_oom() -> None:
    raise normalize_cuda_oom(torch.OutOfMemoryError("injected allocation failure"))


def _corrupt_checkpoint(root: Path) -> None:
    payload = root / "training_state.pt"
    payload.write_bytes(b"corrupted")
    expected = hashlib.sha256(b"valid-checkpoint").hexdigest()
    require_file_sha256(payload, expected_sha256=expected)


def _insufficient_disk(root: Path) -> None:
    class Usage(NamedTuple):
        total: int
        used: int
        free: int

    def usage(_: Path) -> Usage:
        return Usage(total=100, used=99, free=1)

    require_storage_capacity(root, minimum_free_bytes=2, disk_usage=usage)


def _child_exit() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "raise SystemExit(17)"],
        check=False,
        timeout=10,
    )
    require_child_success(completed.returncode)


def run_smoke(*, project_root: Path, generated_at: datetime | None = None) -> M5FailurePathEvidence:
    """Execute the frozen failure matrix without allocating CUDA memory."""

    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise RuntimeError("M5 failure-path acceptance requires a clean Git worktree")
    with tempfile.TemporaryDirectory(prefix="tinyllm-m5-failure-") as temporary:
        root = Path(temporary)
        cases = (
            _capture("cuda_oom", _raise_oom),
            _capture("non_finite", lambda: require_finite_metric("loss", float("nan"))),
            _capture("corrupt_checkpoint", lambda: _corrupt_checkpoint(root)),
            _capture("disk_insufficient", lambda: _insufficient_disk(root)),
            _capture(
                "dataset_drift",
                lambda: require_dataset_identity(
                    actual_version="changed",
                    expected_version="frozen",
                    actual_manifest_sha256="a" * 64,
                    expected_manifest_sha256="b" * 64,
                ),
            ),
            _capture("world_size_mismatch", lambda: require_world_size(actual=2, expected=4)),
            _capture("child_process_exit", _child_exit),
        )
    return M5FailurePathEvidence(
        status="passed",
        generated_at=generated_at or datetime.now(UTC),
        git_commit=git_commit,
        git_dirty=False,
        model_generated=False,
        quality_metric=False,
        cases=cases,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    try:
        result = run_smoke(project_root=project_root)
        payload = result.model_dump_json(indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        print(result.model_dump_json())
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
