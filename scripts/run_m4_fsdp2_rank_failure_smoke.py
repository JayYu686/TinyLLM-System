#!/usr/bin/env python3
"""Validate one expected nonzero-Rank FSDP2 exit on explicit idle GPUs."""

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
from tinyllm.training import FSDP2RankFailureEvidence, load_fsdp2_config
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


def _torchrun_command(
    config_path: Path,
    output_root: Path,
    world_size: int,
    *,
    fail_rank: int,
    fail_after_step: int,
) -> list[str]:
    executable = Path(sys.executable).with_name("torchrun")
    if not executable.is_file():
        raise RuntimeError("torchrun is unavailable beside the active Python interpreter")
    return [
        str(executable),
        "--standalone",
        f"--nproc-per-node={world_size}",
        "-m",
        "tinyllm.training.fsdp2_worker",
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--fail-rank",
        str(fail_rank),
        "--fail-after-step",
        str(fail_after_step),
    ]


def run_rank_failure_smoke(
    *,
    config_path: Path,
    output_root: Path,
    evidence_dir: Path,
    gpu_indices: tuple[int, ...],
    fail_rank: int,
    fail_after_step: int,
    timeout_seconds: int,
) -> FSDP2RankFailureEvidence:
    """Preflight GPUs and prove that torchrun detects the forced Rank exit."""

    project_root = Path(__file__).resolve().parents[1]
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    evidence_dir = evidence_dir.resolve()
    config = load_fsdp2_config(config_path)
    if config.distributed.backend != "nccl" or config.distributed.device_type != "cuda":
        raise RuntimeError("M4 Rank-failure Smoke requires backend=nccl and device_type=cuda")
    if config.precision.dtype != "bf16":
        raise RuntimeError("M4 RTX 3090 Rank-failure Smoke requires BF16")
    if not config.distributed.activation_checkpointing:
        raise RuntimeError("M4 Rank-failure Smoke requires Activation Checkpointing")
    if config.distributed.world_size != len(gpu_indices):
        raise RuntimeError("GPU index count must equal the resolved FSDP2 world_size")
    if not 1 <= fail_rank < config.distributed.world_size:
        raise RuntimeError("failure Rank must be a nonzero member of world_size")
    if not 1 <= fail_after_step < config.training.max_steps:
        raise RuntimeError("failure Step must be before training.max_steps")
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise RuntimeError("formal M4 Rank-failure Smoke requires a clean Git worktree")
    if evidence_dir.exists():
        raise RuntimeError("evidence directory already exists")

    existing_runs = set(output_root.iterdir()) if output_root.is_dir() else set()
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

    command = _torchrun_command(
        config_path,
        output_root,
        len(gpu_indices),
        fail_rank=fail_rank,
        fail_after_step=fail_after_step,
    )
    _write_json(
        evidence_dir / "command.json",
        {
            "schema_version": "1.0",
            "git_commit": git_commit,
            "config": config_path.name,
            "gpu_indices": gpu_indices,
            "world_size": len(gpu_indices),
            "expected_failure_rank": fail_rank,
            "expected_exit_code": 17,
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
        raise RuntimeError("M4 FSDP2 Rank-failure Smoke timed out") from exc

    (evidence_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (evidence_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    expected_rank = f"rank      : {fail_rank}"
    if completed.returncode == 0:
        reason = "unexpected_zero_exit"
    elif expected_rank not in completed.stderr or "exitcode  : 17" not in completed.stderr:
        reason = "forced_exit_not_observed"
    else:
        reason = None
    if reason is not None:
        _write_json(
            evidence_dir / "summary.json",
            {
                "schema_version": "1.0",
                "status": "fail",
                "reason": reason,
                "torchrun_exit_code": completed.returncode,
            },
        )
        raise RuntimeError("torchrun did not report the expected Rank exit")

    created_runs = set(output_root.iterdir()) - existing_runs
    run_dirs = tuple(path for path in created_runs if path.is_dir())
    if len(run_dirs) != 1:
        raise RuntimeError("expected Rank failure did not create exactly one Run")
    artifact_dir = run_dirs[0]
    failure_files = tuple((artifact_dir / "failures").glob("*.json"))
    if len(failure_files) != 1:
        raise RuntimeError("expected Rank failure did not retain one diagnostic")
    failure = FSDP2RankFailureEvidence.model_validate_json(
        failure_files[0].read_text(encoding="utf-8")
    )
    if failure.rank != fail_rank or failure.global_step != fail_after_step:
        raise RuntimeError("Rank-failure diagnostic does not match the requested injection")
    if failure.git_commit != git_commit or failure.run_id != artifact_dir.name:
        raise RuntimeError("Rank-failure diagnostic lineage does not match the supervised Run")

    hardware = cast(
        dict[str, object],
        json.loads((artifact_dir / "hardware.json").read_text(encoding="utf-8")),
    )
    ranks = cast(list[dict[str, object]], hardware["ranks"])
    recorded_indices = tuple(int(cast(int, item["physical_gpu_index"])) for item in ranks)
    if recorded_indices != gpu_indices:
        raise RuntimeError("Run hardware lineage does not match requested physical GPU indices")
    run = cast(
        dict[str, object],
        json.loads((artifact_dir / "run.json").read_text(encoding="utf-8")),
    )
    if run.get("status") != "failure_injected" or run.get("resumable") is not False:
        raise RuntimeError("Rank-failure Run did not retain the non-resumable M4.1 boundary")
    metric_records = len((artifact_dir / "metrics.jsonl").read_text().splitlines())
    if metric_records != fail_after_step:
        raise RuntimeError("Rank-failure Run retained an unexpected metric boundary")
    if (artifact_dir / "correctness.json").exists():
        raise RuntimeError("failed Rank run must not publish a correctness result")
    if tuple((artifact_dir / "checkpoints").iterdir()):
        raise RuntimeError("M4.1 Rank-failure run must not publish a Checkpoint")

    finished_at = datetime.now(UTC)
    _write_json(
        evidence_dir / "summary.json",
        {
            "schema_version": "1.0",
            "status": "pass",
            "expected_process_failure": True,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "torchrun_exit_code": completed.returncode,
            "gpu_indices": gpu_indices,
            "failure": failure.to_dict(),
        },
    )
    return failure


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded M4 CUDA/NCCL Rank-failure interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--gpu-indices", type=parse_gpu_indices, required=True)
    parser.add_argument("--fail-rank", type=int, default=1)
    parser.add_argument("--fail-after-step", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser


def main() -> int:
    """Run the expected failure and print its strict diagnostic."""

    args = build_parser().parse_args()
    try:
        result = run_rank_failure_smoke(
            config_path=args.config,
            output_root=args.output_root,
            evidence_dir=args.evidence_dir,
            gpu_indices=args.gpu_indices,
            fail_rank=args.fail_rank,
            fail_after_step=args.fail_after_step,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"M4 FSDP2 Rank-failure Smoke failed: {exc}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
