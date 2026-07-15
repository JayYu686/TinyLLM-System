#!/usr/bin/env python3
"""Run the formal M3.2 interruption and Rank-failure recovery gate on two idle GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from tinyllm.lineage import read_git_identity
from tinyllm.training import DDPRecoveryResult, load_training_config
from tinyllm.training.smoke_preflight import (
    GpuPreflight,
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
    extra: tuple[str, ...],
) -> list[str]:
    executable = Path(sys.executable).with_name("torchrun")
    if not executable.is_file():
        raise RuntimeError("torchrun is unavailable beside the active Python interpreter")
    return [
        str(executable),
        "--standalone",
        f"--nproc-per-node={world_size}",
        "-m",
        "tinyllm.training.ddp_recovery_worker",
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        *extra,
    ]


def _phase_preflight(
    *,
    phase: str,
    gpu_indices: tuple[int, ...],
    evidence_dir: Path,
) -> tuple[GpuPreflight, ...]:
    captured_at = datetime.now(UTC)
    rows = inspect_gpus(gpu_indices)
    _write_json(
        evidence_dir / f"{phase}.preflight.json",
        {
            "schema_version": "1.0",
            "captured_at": captured_at.isoformat(),
            "gpus": rows,
        },
    )
    validate_gpu_preflight(rows)
    return rows


def _run_phase(
    *,
    phase: str,
    expectation: Literal["succeeded", "interrupted", "rank_failure"],
    config_path: Path,
    output_root: Path,
    evidence_dir: Path,
    gpu_indices: tuple[int, ...],
    extra: tuple[str, ...] = (),
    timeout_seconds: int,
) -> tuple[DDPRecoveryResult | None, Path | None]:
    _phase_preflight(
        phase=phase,
        gpu_indices=gpu_indices,
        evidence_dir=evidence_dir,
    )
    before = {path.resolve() for path in output_root.iterdir() if path.is_dir()}
    command = _torchrun_command(config_path, output_root, len(gpu_indices), extra)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indices)
    environment["OMP_NUM_THREADS"] = "1"
    environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    started_at = datetime.now(UTC)
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{phase} timed out") from exc
    finished_at = datetime.now(UTC)
    (evidence_dir / f"{phase}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (evidence_dir / f"{phase}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    after = {path.resolve() for path in output_root.iterdir() if path.is_dir()}
    created = tuple(sorted(after - before))

    result: DDPRecoveryResult | None = None
    run_dir: Path | None = created[0] if len(created) == 1 else None
    if expectation in {"succeeded", "interrupted"}:
        try:
            result = DDPRecoveryResult.model_validate_json(completed.stdout)
        except ValueError as exc:
            raise RuntimeError(f"{phase} emitted an invalid Rank-zero result") from exc
        run_dir = result.artifact_dir
        if result.status != expectation:
            raise RuntimeError(f"{phase} returned an unexpected status")
        if expectation == "succeeded" and completed.returncode != 0:
            raise RuntimeError(f"{phase} failed with exit code {completed.returncode}")
        if expectation == "interrupted" and completed.returncode == 0:
            raise RuntimeError(f"{phase} did not propagate the intentional interruption")
    else:
        if completed.returncode == 0:
            raise RuntimeError(f"{phase} did not fail after the forced Rank exit")
        if "rank      : 1" not in completed.stderr or "exitcode  : 17" not in completed.stderr:
            raise RuntimeError(f"{phase} did not preserve the forced Rank-1 exit diagnostics")
        if run_dir is None:
            raise RuntimeError(f"{phase} did not create exactly one recoverable Run")

    _write_json(
        evidence_dir / f"{phase}.json",
        {
            "schema_version": "1.0",
            "phase": phase,
            "expectation": expectation,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
            "return_code": completed.returncode,
            "argv": [Path(command[0]).name, *command[1:]],
            "gpu_indices": gpu_indices,
            "run_id": result.run_id if result is not None else run_dir.name,
            "result": result.to_dict() if result is not None else None,
        },
    )
    return result, run_dir


def _losses(run_dir: Path) -> tuple[float, ...]:
    try:
        rows = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot read canonical Run metrics") from exc
    steps = [row.get("global_step") for row in rows]
    if steps != list(range(1, len(rows) + 1)):
        raise RuntimeError("Run metrics contain missing or repeated optimizer steps")
    return tuple(float(row["loss"]) for row in rows)


def _max_abs_difference(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise RuntimeError("recovery comparisons require equal non-empty metric series")
    return max(abs(first - second) for first, second in zip(left, right, strict=True))


def run_smoke(
    *,
    config_path: Path,
    output_root: Path,
    evidence_dir: Path,
    gpu_indices: tuple[int, ...],
    timeout_seconds: int,
) -> dict[str, object]:
    """Execute baselines first, freeze tolerance, then exercise both recovery paths."""

    project_root = Path(__file__).resolve().parents[1]
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    evidence_dir = evidence_dir.resolve()
    config = load_training_config(config_path)
    if config.distributed.strategy != "ddp" or config.distributed.backend != "nccl":
        raise RuntimeError("formal M3.2 GPU Smoke requires DDP/NCCL")
    if config.precision.dtype != "bf16" or config.distributed.world_size != 2:
        raise RuntimeError("formal M3.2 GPU Smoke requires two-GPU BF16")
    if len(gpu_indices) != 2:
        raise RuntimeError("exactly two physical GPU indices are required")
    if evidence_dir.exists():
        raise RuntimeError("evidence directory already exists")
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise RuntimeError("formal M3.2 GPU Smoke requires a clean Git worktree")
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True)
    started_at = datetime.now(UTC)

    baseline_a, baseline_a_dir = _run_phase(
        phase="baseline-a",
        expectation="succeeded",
        config_path=config_path,
        output_root=output_root,
        evidence_dir=evidence_dir,
        gpu_indices=gpu_indices,
        timeout_seconds=timeout_seconds,
    )
    baseline_b, baseline_b_dir = _run_phase(
        phase="baseline-b",
        expectation="succeeded",
        config_path=config_path,
        output_root=output_root,
        evidence_dir=evidence_dir,
        gpu_indices=gpu_indices,
        timeout_seconds=timeout_seconds,
    )
    assert baseline_a is not None and baseline_b is not None
    assert baseline_a_dir is not None and baseline_b_dir is not None
    baseline_delta = _max_abs_difference(_losses(baseline_a_dir), _losses(baseline_b_dir))
    loss_tolerance = max(1.0e-6, baseline_delta * 2.0)
    if baseline_a.model_parameter_sha256 != baseline_b.model_parameter_sha256:
        raise RuntimeError("uninterrupted baseline final parameter hashes differ")
    tolerance = {
        "schema_version": "1.0",
        "frozen_before_recovery_phases": True,
        "rule": "max(1e-6, baseline_max_abs_loss_diff * 2)",
        "baseline_max_abs_loss_diff": baseline_delta,
        "recovery_loss_atol": loss_tolerance,
        "require_exact_final_parameter_hash": True,
    }
    _write_json(evidence_dir / "tolerance.json", tolerance)

    interrupted, interrupted_dir = _run_phase(
        phase="coordinated-stop",
        expectation="interrupted",
        config_path=config_path,
        output_root=output_root,
        evidence_dir=evidence_dir,
        gpu_indices=gpu_indices,
        extra=("--stop-after-step", "6"),
        timeout_seconds=timeout_seconds,
    )
    assert interrupted is not None and interrupted_dir is not None
    recovered, _ = _run_phase(
        phase="coordinated-resume",
        expectation="succeeded",
        config_path=config_path,
        output_root=output_root,
        evidence_dir=evidence_dir,
        gpu_indices=gpu_indices,
        extra=("--resume-run", str(interrupted_dir)),
        timeout_seconds=timeout_seconds,
    )
    assert recovered is not None

    _, failed_dir = _run_phase(
        phase="rank-failure",
        expectation="rank_failure",
        config_path=config_path,
        output_root=output_root,
        evidence_dir=evidence_dir,
        gpu_indices=gpu_indices,
        extra=("--fail-rank", "1", "--fail-after-step", "8"),
        timeout_seconds=timeout_seconds,
    )
    assert failed_dir is not None
    failure_records = tuple((failed_dir / "failures").glob("*.json"))
    if len(failure_records) != 1:
        raise RuntimeError("forced Rank failure did not publish exactly one diagnostic")
    failure_record = json.loads(failure_records[0].read_text(encoding="utf-8"))
    if failure_record.get("exit_code") != 17 or not failure_record.get("resumable"):
        raise RuntimeError("forced Rank failure diagnostic is incomplete")
    failure_recovered, _ = _run_phase(
        phase="rank-failure-resume",
        expectation="succeeded",
        config_path=config_path,
        output_root=output_root,
        evidence_dir=evidence_dir,
        gpu_indices=gpu_indices,
        extra=("--resume-run", str(failed_dir)),
        timeout_seconds=timeout_seconds,
    )
    assert failure_recovered is not None

    baseline_losses = _losses(baseline_a_dir)
    recovery_delta = _max_abs_difference(baseline_losses, _losses(interrupted_dir))
    failure_recovery_delta = _max_abs_difference(baseline_losses, _losses(failed_dir))
    comparisons = (recovered, failure_recovered)
    if any(
        result.model_parameter_sha256 != baseline_a.model_parameter_sha256 for result in comparisons
    ):
        raise RuntimeError("recovered final parameter hash differs from the baseline")
    if recovery_delta > loss_tolerance or failure_recovery_delta > loss_tolerance:
        raise RuntimeError("recovered loss series exceeds the frozen baseline tolerance")

    finished_at = datetime.now(UTC)
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "status": "pass",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "git_commit": git_commit,
        "config_sha256": baseline_a.config_sha256,
        "gpu_indices": gpu_indices,
        "world_size": 2,
        "baseline_run_ids": (baseline_a.run_id, baseline_b.run_id),
        "coordinated_recovery_run_id": recovered.run_id,
        "rank_failure_recovery_run_id": failure_recovered.run_id,
        "coordinated_resume_from_step": recovered.resumed_from_step,
        "rank_failure_resume_from_step": failure_recovered.resumed_from_step,
        "forced_rank_exit_code": failure_record["exit_code"],
        "baseline_max_abs_loss_diff": baseline_delta,
        "recovery_loss_atol": loss_tolerance,
        "coordinated_recovery_max_abs_loss_diff": recovery_delta,
        "rank_failure_recovery_max_abs_loss_diff": failure_recovery_delta,
        "exact_final_parameter_hash": baseline_a.model_parameter_sha256,
        "optimizer_steps": config.training.max_steps,
        "metrics_unique_and_complete": True,
        "checkpoint_integrity": "pass",
    }
    _write_json(evidence_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the formal M3.2 GPU gate interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--gpu-indices", type=parse_gpu_indices, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser


def main() -> int:
    """Run the gate and print its machine-readable summary."""

    args = build_parser().parse_args()
    try:
        summary = run_smoke(
            config_path=args.config,
            output_root=args.output_root,
            evidence_dir=args.evidence_dir,
            gpu_indices=args.gpu_indices,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"M3.2 DDP recovery Smoke failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
