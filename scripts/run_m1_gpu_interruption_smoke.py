#!/usr/bin/env python3
"""Run BF16 repeat baselines and real SIGTERM/SIGKILL recovery on one RTX 3090."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training import CheckpointStore, load_training_config


@dataclass(frozen=True, slots=True)
class WorkerRun:
    """Collected stdout events and exit status from one real worker process."""

    events: tuple[dict[str, Any], ...]
    returncode: int
    stderr: str


def _query_gpu(physical_gpu_index: int) -> dict[str, object]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu_index}",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 6:
        raise RuntimeError("unexpected nvidia-smi preflight output")
    return {
        "physical_gpu_index": int(fields[0]),
        "name": fields[1],
        "memory_used_mib": int(fields[2]),
        "memory_total_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
        "temperature_c": int(fields[5]),
    }


def _wait_for_idle_gpu(physical_gpu_index: int) -> dict[str, object]:
    last: dict[str, object] | None = None
    for _ in range(20):
        last = _query_gpu(physical_gpu_index)
        if (
            int(cast(int, last["memory_used_mib"])) <= 1024
            and int(cast(int, last["utilization_percent"])) <= 10
            and int(cast(int, last["temperature_c"])) < 70
        ):
            return last
        time.sleep(0.25)
    raise RuntimeError(f"GPU preflight failed: {last}")


def _worker_command(
    *,
    project_root: Path,
    config_path: Path,
    artifact_dir: Path,
    run_id: str,
    dataset_version: str,
    git_commit: str,
    resume: bool,
    step_delay_ms: int,
) -> list[str]:
    command = [
        sys.executable,
        str(project_root / "scripts" / "m1_gpu_worker.py"),
        "--config",
        str(config_path),
        "--artifact-dir",
        str(artifact_dir),
        "--run-id",
        run_id,
        "--dataset-version",
        dataset_version,
        "--git-commit",
        git_commit,
        "--step-delay-ms",
        str(step_delay_ms),
    ]
    if resume:
        command.append("--resume")
    return command


def _run_worker(
    *,
    project_root: Path,
    config_path: Path,
    artifact_dir: Path,
    run_id: str,
    dataset_version: str,
    git_commit: str,
    resume: bool,
    step_delay_ms: int,
    physical_gpu_index: int,
    inject_signal: Literal["SIGTERM", "SIGKILL"] | None = None,
    signal_after_step: int | None = None,
) -> WorkerRun:
    _wait_for_idle_gpu(physical_gpu_index)
    environment = os.environ.copy()
    environment["TINYLLM_PHYSICAL_GPU_INDEX"] = str(physical_gpu_index)
    process = subprocess.Popen(
        _worker_command(
            project_root=project_root,
            config_path=config_path,
            artifact_dir=artifact_dir,
            run_id=run_id,
            dataset_version=dataset_version,
            git_commit=git_commit,
            resume=resume,
            step_delay_ms=step_delay_ms,
        ),
        cwd=project_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("worker pipes were not created")
    events: list[dict[str, Any]] = []
    signal_sent = False
    for line in process.stdout:
        event = json.loads(line)
        if not isinstance(event, dict):
            process.kill()
            raise RuntimeError("worker emitted a non-object event")
        events.append(cast(dict[str, Any], event))
        if (
            inject_signal is not None
            and not signal_sent
            and event.get("event") == "optimizer_step"
            and event.get("global_step") == signal_after_step
        ):
            process.send_signal(signal.SIGTERM if inject_signal == "SIGTERM" else signal.SIGKILL)
            signal_sent = True
    returncode = process.wait(timeout=30)
    stderr = process.stderr.read()
    if inject_signal is not None and not signal_sent:
        raise RuntimeError(f"worker completed before {inject_signal} injection")
    return WorkerRun(events=tuple(events), returncode=returncode, stderr=stderr)


def _metrics(run: WorkerRun) -> list[dict[str, Any]]:
    return [event for event in run.events if event.get("event") == "optimizer_step"]


def _event(run: WorkerRun, name: str) -> dict[str, Any]:
    matches = [event for event in run.events if event.get("event") == name]
    if not matches:
        raise RuntimeError(f"worker did not emit {name}")
    return matches[-1]


def _load_final_payload(artifact_dir: Path) -> dict[str, Any]:
    store = CheckpointStore(artifact_dir / "checkpoints", keep_last=2)
    return store.load_training_state(store.latest(), map_location="cpu")


def _max_model_abs_diff(left: object, right: object) -> float:
    if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
        return float("inf")
    maximum = 0.0
    for key in left:
        left_tensor = left[key]
        right_tensor = right[key]
        if not isinstance(left_tensor, Tensor) or not isinstance(right_tensor, Tensor):
            return float("inf")
        if left_tensor.shape != right_tensor.shape:
            return float("inf")
        maximum = max(
            maximum,
            float((left_tensor.float() - right_tensor.float()).abs().max().item()),
        )
    return maximum


def _max_loss_abs_diff(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    if len(left) != len(right):
        return float("inf")
    return max(
        (abs(float(a["loss"]) - float(b["loss"])) for a, b in zip(left, right, strict=True)),
        default=0.0,
    )


def _canonical_interrupted_metrics(
    initial: WorkerRun,
    resumed: WorkerRun,
) -> tuple[list[dict[str, Any]], int]:
    resumed_event = _event(resumed, "resumed")
    resume_step = int(resumed_event["global_step"])
    prefix = [metric for metric in _metrics(initial) if int(metric["global_step"]) <= resume_step]
    return prefix + _metrics(resumed), resume_step


def _compare_to_baseline(
    *,
    baseline_metrics: list[dict[str, Any]],
    candidate_metrics: list[dict[str, Any]],
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    loss_abs_tolerance: float,
    parameter_abs_tolerance: float,
) -> dict[str, object]:
    steps_equal = [metric["global_step"] for metric in candidate_metrics] == [
        metric["global_step"] for metric in baseline_metrics
    ]
    lr_equal = [metric["learning_rate"] for metric in candidate_metrics] == [
        metric["learning_rate"] for metric in baseline_metrics
    ]
    loss_max_abs_diff = _max_loss_abs_diff(baseline_metrics, candidate_metrics)
    parameter_max_abs_diff = _max_model_abs_diff(
        baseline_payload["model"], candidate_payload["model"]
    )
    trainer_state_equal = baseline_payload["trainer_state"] == candidate_payload["trainer_state"]
    sampler_state_equal = baseline_payload["sampler"] == candidate_payload["sampler"]
    passed = (
        steps_equal
        and lr_equal
        and loss_max_abs_diff <= loss_abs_tolerance
        and parameter_max_abs_diff <= parameter_abs_tolerance
        and trainer_state_equal
        and sampler_state_equal
    )
    return {
        "status": "pass" if passed else "fail",
        "steps_equal": steps_equal,
        "lr_equal": lr_equal,
        "loss_max_abs_diff": loss_max_abs_diff,
        "parameter_max_abs_diff": parameter_max_abs_diff,
        "trainer_state_equal": trainer_state_equal,
        "sampler_state_equal": sampler_state_equal,
    }


def run_gpu_interruption_smoke(
    *,
    config_path: Path,
    artifact_root: Path,
    physical_gpu_index: int,
    step_delay_ms: int,
) -> dict[str, Any]:
    """Execute ordered baselines, freeze tolerance, then inject both process signals."""

    project_root = Path(__file__).resolve().parents[1]
    config_path = config_path.resolve()
    config = load_training_config(config_path)
    if config.precision.dtype != "bf16" or config.precision.use_grad_scaler:
        raise ValueError("M1.4 smoke config must request BF16 without GradScaler")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("expose exactly one BF16-capable CUDA device to the smoke controller")
    config_hash = canonical_config_hash(config)
    git_commit, git_dirty = read_git_identity(project_root)
    dataset_version = f"toy-gpu-resume-{config_hash[:8]}"
    group_label = (
        datetime.now(UTC).strftime("m1-4-%Y%m%dT%H%M%SZ-")
        + config_hash[:8]
        + "-"
        + uuid.uuid4().hex[:4]
    )
    group_dir = artifact_root / group_label
    group_dir.mkdir(parents=True, exist_ok=False)
    initial_preflight = _wait_for_idle_gpu(physical_gpu_index)

    def identity(label: str) -> tuple[str, Path]:
        run_id = generate_run_id(label, config_hash, now=datetime.now(UTC))
        return run_id, group_dir / run_id

    baseline_a_id, baseline_a_dir = identity("m1-bf16-baseline-a")
    baseline_a = _run_worker(
        project_root=project_root,
        config_path=config_path,
        artifact_dir=baseline_a_dir,
        run_id=baseline_a_id,
        dataset_version=dataset_version,
        git_commit=git_commit,
        resume=False,
        step_delay_ms=0,
        physical_gpu_index=physical_gpu_index,
    )
    baseline_b_id, baseline_b_dir = identity("m1-bf16-baseline-b")
    baseline_b = _run_worker(
        project_root=project_root,
        config_path=config_path,
        artifact_dir=baseline_b_dir,
        run_id=baseline_b_id,
        dataset_version=dataset_version,
        git_commit=git_commit,
        resume=False,
        step_delay_ms=0,
        physical_gpu_index=physical_gpu_index,
    )
    if baseline_a.returncode != 0 or baseline_b.returncode != 0:
        raise RuntimeError("an uninterrupted BF16 baseline failed")
    baseline_a_metrics = _metrics(baseline_a)
    baseline_b_metrics = _metrics(baseline_b)
    baseline_a_payload = _load_final_payload(baseline_a_dir)
    baseline_b_payload = _load_final_payload(baseline_b_dir)
    baseline_loss_diff = _max_loss_abs_diff(baseline_a_metrics, baseline_b_metrics)
    baseline_parameter_diff = _max_model_abs_diff(
        baseline_a_payload["model"], baseline_b_payload["model"]
    )

    # Frozen in docs/m1_training_contract.md before interrupted comparisons execute.
    loss_abs_tolerance = max(1.0e-6, 2.0 * baseline_loss_diff)
    parameter_abs_tolerance = max(1.0e-7, 2.0 * baseline_parameter_diff)

    signal_after_step = config.checkpoint.save_steps + 2
    if signal_after_step >= config.training.max_steps:
        raise ValueError("smoke config does not leave enough steps after signal injection")

    sigterm_id, sigterm_dir = identity("m1-bf16-sigterm")
    sigterm_initial = _run_worker(
        project_root=project_root,
        config_path=config_path,
        artifact_dir=sigterm_dir,
        run_id=sigterm_id,
        dataset_version=dataset_version,
        git_commit=git_commit,
        resume=False,
        step_delay_ms=step_delay_ms,
        physical_gpu_index=physical_gpu_index,
        inject_signal="SIGTERM",
        signal_after_step=signal_after_step,
    )
    if sigterm_initial.returncode != 143:
        raise RuntimeError(f"SIGTERM worker returned {sigterm_initial.returncode}")
    sigterm_checkpoint = _event(sigterm_initial, "termination_checkpoint_committed")
    sigterm_resumed = _run_worker(
        project_root=project_root,
        config_path=config_path,
        artifact_dir=sigterm_dir,
        run_id=sigterm_id,
        dataset_version=dataset_version,
        git_commit=git_commit,
        resume=True,
        step_delay_ms=0,
        physical_gpu_index=physical_gpu_index,
    )
    if sigterm_resumed.returncode != 0:
        raise RuntimeError("SIGTERM resume worker failed")
    sigterm_metrics, sigterm_resume_step = _canonical_interrupted_metrics(
        sigterm_initial, sigterm_resumed
    )
    sigterm_payload = _load_final_payload(sigterm_dir)
    sigterm_comparison = _compare_to_baseline(
        baseline_metrics=baseline_a_metrics,
        candidate_metrics=sigterm_metrics,
        baseline_payload=baseline_a_payload,
        candidate_payload=sigterm_payload,
        loss_abs_tolerance=loss_abs_tolerance,
        parameter_abs_tolerance=parameter_abs_tolerance,
    )

    sigkill_id, sigkill_dir = identity("m1-bf16-sigkill")
    sigkill_initial = _run_worker(
        project_root=project_root,
        config_path=config_path,
        artifact_dir=sigkill_dir,
        run_id=sigkill_id,
        dataset_version=dataset_version,
        git_commit=git_commit,
        resume=False,
        step_delay_ms=step_delay_ms,
        physical_gpu_index=physical_gpu_index,
        inject_signal="SIGKILL",
        signal_after_step=signal_after_step,
    )
    if sigkill_initial.returncode != -signal.SIGKILL:
        raise RuntimeError(f"SIGKILL worker returned {sigkill_initial.returncode}")
    sigkill_resumed = _run_worker(
        project_root=project_root,
        config_path=config_path,
        artifact_dir=sigkill_dir,
        run_id=sigkill_id,
        dataset_version=dataset_version,
        git_commit=git_commit,
        resume=True,
        step_delay_ms=0,
        physical_gpu_index=physical_gpu_index,
    )
    if sigkill_resumed.returncode != 0:
        raise RuntimeError("SIGKILL resume worker failed")
    sigkill_metrics, sigkill_resume_step = _canonical_interrupted_metrics(
        sigkill_initial, sigkill_resumed
    )
    sigkill_payload = _load_final_payload(sigkill_dir)
    sigkill_comparison = _compare_to_baseline(
        baseline_metrics=baseline_a_metrics,
        candidate_metrics=sigkill_metrics,
        baseline_payload=baseline_a_payload,
        candidate_payload=sigkill_payload,
        loss_abs_tolerance=loss_abs_tolerance,
        parameter_abs_tolerance=parameter_abs_tolerance,
    )

    baseline_a_final = _event(baseline_a, "worker_succeeded")
    baseline_b_final = _event(baseline_b, "worker_succeeded")
    sigterm_final = _event(sigterm_resumed, "worker_succeeded")
    sigkill_final = _event(sigkill_resumed, "worker_succeeded")
    baseline_passed = (
        len(baseline_a_metrics) == config.training.max_steps
        and len(baseline_b_metrics) == config.training.max_steps
        and baseline_loss_diff <= loss_abs_tolerance
        and baseline_parameter_diff <= parameter_abs_tolerance
    )
    passed = (
        baseline_passed
        and sigterm_comparison["status"] == "pass"
        and sigkill_comparison["status"] == "pass"
        and int(sigterm_checkpoint["global_step"]) == signal_after_step
        and sigterm_resume_step == signal_after_step
        and sigkill_resume_step == config.checkpoint.save_steps
    )
    payload = {
        "schema_version": "1.0",
        "smoke": "m1.4-rtx3090-bf16-process-interruption",
        "status": "pass" if passed else "fail",
        "git": {"commit": git_commit, "dirty": git_dirty},
        "config": {
            "path": config_path.name,
            "resolved_sha256": config_hash,
            "max_steps": config.training.max_steps,
            "save_steps": config.checkpoint.save_steps,
            "dtype": config.precision.dtype,
            "tf32": config.precision.allow_tf32,
        },
        "gpu_preflight": initial_preflight,
        "software": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "baseline_repeat": {
            "runs": 2,
            "steps_per_run": config.training.max_steps,
            "loss_max_abs_diff": baseline_loss_diff,
            "parameter_max_abs_diff": baseline_parameter_diff,
            "model_hash_equal": (
                baseline_a_final["model_sha256"] == baseline_b_final["model_sha256"]
            ),
        },
        "tolerance_frozen_before_interruption": {
            "rule": "max(floor, 2x baseline repeat maximum absolute difference)",
            "loss_abs_floor": 1.0e-6,
            "parameter_abs_floor": 1.0e-7,
            "loss_abs_tolerance": loss_abs_tolerance,
            "parameter_abs_tolerance": parameter_abs_tolerance,
        },
        "sigterm": {
            "signal_sent_after_step": signal_after_step,
            "initial_exit_code": sigterm_initial.returncode,
            "checkpoint_step": int(sigterm_checkpoint["global_step"]),
            "resume_step": sigterm_resume_step,
            "first_resumed_step": sigterm_resume_step + 1,
            "rolled_back_steps": 0,
            "final_model_sha256": sigterm_final["model_sha256"],
            "comparison": sigterm_comparison,
        },
        "sigkill": {
            "signal_sent_after_step": signal_after_step,
            "initial_returncode": sigkill_initial.returncode,
            "resume_step": sigkill_resume_step,
            "first_resumed_step": sigkill_resume_step + 1,
            "rolled_back_steps": signal_after_step - sigkill_resume_step,
            "final_model_sha256": sigkill_final["model_sha256"],
            "comparison": sigkill_comparison,
        },
        "private_artifacts": {
            "retained": True,
            "group_label": group_label,
            "run_directories": 4,
            "contains_full_events_metrics_and_checkpoints": True,
        },
        "not_evaluated": [
            "training_throughput",
            "cross_gpu_or_cross_version_reproducibility",
            "v100_fp16_grad_scaler",
            "distributed_resume",
        ],
    }
    summary_path = group_dir / "controller-summary.json"
    temporary_summary = group_dir / f".controller-summary.tmp-{uuid.uuid4().hex}"
    temporary_summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_summary, summary_path)
    return payload


def main() -> int:
    """Parse arguments and print a sanitized public summary."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain/tinygpt_debug_rtx3090_bf16_smoke.yaml"),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--step-delay-ms", type=int, default=100)
    args = parser.parse_args()
    try:
        payload = run_gpu_interruption_smoke(
            config_path=args.config,
            artifact_root=args.artifact_root,
            physical_gpu_index=args.physical_gpu_index,
            step_delay_ms=args.step_delay_ms,
        )
    except Exception as exc:
        payload = {
            "schema_version": "1.0",
            "smoke": "m1.4-rtx3090-bf16-process-interruption",
            "status": "fail",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
