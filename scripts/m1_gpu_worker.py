#!/usr/bin/env python3
"""Run one resumable M1 CUDA worker process and emit private JSONL events."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import signal
import sys
import time
import uuid
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import torch

from tinyllm.schemas import canonical_config_hash
from tinyllm.training import (
    CheckpointContext,
    CheckpointStore,
    ResumeMode,
    SingleDeviceTrainer,
    build_m1_cuda_trainer,
    load_training_config,
    restore_trainer,
)
from tinyllm.training.config import M1TrainingConfig


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class PrivateEventWriter:
    """Persist full private events while forwarding one JSON object per stdout line."""

    def __init__(self, artifact_dir: Path) -> None:
        self.events_path = artifact_dir / "events.jsonl"
        self.metrics_path = artifact_dir / "metrics.jsonl"

    def emit(self, payload: dict[str, object], *, metric: bool = False) -> None:
        """Append, fsync, and expose one event for the supervising process."""

        _append_jsonl(self.events_path, payload)
        if metric:
            _append_jsonl(self.metrics_path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)


def _runtime_environment() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": "cuda:0",
        "visible_device_count": torch.cuda.device_count(),
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "physical_gpu_index": os.environ.get("TINYLLM_PHYSICAL_GPU_INDEX", "unknown"),
    }


def _save(
    *,
    store: CheckpointStore,
    trainer: SingleDeviceTrainer,
    config: M1TrainingConfig,
    context: CheckpointContext,
    pin_reason: Literal["interruption", "final"] | None,
) -> str:
    if trainer.sampler is None:
        raise RuntimeError("M1 GPU worker requires a stateful sampler")
    if pin_reason not in {None, "interruption", "final"}:
        raise ValueError("unsupported worker pin reason")
    manifest = store.save(
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=None,
        sampler=trainer.sampler,
        trainer_state=trainer.state,
        config=config,
        context=context,
        pin_reason=pin_reason,
    )
    return manifest.checkpoint_id


def run_worker(
    *,
    config_path: Path,
    artifact_dir: Path,
    run_id: str,
    dataset_version: str,
    git_commit: str,
    resume: bool,
    step_delay_ms: int,
) -> int:
    """Run until success or a safe SIGTERM boundary, returning a process exit code."""

    if step_delay_ms < 0:
        raise ValueError("step_delay_ms must be non-negative")
    config = load_training_config(config_path)
    environment = _runtime_environment()
    context = CheckpointContext(
        run_id=run_id,
        dataset_version=dataset_version,
        git_commit=git_commit,
        environment=environment,
    )
    if resume:
        if not artifact_dir.is_dir():
            raise FileNotFoundError("resume artifact directory does not exist")
    else:
        artifact_dir.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(config_path, artifact_dir / "config.original.yaml")
        _atomic_json(artifact_dir / "config.resolved.json", config.to_dict())
        _atomic_json(artifact_dir / "environment.json", environment)
        _atomic_json(
            artifact_dir / "hardware.json",
            {
                "schema_version": "1.0",
                "gpu_name": environment["gpu_name"],
                "compute_capability": environment["compute_capability"],
                "physical_gpu_index": environment["physical_gpu_index"],
            },
        )

    writer = PrivateEventWriter(artifact_dir)
    store = CheckpointStore(
        artifact_dir / "checkpoints",
        keep_last=config.checkpoint.keep_last,
    )
    termination_requested = False

    def request_termination(signum: int, frame: object) -> None:
        nonlocal termination_requested
        del signum, frame
        termination_requested = True

    signal.signal(signal.SIGTERM, request_termination)
    trainer = build_m1_cuda_trainer(config)
    resume_result = None
    if resume:
        resume_result = restore_trainer(
            store=store,
            trainer=trainer,
            mode=ResumeMode.EXACT,
            context=context,
        )
        writer.emit(
            {
                "event": "resumed",
                "checkpoint_id": resume_result.checkpoint_id,
                "global_step": trainer.state.global_step,
                "skipped_invalid": list(resume_result.skipped_invalid_checkpoints),
            }
        )
    elif any((artifact_dir / "checkpoints").glob("checkpoint-step-*")):
        raise RuntimeError("fresh worker cannot start with existing checkpoints")

    _atomic_json(
        artifact_dir / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "running",
            "config_hash": canonical_config_hash(config),
            "dataset_version": dataset_version,
            "git_commit": git_commit,
            "resume_source": resume_result.checkpoint_id if resume_result is not None else None,
        },
    )
    writer.emit(
        {
            "event": "worker_started",
            "global_step": trainer.state.global_step,
            "resume": resume,
        }
    )

    while trainer.state.global_step < config.training.max_steps:
        next_step = trainer.state.global_step + 1
        result = trainer.train(target_global_step=next_step)
        metric = result.metrics[0]
        writer.emit(metric.to_dict(), metric=True)
        if (
            trainer.state.global_step % config.checkpoint.save_steps == 0
            and trainer.state.global_step < config.training.max_steps
        ):
            checkpoint_id = _save(
                store=store,
                trainer=trainer,
                config=config,
                context=context,
                pin_reason=None,
            )
            writer.emit(
                {
                    "event": "checkpoint_committed",
                    "checkpoint_id": checkpoint_id,
                    "global_step": trainer.state.global_step,
                    "pin_reason": None,
                }
            )
        if step_delay_ms:
            time.sleep(step_delay_ms / 1000)
        if termination_requested:
            checkpoint_id = _save(
                store=store,
                trainer=trainer,
                config=config,
                context=context,
                pin_reason="interruption",
            )
            writer.emit(
                {
                    "event": "termination_checkpoint_committed",
                    "checkpoint_id": checkpoint_id,
                    "global_step": trainer.state.global_step,
                    "signal": "SIGTERM",
                }
            )
            _atomic_json(
                artifact_dir / "run.json",
                {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "status": "terminated",
                    "global_step": trainer.state.global_step,
                    "checkpoint_id": checkpoint_id,
                },
            )
            return 143

    checkpoint_id = _save(
        store=store,
        trainer=trainer,
        config=config,
        context=context,
        pin_reason="final",
    )
    model_sha256 = _model_digest(trainer.model)
    writer.emit(
        {
            "event": "worker_succeeded",
            "checkpoint_id": checkpoint_id,
            "global_step": trainer.state.global_step,
            "model_sha256": model_sha256,
        }
    )
    _atomic_json(
        artifact_dir / "run.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "succeeded",
            "global_step": trainer.state.global_step,
            "checkpoint_id": checkpoint_id,
            "model_sha256": model_sha256,
        },
    )
    return 0


def main() -> int:
    """Parse the private worker interface and return its stable process status."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--step-delay-ms", type=int, default=0)
    args = parser.parse_args()
    try:
        return run_worker(
            config_path=args.config,
            artifact_dir=args.artifact_dir,
            run_id=args.run_id,
            dataset_version=args.dataset_version,
            git_commit=args.git_commit,
            resume=args.resume,
            step_delay_ms=args.step_delay_ms,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "worker_failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
