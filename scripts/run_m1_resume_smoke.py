#!/usr/bin/env python3
"""Compare uninterrupted CPU training with Exact/Warm/Transfer checkpoint restore."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import tempfile
from argparse import ArgumentParser
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy
import torch
from torch import Tensor

from tinyllm.data import ToyTokenDataset
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training import (
    CheckpointContext,
    CheckpointError,
    CheckpointStore,
    ResumeMode,
    SingleDeviceTrainer,
    build_m1_cpu_trainer,
    load_training_config,
    restore_trainer,
)
from tinyllm.training.checkpoint import capture_rng_state
from tinyllm.training.config import M1TrainingConfig


def _nested_equal(left: object, right: object) -> bool:
    if isinstance(left, Tensor):
        return isinstance(right, Tensor) and torch.equal(left, right)
    if isinstance(left, numpy.ndarray):
        return isinstance(right, numpy.ndarray) and numpy.array_equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(
                _nested_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    return bool(left == right)


def _model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tensor_digest(value: Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _save_checkpoint(
    *,
    store: CheckpointStore,
    trainer: SingleDeviceTrainer,
    config: M1TrainingConfig,
    context: CheckpointContext,
    pinned: bool = False,
) -> str:
    if trainer.sampler is None:
        raise RuntimeError("resume smoke requires the stateful sampler")
    manifest = store.save(
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=None,
        sampler=trainer.sampler,
        trainer_state=trainer.state,
        config=config,
        context=context,
        pin_reason="interruption" if pinned else None,
    )
    return manifest.checkpoint_id


def _probe_failure(
    *,
    store: CheckpointStore,
    config: M1TrainingConfig,
    context: CheckpointContext,
    checkpoint_id: str,
) -> dict[str, str | None]:
    target = build_m1_cpu_trainer(config)
    try:
        restore_trainer(
            store=store,
            trainer=target,
            mode=ResumeMode.EXACT,
            context=context,
            checkpoint_id=checkpoint_id,
        )
    except CheckpointError as exc:
        reason = exc.context.get("reason")
        return {"code": exc.code.value, "reason": reason if isinstance(reason, str) else None}
    return {"code": None, "reason": None}


def run_resume_smoke(config_path: Path) -> dict[str, Any]:
    """Run real CPU restore comparisons and return machine-readable evidence."""

    project_root = Path(__file__).resolve().parents[1]
    config = load_training_config(config_path)
    config_hash = canonical_config_hash(config)
    git_commit, git_dirty = read_git_identity(project_root)
    run_id = generate_run_id("m1-exact-resume-smoke", config_hash, now=datetime.now(UTC))
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": "cpu",
    }
    context = CheckpointContext(
        run_id=run_id,
        dataset_version=f"toy-resume-{config_hash[:8]}",
        git_commit=git_commit,
        environment=environment,
    )
    interrupt_step = min(10, max(1, config.training.max_steps // 2))
    earlier_step = max(1, interrupt_step // 2)

    uninterrupted = build_m1_cpu_trainer(config)
    uninterrupted.train(target_global_step=interrupt_step)
    if uninterrupted.sampler is None:
        raise RuntimeError("resume smoke requires the stateful sampler")
    uninterrupted_boundary = uninterrupted.sampler.state_dict()
    uninterrupted_tail = uninterrupted.train()

    interrupted = build_m1_cpu_trainer(config)
    with tempfile.TemporaryDirectory(prefix="tinyllm-m1-resume-") as temporary:
        store = CheckpointStore(Path(temporary) / "checkpoints", keep_last=2)
        interrupted.train(target_global_step=earlier_step)
        earlier_id = _save_checkpoint(
            store=store,
            trainer=interrupted,
            config=config,
            context=context,
        )
        interrupted.train(target_global_step=interrupt_step)
        interruption_id = _save_checkpoint(
            store=store,
            trainer=interrupted,
            config=config,
            context=context,
            pinned=True,
        )
        payload = store.load_training_state(interruption_id)

        resumed = build_m1_cpu_trainer(config)
        random.random()
        numpy.random.random()
        torch.rand(4)
        exact_result = restore_trainer(
            store=store,
            trainer=resumed,
            mode=ResumeMode.EXACT,
            context=context,
        )
        restored_rng_equal = _nested_equal(capture_rng_state(), payload["rng"])
        restored_boundary = resumed.sampler.state_dict() if resumed.sampler is not None else None

        dataset = ToyTokenDataset(
            vocab_size=config.data.vocab_size,
            sequence_length=config.data.sequence_length,
            num_samples=config.data.num_samples,
            seed=config.run.seed,
        )
        cursor = int(payload["sampler"]["cursor"])
        next_indices = [
            (cursor + offset) % len(dataset) for offset in range(config.training.micro_batch_size)
        ]
        next_batch = torch.stack([dataset[index] for index in next_indices])
        next_batch_sha256 = _tensor_digest(next_batch)

        resumed_tail = resumed.train()
        final_parameters_equal = all(
            torch.equal(resumed_parameter, uninterrupted_parameter)
            for resumed_parameter, uninterrupted_parameter in zip(
                resumed.model.parameters(), uninterrupted.model.parameters(), strict=True
            )
        )
        optimizer_equal = _nested_equal(
            resumed.optimizer.state_dict(), uninterrupted.optimizer.state_dict()
        )
        scheduler_equal = _nested_equal(
            resumed.scheduler.state_dict(),  # type: ignore[no-untyped-call]
            uninterrupted.scheduler.state_dict(),  # type: ignore[no-untyped-call]
        )

        warm_target = build_m1_cpu_trainer(
            config.model_copy(update={"run": config.run.model_copy(update={"seed": 99})})
        )
        warm_result = restore_trainer(
            store=store,
            trainer=warm_target,
            mode=ResumeMode.WARM,
            checkpoint_id=interruption_id,
        )
        warm_weights_equal = all(
            torch.equal(warm_parameter, source_parameter)
            for warm_parameter, source_parameter in zip(
                warm_target.model.parameters(), interrupted.model.parameters(), strict=True
            )
        )

        transfer_config = config.model_copy(
            update={
                "model": config.model.model_copy(
                    update={"vocab_size": config.model.vocab_size + 4}
                ),
                "data": config.data.model_copy(update={"vocab_size": config.data.vocab_size + 4}),
            }
        )
        transfer_target = build_m1_cpu_trainer(transfer_config)
        transfer_result = restore_trainer(
            store=store,
            trainer=transfer_target,
            mode=ResumeMode.TRANSFER,
            checkpoint_id=interruption_id,
        )

        drift_config = config.model_copy(
            update={
                "training": config.training.model_copy(
                    update={"learning_rate": config.training.learning_rate * 2}
                )
            }
        )
        failure_matrix = {
            "config_drift": _probe_failure(
                store=store,
                config=drift_config,
                context=context,
                checkpoint_id=interruption_id,
            ),
            "data_version": _probe_failure(
                store=store,
                config=config,
                context=replace(context, dataset_version="wrong-data-version"),
                checkpoint_id=interruption_id,
            ),
            "world_size": _probe_failure(
                store=store,
                config=config,
                context=replace(context, world_size=2),
                checkpoint_id=interruption_id,
            ),
        }

        with (store.root / interruption_id / "training_state.pt").open("ab") as stream:
            stream.write(b"deliberate-resume-corruption")
        failure_matrix["bad_hash"] = _probe_failure(
            store=store,
            config=config,
            context=context,
            checkpoint_id=interruption_id,
        )
        fallback = store.latest_valid()

    metrics_equal = resumed_tail.metrics == uninterrupted_tail.metrics
    state_equal = resumed_tail.state == uninterrupted_tail.state
    no_repeated_step = bool(resumed_tail.metrics) and (
        resumed_tail.metrics[0].global_step == interrupt_step + 1
        and len(resumed_tail.metrics) == config.training.max_steps - interrupt_step
    )
    sampler_equal = uninterrupted_boundary == payload["sampler"] == restored_boundary
    failure_codes = {name: probe["code"] for name, probe in failure_matrix.items()}
    failures_pass = failure_codes == {
        "config_drift": "CHECKPOINT_INCOMPATIBLE",
        "data_version": "CHECKPOINT_INCOMPATIBLE",
        "world_size": "CHECKPOINT_INCOMPATIBLE",
        "bad_hash": "CHECKPOINT_CORRUPT",
    }
    passed = all(
        (
            exact_result.checkpoint_id == interruption_id,
            sampler_equal,
            restored_rng_equal,
            metrics_equal,
            state_equal,
            no_repeated_step,
            final_parameters_equal,
            optimizer_equal,
            scheduler_equal,
            warm_weights_equal,
            warm_target.is_pristine,
            bool(transfer_result.loaded_model_keys),
            bool(transfer_result.incompatible_checkpoint_keys),
            failures_pass,
            fallback.checkpoint_id == earlier_id,
            fallback.skipped_invalid_checkpoints == (interruption_id,),
        )
    )
    return {
        "schema_version": "1.0",
        "smoke": "m1.3-resume-semantics-cpu",
        "status": "pass" if passed else "fail",
        "config": {"path": config_path.name, "resolved_sha256": config_hash},
        "git": {"commit": git_commit, "dirty": git_dirty},
        "software": environment,
        "run_id": run_id,
        "exact_resume": {
            "checkpoint_id": interruption_id,
            "interruption_global_step": interrupt_step,
            "final_global_step": resumed_tail.state.global_step,
            "first_resumed_global_step": (
                resumed_tail.metrics[0].global_step if resumed_tail.metrics else None
            ),
            "resumed_optimizer_steps": len(resumed_tail.metrics),
            "sampler_boundary_equal": sampler_equal,
            "next_batch_indices": next_indices,
            "next_batch_sha256": next_batch_sha256,
            "rng_equal_at_restore": restored_rng_equal,
            "loss_lr_and_metrics_bitwise_equal": metrics_equal,
            "trainer_state_equal": state_equal,
            "parameters_bitwise_equal": final_parameters_equal,
            "optimizer_state_bitwise_equal": optimizer_equal,
            "scheduler_state_bitwise_equal": scheduler_equal,
            "uninterrupted_model_sha256": _model_digest(uninterrupted.model),
            "resumed_model_sha256": _model_digest(resumed.model),
        },
        "warm_resume": {
            "model_keys_loaded": len(warm_result.loaded_model_keys),
            "weights_equal_to_source": warm_weights_equal,
            "target_global_step": warm_target.state.global_step,
            "runtime_state_reset": warm_target.is_pristine,
        },
        "transfer_resume": {
            "model_keys_loaded": len(transfer_result.loaded_model_keys),
            "missing_keys": list(transfer_result.missing_model_keys),
            "incompatible_keys": list(transfer_result.incompatible_checkpoint_keys),
            "target_global_step": transfer_target.state.global_step,
        },
        "failure_matrix": failure_matrix,
        "auto_fallback": {
            "selected": fallback.checkpoint_id,
            "skipped_invalid": list(fallback.skipped_invalid_checkpoints),
        },
        "not_evaluated": [
            "sigterm_or_sigkill_process_recovery",
            "cuda_bf16_resume_tolerance",
            "distributed_checkpoint_resume",
        ],
    }


def main() -> int:
    """Parse arguments, run the CPU resume smoke, and print JSON."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain/tinygpt_debug_cpu_smoke.yaml"),
    )
    args = parser.parse_args()
    payload = run_resume_smoke(args.config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
