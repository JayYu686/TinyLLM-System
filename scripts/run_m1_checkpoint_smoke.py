#!/usr/bin/env python3
"""Exercise atomic M1.2 checkpoint publication, validation, and retention on CPU."""

from __future__ import annotations

import json
import platform
import tempfile
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training import (
    CheckpointContext,
    CheckpointError,
    CheckpointStore,
    build_m1_cpu_trainer,
    load_training_config,
)


def run_checkpoint_smoke(config_path: Path) -> dict[str, Any]:
    """Create four checkpoints, apply retention, and detect deliberate corruption."""

    project_root = Path(__file__).resolve().parents[1]
    config = load_training_config(config_path)
    config_hash = canonical_config_hash(config)
    git_commit, git_dirty = read_git_identity(project_root)
    run_id = generate_run_id(
        "m1-checkpoint-smoke",
        config_hash,
        now=datetime.now(UTC),
    )
    context = CheckpointContext(
        run_id=run_id,
        dataset_version=f"toy-checkpoint-{config_hash[:8]}",
        git_commit=git_commit,
        environment={
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": "cpu",
        },
    )
    trainer = build_m1_cpu_trainer(config)
    assert trainer.sampler is not None

    with tempfile.TemporaryDirectory(prefix="tinyllm-m1-checkpoint-") as temporary:
        store = CheckpointStore(Path(temporary) / "checkpoints", keep_last=2)
        for step in range(1, 5):
            trainer.train(target_global_step=step)
            store.save(
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=None,
                sampler=trainer.sampler,
                trainer_state=trainer.state,
                config=config,
                context=context,
                pin_reason="interruption" if step == 2 else None,
            )

        retained = sorted(path.name for path in store.root.glob("checkpoint-step-*"))
        latest_id = store.latest()
        latest_manifest = store.validate(latest_id)
        payload = store.load_training_state(latest_id)
        with (store.root / latest_id / "training_state.pt").open("ab") as stream:
            stream.write(b"deliberate-corruption-probe")
        corruption_code: str | None = None
        try:
            store.validate(latest_id)
        except CheckpointError as exc:
            corruption_code = exc.code

    expected_retained = [
        "checkpoint-step-00000002",
        "checkpoint-step-00000003",
        "checkpoint-step-00000004",
    ]
    passed = (
        retained == expected_retained
        and latest_id == "checkpoint-step-00000004"
        and payload["trainer_state"] == trainer.state.to_dict()
        and corruption_code == "CHECKPOINT_CORRUPT"
    )
    return {
        "schema_version": "1.0",
        "smoke": "m1.2-atomic-checkpoint-cpu",
        "status": "pass" if passed else "fail",
        "config": {
            "path": config_path.name,
            "resolved_sha256": config_hash,
        },
        "git": {"commit": git_commit, "dirty": git_dirty},
        "software": context.environment,
        "run_id": run_id,
        "retention": {
            "keep_last_ordinary": 2,
            "pinned": ["checkpoint-step-00000002"],
            "retained": retained,
            "latest": latest_id,
        },
        "latest_manifest": latest_manifest.to_dict(),
        "saved_sampler": payload["sampler"],
        "saved_trainer_state": payload["trainer_state"],
        "corruption_probe": {"detected_error_code": corruption_code},
        "not_evaluated": [
            "restoring_state_into_a_new_trainer",
            "exact_resume_equivalence",
            "signal_interruption_recovery",
            "cuda_bf16",
        ],
    }


def main() -> int:
    """Parse arguments, run the checkpoint smoke, and print JSON."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain/tinygpt_debug_cpu_smoke.yaml"),
    )
    args = parser.parse_args()
    payload = run_checkpoint_smoke(args.config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
