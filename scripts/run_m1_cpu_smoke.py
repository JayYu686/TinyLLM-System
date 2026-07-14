#!/usr/bin/env python3
"""Run the test-scale M1.1 CPU/FP32 correctness smoke from YAML."""

from __future__ import annotations

import json
import platform
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch

from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash
from tinyllm.training import build_m1_cpu_trainer, load_training_config


def run_cpu_smoke(config_path: Path) -> dict[str, Any]:
    """Execute the smoke and return a sanitized machine-readable result."""

    project_root = Path(__file__).resolve().parents[1]
    config = load_training_config(config_path)
    trainer = build_m1_cpu_trainer(config)
    result = trainer.train()
    window_size = min(5, len(result.metrics))
    if window_size == 0:
        raise RuntimeError("CPU smoke produced no optimizer-step metrics")
    first_loss = sum(metric.loss for metric in result.metrics[:window_size]) / window_size
    last_loss = sum(metric.loss for metric in result.metrics[-window_size:]) / window_size
    git_commit, git_dirty = read_git_identity(project_root)
    return {
        "schema_version": "1.0",
        "smoke": "m1.1-cpu-fp32",
        "status": "pass" if last_loss < first_loss else "fail",
        "config": {
            "path": config_path.name,
            "resolved_sha256": canonical_config_hash(config),
        },
        "git": {"commit": git_commit, "dirty": git_dirty},
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
        },
        "hardware": {"device": "cpu"},
        "model": {
            "name": "TinyGPT test-scale fixture",
            "parameter_count": sum(parameter.numel() for parameter in trainer.model.parameters()),
        },
        "state": result.state.to_dict(),
        "loss": {
            "window_steps": window_size,
            "first_window_mean": first_loss,
            "last_window_mean": last_loss,
            "last_over_first": last_loss / first_loss,
        },
        "metrics": [metric.to_dict() for metric in result.metrics],
        "not_evaluated": [
            "checkpoint_and_resume",
            "cuda_bf16",
            "throughput_benchmark",
        ],
    }


def main() -> int:
    """Parse arguments, execute the smoke, and print deterministic JSON."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain/tinygpt_debug_cpu_smoke.yaml"),
    )
    args = parser.parse_args()
    payload = run_cpu_smoke(args.config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
