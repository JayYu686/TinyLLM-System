#!/usr/bin/env python3
"""Preflight and launch one real M5.2 Qwen3-0.6B ablation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tinyllm.data import open_m5_ablation_mixture
from tinyllm.evaluation import acquire_baseline_model, load_baseline_config
from tinyllm.lineage import read_git_identity
from tinyllm.training.m5_ablation import M5AblationError, run_m5_ablation
from tinyllm.training.m5_config import load_m5_sft_config
from tinyllm.training.smoke_preflight import inspect_gpus, validate_gpu_preflight


def _worker(args: argparse.Namespace) -> int:
    result = run_m5_ablation(
        config_path=args.config,
        mixture_root=args.mixture_root,
        model_dir=args.model_dir,
        output_root=args.output_root,
        physical_gpu_index=args.gpu_index,
        resume_run=args.resume_run,
        stop_after_tokens=args.stop_after_tokens,
    )
    print(result.model_dump_json())
    return 0


def _supervise(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    _, dirty = read_git_identity(project_root)
    if dirty:
        raise M5AblationError("formal M5.2 training requires a clean Git worktree")
    validate_gpu_preflight(inspect_gpus((args.gpu_index,)))
    config = load_m5_sft_config(args.config)
    mixture = open_m5_ablation_mixture(args.mixture_root)
    mixture_sha256 = hashlib.sha256((args.mixture_root / "manifest.json").read_bytes()).hexdigest()
    if (
        config.data.dataset_version != mixture.manifest.pilot_dataset_version
        or config.data.mix_manifest_sha256 != mixture_sha256
    ):
        raise M5AblationError("M5 config does not name the verified private mixture")
    baseline_config = load_baseline_config(args.baseline_config)
    verified_model_dir = acquire_baseline_model(
        baseline_config,
        cache_root=args.artifact_root / "cache",
        offline=True,
    )
    if verified_model_dir.resolve() != args.model_dir.resolve():
        raise M5AblationError("M5 model path differs from the verified pinned snapshot")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--mixture-root",
        str(args.mixture_root),
        "--model-dir",
        str(args.model_dir),
        "--output-root",
        str(args.output_root),
        "--artifact-root",
        str(args.artifact_root),
        "--baseline-config",
        str(args.baseline_config),
        "--gpu-index",
        str(args.gpu_index),
        "--worker",
    ]
    if args.resume_run is not None:
        command.extend(("--resume-run", str(args.resume_run)))
    if args.stop_after_tokens is not None:
        command.extend(("--stop-after-tokens", str(args.stop_after_tokens)))
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=False,
        text=True,
        timeout=args.timeout_seconds,
    )
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit M5.2 single-GPU launch interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mixture-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/eval/m2_baseline.yaml"),
    )
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--stop-after-tokens", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=43_200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Run the supervisor or its isolated CUDA worker."""

    args = build_parser().parse_args()
    try:
        return _worker(args) if args.worker else _supervise(args)
    except (M5AblationError, OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(
            json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
