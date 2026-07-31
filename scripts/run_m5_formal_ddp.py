#!/usr/bin/env python3
"""Preflight and launch the formal four-GPU Qwen3-0.6B Full-SFT Run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tinyllm.data import open_m5_formal_dataset
from tinyllm.evaluation import acquire_baseline_model, load_baseline_config
from tinyllm.lineage import read_git_identity
from tinyllm.training.m5_config import load_m5_sft_config
from tinyllm.training.m5_formal import M5FormalTrainingError, run_m5_formal_ddp
from tinyllm.training.smoke_preflight import inspect_gpus, validate_gpu_preflight


def _gpu_indices(value: str) -> tuple[int, int, int, int]:
    try:
        indices = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU indices must be comma-separated integers") from exc
    if len(indices) != 4 or len(set(indices)) != 4 or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("formal M5 requires four distinct GPU indices")
    return indices


def _worker(args: argparse.Namespace) -> int:
    result = run_m5_formal_ddp(
        config_path=args.config,
        dataset_root=args.dataset_root,
        model_dir=args.model_dir,
        output_root=args.output_root,
        physical_gpu_indices=args.gpu_indices,
        resume_run=args.resume_run,
        stop_after_tokens=args.stop_after_tokens,
    )
    return 0 if result is not None or int(os.environ.get("RANK", "-1")) != 0 else 4


def _supervise(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    _, dirty = read_git_identity(project_root)
    if dirty:
        raise M5FormalTrainingError("formal M5 training requires a clean Git worktree")
    validate_gpu_preflight(inspect_gpus(args.gpu_indices))
    config = load_m5_sft_config(args.config)
    opened = open_m5_formal_dataset(args.dataset_root)
    manifest_sha256 = hashlib.sha256((args.dataset_root / "manifest.json").read_bytes()).hexdigest()
    if (
        config.data.dataset_version != opened.manifest.dataset_version
        or config.data.mix_manifest_sha256 != manifest_sha256
        or config.parallel.world_size != 4
    ):
        raise M5FormalTrainingError("formal M5 config does not bind the verified Dataset")
    baseline = load_baseline_config(args.baseline_config)
    verified_model_dir = acquire_baseline_model(
        baseline,
        cache_root=args.artifact_root / "cache",
        offline=True,
    )
    if verified_model_dir.resolve() != args.model_dir.resolve():
        raise M5FormalTrainingError("formal M5 model differs from the pinned Base snapshot")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in args.gpu_indices)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "4",
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--dataset-root",
        str(args.dataset_root),
        "--model-dir",
        str(args.model_dir),
        "--output-root",
        str(args.output_root),
        "--artifact-root",
        str(args.artifact_root),
        "--baseline-config",
        str(args.baseline_config),
        "--gpu-indices",
        ",".join(str(value) for value in args.gpu_indices),
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
    """Build the explicit four-GPU launch interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft/m5_formal_qwen3_0_6b.yaml"),
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/eval/m2_baseline.yaml"),
    )
    parser.add_argument("--gpu-indices", type=_gpu_indices, required=True)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--stop-after-tokens", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=43_200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Run the supervisor or one torchrun worker."""

    args = build_parser().parse_args()
    try:
        return _worker(args) if args.worker else _supervise(args)
    except (
        M5FormalTrainingError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(
            json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
