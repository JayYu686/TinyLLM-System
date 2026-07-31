#!/usr/bin/env python3
"""Verify and launch formal Qwen3-8B single-GPU BF16 LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tinyllm.data import open_m5_formal_dataset
from tinyllm.lineage import read_git_identity
from tinyllm.training.m5_config import load_m5_sft_config
from tinyllm.training.m5_lora import M5LoRAError, run_m5_lora
from tinyllm.training.smoke_preflight import inspect_gpus, validate_gpu_preflight


def _worker(args: argparse.Namespace) -> int:
    run_m5_lora(
        config_path=args.config,
        dataset_root=args.dataset_root,
        model_dir=args.model_dir,
        output_root=args.output_root,
        physical_gpu_index=args.gpu_index,
        resume_run=args.resume_run,
        stop_after_tokens=args.stop_after_tokens,
    )
    return 0


def _supervise(args: argparse.Namespace) -> int:
    from tinyllm.training.m4_model import inspect_qwen3_8b_artifact
    from tinyllm.training.m4_qwen_config import load_m4_qwen_config

    project_root = Path(__file__).resolve().parents[1]
    _, dirty = read_git_identity(project_root)
    if dirty:
        raise M5LoRAError("M5 LoRA training requires a clean Git worktree")
    validate_gpu_preflight(inspect_gpus((args.gpu_index,)))
    config = load_m5_sft_config(args.config)
    opened = open_m5_formal_dataset(args.dataset_root)
    manifest_sha256 = hashlib.sha256((args.dataset_root / "manifest.json").read_bytes()).hexdigest()
    if (
        config.data.dataset_version != opened.manifest.dataset_version
        or config.data.mix_manifest_sha256 != manifest_sha256
        or config.model.adaptation != "lora"
        or config.parallel.world_size != 1
    ):
        raise M5LoRAError("M5 LoRA config does not bind the verified Dataset and route")
    m4_config = load_m4_qwen_config(args.m4_config)
    artifact = inspect_qwen3_8b_artifact(model_dir=args.model_dir, config=m4_config)
    if (
        artifact.revision != config.model.revision
        or artifact.repository != config.model.repository
        or artifact.num_attention_heads != 32
        or artifact.num_key_value_heads != 8
    ):
        raise M5LoRAError("M5 LoRA model differs from the verified M4 snapshot")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--dataset-root",
        str(args.dataset_root),
        "--model-dir",
        str(args.model_dir),
        "--output-root",
        str(args.output_root),
        "--m4-config",
        str(args.m4_config),
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
    """Build the explicit single-GPU launch interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft/m5_formal_qwen3_8b_lora.yaml"),
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--m4-config",
        type=Path,
        default=Path("configs/fsdp2/qwen3_8b_four_gpu_formal.yaml"),
    )
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--stop-after-tokens", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=43_200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Run the supervisor or one isolated CUDA worker."""

    args = build_parser().parse_args()
    try:
        return _worker(args) if args.worker else _supervise(args)
    except (
        M5LoRAError,
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
