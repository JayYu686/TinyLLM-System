#!/usr/bin/env python3
"""Preflight and launch staged M10 Qwen3-0.6B Agent Full SFT."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tinyllm.lineage import read_git_identity
from tinyllm.training.m10_sft import (
    M10FullSFTError,
    preflight_m10_full_sft,
)
from tinyllm.training.m10_sft_worker import run_m10_full_sft
from tinyllm.training.smoke_preflight import inspect_gpus, validate_gpu_preflight


def _worker(args: argparse.Namespace) -> int:
    result = run_m10_full_sft(
        config_path=args.config,
        mixture_root=args.mixture_root,
        artifact_root=args.artifact_root,
        output_root=args.output_root,
        physical_gpu_index=args.gpu_index,
        resume_run=args.resume_run,
        stop_after_tokens=args.stop_after_tokens,
        continuation_gate_path=args.continuation_gate,
    )
    print(result.model_dump_json())
    return 0


def _supervise(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    _, dirty = read_git_identity(project_root)
    if dirty:
        raise M10FullSFTError("formal M10.2 training requires a clean Git worktree")
    validate_gpu_preflight(inspect_gpus((args.gpu_index,)))
    config, resolved, manifest_sha256 = preflight_m10_full_sft(
        config_path=args.config,
        mixture_root=args.mixture_root,
        artifact_root=args.artifact_root,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "ready",
                    "parent_production_record_sha256": (
                        config.model.parent_production_record_sha256
                    ),
                    "dataset_version": config.data.dataset_version,
                    "dataset_manifest_sha256": manifest_sha256,
                    "parent_production_version": resolved.model_version,
                    "parent_model_artifact_sha256": resolved.model_artifact_sha256,
                    "physical_gpu_index": args.gpu_index,
                    "next_stage_tokens": 1_000_000,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.resume_run is None and args.stop_after_tokens != 1_000_000:
        raise M10FullSFTError("fresh M10.2 launch requires --stop-after-tokens 1000000")
    if args.resume_run is not None and not args.resume_run.is_dir():
        raise M10FullSFTError("M10 Resume Run directory is missing")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--mixture-root",
        str(args.mixture_root),
        "--artifact-root",
        str(args.artifact_root),
        "--output-root",
        str(args.output_root),
        "--gpu-index",
        str(args.gpu_index),
        "--worker",
    ]
    if args.resume_run is not None:
        command.extend(("--resume-run", str(args.resume_run)))
    if args.continuation_gate is not None:
        command.extend(("--continuation-gate", str(args.continuation_gate)))
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
    """Build the explicit M10.2 single-GPU launch interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft/m10_agent_full_sft_qwen3_0_6b.yaml"),
    )
    parser.add_argument("--mixture-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--continuation-gate", type=Path)
    parser.add_argument("--stop-after-tokens", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=43_200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Run preflight, supervisor, or its isolated CUDA worker."""

    args = build_parser().parse_args()
    try:
        return _worker(args) if args.worker else _supervise(args)
    except (
        M10FullSFTError,
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
