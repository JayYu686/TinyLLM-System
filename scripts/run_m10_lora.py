#!/usr/bin/env python3
"""Preflight, Probe, and launch staged M10 Qwen3-8B Agent LoRA."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tinyllm.lineage import read_git_identity
from tinyllm.training.m10_lora import (
    M10LoRAError,
    preflight_m10_lora,
    require_m10_lora_storage,
)
from tinyllm.training.m10_lora_schema import M10LoRAMemoryProbeResult, M10LoRARunResult
from tinyllm.training.m10_lora_worker import (
    run_m10_lora,
    run_m10_lora_memory_probe,
)
from tinyllm.training.smoke_preflight import inspect_gpus, validate_gpu_preflight


def _worker(args: argparse.Namespace) -> int:
    result: M10LoRAMemoryProbeResult | M10LoRARunResult
    if args.probe_output is not None:
        result = run_m10_lora_memory_probe(
            config_path=args.config,
            mixture_root=args.mixture_root,
            artifact_root=args.artifact_root,
            output_path=args.probe_output,
            physical_gpu_index=args.gpu_index,
        )
    else:
        if args.memory_probe is None:
            raise M10LoRAError("formal M10 Agent LoRA training requires --memory-probe")
        result = run_m10_lora(
            config_path=args.config,
            mixture_root=args.mixture_root,
            artifact_root=args.artifact_root,
            output_root=args.output_root,
            physical_gpu_index=args.gpu_index,
            memory_probe_path=args.memory_probe,
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
        raise M10LoRAError("formal M10 Agent LoRA execution requires a clean Git worktree")
    validate_gpu_preflight(inspect_gpus((args.gpu_index,)))
    require_m10_lora_storage(args.output_root)
    config, parent, manifest_sha256 = preflight_m10_lora(
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
                    "dataset_version": config.data.dataset_version,
                    "dataset_manifest_sha256": manifest_sha256,
                    "parent_evaluation_subject": parent.model_version,
                    "parent_evaluation_subject_sha256": parent.evaluation_subject_sha256,
                    "parent_model_artifact_sha256": parent.model_artifact_sha256,
                    "physical_gpu_index": args.gpu_index,
                    "next_action": "memory_probe_10_steps",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.probe_output is not None:
        if any(
            value is not None
            for value in (
                args.memory_probe,
                args.resume_run,
                args.stop_after_tokens,
                args.continuation_gate,
            )
        ):
            raise M10LoRAError("M10 Agent LoRA Probe cannot consume training-stage options")
        args.probe_output.parent.mkdir(parents=True, exist_ok=True)
    else:
        if args.memory_probe is None:
            raise M10LoRAError("formal M10 Agent LoRA training requires --memory-probe")
        if args.resume_run is None and args.stop_after_tokens != 1_000_000:
            raise M10LoRAError("fresh M10 Agent LoRA launch requires 1M stop boundary")
        if args.resume_run is not None and not args.resume_run.is_dir():
            raise M10LoRAError("M10 Agent LoRA Resume Run directory is missing")
        if args.resume_run is not None and args.continuation_gate is None:
            raise M10LoRAError("M10 Agent LoRA Resume requires --continuation-gate")
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
    optional_paths = (
        ("--probe-output", args.probe_output),
        ("--memory-probe", args.memory_probe),
        ("--resume-run", args.resume_run),
        ("--continuation-gate", args.continuation_gate),
    )
    for flag, value in optional_paths:
        if value is not None:
            command.extend((flag, str(value)))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft/m10_agent_lora_qwen3_8b.yaml"),
    )
    parser.add_argument("--mixture-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--probe-output", type=Path)
    parser.add_argument("--memory-probe", type=Path)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--continuation-gate", type=Path)
    parser.add_argument("--stop-after-tokens", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=43_200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return _worker(args) if args.worker else _supervise(args)
    except (
        M10LoRAError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.TimeoutExpired,
    ) as exc:
        message = str(exc)
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": message,
                    "failure_kind": (
                        "verified_bf16_oom" if "out of memory" in message.lower() else "runtime"
                    ),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
