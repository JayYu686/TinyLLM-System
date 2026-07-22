#!/usr/bin/env python3
"""Preflight and run the private 200-item M5.2 dual-mode Reasoning Dev evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, cast

from tinyllm.evaluation import acquire_baseline_model, load_baseline_config
from tinyllm.evaluation.m5_reasoning import (
    M5ReasoningEvaluationError,
    run_m5_reasoning_evaluation,
)
from tinyllm.lineage import read_git_identity
from tinyllm.training.m5_ablation_schema import M5AblationRunResult
from tinyllm.training.smoke_preflight import inspect_gpus, validate_gpu_preflight


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink():
            raise M5ReasoningEvaluationError("M5 Candidate export contains a non-regular file")
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def _candidate_lineage(args: argparse.Namespace) -> M5AblationRunResult | None:
    if args.model_kind == "base":
        if args.training_run is not None:
            raise M5ReasoningEvaluationError("Base evaluation cannot receive a training Run")
        return None
    if args.training_run is None:
        raise M5ReasoningEvaluationError("Candidate evaluation requires a training Run")
    try:
        result = M5AblationRunResult.model_validate_json(
            (args.training_run / "result.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise M5ReasoningEvaluationError("M5 Candidate training result is invalid") from exc
    if (
        result.status != "succeeded"
        or args.model_dir.resolve() != (args.training_run / "exports" / "model").resolve()
        or _export_sha256(args.model_dir) != result.export_sha256
    ):
        raise M5ReasoningEvaluationError("M5 Candidate export differs from training lineage")
    return result


def _worker(args: argparse.Namespace) -> int:
    lineage = _candidate_lineage(args)
    summary = run_m5_reasoning_evaluation(
        config_path=args.config,
        reasoning_config_path=args.reasoning_config,
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        physical_gpu_index=args.gpu_index,
        model_kind=cast(Literal["base", "ablation_candidate"], args.model_kind),
        training_run_id=lineage.run_id if lineage is not None else None,
        training_seed=lineage.seed if lineage is not None else None,
        thinking_fraction_basis_points=(
            lineage.thinking_fraction_basis_points if lineage is not None else None
        ),
    )
    print(summary.model_dump_json())
    return 0


def _supervise(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    _, dirty = read_git_identity(project_root)
    if dirty:
        raise M5ReasoningEvaluationError("formal M5 evaluation requires a clean Git worktree")
    validate_gpu_preflight(inspect_gpus((args.gpu_index,)))
    baseline_config = load_baseline_config(args.baseline_config)
    tokenizer_dir = acquire_baseline_model(
        baseline_config,
        cache_root=args.artifact_root / "cache",
        offline=True,
    )
    if tokenizer_dir.resolve() != args.tokenizer_dir.resolve():
        raise M5ReasoningEvaluationError("M5 tokenizer path differs from pinned Base snapshot")
    if args.model_kind == "base" and args.model_dir.resolve() != tokenizer_dir.resolve():
        raise M5ReasoningEvaluationError("M5 Base evaluation requires the pinned Base snapshot")
    _candidate_lineage(args)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--reasoning-config",
        str(args.reasoning_config),
        "--baseline-config",
        str(args.baseline_config),
        "--artifact-root",
        str(args.artifact_root),
        "--model-dir",
        str(args.model_dir),
        "--tokenizer-dir",
        str(args.tokenizer_dir),
        "--output-dir",
        str(args.output_dir),
        "--model-kind",
        str(args.model_kind),
        "--gpu-index",
        str(args.gpu_index),
        "--worker",
    ]
    if args.training_run is not None:
        command.extend(("--training-run", str(args.training_run)))
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
    """Build the explicit Base/Candidate M5 evaluation interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/m5_reasoning_dev.yaml"),
    )
    parser.add_argument(
        "--reasoning-config",
        type=Path,
        default=Path("configs/data/m5_reasoning.yaml"),
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/eval/m2_baseline.yaml"),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-kind",
        choices=("base", "ablation_candidate"),
        required=True,
    )
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=14_400)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Run the M5 evaluation supervisor or its isolated worker."""

    args = build_parser().parse_args()
    try:
        return _worker(args) if args.worker else _supervise(args)
    except (M5ReasoningEvaluationError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
