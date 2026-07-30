#!/usr/bin/env python3
"""Validate lineage and run one private M5.2-R2 length replay on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tinyllm.evaluation import acquire_baseline_model, load_baseline_config
from tinyllm.evaluation.m5_r2_diagnostic import (
    M5R2DiagnosticError,
    expected_m5_r2_source_identity,
    load_m5_r2_replay_config,
    run_m5_r2_length_replay,
)
from tinyllm.evaluation.m5_r2_schema import M5R2ReplayConfig
from tinyllm.lineage import read_git_identity
from tinyllm.training.m5_ablation_schema import M5AblationRunResult
from tinyllm.training.smoke_preflight import inspect_gpus, validate_gpu_preflight


class M5R2PreflightError(RuntimeError):
    """Raised before the isolated replay worker is started."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    try:
        paths = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise M5R2DiagnosticError("M5 R2 Candidate export cannot be read") from exc
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise M5R2DiagnosticError("M5 R2 Candidate export contains a non-regular file")
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def _validate_lineage(
    args: argparse.Namespace,
    config: M5R2ReplayConfig,
) -> M5AblationRunResult:
    try:
        result = M5AblationRunResult.model_validate_json(
            (args.training_run / "result.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise M5R2DiagnosticError("M5 R2 Candidate training result is invalid") from exc
    expected_run, _, _, expected_export = expected_m5_r2_source_identity(config, result.seed)
    export_sha256 = _export_sha256(args.model_dir)
    if (
        result.status != "succeeded"
        or result.seed not in {42, 20260727}
        or result.thinking_fraction_basis_points != 3000
        or result.run_id != expected_run
        or result.mixture_version != config.source_mixture_version
        or result.mixture_manifest_sha256 != config.source_mixture_manifest_sha256
        or args.model_dir.resolve() != (args.training_run / "exports" / "model").resolve()
        or export_sha256 != result.export_sha256
        or export_sha256 != expected_export
    ):
        raise M5R2DiagnosticError("M5 R2 Candidate export or R1 lineage differs")
    return result


def _worker(args: argparse.Namespace) -> int:
    _validate_lineage(args, load_m5_r2_replay_config(args.config))
    summary = run_m5_r2_length_replay(
        replay_config_path=args.config,
        evaluation_config_path=args.evaluation_config,
        reasoning_config_path=args.reasoning_config,
        source_evaluation_dir=args.source_evaluation,
        training_run_dir=args.training_run,
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        physical_gpu_index=args.gpu_index,
    )
    print(summary.model_dump_json())
    return 0


def _supervise(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    _, dirty = read_git_identity(project_root)
    if dirty:
        raise M5R2DiagnosticError("formal M5 R2 replay requires a clean Git worktree")
    config = load_m5_r2_replay_config(args.config)
    if (
        not args.source_evaluation.is_dir()
        or not (args.source_evaluation / "summary.json").is_file()
        or not (args.source_evaluation / "results.jsonl").is_file()
    ):
        raise M5R2DiagnosticError("M5 R2 source evaluation directory is incomplete")
    _validate_lineage(args, config)
    try:
        validate_gpu_preflight(inspect_gpus((args.gpu_index,)))
    except RuntimeError as exc:
        raise M5R2PreflightError(str(exc)) from exc
    baseline_config = load_baseline_config(args.baseline_config)
    tokenizer_dir = acquire_baseline_model(
        baseline_config,
        cache_root=args.artifact_root / "cache",
        offline=True,
    )
    if tokenizer_dir.resolve() != args.tokenizer_dir.resolve():
        raise M5R2DiagnosticError("M5 R2 tokenizer differs from the pinned Base snapshot")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--evaluation-config",
        str(args.evaluation_config),
        "--reasoning-config",
        str(args.reasoning_config),
        "--baseline-config",
        str(args.baseline_config),
        "--artifact-root",
        str(args.artifact_root),
        "--source-evaluation",
        str(args.source_evaluation),
        "--training-run",
        str(args.training_run),
        "--model-dir",
        str(args.model_dir),
        "--tokenizer-dir",
        str(args.tokenizer_dir),
        "--output-dir",
        str(args.output_dir),
        "--gpu-index",
        str(args.gpu_index),
        "--worker",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            check=False,
            text=True,
            timeout=args.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise M5R2PreflightError("M5 R2 replay worker timed out") from exc
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit one-Seed R2 replay interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/m5_r2_length_replay.yaml"),
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=Path("configs/eval/m5_reasoning_dev.yaml"),
    )
    parser.add_argument(
        "--reasoning-config",
        type=Path,
        default=Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml"),
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/eval/m2_baseline.yaml"),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-evaluation", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=14_400)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Run the R2 supervisor or isolated CUDA worker with stable exit codes."""

    args = build_parser().parse_args()
    try:
        return _worker(args) if args.worker else _supervise(args)
    except M5R2PreflightError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 3
    except M5R2DiagnosticError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 6 if args.worker else 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 6 if args.worker else 2


if __name__ == "__main__":
    raise SystemExit(main())
