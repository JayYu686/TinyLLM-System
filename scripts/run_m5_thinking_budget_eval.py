#!/usr/bin/env python3
"""Run the versioned Qwen-official Thinking Budget evaluation on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from tinyllm.evaluation import acquire_baseline_model, load_baseline_config
from tinyllm.evaluation.m5_thinking_budget import (
    M5ThinkingBudgetError,
    load_m5_thinking_budget_config,
    run_m5_thinking_budget_evaluation,
)
from tinyllm.lineage import read_git_identity
from tinyllm.training.m5_ablation_schema import M5AblationRunResult
from tinyllm.training.m5_formal_schema import M5FormalRunResult
from tinyllm.training.m5_lora_schema import M5LoRARunResult
from tinyllm.training.smoke_preflight import (
    MAX_TEMPERATURE_C,
    MAX_UTILIZATION_PERCENT,
    inspect_gpus,
)


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
            raise M5ThinkingBudgetError("Candidate export contains a non-regular file")
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _CandidateLineage:
    run_id: str
    seed: int
    thinking_fraction_basis_points: Literal[0, 3000, 5000]
    adapter_sha256: str | None = None
    training_checkpoint_id: str | None = None
    training_tokens: int | None = None


def _candidate_lineage(args: argparse.Namespace) -> _CandidateLineage | None:
    if args.model_kind == "base":
        if args.training_run is not None or args.training_checkpoint is not None:
            raise M5ThinkingBudgetError("Base evaluation cannot receive a training Run")
        return None
    if args.model_kind != "formal_candidate" and args.training_checkpoint is not None:
        raise M5ThinkingBudgetError("only formal Candidate can receive a staged Checkpoint")
    if args.training_run is None:
        raise M5ThinkingBudgetError("Candidate evaluation requires a training Run")
    try:
        if args.model_kind == "lora_candidate":
            lora_result = M5LoRARunResult.model_validate_json(
                (args.training_run / "result.json").read_bytes()
            )
            if args.adapter_dir is None:
                raise M5ThinkingBudgetError("LoRA Candidate requires --adapter-dir")
            if (
                lora_result.status != "succeeded"
                or lora_result.adapter_sha256 is None
                or args.adapter_dir.resolve()
                != (args.training_run / "exports" / "adapter").resolve()
                or _export_sha256(args.adapter_dir) != lora_result.adapter_sha256
            ):
                raise M5ThinkingBudgetError("LoRA Adapter differs from training lineage")
            return _CandidateLineage(
                run_id=lora_result.run_id,
                seed=lora_result.seed,
                thinking_fraction_basis_points=lora_result.thinking_fraction_basis_points,
                adapter_sha256=lora_result.adapter_sha256,
            )
        if args.model_kind == "formal_candidate":
            formal_result = M5FormalRunResult.model_validate_json(
                (args.training_run / "result.json").read_bytes()
            )
            if args.training_checkpoint is None:
                raise M5ThinkingBudgetError("formal Candidate requires --training-checkpoint")
            try:
                checkpoint_index = formal_result.evaluation_checkpoints.index(
                    args.training_checkpoint
                )
            except ValueError as exc:
                raise M5ThinkingBudgetError(
                    "formal Candidate Checkpoint differs from training lineage"
                ) from exc
            expected_model_dir = (
                args.training_run / "evaluations" / args.training_checkpoint / "model"
            )
            if (
                formal_result.status != "succeeded"
                or args.model_dir.resolve() != expected_model_dir.resolve()
                or _export_sha256(args.model_dir)
                != formal_result.evaluation_export_sha256s[checkpoint_index]
            ):
                raise M5ThinkingBudgetError("formal Candidate export differs from training lineage")
            return _CandidateLineage(
                run_id=formal_result.run_id,
                seed=formal_result.seed,
                thinking_fraction_basis_points=formal_result.thinking_fraction_basis_points,
                training_checkpoint_id=args.training_checkpoint,
                training_tokens=int(args.training_checkpoint.removeprefix("checkpoint-tokens-")),
            )
        ablation_result = M5AblationRunResult.model_validate_json(
            (args.training_run / "result.json").read_bytes()
        )
    except (OSError, ValueError) as exc:
        raise M5ThinkingBudgetError("Candidate training result is invalid") from exc
    if (
        ablation_result.status != "succeeded"
        or args.model_dir.resolve() != (args.training_run / "exports" / "model").resolve()
        or _export_sha256(args.model_dir) != ablation_result.export_sha256
    ):
        raise M5ThinkingBudgetError("Candidate export differs from training lineage")
    return _CandidateLineage(
        run_id=ablation_result.run_id,
        seed=ablation_result.seed,
        thinking_fraction_basis_points=ablation_result.thinking_fraction_basis_points,
    )


def _worker(args: argparse.Namespace) -> int:
    lineage = _candidate_lineage(args)
    result = run_m5_thinking_budget_evaluation(
        config_path=args.config,
        reasoning_config_path=args.reasoning_config,
        model_dir=args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        physical_gpu_index=args.gpu_index,
        model_kind=cast(
            Literal[
                "base",
                "ablation_candidate",
                "formal_candidate",
                "lora_candidate",
            ],
            args.model_kind,
        ),
        training_run_id=lineage.run_id if lineage is not None else None,
        training_seed=lineage.seed if lineage is not None else None,
        thinking_fraction_basis_points=(
            lineage.thinking_fraction_basis_points if lineage is not None else None
        ),
        training_checkpoint_id=(lineage.training_checkpoint_id if lineage is not None else None),
        training_tokens=lineage.training_tokens if lineage is not None else None,
        adapter_dir=args.adapter_dir,
        adapter_sha256=lineage.adapter_sha256 if lineage is not None else None,
        preflight_memory_used_mib=args.preflight_memory_used_mib,
        preflight_utilization_percent=args.preflight_utilization_percent,
        preflight_temperature_c=args.preflight_temperature_c,
    )
    print(result.model_dump_json())
    return 0


def _supervise(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    _, dirty = read_git_identity(project_root)
    if dirty:
        raise M5ThinkingBudgetError("formal evaluation requires a clean Git worktree")
    preflight = inspect_gpus((args.gpu_index,))[0]
    if (
        preflight["memory_used_mib"] > args.max_preflight_memory_mib
        or preflight["utilization_percent"] > MAX_UTILIZATION_PERCENT
        or preflight["temperature_c"] > MAX_TEMPERATURE_C
    ):
        raise M5ThinkingBudgetError(
            f"evaluation GPU preflight rejected physical index: {args.gpu_index}"
        )
    config = load_m5_thinking_budget_config(args.config)
    if args.model_kind == "lora_candidate":
        if (
            config.model_repository != "Qwen/Qwen3-8B"
            or args.model_dir.name != config.base_revision
            or args.tokenizer_dir.resolve() != args.model_dir.resolve()
        ):
            raise M5ThinkingBudgetError("LoRA evaluation requires the pinned 8B Base snapshot")
    else:
        baseline = load_baseline_config(args.baseline_config)
        tokenizer_dir = acquire_baseline_model(
            baseline,
            cache_root=args.artifact_root / "cache",
            offline=True,
        )
        if tokenizer_dir.resolve() != args.tokenizer_dir.resolve():
            raise M5ThinkingBudgetError("tokenizer differs from the pinned Base snapshot")
        if args.model_kind == "base" and args.model_dir.resolve() != tokenizer_dir.resolve():
            raise M5ThinkingBudgetError("Base evaluation requires the pinned Base snapshot")
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
        "--preflight-memory-used-mib",
        str(preflight["memory_used_mib"]),
        "--preflight-utilization-percent",
        str(preflight["utilization_percent"]),
        "--preflight-temperature-c",
        str(preflight["temperature_c"]),
    ]
    if args.training_run is not None:
        command.extend(("--training-run", str(args.training_run)))
    if args.training_checkpoint is not None:
        command.extend(("--training-checkpoint", str(args.training_checkpoint)))
    if args.adapter_dir is not None:
        command.extend(("--adapter-dir", str(args.adapter_dir)))
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
    """Build the Base/Candidate protocol-v2 interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/m5_thinking_budget_v2.yaml"),
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
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-kind",
        choices=("base", "ablation_candidate", "formal_candidate", "lora_candidate"),
        required=True,
    )
    parser.add_argument("--training-run", type=Path)
    parser.add_argument("--training-checkpoint")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument(
        "--max-preflight-memory-mib",
        type=int,
        choices=(1024, 3072),
        default=1024,
    )
    parser.add_argument("--preflight-memory-used-mib", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--preflight-utilization-percent", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--preflight-temperature-c", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--timeout-seconds", type=int, default=14_400)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Run the supervisor or isolated CUDA worker."""

    args = build_parser().parse_args()
    try:
        return _worker(args) if args.worker else _supervise(args)
    except (M5ThinkingBudgetError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
