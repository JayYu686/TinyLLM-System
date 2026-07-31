#!/usr/bin/env python3
"""Generate, verify, and select M5.2-R3 formal sources in resumable GPU shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_m5_r3_p1 import (
    _generation_record,
    _preflight_task_sets,
    _sha256_file,
    _verify_frozen_inputs,
    _verify_model_directory,
    _verify_policy_python,
)
from tinyllm.data import ReasoningTask, TokenizersBackend, load_m2_tokenization_config
from tinyllm.data.m5_r3_formal import (
    M5R3FormalSourceError,
    build_m5_r3_formal_source,
    check_m5_r3_formal_contamination,
    generate_m5_r3_formal_contexts,
    load_m5_r3_formal_source_config,
    m5_r3_formal_source_config_sha256,
)
from tinyllm.data.m5_r3_formal_schema import (
    M5R3FormalShardArtifact,
    M5R3FormalSourceResult,
)
from tinyllm.data.m5_r3_p1 import m5_r3_p1_stage_seed
from tinyllm.data.m5_r3_p1_schema import (
    M5R3P1StageGeneration,
    M5R3P1TaskContext,
)
from tinyllm.data.m5_r3_p2 import (
    build_m5_r3_p2_fallback_solver_prompt,
    build_m5_r3_p2_isolated_compressor_prompt,
)
from tinyllm.data.m5_r3_review_schema import M5R3ContentReviewResult
from tinyllm.data.reasoning import parse_teacher_output
from tinyllm.data.reasoning_schema import canonical_json
from tinyllm.lineage import read_git_identity
from tinyllm.training.smoke_preflight import inspect_gpus, validate_gpu_preflight


class M5R3FormalEnvironmentError(RuntimeError):
    """Raised when one formal-source worker cannot safely use its environment."""


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parent_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        config=args.p1_config,
        reasoning_config=args.reasoning_config,
        tokenization_config=args.tokenization_config,
        p0_config=args.p0_config,
        p0_r1_config=args.p0_r1_config,
        r2_decision=args.r2_decision,
        p0_result=args.p0_result,
        p0_r1_result=args.p0_r1_result,
        historical_pilot_artifact=args.historical_pilot_artifact,
    )


def _verify_formal_inputs(
    args: argparse.Namespace,
) -> tuple[
    tuple[M5R3P1TaskContext, ...],
    tuple[ReasoningTask, ...],
    tuple[ReasoningTask, ...],
    tuple[ReasoningTask, ...],
    tuple[ReasoningTask, ...],
    tuple[M5R3P1TaskContext, ...],
]:
    config = load_m5_r3_formal_source_config(args.config)
    if _sha256_file(args.content_review_result) != config.parent_content_review_sha256:
        raise M5R3FormalSourceError("M5 R3 formal content-review SHA256 differs")
    try:
        review = M5R3ContentReviewResult.model_validate_json(
            args.content_review_result.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise M5R3FormalSourceError("M5 R3 formal content review is invalid") from exc
    if review.status != "approved" or not review.formal_source_expansion_authorized:
        raise M5R3FormalSourceError("M5 R3 formal source expansion is not authorized")

    parent_args = _parent_namespace(args)
    _verify_frozen_inputs(parent_args)
    p1_contexts, dev, historical, p0, p0_r1 = _preflight_task_sets(parent_args)
    contexts = generate_m5_r3_formal_contexts(config)
    contamination = check_m5_r3_formal_contamination(
        contexts,
        dev_tasks=dev,
        historical_tasks=historical,
        p0_tasks=p0,
        p0_r1_tasks=p0_r1,
        p1_tasks=(item.task for item in p1_contexts),
    )
    if contamination.status != "pass":
        raise M5R3FormalSourceError("M5 R3 formal contamination preflight failed")
    return contexts, dev, historical, p0, p0_r1, p1_contexts


def _worker(args: argparse.Namespace) -> int:
    import torch
    import transformers  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    project_root = Path(__file__).resolve().parents[1]
    config = load_m5_r3_formal_source_config(args.config)
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5R3FormalSourceError("M5 R3 formal worker requires a clean Git worktree")
    _verify_model_directory(args.model_dir, config.solver.revision)
    all_contexts, *_parents = _verify_formal_inputs(args)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M5R3FormalEnvironmentError("M5 R3 formal worker requires one visible CUDA device")
    shard_contexts = tuple(
        context
        for index, context in enumerate(all_contexts)
        if index % args.shard_count == args.shard_index
    )
    if not shard_contexts:
        raise M5R3FormalSourceError("M5 R3 formal shard has no tasks")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    records: list[M5R3P1StageGeneration] = []
    global_index = {context.task.id: index for index, context in enumerate(all_contexts)}

    def generate(
        context: M5R3P1TaskContext,
        *,
        stage: str,
        prompt: str,
        seed: int,
        enable_thinking: bool,
        max_new_tokens: int,
        do_sample: bool,
    ) -> M5R3P1StageGeneration:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        encoded: dict[str, Any] = tokenizer(rendered, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        input_count = int(input_ids.shape[1])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generation_args: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            generation_args.update(
                {
                    "temperature": config.solver.temperature,
                    "top_p": config.solver.top_p,
                    "top_k": config.solver.top_k,
                    "repetition_penalty": config.solver.repetition_penalty,
                }
            )
        try:
            with torch.inference_mode():
                output_ids = model.generate(**generation_args)
            generated = output_ids[0, input_count:]
            generated_count = int(generated.numel())
            raw_output = cast(str, tokenizer.decode(generated, skip_special_tokens=True))
            return _generation_record(
                context=context,
                stage=stage,
                seed=seed,
                prompt=prompt,
                status="succeeded",
                finish_reason=("length" if generated_count >= max_new_tokens else "stop"),
                raw_output=raw_output,
                input_token_count=input_count,
                generated_token_count=generated_count,
            )
        except RuntimeError:
            torch.cuda.empty_cache()
            return _generation_record(
                context=context,
                stage=stage,
                seed=seed,
                prompt=prompt,
                status="failed",
                finish_reason="error",
                error_code="generation_runtime_error",
            )

    for completed, context in enumerate(shard_contexts, 1):
        index = global_index[context.task.id]
        solver_prompt = build_m5_r3_p2_fallback_solver_prompt(context)
        solver = generate(
            context,
            stage="solver",
            prompt=solver_prompt,
            seed=m5_r3_p1_stage_seed(config.solver.base_seed, index),
            enable_thinking=True,
            max_new_tokens=config.solver.max_new_tokens,
            do_sample=True,
        )
        records.append(solver)
        if (
            solver.status == "succeeded"
            and solver.finish_reason == "stop"
            and solver.raw_output is not None
        ):
            parsed, reason = parse_teacher_output(solver.raw_output)
            if parsed is not None and reason is None:
                try:
                    answer = canonical_json(json.loads(parsed.final_answer))
                except json.JSONDecodeError:
                    answer = ""
                if answer == context.task.expected_answer_json:
                    compressor_prompt = build_m5_r3_p2_isolated_compressor_prompt(
                        context,
                        parsed.reasoning_content,
                        answer,
                    )
                    records.append(
                        generate(
                            context,
                            stage="compressor",
                            prompt=compressor_prompt,
                            seed=m5_r3_p1_stage_seed(config.compressor.base_seed, index),
                            enable_thinking=False,
                            max_new_tokens=config.compressor.max_new_tokens,
                            do_sample=False,
                        )
                    )
        if completed % 5 == 0 or completed == len(shard_contexts):
            print(
                json.dumps(
                    {
                        "completed_tasks": completed,
                        "shard_index": args.shard_index,
                        "shard_tasks": len(shard_contexts),
                        "solver_attempts": sum(item.stage == "solver" for item in records),
                        "compressor_attempts": sum(item.stage == "compressor" for item in records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    artifact = M5R3FormalShardArtifact(
        schema_version="1.0",
        expansion_version=config.expansion_version,
        config_sha256=m5_r3_formal_source_config_sha256(config),
        git_commit=git_commit,
        git_dirty=False,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        physical_gpu_index=args.gpu_index,
        gpu_name=torch.cuda.get_device_name(device),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        task_ids=tuple(item.task.id for item in shard_contexts),
        contexts=shard_contexts,
        generations=tuple(records),
        duration_seconds=time.monotonic() - started,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
    )
    _atomic_json(args.shard_output, artifact.to_dict())
    print(
        json.dumps(
            {
                "status": "generated",
                "shard_index": args.shard_index,
                "task_count": len(shard_contexts),
                "artifact_sha256": _sha256_file(args.shard_output),
            },
            sort_keys=True,
        )
    )
    return 0


def _supervise_shard(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    _, dirty = read_git_identity(project_root)
    if dirty:
        raise M5R3FormalSourceError("M5 R3 formal shard requires a clean Git worktree")
    if args.shard_output is None:
        raise M5R3FormalSourceError("M5 R3 formal shard output is required")
    if args.shard_output.exists():
        raise M5R3FormalSourceError("M5 R3 formal shard output already exists")
    _verify_formal_inputs(args)
    _verify_policy_python(args.policy_python, project_root)
    try:
        validate_gpu_preflight(inspect_gpus((args.gpu_index,)))
    except RuntimeError as exc:
        raise M5R3FormalEnvironmentError(str(exc)) from exc
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    command = _common_command(args, interpreter=Path(sys.executable)) + [
        "--gpu-index",
        str(args.gpu_index),
        "--shard-index",
        str(args.shard_index),
        "--shard-count",
        str(args.shard_count),
        "--shard-output",
        str(args.shard_output),
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
        raise M5R3FormalEnvironmentError("M5 R3 formal shard exceeded its timeout") from exc
    return completed.returncode


def _load_shards(
    paths: tuple[Path, ...],
    *,
    config_sha256: str,
    finalizer_git_commit: str,
    project_root: Path,
    expected_contexts: tuple[M5R3P1TaskContext, ...],
) -> tuple[M5R3FormalShardArtifact, ...]:
    try:
        shards = tuple(
            M5R3FormalShardArtifact.model_validate_json(path.read_text(encoding="utf-8"))
            for path in paths
        )
    except (OSError, ValueError) as exc:
        raise M5R3FormalSourceError("M5 R3 formal shard artifact is invalid") from exc
    if not shards:
        raise M5R3FormalSourceError("M5 R3 formal finalizer requires shard artifacts")
    shard_count = shards[0].shard_count
    generation_git_commit = shards[0].git_commit
    ordered = shards
    expected_ids = tuple(context.task.id for context in expected_contexts)
    merged_ids = tuple(task_id for shard in ordered for task_id in shard.task_ids)
    # Interleaved shards are reconstructed into canonical global order below.
    if (
        len(paths) != shard_count
        or tuple(item.shard_index for item in ordered) != tuple(range(shard_count))
        or any(item.shard_count != shard_count for item in ordered)
        or any(item.config_sha256 != config_sha256 for item in ordered)
        or any(item.git_commit != generation_git_commit for item in ordered)
        or len(set(merged_ids)) != 240
        or set(merged_ids) != set(expected_ids)
    ):
        raise M5R3FormalSourceError("M5 R3 formal shard lineage or coverage differs")
    ancestry = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            generation_git_commit,
            finalizer_git_commit,
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        raise M5R3FormalSourceError(
            "M5 R3 formal shard commit is not an ancestor of the finalizer commit"
        )
    return ordered


def _finalize(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5R3FormalSourceError("M5 R3 formal finalizer requires a clean Git worktree")
    if (
        args.raw_output is None
        or args.selected_output is None
        or args.public_output is None
        or not args.shard_artifacts
    ):
        raise M5R3FormalSourceError("M5 R3 formal finalizer outputs and shards are required")
    if any(path.exists() for path in (args.raw_output, args.selected_output, args.public_output)):
        raise M5R3FormalSourceError("M5 R3 formal finalizer output already exists")
    config = load_m5_r3_formal_source_config(args.config)
    contexts, dev, historical, p0, p0_r1, p1_contexts = _verify_formal_inputs(args)
    config_sha256 = m5_r3_formal_source_config_sha256(config)
    paths = tuple(args.shard_artifacts)
    shards = _load_shards(
        paths,
        config_sha256=config_sha256,
        finalizer_git_commit=git_commit,
        project_root=project_root,
        expected_contexts=contexts,
    )
    context_by_id = {context.task.id: context for shard in shards for context in shard.contexts}
    ordered_contexts = tuple(context_by_id[context.task.id] for context in contexts)
    generations = tuple(generation for shard in shards for generation in shard.generations)
    if sum(item.stage == "solver" for item in generations) != 240:
        raise M5R3FormalSourceError("M5 R3 formal solver coverage differs")

    tokenization = load_m2_tokenization_config(args.tokenization_config)
    tokenizer = TokenizersBackend.from_files(
        args.tokenizer_dir / tokenization.tokenizer.tokenizer_file,
        args.tokenizer_dir / tokenization.tokenizer.tokenizer_config_file,
        tokenization.tokenizer,
    )
    build = build_m5_r3_formal_source(
        ordered_contexts,
        generations,
        config=config,
        dev_tasks=dev,
        historical_tasks=historical,
        p0_tasks=p0,
        p0_r1_tasks=p0_r1,
        p1_tasks=(item.task for item in p1_contexts),
        tokenizer=tokenizer,
    )
    shard_hashes = tuple(_sha256_file(path) for path in paths)
    raw_payload = {
        "schema_version": "1.0",
        "expansion_version": config.expansion_version,
        "config_sha256": config_sha256,
        "parent_content_review_sha256": config.parent_content_review_sha256,
        "shard_artifact_sha256s": shard_hashes,
        "task_set_sha256": build.task_set_sha256,
        "accepted_samples_sha256": build.accepted_samples_sha256,
        "selected_samples_sha256": build.selected_samples_sha256,
        "contexts": [item.to_dict() for item in build.contexts],
        "generations": [item.to_dict() for item in build.generations],
        "samples": [item.to_dict() for item in build.samples],
        "selected_samples": [item.to_dict() for item in build.selected_samples],
        "audits": [item.to_dict() for item in build.audits],
        "stratum_results": [item.to_dict() for item in build.stratum_results],
        "rejection_counts": build.rejection_counts,
        "contamination": build.contamination.to_dict(),
    }
    selected_payload = {
        "schema_version": "1.0",
        "dataset_version": "m5-r3-formal-selected-v1",
        "source_expansion_version": config.expansion_version,
        "config_sha256": config_sha256,
        "task_set_sha256": build.task_set_sha256,
        "selected_samples_sha256": build.selected_samples_sha256,
        "sample_count": len(build.selected_samples),
        "samples": [item.to_dict() for item in build.selected_samples],
    }
    _atomic_json(args.raw_output, raw_payload)
    _atomic_json(args.selected_output, selected_payload)
    passed = (
        len(build.selected_samples) == 160
        and all(item.gate_passed for item in build.stratum_results)
        and build.contamination.status == "pass"
    )
    torch_versions = {item.torch_version for item in shards}
    transformer_versions = {item.transformers_version for item in shards}
    if len(torch_versions) != 1 or len(transformer_versions) != 1:
        raise M5R3FormalSourceError("M5 R3 formal shard software versions differ")
    result = M5R3FormalSourceResult(
        schema_version="1.0",
        status="pass" if passed else "fail",
        expansion_version=config.expansion_version,
        generated_at=datetime.now(UTC),
        config_sha256=config_sha256,
        parent_content_review_sha256=config.parent_content_review_sha256,
        git_commit=shards[0].git_commit,
        finalizer_git_commit=git_commit,
        git_dirty=False,
        solver=config.solver,
        compressor=config.compressor,
        tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        physical_gpu_indices=tuple(item.physical_gpu_index for item in shards),
        gpu_names=tuple(item.gpu_name for item in shards),
        torch_version=next(iter(torch_versions)),
        transformers_version=next(iter(transformer_versions)),
        policy_tokenizers_version="0.21.4",
        shard_count=len(shards),
        shard_artifact_sha256s=shard_hashes,
        input_tasks=240,
        solver_attempts=240,
        compressor_attempts=sum(item.stage == "compressor" for item in generations),
        accepted_samples=len(build.samples),
        rejected_tasks=240 - len(build.samples),
        selected_samples=len(build.selected_samples),
        stratum_results=build.stratum_results,
        rejection_counts=build.rejection_counts,
        contamination=build.contamination,
        task_set_sha256=build.task_set_sha256,
        accepted_samples_sha256=build.accepted_samples_sha256,
        selected_samples_sha256=build.selected_samples_sha256,
        duration_seconds=sum(item.duration_seconds for item in shards),
        peak_allocated_bytes=max(item.peak_allocated_bytes for item in shards),
        peak_reserved_bytes=max(item.peak_reserved_bytes for item in shards),
        raw_artifact_sha256=_sha256_file(args.raw_output),
        selected_artifact_sha256=_sha256_file(args.selected_output),
        formal_source_expansion_complete=passed,
        r3_mixture_authorized=passed,
        r3_training_authorized=False,
        consumes_m6_frozen_results=False,
    )
    _atomic_json(args.public_output, result.to_dict())
    print(result.model_dump_json())
    return 0 if passed else 6


def _common_command(args: argparse.Namespace, *, interpreter: Path) -> list[str]:
    return [
        str(interpreter),
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--content-review-result",
        str(args.content_review_result),
        "--p1-config",
        str(args.p1_config),
        "--reasoning-config",
        str(args.reasoning_config),
        "--tokenization-config",
        str(args.tokenization_config),
        "--p0-config",
        str(args.p0_config),
        "--p0-r1-config",
        str(args.p0_r1_config),
        "--r2-decision",
        str(args.r2_decision),
        "--p0-result",
        str(args.p0_result),
        "--p0-r1-result",
        str(args.p0_r1_result),
        "--historical-pilot-artifact",
        str(args.historical_pilot_artifact),
        "--model-dir",
        str(args.model_dir),
        "--tokenizer-dir",
        str(args.tokenizer_dir),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/m5_r3_formal_source.yaml"),
    )
    parser.add_argument(
        "--content-review-result",
        type=Path,
        default=Path("reports/m5/raw/m5_r3_p2_content_review.json"),
    )
    parser.add_argument(
        "--p1-config",
        type=Path,
        default=Path("configs/data/m5_r3_teacher_source_strategy.yaml"),
    )
    parser.add_argument(
        "--reasoning-config",
        type=Path,
        default=Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml"),
    )
    parser.add_argument(
        "--tokenization-config",
        type=Path,
        default=Path("configs/data/m2_tokenization.yaml"),
    )
    parser.add_argument("--p0-config", type=Path, default=Path("configs/data/m5_r3_p0.yaml"))
    parser.add_argument(
        "--p0-r1-config",
        type=Path,
        default=Path("configs/data/m5_r3_p0_r1.yaml"),
    )
    parser.add_argument(
        "--r2-decision",
        type=Path,
        default=Path("reports/m5/raw/m5_r2_length_diagnostic.json"),
    )
    parser.add_argument(
        "--p0-result",
        type=Path,
        default=Path("reports/m5/raw/m5_r3_p0.json"),
    )
    parser.add_argument(
        "--p0-r1-result",
        type=Path,
        default=Path("reports/m5/raw/m5_r3_p0_r1.json"),
    )
    parser.add_argument("--historical-pilot-artifact", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-output", type=Path)
    parser.add_argument("--shard-artifact", dest="shard_artifacts", action="append", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--selected-output", type=Path)
    parser.add_argument("--public-output", type=Path)
    parser.add_argument("--policy-python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--finalize", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.worker and args.finalize:
            raise M5R3FormalSourceError("M5 R3 formal internal modes are mutually exclusive")
        if args.finalize:
            return _finalize(args)
        if args.gpu_index is None:
            raise M5R3FormalSourceError("M5 R3 formal shard GPU index is required")
        if not 0 <= args.shard_index < args.shard_count <= 8:
            raise M5R3FormalSourceError("M5 R3 formal shard coordinates differ")
        if args.worker:
            return _worker(args)
        return _supervise_shard(args)
    except M5R3FormalEnvironmentError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 3
    except (M5R3FormalSourceError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
