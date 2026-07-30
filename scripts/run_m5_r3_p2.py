#!/usr/bin/env python3
"""Run the parent-bound M5.2-R3 P2 fallback and isolated-compressor pilot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from scripts.run_m5_r3_p1 import (
    _atomic_json,
    _generation_record,
    _preflight_task_sets,
    _sha256_file,
    _verify_frozen_inputs,
    _verify_model_directory,
    _verify_policy_python,
)
from tinyllm.data import TokenizersBackend, load_m2_tokenization_config
from tinyllm.data.m5_r3_p1 import build_m5_r3_p1_dataset, m5_r3_p1_stage_seed
from tinyllm.data.m5_r3_p1_schema import (
    M5R3P1Result,
    M5R3P1StageGeneration,
    M5R3P1TaskContext,
)
from tinyllm.data.m5_r3_p2 import (
    M5R3P2Error,
    build_m5_r3_p2_fallback_solver_prompt,
    build_m5_r3_p2_isolated_compressor_prompt,
    classify_m5_r3_p2_solver,
    load_m5_r3_p2_config,
    m5_r3_p2_config_sha256,
    select_m5_r3_p2_generations,
)
from tinyllm.data.m5_r3_p2_schema import (
    M5R3P2GenerationDelta,
    M5R3P2Result,
)
from tinyllm.data.m5_r3_source_strategy import (
    load_m5_r3_teacher_source_strategy_config,
    m5_r3_teacher_source_strategy_config_sha256,
)
from tinyllm.data.reasoning import parse_teacher_output
from tinyllm.data.reasoning_schema import canonical_json
from tinyllm.lineage import read_git_identity
from tinyllm.training.smoke_preflight import inspect_gpus, validate_gpu_preflight


class M5R3P2EnvironmentError(RuntimeError):
    """Raised when P2 cannot safely use its environment or pinned model."""


def _parent_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Expose P1 argument names to the frozen parent preflight implementation."""

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


def _load_parent(
    args: argparse.Namespace,
) -> tuple[
    M5R3P1Result,
    tuple[M5R3P1TaskContext, ...],
    tuple[M5R3P1StageGeneration, ...],
]:
    """Verify and reconstruct the real rejected P1 public/private parent pair."""

    config = load_m5_r3_p2_config(args.config)
    if (
        _sha256_file(args.parent_p1_result) != config.parent_p1_result_sha256
        or _sha256_file(args.parent_p1_generation_artifact)
        != config.parent_p1_generation_artifact_sha256
    ):
        raise M5R3P2Error("M5 R3 P2 parent P1 SHA256 differs")
    try:
        result = M5R3P1Result.model_validate_json(args.parent_p1_result.read_text(encoding="utf-8"))
        payload = cast(
            dict[str, object],
            json.loads(args.parent_p1_generation_artifact.read_text(encoding="utf-8")),
        )
        contexts = tuple(
            M5R3P1TaskContext.model_validate_json(json.dumps(value, sort_keys=True))
            for value in cast(list[object], payload["contexts"])
        )
        generations = tuple(
            M5R3P1StageGeneration.model_validate_json(json.dumps(value, sort_keys=True))
            for value in cast(list[object], payload["generations"])
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise M5R3P2Error("M5 R3 P2 parent P1 artifact cannot be reconstructed") from exc
    p1_config = load_m5_r3_teacher_source_strategy_config(args.p1_config)
    expected_contexts, _dev, _historical, _p0, _p0_r1 = _preflight_task_sets(
        _parent_namespace(args)
    )
    if (
        result.status != "fail"
        or result.accepted_samples != 11
        or result.formal_source_expansion_authorized
        or result.task_set_sha256 != config.task_set_sha256
        or payload.get("schema_version") != "1.0"
        or payload.get("pilot_version") != p1_config.pilot.pilot_version
        or payload.get("config_sha256") != m5_r3_teacher_source_strategy_config_sha256(p1_config)
        or payload.get("git_commit") != result.git_commit
        or payload.get("git_dirty") is not False
        or contexts != expected_contexts
        or sum(item.stage == "solver" for item in generations) != 40
    ):
        raise M5R3P2Error("M5 R3 P2 parent P1 identity differs")
    return result, contexts, generations


def _verify_inputs(args: argparse.Namespace) -> None:
    """Fail closed on every P2, P1, model, and parent evidence identity."""

    config = load_m5_r3_p2_config(args.config)
    _verify_frozen_inputs(_parent_namespace(args))
    _load_parent(args)
    if config.fallback_solver.revision != config.isolated_compressor.revision:
        raise M5R3P2Error("M5 R3 P2 stage revisions differ")


def _worker(args: argparse.Namespace) -> int:
    import tokenizers as teacher_tokenizers  # type: ignore[import-untyped]
    import torch
    import transformers  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.generation_output is None:
        raise M5R3P2Error("M5 R3 P2 worker requires a generation output")
    project_root = Path(__file__).resolve().parents[1]
    config = load_m5_r3_p2_config(args.config)
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5R3P2Error("M5 R3 P2 requires a clean Git worktree")
    _verify_inputs(args)
    _verify_model_directory(args.model_dir, config.fallback_solver.revision)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M5R3P2EnvironmentError("M5 R3 P2 worker requires one visible CUDA device")
    _parent_result, contexts, parent_generations = _load_parent(args)
    parent_solvers = {item.task_id: item for item in parent_generations if item.stage == "solver"}
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
    fallbacks: list[M5R3P1StageGeneration] = []
    compressors: list[M5R3P1StageGeneration] = []

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
                    "temperature": config.fallback_solver.temperature,
                    "top_p": config.fallback_solver.top_p,
                    "top_k": config.fallback_solver.top_k,
                    "repetition_penalty": config.fallback_solver.repetition_penalty,
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

    for index, context in enumerate(contexts):
        parent_solver = parent_solvers[context.task.id]
        selected_solver = parent_solver
        if classify_m5_r3_p2_solver(context, parent_solver) is not None:
            fallback_prompt = build_m5_r3_p2_fallback_solver_prompt(context)
            selected_solver = generate(
                context,
                stage="solver",
                prompt=fallback_prompt,
                seed=m5_r3_p1_stage_seed(config.fallback_solver.base_seed, index),
                enable_thinking=True,
                max_new_tokens=config.fallback_solver.max_new_tokens,
                do_sample=True,
            )
            fallbacks.append(selected_solver)
        if classify_m5_r3_p2_solver(context, selected_solver) is None:
            assert selected_solver.raw_output is not None
            parsed, reason = parse_teacher_output(selected_solver.raw_output)
            if parsed is None or reason is not None:
                raise M5R3P2Error("M5 R3 P2 accepted solver cannot be parsed")
            verified_answer = canonical_json(json.loads(parsed.final_answer))
            compressor_prompt = build_m5_r3_p2_isolated_compressor_prompt(
                context,
                parsed.reasoning_content,
                verified_answer,
            )
            compressors.append(
                generate(
                    context,
                    stage="compressor",
                    prompt=compressor_prompt,
                    seed=m5_r3_p1_stage_seed(
                        config.isolated_compressor.base_seed,
                        index,
                    ),
                    enable_thinking=False,
                    max_new_tokens=config.isolated_compressor.max_new_tokens,
                    do_sample=False,
                )
            )
        if (index + 1) % 5 == 0:
            print(
                json.dumps(
                    {
                        "completed_tasks": index + 1,
                        "fallback_solver_attempts": len(fallbacks),
                        "isolated_compressor_attempts": len(compressors),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    delta = M5R3P2GenerationDelta(
        pilot_version=config.pilot_version,
        config_sha256=m5_r3_p2_config_sha256(config),
        parent_p1_generation_artifact_sha256=(config.parent_p1_generation_artifact_sha256),
        task_set_sha256=config.task_set_sha256,
        git_commit=git_commit,
        git_dirty=False,
        physical_gpu_index=args.gpu_index,
        gpu_name=torch.cuda.get_device_name(device),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        teacher_tokenizers_version=teacher_tokenizers.__version__,
        duration_seconds=time.monotonic() - started,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        fallback_solvers=tuple(fallbacks),
        isolated_compressors=tuple(compressors),
    )
    _atomic_json(args.generation_output, delta.to_dict())
    print(
        json.dumps(
            {
                "status": "generated",
                "fallback_solver_attempts": len(fallbacks),
                "isolated_compressor_attempts": len(compressors),
                "generation_delta_sha256": _sha256_file(args.generation_output),
            },
            sort_keys=True,
        )
    )
    return 0


def _finalize(args: argparse.Namespace) -> int:
    if args.generation_output is None:
        raise M5R3P2Error("M5 R3 P2 finalizer requires a generation output")
    project_root = Path(__file__).resolve().parents[1]
    config = load_m5_r3_p2_config(args.config)
    p1_config = load_m5_r3_teacher_source_strategy_config(args.p1_config)
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5R3P2Error("M5 R3 P2 requires a clean Git worktree")
    _verify_inputs(args)
    _parent_result, contexts, parent_generations = _load_parent(args)
    try:
        delta = M5R3P2GenerationDelta.model_validate_json(
            args.generation_output.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise M5R3P2Error("M5 R3 P2 generation delta cannot be reconstructed") from exc
    if (
        delta.pilot_version != config.pilot_version
        or delta.config_sha256 != m5_r3_p2_config_sha256(config)
        or delta.git_commit != git_commit
        or delta.physical_gpu_index != args.gpu_index
    ):
        raise M5R3P2Error("M5 R3 P2 generation delta identity differs")
    selection = select_m5_r3_p2_generations(
        contexts,
        parent_generations,
        delta.fallback_solvers,
        delta.isolated_compressors,
        p1_config=p1_config,
        p2_config=config,
    )
    _contexts, dev, historical, p0, p0_r1 = _preflight_task_sets(_parent_namespace(args))
    tokenization = load_m2_tokenization_config(args.tokenization_config)
    tokenizer = TokenizersBackend.from_files(
        args.tokenizer_dir / tokenization.tokenizer.tokenizer_file,
        args.tokenizer_dir / tokenization.tokenizer.tokenizer_config_file,
        tokenization.tokenizer,
    )
    build = build_m5_r3_p1_dataset(
        contexts,
        selection.generations,
        config=p1_config,
        dev_tasks=dev,
        historical_tasks=historical,
        p0_tasks=p0,
        p0_r1_tasks=p0_r1,
        tokenizer=tokenizer,
        expected_stage_seeds=selection.expected_stage_seeds,
        expected_stage_prompt_sha256=selection.expected_stage_prompt_sha256,
        compressor_prompt_builder=build_m5_r3_p2_isolated_compressor_prompt,
    )
    raw_payload = {
        "schema_version": "1.0",
        "pilot_version": config.pilot_version,
        "config_sha256": m5_r3_p2_config_sha256(config),
        "parent_p1_result_sha256": config.parent_p1_result_sha256,
        "parent_p1_generation_artifact_sha256": (config.parent_p1_generation_artifact_sha256),
        "generation_delta_sha256": _sha256_file(args.generation_output),
        "task_set_sha256": build.task_set_sha256,
        "samples_sha256": build.samples_sha256,
        "selected_generations": [item.to_dict() for item in build.generations],
        "samples": [item.to_dict() for item in build.samples],
        "audits": [item.to_dict() for item in build.audits],
        "contamination": build.contamination.to_dict(),
        "family_results": [item.to_dict() for item in build.family_results],
        "control": build.control.to_dict(),
        "rejection_counts": build.rejection_counts,
        "fallback_trigger_counts": selection.fallback_trigger_counts,
    }
    _atomic_json(args.raw_output, raw_payload)
    passed = (
        all(item.gate_passed for item in build.family_results)
        and build.control.status == "pass"
        and build.contamination.status == "pass"
    )
    result = M5R3P2Result(
        status="pass" if passed else "fail",
        pilot_version=config.pilot_version,
        generated_at=datetime.now(UTC),
        config_sha256=m5_r3_p2_config_sha256(config),
        parent_p1_result_sha256=config.parent_p1_result_sha256,
        parent_p1_generation_artifact_sha256=(config.parent_p1_generation_artifact_sha256),
        git_commit=git_commit,
        git_dirty=False,
        fallback_solver=config.fallback_solver,
        isolated_compressor=config.isolated_compressor,
        physical_gpu_index=delta.physical_gpu_index,
        gpu_name=delta.gpu_name,
        torch_version=delta.torch_version,
        transformers_version=delta.transformers_version,
        teacher_tokenizers_version=delta.teacher_tokenizers_version,
        policy_tokenizers_version="0.21.4",
        input_tasks=40,
        parent_solver_attempts=40,
        fallback_solver_attempts=len(delta.fallback_solvers),
        isolated_compressor_attempts=len(delta.isolated_compressors),
        accepted_samples=len(build.samples),
        rejected_tasks=40 - len(build.samples),
        family_results=build.family_results,
        rejection_counts=build.rejection_counts,
        fallback_trigger_counts=selection.fallback_trigger_counts,
        control=build.control,
        contamination=build.contamination,
        task_set_sha256=cast(Any, build.task_set_sha256),
        samples_sha256=build.samples_sha256,
        duration_seconds=delta.duration_seconds,
        peak_allocated_bytes=delta.peak_allocated_bytes,
        peak_reserved_bytes=delta.peak_reserved_bytes,
        generation_delta_sha256=_sha256_file(args.generation_output),
        raw_artifact_sha256=_sha256_file(args.raw_output),
        formal_source_expansion_authorized=passed,
        r3_mixture_authorized=False,
        r3_training_authorized=False,
    )
    _atomic_json(args.public_output, result.to_dict())
    print(result.model_dump_json())
    return 0 if result.status == "pass" else 6


def _subprocess_command(
    args: argparse.Namespace,
    *,
    interpreter: Path,
    mode: str,
    generation_output: Path,
) -> list[str]:
    command = [
        str(interpreter),
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
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
        "--parent-p1-result",
        str(args.parent_p1_result),
        "--parent-p1-generation-artifact",
        str(args.parent_p1_generation_artifact),
        "--model-dir",
        str(args.model_dir),
        "--tokenizer-dir",
        str(args.tokenizer_dir),
        "--gpu-index",
        str(args.gpu_index),
        "--generation-output",
        str(generation_output),
        "--raw-output",
        str(args.raw_output),
        "--public-output",
        str(args.public_output),
        mode,
    ]
    return command


def _supervise(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    _, dirty = read_git_identity(project_root)
    if dirty:
        raise M5R3P2Error("M5 R3 P2 requires a clean Git worktree")
    generation_output = (
        args.generation_output
        if args.generation_output is not None
        else args.raw_output.with_name(f"{args.raw_output.stem}.generations.json")
    )
    if generation_output.exists() or args.raw_output.exists() or args.public_output.exists():
        raise M5R3P2Error("M5 R3 P2 output already exists")
    _verify_inputs(args)
    try:
        _verify_policy_python(args.policy_python, project_root)
        validate_gpu_preflight(inspect_gpus((args.gpu_index,)))
    except RuntimeError as exc:
        raise M5R3P2EnvironmentError(str(exc)) from exc
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    try:
        completed = subprocess.run(
            _subprocess_command(
                args,
                interpreter=Path(sys.executable),
                mode="--worker",
                generation_output=generation_output,
            ),
            cwd=project_root,
            env=environment,
            check=False,
            text=True,
            timeout=args.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise M5R3P2EnvironmentError("M5 R3 P2 exceeded its timeout") from exc
    if completed.returncode != 0:
        return completed.returncode
    finalizer_environment = os.environ.copy()
    finalizer_environment.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        finalized = subprocess.run(
            _subprocess_command(
                args,
                interpreter=args.policy_python,
                mode="--finalize",
                generation_output=generation_output,
            ),
            cwd=project_root,
            env=finalizer_environment,
            check=False,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise M5R3P2EnvironmentError("M5 R3 P2 finalizer exceeded its timeout") from exc
    return finalized.returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the P2 offline-only supervised interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/m5_r3_p2.yaml"))
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
    parser.add_argument(
        "--parent-p1-result",
        type=Path,
        default=Path("reports/m5/raw/m5_r3_p1.json"),
    )
    parser.add_argument("--parent-p1-generation-artifact", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--generation-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--finalize", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Run P2 with stable configuration, environment, and Gate exit codes."""

    args = build_parser().parse_args()
    try:
        if args.worker and args.finalize:
            raise M5R3P2Error("M5 R3 P2 internal modes are mutually exclusive")
        if args.worker:
            return _worker(args)
        if args.finalize:
            return _finalize(args)
        return _supervise(args)
    except M5R3P2EnvironmentError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 3
    except (M5R3P2Error, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
