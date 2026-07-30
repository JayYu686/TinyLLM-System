#!/usr/bin/env python3
"""Run the bounded M5.2-R3 P1 solve/compress Teacher Pilot on one RTX 3090."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from tinyllm.data import (
    ReasoningTask,
    TokenizersBackend,
    generate_reasoning_dev_tasks,
    load_m2_tokenization_config,
    load_m5_reasoning_data_config,
    load_verified_reasoning_pilot,
)
from tinyllm.data.m5_r3_p0 import (
    generate_m5_r3_p0_tasks,
    load_m5_r3_p0_config,
    m5_r3_p0_config_sha256,
)
from tinyllm.data.m5_r3_p1 import (
    M5R3P1Error,
    build_m5_r3_p1_compressor_prompt,
    build_m5_r3_p1_dataset,
    check_m5_r3_p1_contamination,
    generate_m5_r3_p1_contexts,
    m5_r3_p1_stage_seed,
)
from tinyllm.data.m5_r3_p1_schema import (
    M5R3P1Result,
    M5R3P1StageGeneration,
    M5R3P1TaskContext,
)
from tinyllm.data.m5_r3_source_strategy import (
    load_m5_r3_teacher_source_strategy_config,
    m5_r3_teacher_source_strategy_config_sha256,
    review_m5_r3_teacher_source_strategy,
)
from tinyllm.data.reasoning import parse_teacher_output
from tinyllm.data.reasoning_schema import canonical_json
from tinyllm.lineage import read_git_identity
from tinyllm.training.smoke_preflight import inspect_gpus, validate_gpu_preflight


class M5R3P1EnvironmentError(RuntimeError):
    """Raised when P1 cannot safely use the pinned model or selected GPU."""


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise M5R3P1Error("M5 R3 P1 frozen input cannot be read") from exc
    return digest.hexdigest()


def _verify_model_directory(model_dir: Path, expected_revision: str) -> None:
    if model_dir.name != expected_revision or not model_dir.is_dir() or model_dir.is_symlink():
        raise M5R3P1EnvironmentError("M5 R3 P1 Teacher directory must be the pinned snapshot")
    required = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if any(not (model_dir / name).is_file() for name in required):
        raise M5R3P1EnvironmentError("M5 R3 P1 Teacher snapshot is incomplete")
    try:
        decoded = cast(
            dict[str, object],
            json.loads((model_dir / "config.json").read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise M5R3P1EnvironmentError("M5 R3 P1 Teacher config cannot be read") from exc
    expected = {
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "torch_dtype": "bfloat16",
    }
    if {key: decoded.get(key) for key in expected} != expected:
        raise M5R3P1EnvironmentError("M5 R3 P1 Teacher is not pinned Qwen3-8B GQA")


def _verify_frozen_inputs(args: argparse.Namespace) -> None:
    strategy = load_m5_r3_teacher_source_strategy_config(args.config)
    review_m5_r3_teacher_source_strategy(
        config_path=args.config,
        r2_decision_path=args.r2_decision,
        p0_result_path=args.p0_result,
        p0_r1_result_path=args.p0_r1_result,
    )
    p0 = load_m5_r3_p0_config(args.p0_config)
    p0_r1 = load_m5_r3_p0_config(args.p0_r1_config)
    if (
        m5_r3_p0_config_sha256(p0)
        != "ffd32c3d3ac9e7a235243f643be9018c1554909e2214f23c281039b83e5a9219"
        or m5_r3_p0_config_sha256(p0_r1)
        != "6f890910fba0120003133217d788ad4f30e2f0b932f5fd28a47f98bf5513880a"
    ):
        raise M5R3P1Error("M5 R3 P1 parent task config identity differs")
    expected_files = (
        (args.historical_pilot_artifact, p0_r1.historical_pilot_raw_sha256),
        (args.reasoning_config, p0_r1.reasoning_config_sha256),
        (args.tokenization_config, p0_r1.tokenization_config_sha256),
    )
    if any(_sha256_file(path) != expected for path, expected in expected_files):
        raise M5R3P1Error("M5 R3 P1 frozen input SHA256 differs")
    if strategy.pilot.solver.revision != strategy.pilot.compressor.revision:
        raise M5R3P1Error("M5 R3 P1 stage model revisions differ")


def _load_historical_tasks(args: argparse.Namespace) -> tuple[ReasoningTask, ...]:
    load_verified_reasoning_pilot(
        raw_artifact=args.historical_pilot_artifact,
        reasoning_config=args.reasoning_config,
    )
    try:
        payload = cast(
            dict[str, object],
            json.loads(args.historical_pilot_artifact.read_text(encoding="utf-8")),
        )
        return tuple(
            ReasoningTask.model_validate(value) for value in cast(list[object], payload["tasks"])
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise M5R3P1Error("M5 R3 P1 historical tasks cannot be reconstructed") from exc


def _preflight_task_sets(
    args: argparse.Namespace,
) -> tuple[
    tuple[M5R3P1TaskContext, ...],
    tuple[ReasoningTask, ...],
    tuple[ReasoningTask, ...],
    tuple[ReasoningTask, ...],
    tuple[ReasoningTask, ...],
]:
    strategy = load_m5_r3_teacher_source_strategy_config(args.config)
    reasoning = load_m5_reasoning_data_config(args.reasoning_config)
    contexts = generate_m5_r3_p1_contexts(strategy)
    dev = generate_reasoning_dev_tasks(reasoning)
    historical = _load_historical_tasks(args)
    p0 = generate_m5_r3_p0_tasks(load_m5_r3_p0_config(args.p0_config))
    p0_r1 = generate_m5_r3_p0_tasks(load_m5_r3_p0_config(args.p0_r1_config))
    contamination = check_m5_r3_p1_contamination(
        contexts,
        dev_tasks=dev,
        historical_tasks=historical,
        p0_tasks=p0,
        p0_r1_tasks=p0_r1,
    )
    if contamination.status != "pass":
        raise M5R3P1Error("M5 R3 P1 contamination preflight failed")
    return contexts, dev, historical, p0, p0_r1


def _generation_record(
    *,
    context: M5R3P1TaskContext,
    stage: str,
    seed: int,
    prompt: str,
    status: str,
    finish_reason: str,
    raw_output: str | None = None,
    input_token_count: int = 0,
    generated_token_count: int = 0,
    error_code: str | None = None,
) -> M5R3P1StageGeneration:
    return M5R3P1StageGeneration(
        generation_id=f"{context.task.id}:{stage}",
        task_id=context.task.id,
        stage=cast(Any, stage),
        seed=seed,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        status=cast(Any, status),
        finish_reason=cast(Any, finish_reason),
        raw_output=raw_output,
        raw_output_sha256=(
            hashlib.sha256(raw_output.encode()).hexdigest() if raw_output is not None else None
        ),
        input_token_count=input_token_count,
        generated_token_count=generated_token_count,
        error_code=error_code,
    )


def _worker(args: argparse.Namespace) -> int:
    import tokenizers as teacher_tokenizers  # type: ignore[import-untyped]
    import torch
    import transformers  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.generation_output is None:
        raise M5R3P1Error("M5 R3 P1 worker requires a generation output")
    project_root = Path(__file__).resolve().parents[1]
    config = load_m5_r3_teacher_source_strategy_config(args.config)
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5R3P1Error("M5 R3 P1 requires a clean Git worktree")
    _verify_frozen_inputs(args)
    _verify_model_directory(args.model_dir, config.pilot.solver.revision)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M5R3P1EnvironmentError("M5 R3 P1 worker requires one visible CUDA device")
    contexts, _dev, _historical, _p0, _p0_r1 = _preflight_task_sets(args)
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
                    "temperature": config.pilot.solver.temperature,
                    "top_p": config.pilot.solver.top_p,
                    "top_k": config.pilot.solver.top_k,
                    "repetition_penalty": config.pilot.solver.repetition_penalty,
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
        solver = generate(
            context,
            stage="solver",
            prompt=context.task.prompt,
            seed=m5_r3_p1_stage_seed(config.pilot.solver.base_seed, index),
            enable_thinking=True,
            max_new_tokens=config.pilot.solver.max_new_tokens,
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
                    compressor_prompt = build_m5_r3_p1_compressor_prompt(
                        context,
                        solver_reasoning=parsed.reasoning_content,
                        verified_final_answer=answer,
                    )
                    records.append(
                        generate(
                            context,
                            stage="compressor",
                            prompt=compressor_prompt,
                            seed=m5_r3_p1_stage_seed(
                                config.pilot.compressor.base_seed,
                                index,
                            ),
                            enable_thinking=False,
                            max_new_tokens=config.pilot.compressor.max_new_tokens,
                            do_sample=False,
                        )
                    )
        if (index + 1) % 5 == 0:
            print(
                json.dumps(
                    {
                        "completed_tasks": index + 1,
                        "solver_attempts": sum(item.stage == "solver" for item in records),
                        "compressor_attempts": sum(item.stage == "compressor" for item in records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    payload = {
        "schema_version": "1.0",
        "pilot_version": config.pilot.pilot_version,
        "config_sha256": m5_r3_teacher_source_strategy_config_sha256(config),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "physical_gpu_index": args.gpu_index,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "teacher_tokenizers_version": teacher_tokenizers.__version__,
        "duration_seconds": time.monotonic() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "contexts": [item.to_dict() for item in contexts],
        "generations": [item.to_dict() for item in records],
    }
    _atomic_json(args.generation_output, payload)
    print(
        json.dumps(
            {
                "status": "generated",
                "solver_attempts": sum(item.stage == "solver" for item in records),
                "compressor_attempts": sum(item.stage == "compressor" for item in records),
                "generation_artifact_sha256": _sha256_file(args.generation_output),
            },
            sort_keys=True,
        )
    )
    return 0


def _finalize(args: argparse.Namespace) -> int:
    if args.generation_output is None:
        raise M5R3P1Error("M5 R3 P1 finalizer requires a generation output")
    project_root = Path(__file__).resolve().parents[1]
    config = load_m5_r3_teacher_source_strategy_config(args.config)
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5R3P1Error("M5 R3 P1 requires a clean Git worktree")
    _verify_frozen_inputs(args)
    expected_contexts, dev, historical, p0, p0_r1 = _preflight_task_sets(args)
    try:
        payload = cast(
            dict[str, object],
            json.loads(args.generation_output.read_text(encoding="utf-8")),
        )
        contexts = tuple(
            M5R3P1TaskContext.model_validate(value)
            for value in cast(list[object], payload["contexts"])
        )
        generations = tuple(
            M5R3P1StageGeneration.model_validate(value)
            for value in cast(list[object], payload["generations"])
        )
        physical_gpu_index = int(cast(int, payload["physical_gpu_index"]))
        gpu_name = str(payload["gpu_name"])
        torch_version = str(payload["torch_version"])
        transformers_version = str(payload["transformers_version"])
        teacher_tokenizers_version = str(payload["teacher_tokenizers_version"])
        duration_seconds = float(cast(float, payload["duration_seconds"]))
        peak_allocated_bytes = int(cast(int, payload["peak_allocated_bytes"]))
        peak_reserved_bytes = int(cast(int, payload["peak_reserved_bytes"]))
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise M5R3P1Error("M5 R3 P1 generation artifact cannot be reconstructed") from exc
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("pilot_version") != config.pilot.pilot_version
        or payload.get("config_sha256") != m5_r3_teacher_source_strategy_config_sha256(config)
        or payload.get("git_commit") != git_commit
        or payload.get("git_dirty") is not False
        or physical_gpu_index != args.gpu_index
        or contexts != expected_contexts
        or sum(item.stage == "solver" for item in generations) != 40
    ):
        raise M5R3P1Error("M5 R3 P1 generation artifact identity differs")
    tokenization = load_m2_tokenization_config(args.tokenization_config)
    tokenizer = TokenizersBackend.from_files(
        args.tokenizer_dir / tokenization.tokenizer.tokenizer_file,
        args.tokenizer_dir / tokenization.tokenizer.tokenizer_config_file,
        tokenization.tokenizer,
    )
    build = build_m5_r3_p1_dataset(
        contexts,
        generations,
        config=config,
        dev_tasks=dev,
        historical_tasks=historical,
        p0_tasks=p0,
        p0_r1_tasks=p0_r1,
        tokenizer=tokenizer,
    )
    raw_payload = {
        "schema_version": "1.0",
        "pilot_version": config.pilot.pilot_version,
        "config_sha256": m5_r3_teacher_source_strategy_config_sha256(config),
        "generation_artifact_sha256": _sha256_file(args.generation_output),
        "task_set_sha256": build.task_set_sha256,
        "samples_sha256": build.samples_sha256,
        "contexts": [item.to_dict() for item in build.contexts],
        "generations": [item.to_dict() for item in build.generations],
        "samples": [item.to_dict() for item in build.samples],
        "audits": [item.to_dict() for item in build.audits],
        "contamination": build.contamination.to_dict(),
        "family_results": [item.to_dict() for item in build.family_results],
        "control": build.control.to_dict(),
        "rejection_counts": build.rejection_counts,
    }
    _atomic_json(args.raw_output, raw_payload)
    passed = (
        all(item.gate_passed for item in build.family_results)
        and build.control.status == "pass"
        and build.contamination.status == "pass"
    )
    result = M5R3P1Result(
        status="pass" if passed else "fail",
        pilot_version=config.pilot.pilot_version,
        generated_at=datetime.now(UTC),
        config_sha256=m5_r3_teacher_source_strategy_config_sha256(config),
        git_commit=git_commit,
        git_dirty=git_dirty,
        solver=config.pilot.solver,
        compressor=config.pilot.compressor,
        tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        physical_gpu_index=physical_gpu_index,
        gpu_name=gpu_name,
        torch_version=torch_version,
        transformers_version=transformers_version,
        teacher_tokenizers_version=teacher_tokenizers_version,
        policy_tokenizers_version="0.21.4",
        input_tasks=40,
        solver_attempts=sum(item.stage == "solver" for item in generations),
        compressor_attempts=sum(item.stage == "compressor" for item in generations),
        accepted_samples=len(build.samples),
        rejected_tasks=40 - len(build.samples),
        family_results=build.family_results,
        rejection_counts=build.rejection_counts,
        control=build.control,
        contamination=build.contamination,
        task_set_sha256=build.task_set_sha256,
        samples_sha256=build.samples_sha256,
        duration_seconds=duration_seconds,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        raw_artifact_sha256=_sha256_file(args.raw_output),
        formal_source_expansion_authorized=passed,
    )
    _atomic_json(args.public_output, result.to_dict())
    print(result.model_dump_json())
    return 0 if result.status == "pass" else 6


def _verify_policy_python(policy_python: Path, project_root: Path) -> None:
    if not policy_python.is_file() or not os.access(policy_python, os.X_OK):
        raise M5R3P1EnvironmentError("M5 R3 P1 policy Python is unavailable")
    try:
        completed = subprocess.run(
            [
                str(policy_python),
                "-c",
                (
                    "import tokenizers; "
                    "assert tokenizers.__version__ == '0.21.4', tokenizers.__version__; "
                    "import tinyllm.data"
                ),
            ],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise M5R3P1EnvironmentError("M5 R3 P1 policy Python preflight failed") from exc
    if completed.returncode != 0:
        raise M5R3P1EnvironmentError("M5 R3 P1 policy Python must provide tokenizers 0.21.4")


def _subprocess_command(
    args: argparse.Namespace,
    *,
    interpreter: Path,
    mode: str,
    generation_output: Path,
) -> list[str]:
    return [
        str(interpreter),
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
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


def _supervise(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    _, dirty = read_git_identity(project_root)
    if dirty:
        raise M5R3P1Error("M5 R3 P1 requires a clean Git worktree")
    generation_output = (
        args.generation_output
        if args.generation_output is not None
        else args.raw_output.with_name(f"{args.raw_output.stem}.generations.json")
    )
    if generation_output.exists() or args.raw_output.exists() or args.public_output.exists():
        raise M5R3P1Error("M5 R3 P1 output already exists")
    _verify_frozen_inputs(args)
    _verify_policy_python(args.policy_python, project_root)
    try:
        validate_gpu_preflight(inspect_gpus((args.gpu_index,)))
    except RuntimeError as exc:
        raise M5R3P1EnvironmentError(str(exc)) from exc
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
        raise M5R3P1EnvironmentError("M5 R3 P1 exceeded its timeout") from exc
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
        raise M5R3P1EnvironmentError("M5 R3 P1 finalizer exceeded its timeout") from exc
    return finalized.returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the offline-only P1 interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
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
    """Run P1 or one isolated internal stage with stable exit codes."""

    args = build_parser().parse_args()
    try:
        if args.worker and args.finalize:
            raise M5R3P1Error("M5 R3 P1 internal modes are mutually exclusive")
        if args.worker:
            return _worker(args)
        if args.finalize:
            return _finalize(args)
        return _supervise(args)
    except M5R3P1EnvironmentError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 3
    except (M5R3P1Error, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
