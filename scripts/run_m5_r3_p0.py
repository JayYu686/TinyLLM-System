#!/usr/bin/env python3
"""Run the bounded 40-task M5.2-R3-P0 Teacher experiment on one idle RTX 3090."""

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
    TeacherGenerationRecord,
    TokenizersBackend,
    generate_reasoning_dev_tasks,
    load_m2_tokenization_config,
    load_m5_reasoning_data_config,
    load_verified_reasoning_pilot,
)
from tinyllm.data.m5_r3_p0 import (
    M5R3P0Error,
    build_m5_r3_p0_dataset,
    check_m5_r3_p0_contamination,
    generate_m5_r3_p0_tasks,
    load_m5_r3_p0_config,
    m5_r3_p0_config_sha256,
    m5_r3_p0_generation_seed,
)
from tinyllm.data.m5_r3_p0_schema import M5R3P0Result
from tinyllm.lineage import read_git_identity
from tinyllm.training.smoke_preflight import inspect_gpus, validate_gpu_preflight


class M5R3P0EnvironmentError(RuntimeError):
    """Raised when a GPU or pinned local model cannot safely run P0."""


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
        raise M5R3P0Error("M5 R3 P0 frozen input cannot be read") from exc
    return digest.hexdigest()


def _verify_model_directory(model_dir: Path, expected_revision: str) -> None:
    if model_dir.name != expected_revision or not model_dir.is_dir() or model_dir.is_symlink():
        raise M5R3P0EnvironmentError(
            "M5 R3 P0 Teacher directory must be the pinned revision snapshot"
        )
    required = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if any(not (model_dir / name).is_file() for name in required):
        raise M5R3P0EnvironmentError("M5 R3 P0 Teacher snapshot is incomplete")
    try:
        decoded = cast(
            dict[str, object],
            json.loads((model_dir / "config.json").read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise M5R3P0EnvironmentError("M5 R3 P0 Teacher config cannot be read") from exc
    expected = {
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "torch_dtype": "bfloat16",
    }
    if {key: decoded.get(key) for key in expected} != expected:
        raise M5R3P0EnvironmentError("M5 R3 P0 Teacher is not the pinned Qwen3-8B GQA model")


def _verify_frozen_inputs(args: argparse.Namespace) -> None:
    config = load_m5_r3_p0_config(args.config)
    expected = (
        (args.source_audit_config, config.parent_source_audit_config_sha256),
        (args.source_audit_result, config.parent_source_audit_result_sha256),
        (args.historical_pilot_artifact, config.historical_pilot_raw_sha256),
        (args.reasoning_config, config.reasoning_config_sha256),
        (args.tokenization_config, config.tokenization_config_sha256),
    )
    for path, sha256 in expected:
        if _sha256_file(path) != sha256:
            raise M5R3P0Error("M5 R3 P0 frozen input SHA256 differs")


def _load_historical_tasks(args: argparse.Namespace) -> tuple[ReasoningTask, ...]:
    load_verified_reasoning_pilot(
        raw_artifact=args.historical_pilot_artifact,
        reasoning_config=args.reasoning_config,
    )
    try:
        historical_payload = cast(
            dict[str, object],
            json.loads(args.historical_pilot_artifact.read_text(encoding="utf-8")),
        )
        return tuple(
            ReasoningTask.model_validate(value)
            for value in cast(list[object], historical_payload["tasks"])
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise M5R3P0Error("M5 R3 P0 historical tasks cannot be reconstructed") from exc


def _worker(args: argparse.Namespace) -> int:
    import tokenizers as teacher_tokenizers  # type: ignore[import-untyped]
    import torch
    import transformers  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    project_root = Path(__file__).resolve().parents[1]
    if args.generation_output is None:
        raise M5R3P0Error("M5 R3 P0 worker requires a generation output")
    config = load_m5_r3_p0_config(args.config)
    reasoning_config = load_m5_reasoning_data_config(args.reasoning_config)
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5R3P0Error("M5 R3 P0 requires a clean Git worktree")
    _verify_frozen_inputs(args)
    _verify_model_directory(args.model_dir, config.teacher.revision)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M5R3P0EnvironmentError("M5 R3 P0 worker requires one visible CUDA device")
    historical_tasks = _load_historical_tasks(args)
    tasks = generate_m5_r3_p0_tasks(config)
    dev_tasks = generate_reasoning_dev_tasks(reasoning_config)
    contamination = check_m5_r3_p0_contamination(
        tasks,
        dev_tasks=dev_tasks,
        historical_tasks=historical_tasks,
    )
    if contamination.status != "pass":
        raise M5R3P0Error("M5 R3 P0 contamination preflight failed")
    teacher_tokenizer = AutoTokenizer.from_pretrained(
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
    generations: list[TeacherGenerationRecord] = []
    for task_index, task in enumerate(tasks):
        rendered = teacher_tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        encoded: dict[str, Any] = teacher_tokenizer(rendered, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        input_token_count = int(input_ids.shape[1])
        task_records: list[TeacherGenerationRecord] = []
        for candidate_index in range(config.sampling.candidate_count):
            seed = m5_r3_p0_generation_seed(
                config.sampling.base_seed,
                task_index,
                candidate_index,
            )
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            try:
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=config.sampling.do_sample,
                        temperature=config.sampling.temperature,
                        top_p=config.sampling.top_p,
                        top_k=config.sampling.top_k,
                        repetition_penalty=config.sampling.repetition_penalty,
                        max_new_tokens=config.sampling.max_new_tokens,
                        pad_token_id=teacher_tokenizer.eos_token_id,
                        eos_token_id=teacher_tokenizer.eos_token_id,
                        use_cache=True,
                    )
                generated = output_ids[0, input_token_count:]
                generated_count = int(generated.numel())
                raw_output = cast(
                    str,
                    teacher_tokenizer.decode(generated, skip_special_tokens=True),
                )
                record = TeacherGenerationRecord(
                    generation_id=f"{task.id}:candidate-{candidate_index}",
                    task_id=task.id,
                    candidate_index=candidate_index,
                    seed=seed,
                    prompt_sha256=task.prompt_sha256,
                    status="succeeded",
                    finish_reason=(
                        "length" if generated_count >= config.sampling.max_new_tokens else "stop"
                    ),
                    raw_output=raw_output,
                    raw_output_sha256=hashlib.sha256(raw_output.encode()).hexdigest(),
                    observed_token_count=input_token_count + generated_count,
                )
            except RuntimeError:
                record = TeacherGenerationRecord(
                    generation_id=f"{task.id}:candidate-{candidate_index}",
                    task_id=task.id,
                    candidate_index=candidate_index,
                    seed=seed,
                    prompt_sha256=task.prompt_sha256,
                    status="failed",
                    finish_reason="error",
                    observed_token_count=0,
                    error_code="generation_runtime_error",
                )
                torch.cuda.empty_cache()
            task_records.append(record)
        generations.extend(task_records)
        if (task_index + 1) % 5 == 0:
            print(
                json.dumps(
                    {
                        "completed_tasks": task_index + 1,
                        "generation_attempts": len(generations),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    duration_seconds = time.monotonic() - started
    generation_payload = {
        "schema_version": "1.0",
        "pilot_version": config.pilot_version,
        "config_sha256": m5_r3_p0_config_sha256(config),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "physical_gpu_index": args.gpu_index,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "teacher_tokenizers_version": teacher_tokenizers.__version__,
        "duration_seconds": duration_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "tasks": [item.to_dict() for item in tasks],
        "generations": [item.to_dict() for item in generations],
    }
    _atomic_json(args.generation_output, generation_payload)
    print(
        json.dumps(
            {
                "status": "generated",
                "generation_attempts": len(generations),
                "generation_artifact_sha256": _sha256_file(args.generation_output),
            },
            sort_keys=True,
        )
    )
    return 0


def _finalize(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    if args.generation_output is None:
        raise M5R3P0Error("M5 R3 P0 finalizer requires a generation output")
    config = load_m5_r3_p0_config(args.config)
    reasoning_config = load_m5_reasoning_data_config(args.reasoning_config)
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5R3P0Error("M5 R3 P0 requires a clean Git worktree")
    _verify_frozen_inputs(args)
    historical_tasks = _load_historical_tasks(args)
    try:
        generation_payload = cast(
            dict[str, object],
            json.loads(args.generation_output.read_text(encoding="utf-8")),
        )
        tasks = tuple(
            ReasoningTask.model_validate(value)
            for value in cast(list[object], generation_payload["tasks"])
        )
        generations = tuple(
            TeacherGenerationRecord.model_validate(value)
            for value in cast(list[object], generation_payload["generations"])
        )
        physical_gpu_index = int(cast(int, generation_payload["physical_gpu_index"]))
        gpu_name = str(generation_payload["gpu_name"])
        torch_version = str(generation_payload["torch_version"])
        transformers_version = str(generation_payload["transformers_version"])
        teacher_tokenizers_version = str(generation_payload["teacher_tokenizers_version"])
        duration_seconds = float(cast(float, generation_payload["duration_seconds"]))
        peak_allocated_bytes = int(cast(int, generation_payload["peak_allocated_bytes"]))
        peak_reserved_bytes = int(cast(int, generation_payload["peak_reserved_bytes"]))
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise M5R3P0Error("M5 R3 P0 generation artifact cannot be reconstructed") from exc
    if (
        generation_payload.get("schema_version") != "1.0"
        or generation_payload.get("pilot_version") != config.pilot_version
        or generation_payload.get("config_sha256") != m5_r3_p0_config_sha256(config)
        or generation_payload.get("git_commit") != git_commit
        or generation_payload.get("git_dirty") is not False
        or physical_gpu_index != args.gpu_index
        or len(tasks) != 40
        or len(generations) != 80
    ):
        raise M5R3P0Error("M5 R3 P0 generation artifact identity differs")
    dev_tasks = generate_reasoning_dev_tasks(reasoning_config)
    tokenization = load_m2_tokenization_config(args.tokenization_config)
    policy_tokenizer = TokenizersBackend.from_files(
        args.tokenizer_dir / tokenization.tokenizer.tokenizer_file,
        args.tokenizer_dir / tokenization.tokenizer.tokenizer_config_file,
        tokenization.tokenizer,
    )
    build = build_m5_r3_p0_dataset(
        tasks,
        generations,
        config=config,
        reasoning_config=reasoning_config,
        dev_tasks=dev_tasks,
        historical_tasks=historical_tasks,
        tokenizer=policy_tokenizer,
    )
    raw_payload = {
        "schema_version": "1.0",
        "pilot_version": config.pilot_version,
        "config_sha256": m5_r3_p0_config_sha256(config),
        "generation_artifact_sha256": _sha256_file(args.generation_output),
        "task_set_sha256": build.task_set_sha256,
        "samples_sha256": build.samples_sha256,
        "tasks": [item.to_dict() for item in build.tasks],
        "generations": [item.to_dict() for item in build.generations],
        "samples": [item.to_dict() for item in build.samples],
        "candidate_audits": [item.to_dict() for item in build.candidate_audits],
        "verifications": [item.to_dict() for item in build.verifications],
        "contamination": build.contamination.to_dict(),
        "family_results": [item.to_dict() for item in build.family_results],
        "rejection_counts": build.rejection_counts,
    }
    _atomic_json(args.raw_output, raw_payload)
    result = M5R3P0Result(
        status=("pass" if all(item.gate_passed for item in build.family_results) else "fail"),
        pilot_version=config.pilot_version,
        generated_at=datetime.now(UTC),
        config_sha256=m5_r3_p0_config_sha256(config),
        git_commit=git_commit,
        git_dirty=git_dirty,
        model=config.teacher,
        sampling=config.sampling,
        tokenizer_revision=config.tokenizer_revision,
        physical_gpu_index=physical_gpu_index,
        gpu_name=gpu_name,
        torch_version=torch_version,
        transformers_version=transformers_version,
        teacher_tokenizers_version=teacher_tokenizers_version,
        policy_tokenizers_version="0.21.4",
        input_tasks=40,
        generation_attempts=len(build.generations),
        accepted_samples=len(build.samples),
        rejected_tasks=40 - len(build.samples),
        family_results=build.family_results,
        rejection_counts=build.rejection_counts,
        contamination=build.contamination,
        task_set_sha256=build.task_set_sha256,
        samples_sha256=build.samples_sha256,
        duration_seconds=duration_seconds,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        raw_artifact_sha256=_sha256_file(args.raw_output),
    )
    _atomic_json(args.public_output, result.to_dict())
    print(result.model_dump_json())
    return 0 if result.status == "pass" else 6


def _verify_policy_python(policy_python: Path, project_root: Path) -> None:
    if not policy_python.is_file() or not os.access(policy_python, os.X_OK):
        raise M5R3P0EnvironmentError("M5 R3 P0 policy Python is unavailable")
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
        raise M5R3P0EnvironmentError("M5 R3 P0 policy Python preflight failed") from exc
    if completed.returncode != 0:
        raise M5R3P0EnvironmentError("M5 R3 P0 policy Python must provide tokenizers 0.21.4")


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
        "--source-audit-config",
        str(args.source_audit_config),
        "--source-audit-result",
        str(args.source_audit_result),
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
        raise M5R3P0Error("M5 R3 P0 requires a clean Git worktree")
    generation_output = (
        args.generation_output
        if args.generation_output is not None
        else args.raw_output.with_name(f"{args.raw_output.stem}.generations.json")
    )
    if generation_output.exists() or args.raw_output.exists() or args.public_output.exists():
        raise M5R3P0Error("M5 R3 P0 output already exists")
    _verify_frozen_inputs(args)
    _verify_policy_python(args.policy_python, project_root)
    try:
        validate_gpu_preflight(inspect_gpus((args.gpu_index,)))
    except RuntimeError as exc:
        raise M5R3P0EnvironmentError(str(exc)) from exc
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    command = _subprocess_command(
        args,
        interpreter=Path(sys.executable),
        mode="--worker",
        generation_output=generation_output,
    )
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
        raise M5R3P0EnvironmentError("M5 R3 P0 exceeded its timeout") from exc
    if completed.returncode != 0:
        return completed.returncode
    finalizer_environment = os.environ.copy()
    finalizer_environment.pop("CUDA_VISIBLE_DEVICES", None)
    finalizer = _subprocess_command(
        args,
        interpreter=args.policy_python,
        mode="--finalize",
        generation_output=generation_output,
    )
    try:
        finalized = subprocess.run(
            finalizer,
            cwd=project_root,
            env=finalizer_environment,
            check=False,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise M5R3P0EnvironmentError("M5 R3 P0 finalizer exceeded its timeout") from exc
    return finalized.returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit offline-only P0 interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/m5_r3_p0.yaml"),
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
    parser.add_argument(
        "--source-audit-config",
        type=Path,
        default=Path("configs/data/m5_r3_targeted_repair.yaml"),
    )
    parser.add_argument(
        "--source-audit-result",
        type=Path,
        default=Path("reports/m5/raw/m5_r3_source_audit.json"),
    )
    parser.add_argument("--historical-pilot-artifact", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument(
        "--policy-python",
        type=Path,
        default=Path(".venv/bin/python"),
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--generation-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--finalize", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Run P0 or its isolated CUDA worker with stable exit codes."""

    args = build_parser().parse_args()
    try:
        if args.worker and args.finalize:
            raise M5R3P0Error("M5 R3 P0 internal modes are mutually exclusive")
        if args.worker:
            return _worker(args)
        if args.finalize:
            return _finalize(args)
        return _supervise(args)
    except M5R3P0EnvironmentError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 3
    except (M5R3P0Error, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
