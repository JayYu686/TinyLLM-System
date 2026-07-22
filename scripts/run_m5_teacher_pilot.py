#!/usr/bin/env python3
"""Expand the private M5.2 Pilot with a real offline Qwen3-8B Thinking teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from tinyllm.data import (
    M5TeacherPilotResult,
    TeacherGenerationRecord,
    build_reasoning_dataset,
    generate_reasoning_dev_tasks,
    generate_reasoning_pilot_tasks,
    load_m5_reasoning_data_config,
)
from tinyllm.data.reasoning_schema import content_sha256
from tinyllm.lineage import read_git_identity
from tinyllm.training.smoke_preflight import inspect_gpus, validate_gpu_preflight


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generation_seed(base_seed: int, task_index: int, candidate_index: int) -> int:
    """Return the stable non-overlapping seed for one Teacher candidate."""

    if task_index < 0 or candidate_index not in {0, 1}:
        raise ValueError("invalid Teacher task or candidate index")
    return (base_seed + task_index * 2 + candidate_index) % (2**32)


def _verify_model_directory(model_dir: Path, expected_revision: str) -> None:
    if model_dir.name != expected_revision or not model_dir.is_dir() or model_dir.is_symlink():
        raise RuntimeError("teacher model directory must be the pinned revision snapshot")
    required = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if any(not (model_dir / name).is_file() for name in required):
        raise RuntimeError("teacher model snapshot is incomplete")
    decoded = cast(
        dict[str, object],
        json.loads((model_dir / "config.json").read_text(encoding="utf-8")),
    )
    expected = {
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "torch_dtype": "bfloat16",
    }
    if {key: decoded.get(key) for key in expected} != expected:
        raise RuntimeError("teacher model config differs from the pinned Qwen3-8B GQA identity")


def _worker(args: argparse.Namespace) -> int:
    import torch
    import transformers  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    project_root = Path(__file__).resolve().parents[1]
    config = load_m5_reasoning_data_config(args.config)
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise RuntimeError("teacher Pilot requires a clean Git worktree")
    _verify_model_directory(args.model_dir, config.teacher.revision)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("teacher Pilot worker requires exactly one visible CUDA device")
    tasks = generate_reasoning_pilot_tasks(
        seed=config.pilot_task_seed,
        tasks_per_family=args.tasks_per_family,
    )
    dev_tasks = generate_reasoning_dev_tasks(config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    device = torch.device("cuda", 0)
    # Initialize the selected context before querying allocator statistics. On the
    # reviewed CUDA 11.8 stack, resetting an uninitialized explicit device can fail.
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
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        encoded: dict[str, Any] = tokenizer(rendered, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        input_token_count = int(input_ids.shape[1])
        task_records: list[TeacherGenerationRecord] = []
        for candidate_index in range(config.sampling.candidate_count):
            seed = generation_seed(config.sampling.base_seed, task_index, candidate_index)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            try:
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        temperature=config.sampling.temperature,
                        top_p=config.sampling.top_p,
                        top_k=config.sampling.top_k,
                        repetition_penalty=config.sampling.repetition_penalty,
                        max_new_tokens=config.sampling.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                generated = output_ids[0, input_token_count:]
                generated_count = int(generated.numel())
                raw_output = cast(str, tokenizer.decode(generated, skip_special_tokens=True))
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
                    observed_token_count=input_token_count,
                    error_code="generation_runtime_error",
                )
                torch.cuda.empty_cache()
            task_records.append(record)
            partial = build_reasoning_dataset(
                [task],
                task_records,
                config=config,
                dev_tasks=dev_tasks,
            )
            if partial.samples:
                break
        generations.extend(task_records)
        if (task_index + 1) % 10 == 0:
            print(
                json.dumps(
                    {"completed_tasks": task_index + 1, "generation_attempts": len(generations)},
                    sort_keys=True,
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    build = build_reasoning_dataset(tasks, generations, config=config, dev_tasks=dev_tasks)
    duration_seconds = time.monotonic() - started
    raw_payload = {
        "schema_version": "1.0",
        "tasks": [task.to_dict() for task in tasks],
        "generations": [record.to_dict() for record in generations],
        "verifications": [item.to_dict() for item in build.verifications],
        "samples": [sample.to_dict() for sample in build.samples],
        "rejected": [record.to_dict() for record in build.rejected],
        "manifest": build.manifest.to_dict(),
        "contamination_report": build.contamination.to_dict(),
    }
    _write_json(args.raw_output, raw_payload)
    accepted_family_counts = dict(
        sorted(Counter(sample.task_family for sample in build.samples).items())
    )
    accepted_language_counts = dict(
        sorted(Counter(sample.language for sample in build.samples).items())
    )
    acceptance_passed = len(build.samples) * 10_000 >= len(tasks) * 8_000
    family_covered = set(accepted_family_counts) == set(build.manifest.task_family_counts)
    result = M5TeacherPilotResult(
        status="pass" if acceptance_passed and family_covered else "fail",
        generated_at=datetime.now(UTC),
        model=config.teacher,
        sampling=config.sampling,
        config_sha256=content_sha256(config.to_dict()),
        git_commit=git_commit,
        git_dirty=git_dirty,
        physical_gpu_index=args.gpu_index,
        gpu_name=torch.cuda.get_device_name(device),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        tasks_per_family=args.tasks_per_family,
        input_tasks=len(tasks),
        generation_attempts=len(generations),
        accepted_samples=len(build.samples),
        rejected_tasks=build.manifest.rejected_tasks,
        task_family_counts=build.manifest.task_family_counts,
        language_counts=build.manifest.language_counts,
        accepted_task_family_counts=cast(dict[str, int], accepted_family_counts),
        accepted_language_counts=cast(dict[str, int], accepted_language_counts),
        rejection_counts=build.manifest.rejection_counts,
        dataset_version=build.manifest.dataset_version,
        duration_seconds=duration_seconds,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        raw_artifact_sha256=_sha256_file(args.raw_output),
    )
    _write_json(args.public_output, result.to_dict())
    print(result.model_dump_json())
    return 0 if result.status == "pass" else 1


def _supervise(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    _, dirty = read_git_identity(project_root)
    if dirty:
        raise RuntimeError("teacher Pilot requires a clean Git worktree")
    validate_gpu_preflight(inspect_gpus((args.gpu_index,)))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
        "--model-dir",
        str(args.model_dir),
        "--gpu-index",
        str(args.gpu_index),
        "--tasks-per-family",
        str(args.tasks_per_family),
        "--raw-output",
        str(args.raw_output),
        "--public-output",
        str(args.public_output),
        "--worker",
    ]
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
    """Build the explicit offline-only Teacher Pilot interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--tasks-per-family", type=int, default=20)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Preflight one physical GPU and execute the isolated Teacher worker."""

    args = build_parser().parse_args()
    try:
        return _worker(args) if args.worker else _supervise(args)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"M5 teacher Pilot failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
