#!/usr/bin/env python3
"""Run one real, offline Qwen3-8B thinking-teacher smoke on an explicit idle GPU."""

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
    M5TeacherSmokeResult,
    TeacherGenerationRecord,
    build_reasoning_dataset,
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
    missing = sorted(name for name in required if not (model_dir / name).is_file())
    if missing:
        raise RuntimeError(f"teacher model snapshot is incomplete: {missing[0]}")
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
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    project_root = Path(__file__).resolve().parents[1]
    config = load_m5_reasoning_data_config(args.config)
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise RuntimeError("teacher smoke requires a clean Git worktree")
    _verify_model_directory(args.model_dir, config.teacher.revision)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("teacher worker requires exactly one visible CUDA device")

    task = next(
        task
        for task in generate_reasoning_pilot_tasks(
            seed=config.sampling.base_seed,
            tasks_per_family=10,
        )
        if task.task_family == "python" and task.language == "en"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=config.teacher.local_files_only,
        trust_remote_code=config.teacher.trust_remote_code,
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": task.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    encoded: dict[str, Any] = tokenizer(rendered, return_tensors="pt")
    device = torch.device("cuda", 0)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    input_token_count = int(input_ids.shape[1])

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=config.teacher.local_files_only,
        trust_remote_code=config.teacher.trust_remote_code,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    records: list[TeacherGenerationRecord] = []
    generated_counts: list[int] = []
    for candidate_index in range(config.sampling.candidate_count):
        seed = (config.sampling.base_seed + candidate_index) % (2**32)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        generated = output_ids[0, input_token_count:]
        generated_count = int(generated.numel())
        generated_counts.append(generated_count)
        raw_output = cast(str, tokenizer.decode(generated, skip_special_tokens=True))
        records.append(
            TeacherGenerationRecord(
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
                raw_output_sha256=hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
                observed_token_count=input_token_count + generated_count,
            )
        )
    torch.cuda.synchronize(device)
    build = build_reasoning_dataset([task], records, config=config)
    duration_seconds = time.monotonic() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    raw_payload = {
        "schema_version": "1.0",
        "task": task.to_dict(),
        "generations": [record.to_dict() for record in records],
        "verifications": [result.to_dict() for result in build.verifications],
        "samples": [sample.to_dict() for sample in build.samples],
        "rejected": [record.to_dict() for record in build.rejected],
        "manifest": build.manifest.to_dict(),
    }
    _write_json(args.raw_output, raw_payload)
    result = M5TeacherSmokeResult(
        status="pass" if len(build.samples) == 1 else "fail",
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
        input_token_count=input_token_count,
        generated_token_counts=tuple(generated_counts),
        generation_attempts=len(records),
        accepted_samples=len(build.samples),
        rejection_counts=dict(sorted(Counter(record.reason for record in build.rejected).items())),
        dataset_version=(build.manifest.dataset_version if build.samples else None),
        duration_seconds=duration_seconds,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        raw_artifact_sha256=_sha256_file(args.raw_output),
    )
    _write_json(args.public_output, result.to_dict())
    print(result.model_dump_json())
    return 0 if result.status == "pass" else 1


def _supervise(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    _, dirty = read_git_identity(project_root)
    if dirty:
        raise RuntimeError("teacher smoke requires a clean Git worktree")
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
    """Build the explicit, offline-only teacher-smoke interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Preflight one physical GPU and execute the isolated CUDA worker."""

    args = build_parser().parse_args()
    try:
        return _worker(args) if args.worker else _supervise(args)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"M5 teacher smoke failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
