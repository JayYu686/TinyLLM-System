"""Real dual-mode Qwen3 inference and deterministic M5.2 ablation selection."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import yaml
from pydantic import ValidationError

from tinyllm.data import (
    ReasoningTask,
    generate_reasoning_dev_tasks,
    load_m5_reasoning_data_config,
    parse_teacher_output,
    reasoning_config_sha256,
)
from tinyllm.data.reasoning_schema import canonical_json, content_sha256
from tinyllm.evaluation.m5_reasoning_schema import (
    M5AblationArmSummary,
    M5AblationSelection,
    M5FormatRepairGateResult,
    M5ModeSummary,
    M5ReasoningEvaluationConfig,
    M5ReasoningEvaluationSummary,
    M5ReasoningItemResult,
)
from tinyllm.lineage import read_git_identity
from tinyllm.training.m5_ablation_schema import M5AblationRunResult


class M5ReasoningEvaluationError(RuntimeError):
    """Raised when an M5-only evaluation violates its frozen identity or runtime."""


def load_m5_reasoning_evaluation_config(path: Path) -> M5ReasoningEvaluationConfig:
    """Load one strict M5-only dual-mode evaluation YAML."""

    try:
        decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return M5ReasoningEvaluationConfig.model_validate(decoded)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise M5ReasoningEvaluationError("M5 Reasoning evaluation config is invalid") from exc


def score_m5_response(
    task: ReasoningTask,
    *,
    mode: Literal["thinking", "nonthinking"],
    response: str,
    prompt_tokens: int,
    generated_tokens: int,
    finish_reason: Literal["eos", "length"],
) -> M5ReasoningItemResult:
    """Parse and objectively score one JSON-answer Reasoning Dev response."""

    leakage = mode == "nonthinking" and ("<think>" in response or "</think>" in response)
    if mode == "thinking":
        parsed, reason = parse_teacher_output(response)
        format_valid = reason is None and parsed is not None
        final_answer = parsed.final_answer if parsed is not None else ""
    else:
        format_valid = not leakage
        final_answer = response.strip()
    final_json_valid = False
    correct = False
    try:
        decoded = json.loads(final_answer)
        final_json_valid = isinstance(decoded, dict)
        if final_json_valid:
            correct = canonical_json(decoded) == task.expected_answer_json
    except json.JSONDecodeError:
        pass
    return M5ReasoningItemResult(
        item_id=task.id,
        mode=mode,
        prompt_sha256=task.prompt_sha256,
        response=response,
        response_sha256=hashlib.sha256(response.encode()).hexdigest(),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        finish_reason=finish_reason,
        format_valid=format_valid,
        final_json_valid=final_json_valid,
        final_answer_correct=correct,
        visible_reasoning_leakage=leakage,
    )


def summarize_m5_mode(
    mode: Literal["thinking", "nonthinking"],
    results: Sequence[M5ReasoningItemResult],
) -> M5ModeSummary:
    """Aggregate exactly 200 private item results into public basis-point metrics."""

    if len(results) != 200 or any(item.mode != mode for item in results):
        raise M5ReasoningEvaluationError("M5 mode summary requires exactly 200 matching items")
    format_valid = sum(item.format_valid for item in results)
    correct = sum(item.final_answer_correct for item in results)
    return M5ModeSummary(
        mode=mode,
        evaluated_items=200,
        format_valid_items=format_valid,
        final_json_valid_items=sum(item.final_json_valid for item in results),
        final_answer_correct_items=correct,
        visible_reasoning_leakage_items=sum(item.visible_reasoning_leakage for item in results),
        format_valid_basis_points=format_valid * 50,
        final_answer_score_basis_points=correct * 50,
        generated_tokens=sum(item.generated_tokens for item in results),
        length_limited_items=sum(item.finish_reason == "length" for item in results),
    )


def select_m5_ablation(
    base: M5ReasoningEvaluationSummary,
    candidates: Iterable[M5ReasoningEvaluationSummary],
) -> M5AblationSelection:
    """Apply regression, format, score, and lower-ratio tie-break gates exactly."""

    if base.model_kind != "base":
        raise M5ReasoningEvaluationError("M5 selection requires one Base evaluation")
    grouped: dict[int, list[M5ReasoningEvaluationSummary]] = defaultdict(list)
    for candidate in candidates:
        if candidate.model_kind != "ablation_candidate":
            raise M5ReasoningEvaluationError("M5 selection received a non-Candidate result")
        if (
            candidate.suite_version != base.suite_version
            or candidate.config_sha256 != base.config_sha256
            or candidate.model_revision != base.model_revision
            or candidate.attention_architecture != base.attention_architecture
        ):
            raise M5ReasoningEvaluationError(
                "M5 Candidate evaluation protocol or model identity differs from Base"
            )
        ratio = candidate.thinking_fraction_basis_points
        if ratio is None:
            raise M5ReasoningEvaluationError("M5 Candidate is missing its Thinking ratio")
        grouped[cast(int, ratio)].append(candidate)
    if set(grouped) != {0, 3000, 5000} or any(len(values) != 2 for values in grouped.values()):
        raise M5ReasoningEvaluationError("M5 selection requires two seeds for 0/30/50 ratios")
    arms: list[M5AblationArmSummary] = []
    for arm_ratio in (0, 3000, 5000):
        ordered = sorted(grouped[arm_ratio], key=lambda item: cast(int, item.training_seed))
        if ordered[0].training_seed == ordered[1].training_seed:
            raise M5ReasoningEvaluationError("M5 ratio requires two distinct training seeds")
        nonthinking = tuple(item.nonthinking.final_answer_score_basis_points for item in ordered)
        formats = tuple(item.thinking.format_valid_basis_points for item in ordered)
        thinking = tuple(item.thinking.final_answer_score_basis_points for item in ordered)
        arms.append(
            M5AblationArmSummary(
                thinking_fraction_basis_points=arm_ratio,
                training_run_ids=cast(
                    tuple[str, str], tuple(cast(str, item.training_run_id) for item in ordered)
                ),
                training_seeds=cast(
                    tuple[int, int], tuple(cast(int, item.training_seed) for item in ordered)
                ),
                nonthinking_scores_basis_points=cast(tuple[int, int], nonthinking),
                thinking_format_basis_points=cast(tuple[int, int], formats),
                thinking_scores_basis_points=cast(tuple[int, int], thinking),
                nonthinking_regression_gate_passed=all(
                    score >= base.nonthinking.final_answer_score_basis_points - 200
                    for score in nonthinking
                ),
                thinking_format_gate_passed=all(score >= 9900 for score in formats),
                mean_thinking_score_basis_points=sum(thinking) // 2,
            )
        )
    eligible = [
        arm
        for arm in arms
        if arm.nonthinking_regression_gate_passed and arm.thinking_format_gate_passed
    ]
    if not eligible:
        return M5AblationSelection(
            status="no_eligible_arm",
            base_evaluation_id=base.evaluation_id,
            base_nonthinking_score_basis_points=base.nonthinking.final_answer_score_basis_points,
            arms=cast(
                tuple[M5AblationArmSummary, M5AblationArmSummary, M5AblationArmSummary],
                tuple(arms),
            ),
            selection_reason="no_arm_passed_preregistered_gates",
        )
    best_score = max(item.mean_thinking_score_basis_points for item in eligible)
    within_one_point = [
        item for item in eligible if best_score - item.mean_thinking_score_basis_points < 100
    ]
    selected = min(within_one_point, key=lambda item: item.thinking_fraction_basis_points)
    reason: Literal["highest_thinking_score", "lower_ratio_within_one_percentage_point"] = (
        "lower_ratio_within_one_percentage_point"
        if selected.mean_thinking_score_basis_points < best_score
        else "highest_thinking_score"
    )
    return M5AblationSelection(
        status="selected",
        base_evaluation_id=base.evaluation_id,
        base_nonthinking_score_basis_points=base.nonthinking.final_answer_score_basis_points,
        arms=cast(
            tuple[M5AblationArmSummary, M5AblationArmSummary, M5AblationArmSummary],
            tuple(arms),
        ),
        selected_thinking_fraction_basis_points=selected.thinking_fraction_basis_points,
        selection_reason=reason,
    )


def evaluate_m5_format_repair_gate(
    base: M5ReasoningEvaluationSummary,
    candidates: Sequence[M5ReasoningEvaluationSummary],
    training_results: Sequence[M5AblationRunResult],
) -> M5FormatRepairGateResult:
    """Apply the unchanged two-seed regression and 99% format gates to R1."""

    if (
        base.model_kind != "base"
        or base.suite_version != "m5-reasoning-dev-v1-53ddf557"
        or len(candidates) != 2
        or len(training_results) != 2
    ):
        raise M5ReasoningEvaluationError(
            "M5 format-repair gate requires one frozen Base and two R1 runs"
        )
    run_by_id = {item.run_id: item for item in training_results}
    if len(run_by_id) != 2:
        raise M5ReasoningEvaluationError("M5 format-repair training Runs are duplicated")
    paired: list[tuple[M5ReasoningEvaluationSummary, M5AblationRunResult]] = []
    for candidate in candidates:
        run = run_by_id.get(str(candidate.training_run_id))
        if (
            run is None
            or run.status != "succeeded"
            or candidate.model_kind != "ablation_candidate"
            or candidate.training_seed != run.seed
            or candidate.thinking_fraction_basis_points != 3000
            or run.thinking_fraction_basis_points != 3000
            or candidate.suite_version != base.suite_version
            or candidate.config_sha256 != base.config_sha256
            or candidate.model_revision != base.model_revision
            or candidate.attention_architecture != base.attention_architecture
        ):
            raise M5ReasoningEvaluationError(
                "M5 R1 Candidate lineage or evaluation protocol differs"
            )
        paired.append((candidate, run))
    ordered = sorted(paired, key=lambda item: item[1].seed)
    if tuple(item[1].seed for item in ordered) != (42, 20260727):
        raise M5ReasoningEvaluationError("M5 R1 requires Seeds 42 and 20260727")
    mixture_versions = {item[1].mixture_version for item in ordered}
    mixture_hashes = {item[1].mixture_manifest_sha256 for item in ordered}
    if (
        len(mixture_versions) != 1
        or len(mixture_hashes) != 1
        or not next(iter(mixture_versions)).startswith(
            ("m5-format-repair-mixture-v1-", "m5-r3-mixture-v2-")
        )
    ):
        raise M5ReasoningEvaluationError("M5 repair runs do not share one versioned mixture")
    nonthinking = cast(
        tuple[int, int],
        tuple(item[0].nonthinking.final_answer_score_basis_points for item in ordered),
    )
    formats = cast(
        tuple[int, int],
        tuple(item[0].thinking.format_valid_basis_points for item in ordered),
    )
    thinking = cast(
        tuple[int, int],
        tuple(item[0].thinking.final_answer_score_basis_points for item in ordered),
    )
    nonthinking_passed = all(
        score >= base.nonthinking.final_answer_score_basis_points - 200 for score in nonthinking
    )
    format_passed = all(score >= 9900 for score in formats)
    if nonthinking_passed and format_passed:
        status: Literal["passed", "rejected"] = "passed"
        gate_reason: Literal[
            "all_preregistered_gates_passed",
            "nonthinking_regression_gate_failed",
            "thinking_format_gate_failed",
            "multiple_gates_failed",
        ] = "all_preregistered_gates_passed"
    else:
        status = "rejected"
        if not nonthinking_passed and not format_passed:
            gate_reason = "multiple_gates_failed"
        elif not nonthinking_passed:
            gate_reason = "nonthinking_regression_gate_failed"
        else:
            gate_reason = "thinking_format_gate_failed"
    return M5FormatRepairGateResult(
        status=status,
        base_evaluation_id=base.evaluation_id,
        base_nonthinking_score_basis_points=(base.nonthinking.final_answer_score_basis_points),
        mixture_version=next(iter(mixture_versions)),
        mixture_manifest_sha256=next(iter(mixture_hashes)),
        training_run_ids=cast(tuple[str, str], tuple(item[1].run_id for item in ordered)),
        training_seeds=(42, 20260727),
        evaluation_ids=cast(tuple[str, str], tuple(item[0].evaluation_id for item in ordered)),
        nonthinking_scores_basis_points=nonthinking,
        thinking_format_basis_points=formats,
        thinking_scores_basis_points=thinking,
        nonthinking_regression_gate_passed=nonthinking_passed,
        thinking_format_gate_passed=format_passed,
        mean_thinking_score_basis_points=sum(thinking) // 2,
        gate_reason=gate_reason,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Sequence[M5ReasoningItemResult]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finish_reason(token_ids: list[int], eos_ids: set[int]) -> Literal["eos", "length"]:
    return "eos" if any(token in eos_ids for token in token_ids) else "length"


def run_m5_reasoning_evaluation(
    *,
    config_path: Path,
    reasoning_config_path: Path,
    model_dir: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    physical_gpu_index: int,
    model_kind: Literal["base", "ablation_candidate"],
    training_run_id: str | None = None,
    training_seed: int | None = None,
    thinking_fraction_basis_points: Literal[0, 3000, 5000] | None = None,
) -> M5ReasoningEvaluationSummary:
    """Generate and score all 200 M5 Dev items in both explicit Qwen3 modes."""

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M5ReasoningEvaluationError("M5 evaluation requires exactly one visible CUDA GPU")
    project_root = Path(__file__).resolve().parents[3]
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5ReasoningEvaluationError("formal M5 evaluation requires a clean Git worktree")
    config = load_m5_reasoning_evaluation_config(config_path)
    reasoning_config = load_m5_reasoning_data_config(reasoning_config_path)
    tasks = generate_reasoning_dev_tasks(reasoning_config)
    if (
        len(tasks) != config.expected_items
        or reasoning_config_sha256(reasoning_config) != config.task_config_sha256
    ):
        raise M5ReasoningEvaluationError("M5 Dev task identity differs from evaluation config")
    if output_dir.exists():
        raise M5ReasoningEvaluationError("M5 evaluation output directory already exists")
    output_dir.mkdir(parents=True)
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    probe_messages = [{"role": "user", "content": "TinyLLM template probe."}]
    thinking_probe = tokenizer.apply_chat_template(
        probe_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    nonthinking_probe = tokenizer.apply_chat_template(
        probe_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if thinking_probe != (
        "<|im_start|>user\nTinyLLM template probe.<|im_end|>\n<|im_start|>assistant\n"
    ) or nonthinking_probe != (
        "<|im_start|>user\nTinyLLM template probe.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ):
        raise M5ReasoningEvaluationError("Qwen3 dual-mode generation Template drifted")
    try:
        model_config = cast(
            dict[str, object],
            json.loads((model_dir / "config.json").read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise M5ReasoningEvaluationError("M5 evaluated model config cannot be parsed") from exc
    if {
        "model_type": model_config.get("model_type"),
        "num_attention_heads": model_config.get("num_attention_heads"),
        "num_key_value_heads": model_config.get("num_key_value_heads"),
    } != {"model_type": "qwen3", "num_attention_heads": 16, "num_key_value_heads": 8}:
        raise M5ReasoningEvaluationError("M5 evaluated model is not the frozen Qwen3 GQA route")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    all_results: list[M5ReasoningItemResult] = []
    eos_value = tokenizer.eos_token_id
    eos_ids = {eos_value} if isinstance(eos_value, int) else set(cast(list[int], eos_value))
    for mode in ("thinking", "nonthinking"):
        mode_results: list[M5ReasoningItemResult] = []
        for offset in range(0, len(tasks), config.generation.batch_size):
            batch_tasks = tasks[offset : offset + config.generation.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": task.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=mode == "thinking",
                )
                for task in batch_tasks
            ]
            encoded: dict[str, Any] = tokenizer(prompts, padding=True, return_tensors="pt")
            prompt_lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1)]
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            input_width = int(model_inputs["input_ids"].shape[1])
            seed = config.generation.base_seed + offset + (0 if mode == "thinking" else 1_000_000)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    do_sample=mode == "thinking",
                    temperature=config.generation.temperature if mode == "thinking" else None,
                    top_p=config.generation.top_p if mode == "thinking" else None,
                    top_k=config.generation.top_k if mode == "thinking" else None,
                    repetition_penalty=config.generation.repetition_penalty,
                    max_new_tokens=(
                        config.generation.thinking_max_new_tokens
                        if mode == "thinking"
                        else config.generation.nonthinking_max_new_tokens
                    ),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=sorted(eos_ids),
                    use_cache=True,
                )
            sequences = generated[:, input_width:].detach().cpu().tolist()
            for task, prompt_tokens, raw_ids in zip(
                batch_tasks, prompt_lengths, sequences, strict=True
            ):
                token_ids = list(raw_ids)
                finish_reason = _finish_reason(token_ids, eos_ids)
                for index, token_id in enumerate(token_ids):
                    if token_id in eos_ids:
                        token_ids = token_ids[: index + 1]
                        break
                response = str(
                    tokenizer.decode(
                        token_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
                mode_results.append(
                    score_m5_response(
                        task,
                        mode=mode,
                        response=response,
                        prompt_tokens=prompt_tokens,
                        generated_tokens=len(token_ids),
                        finish_reason=finish_reason,
                    )
                )
        all_results.extend(mode_results)
    duration = time.monotonic() - started
    raw_path = output_dir / "results.jsonl"
    _write_jsonl(raw_path, all_results)
    model_identity = training_run_id or config.base_revision
    evaluation_id = (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-m5-reasoning-dev-"
        f"{model_kind}-{hashlib.sha256(model_identity.encode()).hexdigest()[:8]}"
    )
    thinking_results = tuple(item for item in all_results if item.mode == "thinking")
    nonthinking_results = tuple(item for item in all_results if item.mode == "nonthinking")
    summary = M5ReasoningEvaluationSummary(
        status="succeeded",
        evaluation_id=evaluation_id,
        model_kind=model_kind,
        training_run_id=training_run_id,
        training_seed=training_seed,
        thinking_fraction_basis_points=thinking_fraction_basis_points,
        model_revision=config.base_revision,
        attention_architecture=config.attention_architecture,
        suite_version=config.suite_version,
        config_sha256=content_sha256(config.to_dict()),
        git_commit=git_commit,
        git_dirty=False,
        physical_gpu_index=physical_gpu_index,
        gpu_name=torch.cuda.get_device_name(device),
        duration_seconds=duration,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        thinking=summarize_m5_mode("thinking", thinking_results),
        nonthinking=summarize_m5_mode("nonthinking", nonthinking_results),
        raw_results_sha256=_sha256_file(raw_path),
    )
    _atomic_json(output_dir / "summary.json", summary.to_dict())
    return summary
