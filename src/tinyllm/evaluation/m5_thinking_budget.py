"""Qwen-official two-stage Thinking Budget evaluation for M5."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import uuid
from collections.abc import Sequence
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
    reasoning_config_sha256,
)
from tinyllm.data.reasoning_schema import content_sha256
from tinyllm.evaluation.m5_reasoning import score_m5_response
from tinyllm.evaluation.m5_reasoning_schema import M5FormatRepairGateResult
from tinyllm.evaluation.m5_thinking_budget_schema import (
    EARLY_STOPPING_TEXT,
    M5ThinkingBudgetEvaluationConfig,
    M5ThinkingBudgetEvaluationSummary,
    M5ThinkingBudgetGateResult,
    M5ThinkingBudgetItemResult,
    M5ThinkingBudgetModeSummary,
)
from tinyllm.lineage import read_git_identity


class M5ThinkingBudgetError(RuntimeError):
    """Raised when protocol-v2 lineage, configuration, or runtime fails closed."""


def load_m5_thinking_budget_config(path: Path) -> M5ThinkingBudgetEvaluationConfig:
    """Load the strict protocol-v2 YAML."""

    try:
        return M5ThinkingBudgetEvaluationConfig.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise M5ThinkingBudgetError("M5 Thinking Budget config is invalid") from exc


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


def _write_jsonl(path: Path, values: Sequence[M5ThinkingBudgetItemResult]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _eos_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    result = {value} if isinstance(value, int) else set(cast(list[int], value))
    if not result:
        raise M5ThinkingBudgetError("Qwen tokenizer has no EOS token")
    return result


def _trim_at_eos(
    raw_ids: Sequence[int],
    eos_ids: set[int],
    *,
    include_eos: bool,
) -> tuple[list[int], bool]:
    token_ids = list(raw_ids)
    for index, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            end = index + 1 if include_eos else index
            return token_ids[:end], True
    return token_ids, False


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


def build_m5_thinking_budget_item(
    task: ReasoningTask,
    *,
    mode: Literal["thinking", "nonthinking"],
    response: str,
    first_pass_response: str,
    continuation_response: str,
    controller_action: Literal[
        "not_applicable",
        "natural_complete",
        "natural_close_continue",
        "forced_close_continue",
    ],
    prompt_tokens: int,
    first_pass_tokens: int,
    continuation_tokens: int,
    injected_tokens: int,
    finish_reason: Literal["eos", "length"],
) -> M5ThinkingBudgetItemResult:
    """Score one transcript while disclosing every non-model controller token."""

    forced = controller_action == "forced_close_continue"
    natural = mode == "thinking" and not forced
    scored = score_m5_response(
        task,
        mode=mode,
        response=response,
        prompt_tokens=prompt_tokens,
        generated_tokens=first_pass_tokens + continuation_tokens,
        finish_reason=finish_reason,
    )
    return M5ThinkingBudgetItemResult(
        item_id=task.id,
        mode=mode,
        prompt_sha256=task.prompt_sha256,
        response=response,
        response_sha256=hashlib.sha256(response.encode()).hexdigest(),
        first_pass_response=first_pass_response,
        continuation_response=continuation_response,
        controller_injected_text=EARLY_STOPPING_TEXT if forced else "",
        controller_action=controller_action,
        prompt_tokens=prompt_tokens,
        first_pass_tokens=first_pass_tokens,
        continuation_tokens=continuation_tokens,
        injected_tokens=injected_tokens,
        generated_tokens=first_pass_tokens + continuation_tokens,
        finish_reason=finish_reason,
        natural_thinking_closed=natural,
        budget_forced_close=forced,
        format_valid=scored.format_valid,
        final_json_valid=scored.final_json_valid,
        final_answer_correct=scored.final_answer_correct,
        visible_reasoning_leakage=scored.visible_reasoning_leakage,
    )


def summarize_m5_thinking_budget_mode(
    mode: Literal["thinking", "nonthinking"],
    results: Sequence[M5ThinkingBudgetItemResult],
) -> M5ThinkingBudgetModeSummary:
    """Aggregate exactly 200 controlled item results."""

    if len(results) != 200 or any(item.mode != mode for item in results):
        raise M5ThinkingBudgetError("mode summary requires exactly 200 matching items")
    format_valid = sum(item.format_valid for item in results)
    correct = sum(item.final_answer_correct for item in results)
    natural = sum(item.natural_thinking_closed for item in results)
    forced = sum(item.budget_forced_close for item in results)
    return M5ThinkingBudgetModeSummary(
        mode=mode,
        evaluated_items=200,
        format_valid_items=format_valid,
        final_json_valid_items=sum(item.final_json_valid for item in results),
        final_answer_correct_items=correct,
        visible_reasoning_leakage_items=sum(item.visible_reasoning_leakage for item in results),
        natural_thinking_closed_items=natural,
        budget_forced_close_items=forced,
        format_valid_basis_points=format_valid * 50,
        final_answer_score_basis_points=correct * 50,
        natural_close_basis_points=natural * 50,
        forced_close_basis_points=forced * 50,
        generated_tokens=sum(item.generated_tokens for item in results),
        injected_tokens=sum(item.injected_tokens for item in results),
        length_limited_items=sum(item.finish_reason == "length" for item in results),
    )


def evaluate_m5_thinking_budget_gate(
    base: M5ThinkingBudgetEvaluationSummary,
    candidates: Sequence[M5ThinkingBudgetEvaluationSummary],
    source_gate: M5FormatRepairGateResult,
    *,
    base_summary_sha256: str,
    candidate_summary_sha256: tuple[str, str],
    source_gate_sha256: str,
) -> M5ThinkingBudgetGateResult:
    """Apply the frozen protocol-v2 AND gate to the verified R1 pair."""

    if (
        base.model_kind != "base"
        or base.protocol_version != "m5-thinking-budget-v2"
        or len(candidates) != 2
        or source_gate.mixture_version != "m5-format-repair-mixture-v1-1396b60b"
        or source_gate.training_seeds != (42, 20260727)
    ):
        raise M5ThinkingBudgetError(
            "Thinking Budget gate requires one Base and the verified R1 pair"
        )
    ordered = sorted(candidates, key=lambda item: int(item.training_seed or -1))
    if tuple(item.training_seed for item in ordered) != (42, 20260727):
        raise M5ThinkingBudgetError("Thinking Budget gate requires ordered fixed Seeds")
    if tuple(item.training_run_id for item in ordered) != source_gate.training_run_ids:
        raise M5ThinkingBudgetError("Thinking Budget Candidate Runs differ from the R1 gate")
    for candidate in ordered:
        if (
            candidate.model_kind != "ablation_candidate"
            or candidate.protocol_version != base.protocol_version
            or candidate.config_sha256 != base.config_sha256
            or candidate.git_commit != base.git_commit
            or candidate.model_revision != base.model_revision
            or candidate.attention_architecture != base.attention_architecture
            or candidate.suite_version != base.suite_version
            or candidate.thinking_fraction_basis_points != 3000
        ):
            raise M5ThinkingBudgetError(
                "Thinking Budget Candidate lineage or evaluation protocol differs"
            )
    formats = cast(
        tuple[int, int],
        tuple(item.thinking.format_valid_basis_points for item in ordered),
    )
    forced = cast(
        tuple[int, int],
        tuple(item.thinking.forced_close_basis_points for item in ordered),
    )
    thinking = cast(
        tuple[int, int],
        tuple(item.thinking.final_answer_score_basis_points for item in ordered),
    )
    nonthinking = cast(
        tuple[int, int],
        tuple(item.nonthinking.final_answer_score_basis_points for item in ordered),
    )
    gates = (
        all(value >= 9900 for value in formats),
        all(value <= 1000 for value in forced),
        all(value >= 9000 for value in thinking),
        all(
            value >= base.nonthinking.final_answer_score_basis_points - 200 for value in nonthinking
        ),
    )
    if all(gates):
        status: Literal["passed", "rejected"] = "passed"
        reason: Literal[
            "all_protocol_v2_gates_passed",
            "controlled_format_gate_failed",
            "forced_close_gate_failed",
            "thinking_score_gate_failed",
            "nonthinking_regression_gate_failed",
            "multiple_gates_failed",
        ] = "all_protocol_v2_gates_passed"
    else:
        status = "rejected"
        failed = sum(not value for value in gates)
        if failed > 1:
            reason = "multiple_gates_failed"
        elif not gates[0]:
            reason = "controlled_format_gate_failed"
        elif not gates[1]:
            reason = "forced_close_gate_failed"
        elif not gates[2]:
            reason = "thinking_score_gate_failed"
        else:
            reason = "nonthinking_regression_gate_failed"
    return M5ThinkingBudgetGateResult(
        status=status,
        protocol_version=base.protocol_version,
        base_evaluation_id=base.evaluation_id,
        candidate_evaluation_ids=cast(
            tuple[str, str], tuple(item.evaluation_id for item in ordered)
        ),
        evaluation_config_sha256=base.config_sha256,
        evaluation_git_commit=base.git_commit,
        base_summary_sha256=base_summary_sha256,
        candidate_summary_sha256=candidate_summary_sha256,
        source_format_repair_gate_sha256=source_gate_sha256,
        mixture_version=cast(
            Literal["m5-format-repair-mixture-v1-1396b60b"],
            source_gate.mixture_version,
        ),
        mixture_manifest_sha256=cast(
            Literal["2467b5dce0d909b865b73219d2f608bdbc9c6fcc1bb09b93c5ebea8a7b60bd0e"],
            source_gate.mixture_manifest_sha256,
        ),
        training_run_ids=source_gate.training_run_ids,
        training_seeds=(42, 20260727),
        selected_thinking_fraction_basis_points=3000,
        base_nonthinking_score_basis_points=(base.nonthinking.final_answer_score_basis_points),
        controlled_format_basis_points=formats,
        forced_close_basis_points=forced,
        thinking_scores_basis_points=thinking,
        nonthinking_scores_basis_points=nonthinking,
        controlled_format_gate_passed=gates[0],
        forced_close_gate_passed=gates[1],
        thinking_score_gate_passed=gates[2],
        nonthinking_regression_gate_passed=gates[3],
        m5_3_authorized=all(gates),
        gate_reason=reason,
    )


def _validate_qwen_runtime(model_dir: Path, tokenizer: Any) -> None:
    probe = [{"role": "user", "content": "TinyLLM template probe."}]
    thinking = tokenizer.apply_chat_template(
        probe,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    nonthinking = tokenizer.apply_chat_template(
        probe,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if thinking != (
        "<|im_start|>user\nTinyLLM template probe.<|im_end|>\n<|im_start|>assistant\n"
    ) or nonthinking != (
        "<|im_start|>user\nTinyLLM template probe.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ):
        raise M5ThinkingBudgetError("Qwen3 dual-mode generation Template drifted")
    try:
        model_config = cast(
            dict[str, object],
            json.loads((model_dir / "config.json").read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise M5ThinkingBudgetError("evaluated model config cannot be parsed") from exc
    if {
        "model_type": model_config.get("model_type"),
        "num_attention_heads": model_config.get("num_attention_heads"),
        "num_key_value_heads": model_config.get("num_key_value_heads"),
    } != {"model_type": "qwen3", "num_attention_heads": 16, "num_key_value_heads": 8}:
        raise M5ThinkingBudgetError("evaluated model is not the frozen Qwen3 GQA route")


def _generate(
    model: Any,
    *,
    model_inputs: dict[str, Any],
    input_width: int,
    tokenizer: Any,
    eos_ids: set[int],
    seed: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
) -> list[list[int]]:
    _set_seed(seed)
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=sorted(eos_ids),
            use_cache=True,
        )
    return cast(list[list[int]], generated[:, input_width:].detach().cpu().tolist())


def run_m5_thinking_budget_evaluation(
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
) -> M5ThinkingBudgetEvaluationSummary:
    """Evaluate Base or Candidate under the versioned official budget controller."""

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M5ThinkingBudgetError("Thinking Budget evaluation requires one visible CUDA GPU")
    project_root = Path(__file__).resolve().parents[3]
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5ThinkingBudgetError("formal evaluation requires a clean Git worktree")
    config = load_m5_thinking_budget_config(config_path)
    reasoning = load_m5_reasoning_data_config(reasoning_config_path)
    tasks = generate_reasoning_dev_tasks(reasoning)
    if (
        len(tasks) != config.expected_items
        or reasoning_config_sha256(reasoning) != config.task_config_sha256
    ):
        raise M5ThinkingBudgetError("M5 Dev task identity differs from protocol v2")
    if output_dir.exists():
        raise M5ThinkingBudgetError("Thinking Budget output directory already exists")
    output_dir.mkdir(parents=True)

    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    _validate_qwen_runtime(model_dir, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    eos_ids = _eos_ids(tokenizer)
    injection_ids = list(tokenizer.encode(EARLY_STOPPING_TEXT, add_special_tokens=False))
    if not injection_ids or _decode(tokenizer, injection_ids) != EARLY_STOPPING_TEXT:
        raise M5ThinkingBudgetError("official early-stopping text does not round-trip")

    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    all_results: list[M5ThinkingBudgetItemResult] = []
    for mode in ("thinking", "nonthinking"):
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
            first_rows = _generate(
                model,
                model_inputs=model_inputs,
                input_width=input_width,
                tokenizer=tokenizer,
                eos_ids=eos_ids,
                seed=seed,
                max_new_tokens=(
                    config.generation.thinking_budget_tokens
                    if mode == "thinking"
                    else config.generation.nonthinking_max_new_tokens
                ),
                do_sample=mode == "thinking",
                temperature=config.generation.temperature if mode == "thinking" else None,
                top_p=config.generation.top_p if mode == "thinking" else None,
                top_k=config.generation.top_k if mode == "thinking" else None,
            )
            for item_index, (task, prompt_tokens, raw_ids) in enumerate(
                zip(batch_tasks, prompt_lengths, first_rows, strict=True)
            ):
                first_ids, first_eos = _trim_at_eos(raw_ids, eos_ids, include_eos=True)
                first_response = _decode(tokenizer, first_ids)
                if mode == "nonthinking":
                    all_results.append(
                        build_m5_thinking_budget_item(
                            task,
                            mode="nonthinking",
                            response=first_response,
                            first_pass_response=first_response,
                            continuation_response="",
                            controller_action="not_applicable",
                            prompt_tokens=prompt_tokens,
                            first_pass_tokens=len(first_ids),
                            continuation_tokens=0,
                            injected_tokens=0,
                            finish_reason="eos" if first_eos else "length",
                        )
                    )
                    continue

                natural_close = "</think>" in first_response
                if natural_close and first_eos:
                    all_results.append(
                        build_m5_thinking_budget_item(
                            task,
                            mode="thinking",
                            response=first_response,
                            first_pass_response=first_response,
                            continuation_response="",
                            controller_action="natural_complete",
                            prompt_tokens=prompt_tokens,
                            first_pass_tokens=len(first_ids),
                            continuation_tokens=0,
                            injected_tokens=0,
                            finish_reason="eos",
                        )
                    )
                    continue

                prompt_row = model_inputs["input_ids"][item_index]
                mask_row = model_inputs["attention_mask"][item_index].bool()
                unpadded_prompt = prompt_row[mask_row].detach().cpu().tolist()
                first_for_continue, _ = _trim_at_eos(first_ids, eos_ids, include_eos=False)
                forced = not natural_close
                controlled_ids = first_for_continue + (injection_ids if forced else [])
                continuation_input = torch.tensor(
                    [unpadded_prompt + controlled_ids],
                    dtype=torch.long,
                    device=device,
                )
                continuation_mask = torch.ones_like(continuation_input)
                continuation_rows = _generate(
                    model,
                    model_inputs={
                        "input_ids": continuation_input,
                        "attention_mask": continuation_mask,
                    },
                    input_width=int(continuation_input.shape[1]),
                    tokenizer=tokenizer,
                    eos_ids=eos_ids,
                    seed=config.generation.base_seed + 2_000_000 + offset + item_index,
                    max_new_tokens=config.generation.final_answer_max_new_tokens,
                    do_sample=True,
                    temperature=config.generation.temperature,
                    top_p=config.generation.top_p,
                    top_k=config.generation.top_k,
                )
                continuation_ids, continuation_eos = _trim_at_eos(
                    continuation_rows[0],
                    eos_ids,
                    include_eos=True,
                )
                full_ids = controlled_ids + continuation_ids
                response = _decode(tokenizer, full_ids)
                all_results.append(
                    build_m5_thinking_budget_item(
                        task,
                        mode="thinking",
                        response=response,
                        first_pass_response=first_response,
                        continuation_response=_decode(tokenizer, continuation_ids),
                        controller_action=(
                            "forced_close_continue" if forced else "natural_close_continue"
                        ),
                        prompt_tokens=prompt_tokens,
                        first_pass_tokens=len(first_ids),
                        continuation_tokens=len(continuation_ids),
                        injected_tokens=len(injection_ids) if forced else 0,
                        finish_reason="eos" if continuation_eos else "length",
                    )
                )

    duration = time.monotonic() - started
    raw_path = output_dir / "results.jsonl"
    _write_jsonl(raw_path, all_results)
    model_identity = training_run_id or config.base_revision
    evaluation_id = (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-m5-thinking-budget-"
        f"{model_kind}-{hashlib.sha256(model_identity.encode()).hexdigest()[:8]}"
    )
    summary = M5ThinkingBudgetEvaluationSummary(
        status="succeeded",
        evaluation_id=evaluation_id,
        protocol_version=config.protocol_version,
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
        thinking=summarize_m5_thinking_budget_mode(
            "thinking",
            tuple(item for item in all_results if item.mode == "thinking"),
        ),
        nonthinking=summarize_m5_thinking_budget_mode(
            "nonthinking",
            tuple(item for item in all_results if item.mode == "nonthinking"),
        ),
        raw_results_sha256=_sha256_file(raw_path),
    )
    _atomic_json(output_dir / "summary.json", summary.to_dict())
    return summary
