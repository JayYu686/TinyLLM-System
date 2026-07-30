"""Deterministic M5.2-R2 counterfactual length replay and decision logic."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
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
from tinyllm.evaluation.m5_format_analysis import (
    _load_results,
    _sha256_file,
    _validate_private_result_set,
)
from tinyllm.evaluation.m5_r2_schema import (
    M5R2DiagnosticDecision,
    M5R2ReplayConfig,
    M5R2ReplayItemResult,
    M5R2ReplaySummary,
    M5R2SeedProjection,
    M5R2Threshold,
    M5R2ThresholdItemResult,
    M5R2ThresholdSummary,
)
from tinyllm.evaluation.m5_reasoning import score_m5_response
from tinyllm.evaluation.m5_reasoning_schema import (
    M5ReasoningEvaluationConfig,
    M5ReasoningEvaluationSummary,
    M5ReasoningItemResult,
)
from tinyllm.lineage import read_git_identity
from tinyllm.training.m5_ablation_schema import M5AblationRunResult


class M5R2DiagnosticError(RuntimeError):
    """Raised when an R2 replay or decision violates its frozen contract."""


def load_m5_r2_replay_config(path: Path) -> M5R2ReplayConfig:
    """Load the strict R2 diagnostic configuration."""

    try:
        decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise M5R2DiagnosticError("M5 R2 diagnostic config must be a mapping")
        decoded["score_thresholds"] = tuple(decoded.get("score_thresholds", ()))
        return M5R2ReplayConfig.model_validate(decoded)
    except (OSError, yaml.YAMLError, ValidationError, TypeError) as exc:
        raise M5R2DiagnosticError("M5 R2 diagnostic config is invalid") from exc


def replay_batch_offsets(
    results: Sequence[M5ReasoningItemResult],
    tasks: Sequence[ReasoningTask],
    *,
    batch_size: int = 4,
) -> tuple[int, ...]:
    """Return ordered Batch offsets containing original Thinking format failures."""

    if batch_size != 4 or len(tasks) != 200:
        raise M5R2DiagnosticError("M5 R2 requires 200 tasks and Batch Size 4")
    task_index = {task.id: index for index, task in enumerate(tasks)}
    if len(task_index) != len(tasks):
        raise M5R2DiagnosticError("M5 R2 task identities are duplicated")
    try:
        offsets = {
            (task_index[item.item_id] // batch_size) * batch_size
            for item in results
            if item.mode == "thinking" and not item.format_valid
        }
    except KeyError as exc:
        raise M5R2DiagnosticError("M5 R2 source result has an unknown task") from exc
    if not offsets:
        raise M5R2DiagnosticError("M5 R2 source has no Thinking format failures")
    return tuple(sorted(offsets))


def expected_m5_r2_source_identity(
    config: M5R2ReplayConfig,
    training_seed: int,
) -> tuple[str, str, str, str]:
    """Return the frozen Run, evaluation, Raw, and export identities for one Seed."""

    if training_seed == 42:
        return (
            config.seed42_training_run_id,
            config.seed42_source_evaluation_id,
            config.seed42_source_raw_results_sha256,
            config.seed42_model_export_sha256,
        )
    if training_seed == 20260727:
        return (
            config.seed20260727_training_run_id,
            config.seed20260727_source_evaluation_id,
            config.seed20260727_source_raw_results_sha256,
            config.seed20260727_model_export_sha256,
        )
    raise M5R2DiagnosticError("M5 R2 source Seed is not part of the frozen pair")


def validate_m5_r2_replay_pair(
    *,
    source: M5ReasoningItemResult,
    replay_896: M5ReasoningItemResult,
    replay_896_ids: Sequence[int],
    replay_1536_ids: Sequence[int],
) -> None:
    """Fail closed on response/scoring drift or counterfactual prefix drift."""

    if replay_896 != source:
        raise M5R2DiagnosticError(f"M5 R2 INVALID_REPLAY: 896 output drifted for {source.item_id}")
    if list(replay_1536_ids[: len(replay_896_ids)]) != list(replay_896_ids):
        raise M5R2DiagnosticError(f"M5 R2 INVALID_REPLAY: 1536 prefix drifted for {source.item_id}")


def _eos_ids(tokenizer: Any) -> set[int]:
    eos_value = tokenizer.eos_token_id
    values = {eos_value} if isinstance(eos_value, int) else set(cast(list[int], eos_value))
    if not values:
        raise M5R2DiagnosticError("M5 R2 tokenizer has no EOS token")
    return values


def _normalize_generated_ids(
    raw_ids: Sequence[int],
    eos_ids: set[int],
    *,
    limit: int,
) -> tuple[list[int], Literal["eos", "length"]]:
    """Truncate generated IDs at a limit and retain the first EOS token."""

    token_ids = list(raw_ids[:limit])
    for index, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            return token_ids[: index + 1], "eos"
    if len(token_ids) != limit:
        raise M5R2DiagnosticError("M5 R2 generation ended without EOS before its limit")
    return token_ids, "length"


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


def _closing_tag_end_token(
    tokenizer: Any,
    token_ids: Sequence[int],
    response: str,
) -> int | None:
    """Locate the first decoded ``</think>`` end position in generated-token space."""

    if "</think>" not in response:
        return None
    close_ids = list(tokenizer.encode("</think>", add_special_tokens=False))
    if close_ids:
        width = len(close_ids)
        for start in range(0, len(token_ids) - width + 1):
            if list(token_ids[start : start + width]) == close_ids:
                return start + width
    for end in range(1, len(token_ids) + 1):
        if "</think>" in _decode(tokenizer, token_ids[:end]):
            return end
    raise M5R2DiagnosticError("M5 R2 decoded closing tag has no token position")


def score_m5_r2_threshold(
    *,
    task: ReasoningTask,
    tokenizer: Any,
    raw_ids: Sequence[int],
    eos_ids: set[int],
    max_new_tokens: M5R2Threshold,
    prompt_tokens: int,
) -> M5R2ThresholdItemResult:
    """Score one long replay prefix with the unchanged M5 parser and scorer."""

    token_ids, finish_reason = _normalize_generated_ids(
        raw_ids,
        eos_ids,
        limit=max_new_tokens,
    )
    response = _decode(tokenizer, token_ids)
    scored = score_m5_response(
        task,
        mode="thinking",
        response=response,
        prompt_tokens=prompt_tokens,
        generated_tokens=len(token_ids),
        finish_reason=finish_reason,
    )
    return M5R2ThresholdItemResult(
        max_new_tokens=max_new_tokens,
        response=response,
        response_sha256=scored.response_sha256,
        generated_tokens=scored.generated_tokens,
        finish_reason=scored.finish_reason,
        format_valid=scored.format_valid,
        final_json_valid=scored.final_json_valid,
        final_answer_correct=scored.final_answer_correct,
        closing_tag_end_token=_closing_tag_end_token(tokenizer, token_ids, response),
    )


def _set_generation_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _generate_batch(
    *,
    model: Any,
    model_inputs: dict[str, Any],
    input_width: int,
    tokenizer: Any,
    eos_ids: set[int],
    seed: int,
    max_new_tokens: int,
) -> list[list[int]]:
    _set_generation_seed(seed)
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=sorted(eos_ids),
            use_cache=True,
        )
    return cast(list[list[int]], generated[:, input_width:].detach().cpu().tolist())


def _validate_source(
    *,
    source_evaluation_dir: Path,
    source_summary: M5ReasoningEvaluationSummary,
    source_results: Sequence[M5ReasoningItemResult],
    tasks: Sequence[ReasoningTask],
    reasoning_config_hash: str,
    evaluation_config: M5ReasoningEvaluationConfig,
    replay_config: M5R2ReplayConfig,
    training_result: M5AblationRunResult,
) -> None:
    expected_prompt_hashes = {item.id: item.prompt_sha256 for item in tasks}
    expected_run, expected_evaluation, expected_raw, expected_export = (
        expected_m5_r2_source_identity(replay_config, training_result.seed)
    )
    raw_path = source_evaluation_dir / "results.jsonl"
    if _sha256_file(raw_path) != source_summary.raw_results_sha256:
        raise M5R2DiagnosticError("M5 R2 source Raw Result hash differs from Summary")
    _validate_private_result_set(
        summary=source_summary,
        results=source_results,
        expected_prompt_hashes=expected_prompt_hashes,
    )
    if (
        source_summary.model_kind != "ablation_candidate"
        or source_summary.training_seed not in {42, 20260727}
        or source_summary.thinking_fraction_basis_points != 3000
        or source_summary.training_run_id != training_result.run_id
        or source_summary.training_seed != training_result.seed
        or training_result.run_id != expected_run
        or source_summary.evaluation_id != expected_evaluation
        or source_summary.raw_results_sha256 != expected_raw
        or training_result.export_sha256 != expected_export
        or training_result.mixture_version != replay_config.source_mixture_version
        or training_result.mixture_manifest_sha256 != replay_config.source_mixture_manifest_sha256
        or source_summary.model_revision != replay_config.model_revision
        or source_summary.attention_architecture != replay_config.attention_architecture
        or source_summary.suite_version != replay_config.source_suite_version
        or source_summary.config_sha256 != replay_config.source_evaluation_config_sha256
        or content_sha256(evaluation_config.to_dict())
        != replay_config.source_evaluation_config_sha256
        or reasoning_config_hash != evaluation_config.task_config_sha256
        or evaluation_config.generation.batch_size != replay_config.thinking_batch_size
        or evaluation_config.generation.base_seed != replay_config.thinking_base_seed
        or evaluation_config.generation.thinking_max_new_tokens
        != replay_config.original_max_new_tokens
        or training_result.status != "succeeded"
        or training_result.thinking_fraction_basis_points != 3000
        or training_result.model_revision != replay_config.model_revision
        or training_result.attention_architecture != replay_config.attention_architecture
    ):
        raise M5R2DiagnosticError("M5 R2 source lineage or frozen protocol differs")
    failures = tuple(
        item for item in source_results if item.mode == "thinking" and not item.format_valid
    )
    if len(failures) != 200 - source_summary.thinking.format_valid_items or any(
        item.generated_tokens != replay_config.original_max_new_tokens
        or item.finish_reason != "length"
        or item.response.count("<think>") != 1
        or item.response.count("</think>") != 0
        for item in failures
    ):
        raise M5R2DiagnosticError(
            "M5 R2 source failures are not exclusively open length-limited traces"
        )


def _load_source_summary(path: Path) -> M5ReasoningEvaluationSummary:
    try:
        return M5ReasoningEvaluationSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise M5R2DiagnosticError("M5 R2 source Summary is invalid") from exc


def _validate_qwen3_runtime(model_dir: Path, tokenizer: Any) -> None:
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "TinyLLM template probe."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if probe != ("<|im_start|>user\nTinyLLM template probe.<|im_end|>\n<|im_start|>assistant\n"):
        raise M5R2DiagnosticError("M5 R2 Qwen3 Thinking Template drifted")
    try:
        model_config = cast(
            dict[str, object],
            json.loads((model_dir / "config.json").read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise M5R2DiagnosticError("M5 R2 model config cannot be parsed") from exc
    if {
        "model_type": model_config.get("model_type"),
        "num_attention_heads": model_config.get("num_attention_heads"),
        "num_key_value_heads": model_config.get("num_key_value_heads"),
    } != {"model_type": "qwen3", "num_attention_heads": 16, "num_key_value_heads": 8}:
        raise M5R2DiagnosticError("M5 R2 model is not the frozen Qwen3 GQA route")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_private_results(path: Path, values: Sequence[M5R2ReplayItemResult]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _threshold_summaries(
    *,
    source_summary: M5ReasoningEvaluationSummary,
    replay_items: Sequence[M5R2ReplayItemResult],
    thresholds: tuple[Literal[1024], Literal[1280], Literal[1536]],
) -> tuple[M5R2ThresholdSummary, M5R2ThresholdSummary, M5R2ThresholdSummary]:
    failed = tuple(item for item in replay_items if not item.source_format_valid)
    summaries: list[M5R2ThresholdSummary] = []
    for threshold_index, max_new_tokens in enumerate(thresholds):
        scored = tuple(
            cast(
                tuple[
                    M5R2ThresholdItemResult,
                    M5R2ThresholdItemResult,
                    M5R2ThresholdItemResult,
                ],
                item.thresholds,
            )[threshold_index]
            for item in failed
        )
        recovered_format = tuple(item for item in scored if item.format_valid)
        positions = tuple(cast(int, item.closing_tag_end_token) for item in recovered_format)
        recovered_format_count = len(recovered_format)
        recovered_correct = sum(item.final_answer_correct for item in scored)
        summaries.append(
            M5R2ThresholdSummary(
                max_new_tokens=cast(M5R2Threshold, max_new_tokens),
                original_failed_items=len(failed),
                recovered_format_items=recovered_format_count,
                recovered_final_json_items=sum(item.final_json_valid for item in scored),
                recovered_correct_items=recovered_correct,
                unresolved_format_items=len(failed) - recovered_format_count,
                projected_format_valid_items=(
                    source_summary.thinking.format_valid_items + recovered_format_count
                ),
                projected_format_basis_points=(
                    source_summary.thinking.format_valid_items + recovered_format_count
                )
                * 50,
                projected_correct_items=(
                    source_summary.thinking.final_answer_correct_items + recovered_correct
                ),
                projected_score_basis_points=(
                    source_summary.thinking.final_answer_correct_items + recovered_correct
                )
                * 50,
                closing_tag_end_token_min=min(positions) if positions else None,
                closing_tag_end_token_max=max(positions) if positions else None,
            )
        )
    return cast(
        tuple[M5R2ThresholdSummary, M5R2ThresholdSummary, M5R2ThresholdSummary],
        tuple(summaries),
    )


def run_m5_r2_length_replay(
    *,
    replay_config_path: Path,
    evaluation_config_path: Path,
    reasoning_config_path: Path,
    source_evaluation_dir: Path,
    training_run_dir: Path,
    model_dir: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    physical_gpu_index: int,
) -> M5R2ReplaySummary:
    """Replay source-failure Batches at 896/1536 and publish content-free evidence."""

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    from tinyllm.evaluation.m5_reasoning import load_m5_reasoning_evaluation_config

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M5R2DiagnosticError("M5 R2 replay requires exactly one visible CUDA GPU")
    project_root = Path(__file__).resolve().parents[3]
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5R2DiagnosticError("formal M5 R2 replay requires a clean Git worktree")
    if output_dir.exists():
        raise M5R2DiagnosticError("M5 R2 output directory already exists")
    replay_config = load_m5_r2_replay_config(replay_config_path)
    evaluation_config = load_m5_reasoning_evaluation_config(evaluation_config_path)
    reasoning_config = load_m5_reasoning_data_config(reasoning_config_path)
    tasks = generate_reasoning_dev_tasks(reasoning_config)
    source_summary = _load_source_summary(source_evaluation_dir / "summary.json")
    source_results = _load_results(source_evaluation_dir / "results.jsonl")
    try:
        training_result = M5AblationRunResult.model_validate_json(
            (training_run_dir / "result.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise M5R2DiagnosticError("M5 R2 training Run result is invalid") from exc
    if model_dir.resolve() != (training_run_dir / "exports" / "model").resolve():
        raise M5R2DiagnosticError("M5 R2 model path differs from training Run export")
    _validate_source(
        source_evaluation_dir=source_evaluation_dir,
        source_summary=source_summary,
        source_results=source_results,
        tasks=tasks,
        reasoning_config_hash=reasoning_config_sha256(reasoning_config),
        evaluation_config=evaluation_config,
        replay_config=replay_config,
        training_result=training_result,
    )
    offsets = replay_batch_offsets(source_results, tasks)
    source_by_id = {item.item_id: item for item in source_results if item.mode == "thinking"}
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    _validate_qwen3_runtime(model_dir, tokenizer)
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
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    replay_items: list[M5R2ReplayItemResult] = []
    for offset in offsets:
        batch_tasks = tasks[offset : offset + replay_config.thinking_batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": task.prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for task in batch_tasks
        ]
        encoded: dict[str, Any] = tokenizer(prompts, padding=True, return_tensors="pt")
        prompt_lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1)]
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        input_width = int(model_inputs["input_ids"].shape[1])
        seed = replay_config.thinking_base_seed + offset
        generated_896 = _generate_batch(
            model=model,
            model_inputs=model_inputs,
            input_width=input_width,
            tokenizer=tokenizer,
            eos_ids=eos_ids,
            seed=seed,
            max_new_tokens=replay_config.original_max_new_tokens,
        )
        generated_1536 = _generate_batch(
            model=model,
            model_inputs=model_inputs,
            input_width=input_width,
            tokenizer=tokenizer,
            eos_ids=eos_ids,
            seed=seed,
            max_new_tokens=replay_config.diagnostic_max_new_tokens,
        )
        for task, prompt_tokens, raw_896, raw_1536 in zip(
            batch_tasks,
            prompt_lengths,
            generated_896,
            generated_1536,
            strict=True,
        ):
            source = source_by_id[task.id]
            ids_896, finish_896 = _normalize_generated_ids(
                raw_896,
                eos_ids,
                limit=replay_config.original_max_new_tokens,
            )
            response_896 = _decode(tokenizer, ids_896)
            scored_896 = score_m5_response(
                task,
                mode="thinking",
                response=response_896,
                prompt_tokens=prompt_tokens,
                generated_tokens=len(ids_896),
                finish_reason=finish_896,
            )
            ids_1536, _ = _normalize_generated_ids(
                raw_1536,
                eos_ids,
                limit=replay_config.diagnostic_max_new_tokens,
            )
            validate_m5_r2_replay_pair(
                source=source,
                replay_896=scored_896,
                replay_896_ids=ids_896,
                replay_1536_ids=ids_1536,
            )
            threshold_results = (
                tuple(
                    score_m5_r2_threshold(
                        task=task,
                        tokenizer=tokenizer,
                        raw_ids=raw_1536,
                        eos_ids=eos_ids,
                        max_new_tokens=threshold,
                        prompt_tokens=prompt_tokens,
                    )
                    for threshold in replay_config.score_thresholds
                )
                if not source.format_valid
                else None
            )
            replay_items.append(
                M5R2ReplayItemResult(
                    item_id=task.id,
                    task_family=task.task_family,
                    language=task.language,
                    batch_offset=offset,
                    prompt_sha256=task.prompt_sha256,
                    source_response_sha256=source.response_sha256,
                    source_generated_tokens=source.generated_tokens,
                    source_finish_reason=source.finish_reason,
                    source_format_valid=source.format_valid,
                    source_final_json_valid=source.final_json_valid,
                    source_final_answer_correct=source.final_answer_correct,
                    replay_896_response_sha256=scored_896.response_sha256,
                    replay_896_generated_tokens=scored_896.generated_tokens,
                    replay_896_finish_reason=scored_896.finish_reason,
                    replay_896_exact=True,
                    replay_1536_prefix_tokens_compared=len(ids_896),
                    replay_1536_prefix_exact=True,
                    thresholds=cast(
                        tuple[
                            M5R2ThresholdItemResult,
                            M5R2ThresholdItemResult,
                            M5R2ThresholdItemResult,
                        ]
                        | None,
                        threshold_results,
                    ),
                )
            )
    duration = time.monotonic() - started
    output_dir.mkdir(parents=True)
    raw_path = output_dir / "raw_results.jsonl"
    _write_private_results(raw_path, replay_items)
    diagnostic_id = (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-m5-r2-length-replay-"
        f"seed{training_result.seed}-{hashlib.sha256(training_result.run_id.encode()).hexdigest()[:8]}"
    )
    summary = M5R2ReplaySummary(
        status="succeeded",
        diagnostic_id=diagnostic_id,
        diagnostic_version=replay_config.diagnostic_version,
        diagnostic_config_sha256=content_sha256(replay_config.to_dict()),
        source_evaluation_id=source_summary.evaluation_id,
        source_raw_results_sha256=source_summary.raw_results_sha256,
        training_run_id=training_result.run_id,
        training_seed=cast(Literal[42, 20260727], training_result.seed),
        mixture_version=replay_config.source_mixture_version,
        mixture_manifest_sha256=replay_config.source_mixture_manifest_sha256,
        model_export_sha256=cast(str, training_result.export_sha256),
        model_revision=replay_config.model_revision,
        attention_architecture=replay_config.attention_architecture,
        suite_version=replay_config.source_suite_version,
        evaluation_config_sha256=replay_config.source_evaluation_config_sha256,
        git_commit=git_commit,
        git_dirty=False,
        physical_gpu_index=physical_gpu_index,
        gpu_name=torch.cuda.get_device_name(device),
        duration_seconds=duration,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        source_format_valid_items=source_summary.thinking.format_valid_items,
        source_correct_items=source_summary.thinking.final_answer_correct_items,
        original_failed_items=200 - source_summary.thinking.format_valid_items,
        replayed_batches=len(offsets),
        replayed_items=len(replay_items),
        replay_896_exact_items=len(replay_items),
        replay_1536_prefix_exact_items=len(replay_items),
        thresholds=_threshold_summaries(
            source_summary=source_summary,
            replay_items=replay_items,
            thresholds=replay_config.score_thresholds,
        ),
        raw_results_sha256=_sha256_file(raw_path),
    )
    _atomic_json(output_dir / "summary.json", summary.to_dict())
    return summary


def load_m5_r2_summary(path: Path) -> M5R2ReplaySummary:
    """Load one strict public R2 Seed summary."""

    try:
        return M5R2ReplaySummary.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise M5R2DiagnosticError("M5 R2 replay Summary is invalid") from exc


def select_m5_r2_diagnostic(
    summaries: Sequence[M5R2ReplaySummary],
    *,
    summary_sha256: Sequence[str],
    format_gate_basis_points: int = 9900,
    formal_candidate_max_new_tokens: int = 1280,
) -> M5R2DiagnosticDecision:
    """Select the smallest common passing limit for two exact Seed replays."""

    if (
        len(summaries) != 2
        or len(summary_sha256) != 2
        or format_gate_basis_points != 9900
        or formal_candidate_max_new_tokens != 1280
    ):
        raise M5R2DiagnosticError("M5 R2 decision requires the frozen two-Seed policy")
    ordered_pairs = sorted(
        zip(summaries, summary_sha256, strict=True),
        key=lambda item: item[0].training_seed,
    )
    ordered = tuple(item[0] for item in ordered_pairs)
    ordered_hashes = tuple(item[1] for item in ordered_pairs)
    if (
        tuple(item.training_seed for item in ordered) != (42, 20260727)
        or len({item.source_evaluation_id for item in ordered}) != 2
        or len(
            {
                (
                    item.diagnostic_version,
                    item.diagnostic_config_sha256,
                    item.model_revision,
                    item.attention_architecture,
                    item.suite_version,
                    item.evaluation_config_sha256,
                )
                for item in ordered
            }
        )
        != 1
        or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in ordered_hashes)
    ):
        raise M5R2DiagnosticError("M5 R2 two-Seed summaries are incompatible")
    selected: M5R2Threshold | None = None
    selected_index = 2
    for index, threshold in enumerate((1024, 1280, 1536)):
        if all(
            item.thresholds[index].projected_format_basis_points >= format_gate_basis_points
            for item in ordered
        ):
            selected = cast(M5R2Threshold, threshold)
            selected_index = index
            break
    if selected is None:
        status: Literal[
            "supports_eval_protocol_revision",
            "tradeoff_review_required",
            "length_ceiling_insufficient",
        ] = "length_ceiling_insufficient"
        reason: Literal[
            "both_seeds_pass_at_or_below_conditionally_approved_limit",
            "both_seeds_pass_only_at_tradeoff_limit",
            "at_least_one_seed_fails_at_diagnostic_limit",
        ] = "at_least_one_seed_fails_at_diagnostic_limit"
    elif selected <= formal_candidate_max_new_tokens:
        status = "supports_eval_protocol_revision"
        reason = "both_seeds_pass_at_or_below_conditionally_approved_limit"
    else:
        status = "tradeoff_review_required"
        reason = "both_seeds_pass_only_at_tradeoff_limit"
    projections = cast(
        tuple[M5R2SeedProjection, M5R2SeedProjection],
        tuple(
            M5R2SeedProjection(
                training_seed=item.training_seed,
                source_evaluation_id=item.source_evaluation_id,
                projected_format_basis_points=(
                    item.thresholds[selected_index].projected_format_basis_points
                ),
                projected_score_basis_points=(
                    item.thresholds[selected_index].projected_score_basis_points
                ),
                unresolved_format_items=(item.thresholds[selected_index].unresolved_format_items),
            )
            for item in ordered
        ),
    )
    return M5R2DiagnosticDecision(
        status=status,
        diagnostic_version="m5-r2-length-replay-v1",
        summary_sha256=cast(tuple[str, str], ordered_hashes),
        training_seeds=(42, 20260727),
        selected_max_new_tokens=selected,
        projections=projections,
        formal_protocol_changed=False,
        decision_reason=reason,
    )
