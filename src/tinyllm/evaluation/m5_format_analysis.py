"""Deterministic private-to-public analysis of M5.2 Thinking format failures."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from tinyllm.data import (
    generate_reasoning_dev_tasks,
    load_m5_reasoning_data_config,
    parse_teacher_output,
    reasoning_config_sha256,
)
from tinyllm.data.reasoning_schema import (
    REASONING_LANGUAGES,
    REASONING_TASK_FAMILIES,
    ReasoningLanguage,
    ReasoningTaskFamily,
    content_sha256,
)
from tinyllm.evaluation.m5_reasoning import (
    M5ReasoningEvaluationError,
    summarize_m5_mode,
)
from tinyllm.evaluation.m5_reasoning_schema import (
    M5_FORMAT_FAILURE_CATEGORIES,
    M5FormatFailureAnalysis,
    M5FormatFailureCategory,
    M5FormatFailureSlice,
    M5ReasoningEvaluationSummary,
    M5ReasoningItemResult,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_results(path: Path) -> tuple[M5ReasoningItemResult, ...]:
    results: list[M5ReasoningItemResult] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                results.append(M5ReasoningItemResult.model_validate_json(line))
    except (OSError, ValidationError) as exc:
        raise M5ReasoningEvaluationError("private M5 evaluation results cannot be parsed") from exc
    return tuple(results)


def _classify_failure(item: M5ReasoningItemResult) -> M5FormatFailureCategory:
    open_count = item.response.count("<think>")
    close_count = item.response.count("</think>")
    if open_count == 1 and close_count == 0:
        return (
            "length_open_without_close"
            if item.finish_reason == "length"
            else "eos_open_without_close"
        )
    if open_count == 0:
        return "missing_open_tag"
    _, reason = parse_teacher_output(item.response)
    mapping: dict[str, M5FormatFailureCategory] = {
        "empty_final_answer": "empty_final_answer",
        "empty_reasoning": "empty_reasoning",
        "multiple_think_blocks": "multiple_think_blocks",
        "nested_think_tag": "nested_think_tag",
    }
    return mapping.get(str(reason), "other_parse_failure")


def _validate_private_result_set(
    *,
    summary: M5ReasoningEvaluationSummary,
    results: Sequence[M5ReasoningItemResult],
    expected_prompt_hashes: dict[str, str],
) -> None:
    if len(results) != 400:
        raise M5ReasoningEvaluationError("M5 format analysis requires 400 private results")
    identities = {(item.mode, item.item_id) for item in results}
    expected_identities = {
        (mode, item_id)
        for mode in ("thinking", "nonthinking")
        for item_id in expected_prompt_hashes
    }
    if identities != expected_identities or len(identities) != len(results):
        raise M5ReasoningEvaluationError("M5 private result item identity is incomplete")
    for item in results:
        if item.prompt_sha256 != expected_prompt_hashes[item.item_id]:
            raise M5ReasoningEvaluationError("M5 private result Prompt identity drifted")
        if item.response_sha256 != hashlib.sha256(item.response.encode()).hexdigest():
            raise M5ReasoningEvaluationError("M5 private result response hash is invalid")
    thinking = tuple(item for item in results if item.mode == "thinking")
    nonthinking = tuple(item for item in results if item.mode == "nonthinking")
    if (
        summarize_m5_mode("thinking", thinking) != summary.thinking
        or summarize_m5_mode("nonthinking", nonthinking) != summary.nonthinking
    ):
        raise M5ReasoningEvaluationError("M5 private results do not reproduce their summary")


def analyze_m5_format_failures(
    *,
    evaluation_directories: Sequence[Path],
    reasoning_config_path: Path,
) -> M5FormatFailureAnalysis:
    """Analyze four private Candidate result files without returning response content."""

    if len(evaluation_directories) != 4:
        raise M5ReasoningEvaluationError("M5 R1 analysis requires four 30/50 Candidate runs")
    reasoning_config = load_m5_reasoning_data_config(reasoning_config_path)
    tasks = generate_reasoning_dev_tasks(reasoning_config)
    expected_prompt_hashes = {item.id: item.prompt_sha256 for item in tasks}
    task_by_id = {item.id: item for item in tasks}
    slices: list[M5FormatFailureSlice] = []
    input_identities: list[dict[str, object]] = []
    protocol: tuple[str, str, str] | None = None
    for directory in evaluation_directories:
        try:
            summary = M5ReasoningEvaluationSummary.model_validate_json(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise M5ReasoningEvaluationError("M5 format analysis summary cannot be parsed") from exc
        if (
            summary.model_kind != "ablation_candidate"
            or summary.thinking_fraction_basis_points not in {3000, 5000}
            or summary.training_seed not in {42, 20260727}
            or summary.training_run_id is None
            or summary.suite_version != "m5-reasoning-dev-v1-53ddf557"
        ):
            raise M5ReasoningEvaluationError("M5 format analysis received the wrong run identity")
        current_protocol = (
            summary.suite_version,
            summary.config_sha256,
            summary.model_revision,
        )
        if protocol is None:
            protocol = current_protocol
        elif current_protocol != protocol:
            raise M5ReasoningEvaluationError("M5 format analysis protocols differ")
        raw_path = directory / "results.jsonl"
        if _sha256_file(raw_path) != summary.raw_results_sha256:
            raise M5ReasoningEvaluationError("M5 private result file hash differs from summary")
        results = _load_results(raw_path)
        _validate_private_result_set(
            summary=summary,
            results=results,
            expected_prompt_hashes=expected_prompt_hashes,
        )
        failed = tuple(
            item for item in results if item.mode == "thinking" and not item.format_valid
        )
        if len(failed) != 200 - summary.thinking.format_valid_items or not failed:
            raise M5ReasoningEvaluationError("M5 Thinking failure count differs from summary")
        category_counts: Counter[M5FormatFailureCategory] = Counter(
            _classify_failure(item) for item in failed
        )
        family_counts: Counter[ReasoningTaskFamily] = Counter(
            task_by_id[item.item_id].task_family for item in failed
        )
        language_counts: Counter[ReasoningLanguage] = Counter(
            task_by_id[item.item_id].language for item in failed
        )
        generated_tokens = tuple(item.generated_tokens for item in failed)
        slices.append(
            M5FormatFailureSlice(
                thinking_fraction_basis_points=cast(
                    Literal[3000, 5000],
                    summary.thinking_fraction_basis_points,
                ),
                training_seed=cast(Literal[42, 20260727], summary.training_seed),
                training_run_id=summary.training_run_id,
                evaluation_id=summary.evaluation_id,
                raw_results_sha256=summary.raw_results_sha256,
                evaluated_thinking_items=200,
                invalid_format_items=len(failed),
                category_counts={key: category_counts[key] for key in M5_FORMAT_FAILURE_CATEGORIES},
                task_family_counts={key: family_counts[key] for key in REASONING_TASK_FAMILIES},
                language_counts={key: language_counts[key] for key in REASONING_LANGUAGES},
                generated_tokens_min=min(generated_tokens),
                generated_tokens_max=max(generated_tokens),
                generated_tokens_total=sum(generated_tokens),
            )
        )
        input_identities.append(
            {
                "evaluation_id": summary.evaluation_id,
                "raw_results_sha256": summary.raw_results_sha256,
                "thinking_fraction_basis_points": summary.thinking_fraction_basis_points,
                "training_run_id": summary.training_run_id,
                "training_seed": summary.training_seed,
            }
        )
    ordered = tuple(
        sorted(
            slices,
            key=lambda item: (item.thinking_fraction_basis_points, item.training_seed),
        )
    )
    identities = tuple(
        (item.thinking_fraction_basis_points, item.training_seed) for item in ordered
    )
    if identities != (
        (3000, 42),
        (3000, 20260727),
        (5000, 42),
        (5000, 20260727),
    ):
        raise M5ReasoningEvaluationError("M5 format analysis requires both Seeds for 30/50")
    if protocol is None:
        raise AssertionError("four M5 evaluation directories produced no protocol")
    input_identities.sort(
        key=lambda item: (
            cast(int, item["thinking_fraction_basis_points"]),
            cast(int, item["training_seed"]),
        )
    )
    length_open = sum(item.category_counts["length_open_without_close"] for item in ordered)
    eos_open = sum(item.category_counts["eos_open_without_close"] for item in ordered)
    return M5FormatFailureAnalysis(
        suite_version=cast(
            Literal["m5-reasoning-dev-v1-53ddf557"],
            protocol[0],
        ),
        evaluation_config_sha256=protocol[1],
        model_revision=cast(
            Literal["c1899de289a04d12100db370d81485cdf75e47ca"],
            protocol[2],
        ),
        input_set_sha256=content_sha256(
            {
                "reasoning_config_sha256": reasoning_config_sha256(reasoning_config),
                "sources": input_identities,
            }
        ),
        slices=cast(
            tuple[
                M5FormatFailureSlice,
                M5FormatFailureSlice,
                M5FormatFailureSlice,
                M5FormatFailureSlice,
            ],
            ordered,
        ),
        total_invalid_format_items=sum(item.invalid_format_items for item in ordered),
        total_length_open_without_close_items=length_open,
        total_eos_open_without_close_items=eos_open,
        total_open_without_close_items=length_open + eos_open,
    )
