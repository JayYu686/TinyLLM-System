"""Content-free offline diagnostics for M5.2-R2 source evaluations."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from tinyllm.data import generate_reasoning_dev_tasks, load_m5_reasoning_data_config
from tinyllm.data.reasoning_schema import (
    REASONING_LANGUAGES,
    REASONING_TASK_FAMILIES,
    ReasoningLanguage,
    ReasoningTaskFamily,
)
from tinyllm.evaluation.m5_format_analysis import (
    _load_results,
    _sha256_file,
    _validate_private_result_set,
)
from tinyllm.evaluation.m5_r2_diagnostic import (
    M5R2DiagnosticError,
    expected_m5_r2_source_identity,
    load_m5_r2_replay_config,
)
from tinyllm.evaluation.m5_r2_schema import (
    M5R2OfflineAnalysis,
    M5R2OfflineSeedAnalysis,
    M5R2RepetitionDistribution,
)
from tinyllm.evaluation.m5_reasoning_schema import M5ReasoningEvaluationSummary


def build_repetition_distribution(
    *,
    generated_tokens: Sequence[int],
    token_sequences: Sequence[Sequence[int]],
    responses: Sequence[str],
) -> M5R2RepetitionDistribution:
    """Aggregate length, token diversity, 8-gram, and line repetition metrics."""

    if (
        not generated_tokens
        or len(generated_tokens) != len(token_sequences)
        or len(generated_tokens) != len(responses)
        or any(not sequence for sequence in token_sequences)
    ):
        raise M5R2DiagnosticError("M5 R2 repetition inputs are empty or misaligned")
    unique_ratios: list[int] = []
    repeated_8gram_ratios: list[int] = []
    maximum_line_repeat = 1
    for tokens, response in zip(token_sequences, responses, strict=True):
        unique_ratios.append(round(len(set(tokens)) * 10_000 / len(tokens)))
        windows = tuple(tuple(tokens[index : index + 8]) for index in range(len(tokens) - 7))
        repeated_8gram_ratios.append(
            round((len(windows) - len(set(windows))) * 10_000 / len(windows)) if windows else 0
        )
        line_hashes = [
            hashlib.sha256(line.strip().encode()).hexdigest()
            for line in response.splitlines()
            if line.strip()
        ]
        if line_hashes:
            maximum_line_repeat = max(
                maximum_line_repeat,
                max(Counter(line_hashes).values()),
            )
    ordered_lengths = sorted(generated_tokens)
    p90_index = math.ceil(0.9 * len(ordered_lengths)) - 1
    return M5R2RepetitionDistribution(
        items=len(generated_tokens),
        generated_tokens_min=ordered_lengths[0],
        generated_tokens_p50=float(statistics.median(ordered_lengths)),
        generated_tokens_p90=ordered_lengths[p90_index],
        generated_tokens_max=ordered_lengths[-1],
        unique_token_ratio_mean_basis_points=round(sum(unique_ratios) / len(unique_ratios)),
        repeated_8gram_ratio_mean_basis_points=round(
            sum(repeated_8gram_ratios) / len(repeated_8gram_ratios)
        ),
        max_identical_line_hash_repetitions=maximum_line_repeat,
    )


def _load_summary(path: Path) -> M5ReasoningEvaluationSummary:
    try:
        return M5ReasoningEvaluationSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise M5R2DiagnosticError("M5 R2 offline source Summary is invalid") from exc


def analyze_m5_r2_failures(
    *,
    evaluation_directories: Sequence[Path],
    reasoning_config_path: Path,
    replay_config_path: Path,
    tokenizer: Any,
) -> M5R2OfflineAnalysis:
    """Analyze both R1 evaluations without returning task-level private content."""

    if len(evaluation_directories) != 2:
        raise M5R2DiagnosticError("M5 R2 offline analysis requires two R1 evaluations")
    replay_config = load_m5_r2_replay_config(replay_config_path)
    data_config = load_m5_reasoning_data_config(reasoning_config_path)
    tasks = generate_reasoning_dev_tasks(data_config)
    expected_prompt_hashes = {item.id: item.prompt_sha256 for item in tasks}
    task_by_id = {item.id: item for item in tasks}
    slices: list[M5R2OfflineSeedAnalysis] = []
    for directory in evaluation_directories:
        summary = _load_summary(directory / "summary.json")
        raw_path = directory / "results.jsonl"
        if _sha256_file(raw_path) != summary.raw_results_sha256:
            raise M5R2DiagnosticError("M5 R2 offline Raw Result hash differs from Summary")
        results = _load_results(raw_path)
        _validate_private_result_set(
            summary=summary,
            results=results,
            expected_prompt_hashes=expected_prompt_hashes,
        )
        if (
            summary.model_kind != "ablation_candidate"
            or summary.training_seed not in {42, 20260727}
            or summary.thinking_fraction_basis_points != 3000
            or summary.suite_version != "m5-reasoning-dev-v1-53ddf557"
            or summary.config_sha256
            != "3735a34e70c06059fbc09f62d02fabc296fd97e79a45d60f3d00dede21108d51"
        ):
            raise M5R2DiagnosticError("M5 R2 offline source identity differs")
        expected_run, expected_evaluation, expected_raw, _ = expected_m5_r2_source_identity(
            replay_config,
            summary.training_seed,
        )
        if (
            summary.training_run_id != expected_run
            or summary.evaluation_id != expected_evaluation
            or summary.raw_results_sha256 != expected_raw
        ):
            raise M5R2DiagnosticError("M5 R2 offline source is not the frozen R1 pair")
        failures = tuple(
            item for item in results if item.mode == "thinking" and not item.format_valid
        )
        if (
            len(failures) != 200 - summary.thinking.format_valid_items
            or not failures
            or any(
                item.generated_tokens != 896
                or item.finish_reason != "length"
                or item.response.count("<think>") != 1
                or item.response.count("</think>") != 0
                for item in failures
            )
        ):
            raise M5R2DiagnosticError("M5 R2 offline failures differ from the R1 finding")
        failed_families = {task_by_id[item.item_id].task_family for item in failures}
        matched_valid = tuple(
            item
            for item in results
            if item.mode == "thinking"
            and item.format_valid
            and task_by_id[item.item_id].task_family in failed_families
        )
        if not matched_valid:
            raise M5R2DiagnosticError("M5 R2 offline analysis has no matched valid controls")

        def distribution(
            items: Sequence[Any],
        ) -> M5R2RepetitionDistribution:
            return build_repetition_distribution(
                generated_tokens=tuple(int(item.generated_tokens) for item in items),
                token_sequences=tuple(
                    tuple(tokenizer.encode(item.response, add_special_tokens=False))
                    for item in items
                ),
                responses=tuple(str(item.response) for item in items),
            )

        family_counts: Counter[ReasoningTaskFamily] = Counter(
            task_by_id[item.item_id].task_family for item in failures
        )
        language_counts: Counter[ReasoningLanguage] = Counter(
            task_by_id[item.item_id].language for item in failures
        )
        finish_counts: Counter[Literal["eos", "length"]] = Counter(
            item.finish_reason for item in failures
        )
        slices.append(
            M5R2OfflineSeedAnalysis(
                training_seed=cast(Literal[42, 20260727], summary.training_seed),
                source_evaluation_id=summary.evaluation_id,
                source_raw_results_sha256=summary.raw_results_sha256,
                invalid_format_items=len(failures),
                task_family_counts={key: family_counts[key] for key in REASONING_TASK_FAMILIES},
                language_counts={key: language_counts[key] for key in REASONING_LANGUAGES},
                finish_reason_counts={
                    "eos": finish_counts["eos"],
                    "length": finish_counts["length"],
                },
                failure_distribution=distribution(failures),
                matched_valid_items=len(matched_valid),
                matched_valid_distribution=distribution(matched_valid),
            )
        )
    ordered = cast(
        tuple[M5R2OfflineSeedAnalysis, M5R2OfflineSeedAnalysis],
        tuple(sorted(slices, key=lambda item: item.training_seed)),
    )
    if tuple(item.training_seed for item in ordered) != (42, 20260727):
        raise M5R2DiagnosticError("M5 R2 offline analysis requires distinct fixed Seeds")
    return M5R2OfflineAnalysis(
        status="succeeded",
        analysis_version="m5-r2-offline-analysis-v1",
        suite_version=replay_config.source_suite_version,
        evaluation_config_sha256=replay_config.source_evaluation_config_sha256,
        tokenizer_revision=replay_config.model_revision,
        training_seeds=(42, 20260727),
        slices=ordered,
    )
