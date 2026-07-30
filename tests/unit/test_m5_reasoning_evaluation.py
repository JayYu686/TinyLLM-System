from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from tinyllm.data import generate_reasoning_dev_tasks, load_m5_reasoning_data_config
from tinyllm.evaluation.m5_reasoning import (
    M5ReasoningEvaluationError,
    evaluate_m5_format_repair_gate,
    load_m5_reasoning_evaluation_config,
    score_m5_response,
    select_m5_ablation,
    summarize_m5_mode,
)
from tinyllm.evaluation.m5_reasoning_schema import (
    M5ModeSummary,
    M5ReasoningEvaluationSummary,
)
from tinyllm.training.m5_ablation_schema import M5AblationRunResult


def _mode_summary(
    mode: Literal["thinking", "nonthinking"], *, score: int, format_score: int = 10_000
) -> M5ModeSummary:
    return M5ModeSummary(
        mode=mode,
        evaluated_items=200,
        format_valid_items=format_score // 50,
        final_json_valid_items=score // 50,
        final_answer_correct_items=score // 50,
        visible_reasoning_leakage_items=0,
        format_valid_basis_points=format_score,
        final_answer_score_basis_points=score,
        generated_tokens=100,
        length_limited_items=0,
    )


def _summary(
    *,
    kind: Literal["base", "ablation_candidate"],
    identity: str,
    nonthinking: int,
    thinking: int,
    thinking_format: int = 10_000,
    ratio: Literal[0, 3000, 5000] | None = None,
    seed: int | None = None,
) -> M5ReasoningEvaluationSummary:
    return M5ReasoningEvaluationSummary(
        status="succeeded",
        evaluation_id=f"eval-{identity}",
        model_kind=kind,
        training_run_id=f"run-{identity}" if kind == "ablation_candidate" else None,
        training_seed=seed,
        thinking_fraction_basis_points=ratio,
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        attention_architecture="gqa",
        suite_version="m5-reasoning-dev-v1-3eb153c2",
        config_sha256="a" * 64,
        git_commit="b" * 40,
        git_dirty=False,
        physical_gpu_index=9,
        gpu_name="NVIDIA GeForce RTX 3090",
        duration_seconds=1.0,
        peak_allocated_bytes=10,
        peak_reserved_bytes=20,
        thinking=_mode_summary("thinking", score=thinking, format_score=thinking_format),
        nonthinking=_mode_summary("nonthinking", score=nonthinking),
        raw_results_sha256="c" * 64,
    )


def _r1_training_result(*, identity: str, seed: int) -> M5AblationRunResult:
    return M5AblationRunResult(
        status="succeeded",
        mode="fresh",
        run_id=f"run-{identity}",
        config_sha256="d" * 64,
        git_commit="b" * 40,
        git_dirty=False,
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        attention_architecture="gqa",
        mixture_version="m5-format-repair-mixture-v1-a1b2c3d4",
        mixture_manifest_sha256="e" * 64,
        thinking_fraction_basis_points=3000,
        seed=seed,
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        global_step=10,
        supervised_tokens=1_000_000,
        sequence_cursor=100,
        initial_loss=2.0,
        final_loss=1.0,
        duration_seconds=1.0,
        peak_allocated_bytes=10,
        peak_reserved_bytes=20,
        latest_checkpoint="checkpoint-tokens-0001000000",
        export_sha256="f" * 64,
    )


def test_eval_config_freezes_m5_only_suite() -> None:
    config = load_m5_reasoning_evaluation_config(Path("configs/eval/m5_reasoning_dev.yaml"))

    assert config.expected_items == 200
    assert config.suite_version == "m5-reasoning-dev-v1-53ddf557"
    assert config.consume_m6_frozen_results is False
    assert config.generation.thinking_max_new_tokens == 896


def test_dual_mode_scoring_separates_trace_format_from_final_json() -> None:
    reasoning_config = load_m5_reasoning_data_config(Path("configs/data/m5_reasoning.yaml"))
    task = generate_reasoning_dev_tasks(reasoning_config)[0]
    thinking = score_m5_response(
        task,
        mode="thinking",
        response=f"<think>deterministic trace</think>\n\n{task.expected_answer_json}",
        prompt_tokens=10,
        generated_tokens=20,
        finish_reason="eos",
    )
    nonthinking = score_m5_response(
        task,
        mode="nonthinking",
        response=task.expected_answer_json,
        prompt_tokens=10,
        generated_tokens=5,
        finish_reason="eos",
    )
    leaked = score_m5_response(
        task,
        mode="nonthinking",
        response=f"<think>leak</think>{task.expected_answer_json}",
        prompt_tokens=10,
        generated_tokens=8,
        finish_reason="eos",
    )

    assert thinking.format_valid and thinking.final_answer_correct
    assert nonthinking.format_valid and nonthinking.final_answer_correct
    assert leaked.visible_reasoning_leakage and not leaked.format_valid


def test_mode_summary_requires_exactly_200_items() -> None:
    reasoning_config = load_m5_reasoning_data_config(Path("configs/data/m5_reasoning.yaml"))
    tasks = generate_reasoning_dev_tasks(reasoning_config)
    results = tuple(
        score_m5_response(
            task,
            mode="nonthinking",
            response=task.expected_answer_json,
            prompt_tokens=10,
            generated_tokens=5,
            finish_reason="eos",
        )
        for task in tasks
    )

    summary = summarize_m5_mode("nonthinking", results)

    assert summary.final_answer_score_basis_points == 10_000


def test_selection_applies_lower_ratio_tie_break_after_gates() -> None:
    base = _summary(kind="base", identity="base", nonthinking=5000, thinking=3000)
    candidates = (
        _summary(
            kind="ablation_candidate",
            identity="0-a",
            nonthinking=4900,
            thinking=4000,
            ratio=0,
            seed=1,
        ),
        _summary(
            kind="ablation_candidate",
            identity="0-b",
            nonthinking=4900,
            thinking=4000,
            ratio=0,
            seed=2,
        ),
        _summary(
            kind="ablation_candidate",
            identity="30-a",
            nonthinking=4800,
            thinking=4050,
            ratio=3000,
            seed=1,
        ),
        _summary(
            kind="ablation_candidate",
            identity="30-b",
            nonthinking=4800,
            thinking=4050,
            ratio=3000,
            seed=2,
        ),
        _summary(
            kind="ablation_candidate",
            identity="50-a",
            nonthinking=4900,
            thinking=5000,
            thinking_format=9800,
            ratio=5000,
            seed=1,
        ),
        _summary(
            kind="ablation_candidate",
            identity="50-b",
            nonthinking=4900,
            thinking=5000,
            thinking_format=9800,
            ratio=5000,
            seed=2,
        ),
    )

    selection = select_m5_ablation(base, candidates)

    assert selection.status == "selected"
    assert selection.selected_thinking_fraction_basis_points == 0
    assert selection.selection_reason == "lower_ratio_within_one_percentage_point"


def test_selection_rejects_candidate_from_superseded_dev_protocol() -> None:
    base = _summary(kind="base", identity="base", nonthinking=5000, thinking=3000)
    candidate = _summary(
        kind="ablation_candidate",
        identity="candidate",
        nonthinking=5000,
        thinking=4000,
        ratio=0,
        seed=1,
    ).model_copy(update={"suite_version": "m5-reasoning-dev-v1-53ddf557"})

    with pytest.raises(M5ReasoningEvaluationError, match="differs from Base"):
        select_m5_ablation(base, (candidate,))


def test_format_repair_gate_passes_only_both_fixed_seeds_and_unchanged_gates() -> None:
    base = _summary(kind="base", identity="base", nonthinking=5000, thinking=3000).model_copy(
        update={"suite_version": "m5-reasoning-dev-v1-53ddf557"}
    )
    candidates = (
        _summary(
            kind="ablation_candidate",
            identity="r1-a",
            nonthinking=4900,
            thinking=9300,
            thinking_format=9950,
            ratio=3000,
            seed=42,
        ).model_copy(update={"suite_version": "m5-reasoning-dev-v1-53ddf557"}),
        _summary(
            kind="ablation_candidate",
            identity="r1-b",
            nonthinking=4800,
            thinking=9400,
            thinking_format=9900,
            ratio=3000,
            seed=20260727,
        ).model_copy(update={"suite_version": "m5-reasoning-dev-v1-53ddf557"}),
    )

    result = evaluate_m5_format_repair_gate(
        base,
        candidates,
        (
            _r1_training_result(identity="r1-a", seed=42),
            _r1_training_result(identity="r1-b", seed=20260727),
        ),
    )

    assert result.status == "passed"
    assert result.training_seeds == (42, 20260727)
    assert result.nonthinking_regression_gate_passed
    assert result.thinking_format_gate_passed
    assert result.gate_reason == "all_preregistered_gates_passed"


def test_format_repair_gate_rejects_without_lowering_ninety_nine_percent() -> None:
    base = _summary(kind="base", identity="base", nonthinking=5000, thinking=3000).model_copy(
        update={"suite_version": "m5-reasoning-dev-v1-53ddf557"}
    )
    candidates = (
        _summary(
            kind="ablation_candidate",
            identity="r1-a",
            nonthinking=5000,
            thinking=9300,
            thinking_format=9850,
            ratio=3000,
            seed=42,
        ).model_copy(update={"suite_version": "m5-reasoning-dev-v1-53ddf557"}),
        _summary(
            kind="ablation_candidate",
            identity="r1-b",
            nonthinking=5000,
            thinking=9400,
            thinking_format=10_000,
            ratio=3000,
            seed=20260727,
        ).model_copy(update={"suite_version": "m5-reasoning-dev-v1-53ddf557"}),
    )

    result = evaluate_m5_format_repair_gate(
        base,
        candidates,
        (
            _r1_training_result(identity="r1-a", seed=42),
            _r1_training_result(identity="r1-b", seed=20260727),
        ),
    )

    assert result.status == "rejected"
    assert not result.thinking_format_gate_passed
    assert result.gate_reason == "thinking_format_gate_failed"
