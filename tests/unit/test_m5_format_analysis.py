from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest

from tinyllm.data import generate_reasoning_dev_tasks, load_m5_reasoning_data_config
from tinyllm.evaluation.m5_format_analysis import analyze_m5_format_failures
from tinyllm.evaluation.m5_reasoning import (
    M5ReasoningEvaluationError,
    score_m5_response,
    summarize_m5_mode,
)
from tinyllm.evaluation.m5_reasoning_schema import M5ReasoningEvaluationSummary

REASONING_CONFIG = Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_candidate(
    root: Path,
    *,
    ratio: Literal[3000, 5000],
    seed: Literal[42, 20260727],
) -> Path:
    config = load_m5_reasoning_data_config(REASONING_CONFIG)
    tasks = generate_reasoning_dev_tasks(config)
    results = []
    for index, task in enumerate(tasks):
        thinking_response = (
            "<think>unfinished"
            if index == 0
            else f"<think>brief verified reasoning</think>\n{task.expected_answer_json}"
        )
        results.append(
            score_m5_response(
                task,
                mode="thinking",
                response=thinking_response,
                prompt_tokens=20,
                generated_tokens=896 if index == 0 else 32,
                finish_reason="length" if index == 0 else "eos",
            )
        )
        results.append(
            score_m5_response(
                task,
                mode="nonthinking",
                response=task.expected_answer_json,
                prompt_tokens=20,
                generated_tokens=8,
                finish_reason="eos",
            )
        )
    directory = root / f"candidate-{ratio}-{seed}"
    directory.mkdir()
    raw_path = directory / "results.jsonl"
    raw_path.write_text(
        "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in results
        ),
        encoding="utf-8",
    )
    thinking = tuple(item for item in results if item.mode == "thinking")
    nonthinking = tuple(item for item in results if item.mode == "nonthinking")
    summary = M5ReasoningEvaluationSummary(
        status="succeeded",
        evaluation_id=f"evaluation-{ratio}-{seed}",
        model_kind="ablation_candidate",
        training_run_id=f"training-{ratio}-{seed}",
        training_seed=seed,
        thinking_fraction_basis_points=ratio,
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        attention_architecture="gqa",
        suite_version="m5-reasoning-dev-v1-53ddf557",
        config_sha256="a" * 64,
        git_commit="b" * 40,
        git_dirty=False,
        physical_gpu_index=4,
        gpu_name="Synthetic GPU",
        duration_seconds=1.0,
        peak_allocated_bytes=1,
        peak_reserved_bytes=1,
        thinking=summarize_m5_mode("thinking", thinking),
        nonthinking=summarize_m5_mode("nonthinking", nonthinking),
        raw_results_sha256=_sha256_file(raw_path),
    )
    (directory / "summary.json").write_text(
        json.dumps(summary.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


def _four_candidates(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        _write_candidate(root, ratio=3000, seed=42),
        _write_candidate(root, ratio=3000, seed=20260727),
        _write_candidate(root, ratio=5000, seed=42),
        _write_candidate(root, ratio=5000, seed=20260727),
    )


def test_m5_format_analysis_redacts_responses_and_aggregates_failures(
    tmp_path: Path,
) -> None:
    analysis = analyze_m5_format_failures(
        evaluation_directories=_four_candidates(tmp_path),
        reasoning_config_path=REASONING_CONFIG,
    )

    assert analysis.total_invalid_format_items == 4
    assert analysis.total_length_open_without_close_items == 4
    assert analysis.total_eos_open_without_close_items == 0
    assert analysis.total_open_without_close_items == 4
    assert all(item.invalid_format_items == 1 for item in analysis.slices)
    assert all(item.category_counts["length_open_without_close"] == 1 for item in analysis.slices)
    public_text = analysis.model_dump_json()
    assert "unfinished" not in public_text
    assert "response" not in public_text
    assert "item_id" not in public_text


def test_m5_format_analysis_rejects_raw_result_hash_drift(tmp_path: Path) -> None:
    directories = _four_candidates(tmp_path)
    with (directories[0] / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(
        M5ReasoningEvaluationError,
        match="result file hash differs",
    ):
        analyze_m5_format_failures(
            evaluation_directories=directories,
            reasoning_config_path=REASONING_CONFIG,
        )
