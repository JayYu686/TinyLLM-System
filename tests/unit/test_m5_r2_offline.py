from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest

from tinyllm.data import generate_reasoning_dev_tasks, load_m5_reasoning_data_config
from tinyllm.evaluation.m5_r2_offline import (
    analyze_m5_r2_failures,
    build_repetition_distribution,
)
from tinyllm.evaluation.m5_reasoning import score_m5_response, summarize_m5_mode
from tinyllm.evaluation.m5_reasoning_schema import M5ReasoningEvaluationSummary

REASONING_CONFIG = Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml")


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def _write_r1_fixture(
    root: Path,
    *,
    seed: Literal[42, 20260727],
) -> Path:
    data_config = load_m5_reasoning_data_config(REASONING_CONFIG)
    tasks = generate_reasoning_dev_tasks(data_config)
    results = []
    for index, task in enumerate(tasks):
        response = (
            "<think>unfinished\nunfinished"
            if index == 0
            else f"<think>brief reasoning</think>\n{task.expected_answer_json}"
        )
        results.append(
            score_m5_response(
                task,
                mode="thinking",
                response=response,
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
    directory = root / f"r1-{seed}"
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
        evaluation_id=f"r1-evaluation-{seed}",
        model_kind="ablation_candidate",
        training_run_id=f"r1-run-{seed}",
        training_seed=seed,
        thinking_fraction_basis_points=3000,
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        attention_architecture="gqa",
        suite_version="m5-reasoning-dev-v1-53ddf557",
        config_sha256=("3735a34e70c06059fbc09f62d02fabc296fd97e79a45d60f3d00dede21108d51"),
        git_commit="a" * 40,
        git_dirty=False,
        physical_gpu_index=5,
        gpu_name="Synthetic GPU",
        duration_seconds=1.0,
        peak_allocated_bytes=1,
        peak_reserved_bytes=1,
        thinking=summarize_m5_mode("thinking", thinking),
        nonthinking=summarize_m5_mode("nonthinking", nonthinking),
        raw_results_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    )
    (directory / "summary.json").write_text(
        json.dumps(summary.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


def test_repetition_distribution_reports_duplicate_ngrams_and_lines() -> None:
    distribution = build_repetition_distribution(
        generated_tokens=(16,),
        token_sequences=((1, 2, 3, 4, 5, 6, 7, 8) * 2,),
        responses=("same\nsame\nother",),
    )

    assert distribution.generated_tokens_p50 == 16
    assert distribution.unique_token_ratio_mean_basis_points == 5000
    assert distribution.repeated_8gram_ratio_mean_basis_points == 1111
    assert distribution.max_identical_line_hash_repetitions == 2


def test_offline_analysis_validates_sources_and_redacts_private_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directories = (
        _write_r1_fixture(tmp_path, seed=42),
        _write_r1_fixture(tmp_path, seed=20260727),
    )
    raw_by_seed = {
        seed: hashlib.sha256((directory / "results.jsonl").read_bytes()).hexdigest()
        for seed, directory in zip((42, 20260727), directories, strict=True)
    }
    monkeypatch.setattr(
        "tinyllm.evaluation.m5_r2_offline.expected_m5_r2_source_identity",
        lambda _config, seed: (
            f"r1-run-{seed}",
            f"r1-evaluation-{seed}",
            raw_by_seed[seed],
            "f" * 64,
        ),
    )
    result = analyze_m5_r2_failures(
        evaluation_directories=directories,
        reasoning_config_path=REASONING_CONFIG,
        replay_config_path=Path("configs/eval/m5_r2_length_replay.yaml"),
        tokenizer=_CharacterTokenizer(),
    )

    assert result.training_seeds == (42, 20260727)
    assert all(item.invalid_format_items == 1 for item in result.slices)
    assert all(item.finish_reason_counts == {"eos": 0, "length": 1} for item in result.slices)
    public = result.model_dump_json()
    assert "unfinished" not in public
    assert "response" not in public
    assert "item_id" not in public
    assert "prompt" not in public
