from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError

from tinyllm.data import generate_reasoning_dev_tasks, load_m5_reasoning_data_config
from tinyllm.evaluation.m5_r2_diagnostic import (
    M5R2DiagnosticError,
    expected_m5_r2_source_identity,
    load_m5_r2_replay_config,
    replay_batch_offsets,
    score_m5_r2_threshold,
    select_m5_r2_diagnostic,
    validate_m5_r2_replay_pair,
)
from tinyllm.evaluation.m5_r2_schema import (
    M5R2ReplayItemResult,
    M5R2ReplaySummary,
    M5R2ThresholdItemResult,
    M5R2ThresholdSummary,
)
from tinyllm.evaluation.m5_reasoning import score_m5_response

REASONING_CONFIG = Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml")
SHA256 = "a" * 64


class _FakeTokenizer:
    eos_token_id = 99

    def __init__(self, final_answer: str = '{"enabled":true}') -> None:
        self.final_answer = final_answer

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [2] if text == "</think>" else []

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        pieces = {
            1: "<think>brief verified reasoning",
            2: "</think>\n",
            3: self.final_answer,
            7: "still reasoning ",
            99: "",
        }
        return "".join(pieces[item] for item in token_ids)


def _threshold(
    max_new_tokens: Literal[1024, 1280, 1536],
    *,
    source_format_items: int,
    recovered: int,
    source_correct_items: int,
) -> M5R2ThresholdSummary:
    return M5R2ThresholdSummary(
        max_new_tokens=max_new_tokens,
        original_failed_items=200 - source_format_items,
        recovered_format_items=recovered,
        recovered_final_json_items=recovered,
        recovered_correct_items=recovered,
        unresolved_format_items=200 - source_format_items - recovered,
        projected_format_valid_items=source_format_items + recovered,
        projected_format_basis_points=(source_format_items + recovered) * 50,
        projected_correct_items=source_correct_items + recovered,
        projected_score_basis_points=(source_correct_items + recovered) * 50,
        closing_tag_end_token_min=max_new_tokens - 10 if recovered else None,
        closing_tag_end_token_max=max_new_tokens - 1 if recovered else None,
    )


def _summary(
    seed: Literal[42, 20260727],
    projected_format_items: tuple[int, int, int],
) -> M5R2ReplaySummary:
    source_format = 189 if seed == 42 else 187
    source_correct = 186
    recovered = tuple(value - source_format for value in projected_format_items)
    replayed_batches = 3 if seed == 42 else 4
    return M5R2ReplaySummary(
        status="succeeded",
        diagnostic_id=f"r2-seed-{seed}",
        diagnostic_version="m5-r2-length-replay-v1",
        diagnostic_config_sha256="b" * 64,
        source_evaluation_id=f"source-{seed}",
        source_raw_results_sha256="c" * 64,
        training_run_id=f"run-{seed}",
        training_seed=seed,
        mixture_version="m5-format-repair-mixture-v1-1396b60b",
        mixture_manifest_sha256=(
            "2467b5dce0d909b865b73219d2f608bdbc9c6fcc1bb09b93c5ebea8a7b60bd0e"
        ),
        model_export_sha256="f" * 64,
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        attention_architecture="gqa",
        suite_version="m5-reasoning-dev-v1-53ddf557",
        evaluation_config_sha256=(
            "3735a34e70c06059fbc09f62d02fabc296fd97e79a45d60f3d00dede21108d51"
        ),
        git_commit="d" * 40,
        git_dirty=False,
        physical_gpu_index=5,
        gpu_name="Synthetic GPU",
        duration_seconds=1.0,
        peak_allocated_bytes=10,
        peak_reserved_bytes=20,
        source_format_valid_items=source_format,
        source_correct_items=source_correct,
        original_failed_items=200 - source_format,
        replayed_batches=replayed_batches,
        replayed_items=replayed_batches * 4,
        replay_896_exact_items=replayed_batches * 4,
        replay_1536_prefix_exact_items=replayed_batches * 4,
        thresholds=(
            _threshold(
                1024,
                source_format_items=source_format,
                recovered=recovered[0],
                source_correct_items=source_correct,
            ),
            _threshold(
                1280,
                source_format_items=source_format,
                recovered=recovered[1],
                source_correct_items=source_correct,
            ),
            _threshold(
                1536,
                source_format_items=source_format,
                recovered=recovered[2],
                source_correct_items=source_correct,
            ),
        ),
        raw_results_sha256="e" * 64,
    )


def test_r2_config_freezes_counterfactual_limits_and_gate() -> None:
    config = load_m5_r2_replay_config(Path("configs/eval/m5_r2_length_replay.yaml"))

    assert config.original_max_new_tokens == 896
    assert config.score_thresholds == (1024, 1280, 1536)
    assert config.formal_candidate_max_new_tokens == 1280
    assert config.format_gate_basis_points == 9900
    assert config.consume_m6_frozen_results is False
    assert expected_m5_r2_source_identity(config, 42) == (
        "20260727T075422Z-m5-format-repair-r1-seed42-7c825907-1c02",
        "20260727T090313Z-m5-reasoning-dev-ablation_candidate-3707f186",
        "87e478b92c3992fa4f1196d05e32686291f2f2a4b559777e559fbc80988bd50d",
        "46d5ca599bec9cfcad12a3ed001fcb59b4646232d873377d7554219d7ce34f45",
    )
    with pytest.raises(M5R2DiagnosticError, match="Seed"):
        expected_m5_r2_source_identity(config, 7)


def test_replay_batch_offsets_preserve_original_task_order() -> None:
    data_config = load_m5_reasoning_data_config(REASONING_CONFIG)
    tasks = generate_reasoning_dev_tasks(data_config)
    failed = score_m5_response(
        tasks[5],
        mode="thinking",
        response="<think>unfinished",
        prompt_tokens=10,
        generated_tokens=896,
        finish_reason="length",
    )

    assert replay_batch_offsets((failed,), tasks) == (4,)


def test_exact_replay_rejects_scoring_or_response_drift() -> None:
    data_config = load_m5_reasoning_data_config(REASONING_CONFIG)
    task = generate_reasoning_dev_tasks(data_config)[0]
    source = score_m5_response(
        task,
        mode="thinking",
        response="<think>unfinished",
        prompt_tokens=10,
        generated_tokens=896,
        finish_reason="length",
    )
    drifted = score_m5_response(
        task,
        mode="thinking",
        response="<think>different unfinished",
        prompt_tokens=10,
        generated_tokens=896,
        finish_reason="length",
    )

    with pytest.raises(M5R2DiagnosticError, match="896 output drifted"):
        validate_m5_r2_replay_pair(
            source=source,
            replay_896=drifted,
            replay_896_ids=(1, 2),
            replay_1536_ids=(1, 2, 3),
        )


def test_exact_replay_rejects_counterfactual_prefix_drift() -> None:
    data_config = load_m5_reasoning_data_config(REASONING_CONFIG)
    task = generate_reasoning_dev_tasks(data_config)[0]
    source = score_m5_response(
        task,
        mode="thinking",
        response="<think>unfinished",
        prompt_tokens=10,
        generated_tokens=896,
        finish_reason="length",
    )

    with pytest.raises(M5R2DiagnosticError, match="1536 prefix drifted"):
        validate_m5_r2_replay_pair(
            source=source,
            replay_896=source,
            replay_896_ids=(1, 2),
            replay_1536_ids=(1, 9, 3),
        )


def test_threshold_scoring_uses_original_parser_and_reports_close_position() -> None:
    data_config = load_m5_reasoning_data_config(REASONING_CONFIG)
    task = generate_reasoning_dev_tasks(data_config)[0]

    result = score_m5_r2_threshold(
        task=task,
        tokenizer=_FakeTokenizer(task.expected_answer_json),
        raw_ids=(1, 2, 3, 99),
        eos_ids={99},
        max_new_tokens=1024,
        prompt_tokens=10,
    )

    assert result.format_valid
    assert result.final_json_valid
    assert result.final_answer_correct
    assert result.closing_tag_end_token == 2
    assert result.generated_tokens == 4
    assert result.finish_reason == "eos"


def test_threshold_scoring_preserves_unclosed_length_failure() -> None:
    data_config = load_m5_reasoning_data_config(REASONING_CONFIG)
    task = generate_reasoning_dev_tasks(data_config)[0]

    result = score_m5_r2_threshold(
        task=task,
        tokenizer=_FakeTokenizer(),
        raw_ids=(7,) * 1024,
        eos_ids={99},
        max_new_tokens=1024,
        prompt_tokens=10,
    )

    assert not result.format_valid
    assert result.closing_tag_end_token is None
    assert result.generated_tokens == 1024
    assert result.finish_reason == "length"


def test_private_replay_schema_rejects_source_failure_without_thresholds() -> None:
    with pytest.raises(ValidationError, match="must be length-limited and rescored"):
        M5R2ReplayItemResult(
            item_id="m5-reasoning:dev:fixture",
            task_family="python",
            language="en",
            batch_offset=0,
            prompt_sha256=SHA256,
            source_response_sha256=SHA256,
            source_generated_tokens=896,
            source_finish_reason="length",
            source_format_valid=False,
            source_final_json_valid=False,
            source_final_answer_correct=False,
            replay_896_response_sha256=SHA256,
            replay_896_generated_tokens=896,
            replay_896_finish_reason="length",
            replay_896_exact=True,
            replay_1536_prefix_tokens_compared=896,
            replay_1536_prefix_exact=True,
            thresholds=None,
        )


def test_threshold_schema_allows_close_tag_with_invalid_final_envelope() -> None:
    response = "<think>trace</think>"
    result = M5R2ThresholdItemResult(
        max_new_tokens=1024,
        response=response,
        response_sha256=hashlib.sha256(response.encode()).hexdigest(),
        generated_tokens=10,
        finish_reason="eos",
        format_valid=False,
        final_json_valid=False,
        final_answer_correct=False,
        closing_tag_end_token=8,
    )

    assert result.closing_tag_end_token == 8
    assert not result.format_valid


@pytest.mark.parametrize(
    ("seed42_items", "seed2_items", "status", "selected"),
    (
        (
            (190, 198, 200),
            (190, 198, 200),
            "supports_eval_protocol_revision",
            1280,
        ),
        (
            (190, 195, 198),
            (190, 197, 199),
            "tradeoff_review_required",
            1536,
        ),
        (
            (190, 195, 197),
            (190, 197, 198),
            "length_ceiling_insufficient",
            None,
        ),
    ),
)
def test_two_seed_decision_uses_smallest_common_passing_limit(
    seed42_items: tuple[int, int, int],
    seed2_items: tuple[int, int, int],
    status: str,
    selected: int | None,
) -> None:
    decision = select_m5_r2_diagnostic(
        (_summary(20260727, seed2_items), _summary(42, seed42_items)),
        summary_sha256=("1" * 64, "2" * 64),
    )

    assert decision.status == status
    assert decision.selected_max_new_tokens == selected
    assert decision.training_seeds == (42, 20260727)


def test_two_seed_decision_rejects_duplicate_seed() -> None:
    with pytest.raises(M5R2DiagnosticError, match="incompatible"):
        select_m5_r2_diagnostic(
            (_summary(42, (198, 200, 200)), _summary(42, (198, 200, 200))),
            summary_sha256=("1" * 64, "2" * 64),
        )


def test_public_summary_contains_no_private_response_or_path_fields() -> None:
    public = _summary(42, (198, 199, 200)).model_dump_json()

    assert "response" not in public
    assert "item_id" not in public
    assert "/data/" not in public
    assert "prompt" not in public


def test_fake_tokenizer_satisfies_expected_interface() -> None:
    tokenizer = cast(Any, _FakeTokenizer())

    assert tokenizer.encode("</think>", add_special_tokens=False) == [2]
