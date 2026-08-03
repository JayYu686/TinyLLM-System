from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from tinyllm.data import (
    ReasoningTask,
    generate_reasoning_dev_tasks,
    load_m5_reasoning_data_config,
)
from tinyllm.evaluation.m5_reasoning_schema import M5FormatRepairGateResult
from tinyllm.evaluation.m5_thinking_budget import (
    build_m5_thinking_budget_item,
    evaluate_m5_thinking_budget_gate,
    load_m5_thinking_budget_config,
    summarize_m5_thinking_budget_mode,
)
from tinyllm.evaluation.m5_thinking_budget_schema import (
    EARLY_STOPPING_TEXT,
    M5ThinkingBudgetEvaluationSummary,
    M5ThinkingBudgetGateResult,
    M5ThinkingBudgetGenerationConfig,
    M5ThinkingBudgetItemResult,
    M5ThinkingBudgetModeSummary,
)


def _task() -> ReasoningTask:
    config = load_m5_reasoning_data_config(
        Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml")
    )
    return generate_reasoning_dev_tasks(config)[0]


def test_protocol_v2_freezes_qwen_official_budget_and_unchanged_quality_gate() -> None:
    config = load_m5_thinking_budget_config(Path("configs/eval/m5_thinking_budget_v2.yaml"))

    assert config.protocol_version == "m5-thinking-budget-v2"
    assert config.generation.thinking_budget_tokens == 1536
    assert config.generation.final_answer_max_new_tokens == 128
    assert config.generation.early_stopping_text == EARLY_STOPPING_TEXT
    assert config.controlled_format_min_basis_points == 9900
    assert config.max_forced_close_basis_points == 1000
    assert config.min_thinking_score_basis_points == 9000
    assert config.consume_m6_frozen_results is False


def test_protocol_v2_freezes_lora_base_identity() -> None:
    config = load_m5_thinking_budget_config(Path("configs/eval/m5_lora_thinking_budget_v2.yaml"))

    assert config.model_repository == "Qwen/Qwen3-8B"
    assert config.base_revision == "b968826d9c46dd6066d109eabc6255188de91218"
    with pytest.raises(ValidationError, match="repository and Revision"):
        config.__class__.model_validate(
            config.model_dump(mode="json")
            | {"base_revision": "c1899de289a04d12100db370d81485cdf75e47ca"}
        )


def test_forced_close_is_scored_but_never_mislabeled_as_natural() -> None:
    task = _task()
    first = "<think>\nThe budget was exhausted."
    response = first + EARLY_STOPPING_TEXT + task.expected_answer_json

    result = build_m5_thinking_budget_item(
        task,
        mode="thinking",
        response=response,
        first_pass_response=first,
        continuation_response=task.expected_answer_json,
        controller_action="forced_close_continue",
        prompt_tokens=10,
        first_pass_tokens=20,
        continuation_tokens=5,
        injected_tokens=24,
        finish_reason="eos",
    )

    assert result.format_valid
    assert result.final_answer_correct
    assert result.budget_forced_close
    assert not result.natural_thinking_closed
    assert result.controller_injected_text == EARLY_STOPPING_TEXT


def test_natural_close_cannot_claim_injected_tokens() -> None:
    task = _task()

    with pytest.raises(ValidationError, match="natural Thinking result"):
        build_m5_thinking_budget_item(
            task,
            mode="thinking",
            response=f"<think>short</think>\n\n{task.expected_answer_json}",
            first_pass_response=f"<think>short</think>\n\n{task.expected_answer_json}",
            continuation_response="",
            controller_action="natural_complete",
            prompt_tokens=10,
            first_pass_tokens=10,
            continuation_tokens=0,
            injected_tokens=1,
            finish_reason="eos",
        )


def test_summary_keeps_natural_and_forced_rates_separate() -> None:
    task = _task()
    natural = build_m5_thinking_budget_item(
        task,
        mode="thinking",
        response=f"<think>short</think>\n\n{task.expected_answer_json}",
        first_pass_response=f"<think>short</think>\n\n{task.expected_answer_json}",
        continuation_response="",
        controller_action="natural_complete",
        prompt_tokens=10,
        first_pass_tokens=10,
        continuation_tokens=0,
        injected_tokens=0,
        finish_reason="eos",
    )
    forced = build_m5_thinking_budget_item(
        task,
        mode="thinking",
        response=f"<think>long{EARLY_STOPPING_TEXT}{task.expected_answer_json}",
        first_pass_response="<think>long",
        continuation_response=task.expected_answer_json,
        controller_action="forced_close_continue",
        prompt_tokens=10,
        first_pass_tokens=10,
        continuation_tokens=5,
        injected_tokens=24,
        finish_reason="eos",
    )

    summary = summarize_m5_thinking_budget_mode(
        "thinking",
        tuple([natural] * 190 + [forced] * 10),
    )

    assert summary.format_valid_basis_points == 10000
    assert summary.natural_close_basis_points == 9500
    assert summary.forced_close_basis_points == 500
    assert summary.injected_tokens == 240


def test_generation_config_rejects_sampler_drift() -> None:
    config = load_m5_thinking_budget_config(Path("configs/eval/m5_thinking_budget_v2.yaml"))
    payload = config.generation.model_dump(mode="json")
    payload["temperature"] = 0.7

    with pytest.raises(ValidationError, match="sampler differs"):
        M5ThinkingBudgetGenerationConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"response": "tampered"}, "response hash"),
        ({"generated_tokens": 99}, "Token accounting"),
        (
            {"format_valid": False, "final_answer_correct": True},
            "requires valid format",
        ),
    ],
)
def test_item_contract_rejects_integrity_errors(
    updates: dict[str, object],
    message: str,
) -> None:
    task = _task()
    item = build_m5_thinking_budget_item(
        task,
        mode="thinking",
        response=f"<think>short</think>\n\n{task.expected_answer_json}",
        first_pass_response=f"<think>short</think>\n\n{task.expected_answer_json}",
        continuation_response="",
        controller_action="natural_complete",
        prompt_tokens=10,
        first_pass_tokens=10,
        continuation_tokens=0,
        injected_tokens=0,
        finish_reason="eos",
    )

    with pytest.raises(ValidationError, match=message):
        M5ThinkingBudgetItemResult.model_validate(item.model_dump(mode="json") | updates)


def test_nonthinking_item_cannot_claim_controller_activity() -> None:
    task = _task()
    item = build_m5_thinking_budget_item(
        task,
        mode="nonthinking",
        response=task.expected_answer_json,
        first_pass_response=task.expected_answer_json,
        continuation_response="",
        controller_action="not_applicable",
        prompt_tokens=10,
        first_pass_tokens=5,
        continuation_tokens=0,
        injected_tokens=0,
        finish_reason="eos",
    )

    with pytest.raises(ValidationError, match="cannot claim Thinking controller"):
        M5ThinkingBudgetItemResult.model_validate(
            item.model_dump(mode="json") | {"budget_forced_close": True}
        )


def test_mode_summary_rejects_accounting_and_controller_errors() -> None:
    task = _task()
    natural = build_m5_thinking_budget_item(
        task,
        mode="thinking",
        response=f"<think>short</think>\n\n{task.expected_answer_json}",
        first_pass_response=f"<think>short</think>\n\n{task.expected_answer_json}",
        continuation_response="",
        controller_action="natural_complete",
        prompt_tokens=10,
        first_pass_tokens=10,
        continuation_tokens=0,
        injected_tokens=0,
        finish_reason="eos",
    )
    thinking = summarize_m5_thinking_budget_mode("thinking", tuple([natural] * 200))

    with pytest.raises(ValidationError, match="basis points"):
        M5ThinkingBudgetModeSummary.model_validate(
            thinking.model_dump(mode="json") | {"format_valid_basis_points": 9950}
        )
    with pytest.raises(ValidationError, match="without leakage"):
        M5ThinkingBudgetModeSummary.model_validate(
            thinking.model_dump(mode="json") | {"visible_reasoning_leakage_items": 1}
        )

    nonthinking_item = build_m5_thinking_budget_item(
        task,
        mode="nonthinking",
        response=task.expected_answer_json,
        first_pass_response=task.expected_answer_json,
        continuation_response="",
        controller_action="not_applicable",
        prompt_tokens=10,
        first_pass_tokens=5,
        continuation_tokens=0,
        injected_tokens=0,
        finish_reason="eos",
    )
    nonthinking = summarize_m5_thinking_budget_mode("nonthinking", tuple([nonthinking_item] * 200))
    with pytest.raises(ValidationError, match="cannot contain controller"):
        M5ThinkingBudgetModeSummary.model_validate(
            nonthinking.model_dump(mode="json") | {"injected_tokens": 1}
        )


def test_evaluation_summary_rejects_lineage_and_memory_errors() -> None:
    task = _task()
    thinking_item = build_m5_thinking_budget_item(
        task,
        mode="thinking",
        response=f"<think>short</think>\n\n{task.expected_answer_json}",
        first_pass_response=f"<think>short</think>\n\n{task.expected_answer_json}",
        continuation_response="",
        controller_action="natural_complete",
        prompt_tokens=10,
        first_pass_tokens=10,
        continuation_tokens=0,
        injected_tokens=0,
        finish_reason="eos",
    )
    nonthinking_item = build_m5_thinking_budget_item(
        task,
        mode="nonthinking",
        response=task.expected_answer_json,
        first_pass_response=task.expected_answer_json,
        continuation_response="",
        controller_action="not_applicable",
        prompt_tokens=10,
        first_pass_tokens=5,
        continuation_tokens=0,
        injected_tokens=0,
        finish_reason="eos",
    )
    payload: dict[str, object] = {
        "status": "succeeded",
        "evaluation_id": "unit-base",
        "protocol_version": "m5-thinking-budget-v2",
        "model_kind": "base",
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "attention_architecture": "gqa",
        "suite_version": "m5-reasoning-dev-v1-53ddf557",
        "config_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "physical_gpu_index": 7,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "duration_seconds": 1.0,
        "peak_allocated_bytes": 1,
        "peak_reserved_bytes": 2,
        "thinking": summarize_m5_thinking_budget_mode("thinking", tuple([thinking_item] * 200)),
        "nonthinking": summarize_m5_thinking_budget_mode(
            "nonthinking", tuple([nonthinking_item] * 200)
        ),
        "raw_results_sha256": "c" * 64,
    }

    with pytest.raises(ValidationError, match="Base cannot claim training lineage"):
        M5ThinkingBudgetEvaluationSummary.model_validate(
            payload | {"training_run_id": "unexpected"}
        )
    with pytest.raises(ValidationError, match="complete training lineage"):
        M5ThinkingBudgetEvaluationSummary.model_validate(
            payload | {"model_kind": "ablation_candidate"}
        )
    with pytest.raises(ValidationError, match="reserved memory"):
        M5ThinkingBudgetEvaluationSummary.model_validate(payload | {"peak_allocated_bytes": 3})

    shared = M5ThinkingBudgetEvaluationSummary.model_validate(
        payload
        | {
            "preflight_memory_used_mib": 1744,
            "preflight_utilization_percent": 0,
            "preflight_temperature_c": 31,
            "shared_gpu_evaluation": True,
        }
    )
    assert shared.shared_gpu_evaluation is True
    with pytest.raises(ValidationError, match="shared-GPU flag"):
        M5ThinkingBudgetEvaluationSummary.model_validate(
            shared.model_dump(mode="json") | {"shared_gpu_evaluation": False}
        )

    formal = M5ThinkingBudgetEvaluationSummary.model_validate(
        payload
        | {
            "model_kind": "formal_candidate",
            "training_run_id": "formal-run",
            "training_seed": 42,
            "thinking_fraction_basis_points": 3000,
            "training_checkpoint_id": "checkpoint-tokens-0050000000",
            "training_tokens": 50_000_000,
        }
    )
    assert formal.model_kind == "formal_candidate"
    with pytest.raises(ValidationError, match="frozen 0.6B route"):
        M5ThinkingBudgetEvaluationSummary.model_validate(
            formal.model_dump(mode="json")
            | {"model_revision": "b968826d9c46dd6066d109eabc6255188de91218"}
        )

    lora = M5ThinkingBudgetEvaluationSummary.model_validate(
        payload
        | {
            "model_kind": "lora_candidate",
            "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "adaptation": "lora",
            "adapter_sha256": "d" * 64,
            "training_run_id": "formal-lora-run",
            "training_seed": 42,
            "thinking_fraction_basis_points": 3000,
        }
    )
    assert lora.adaptation == "lora"
    with pytest.raises(ValidationError, match="Adapter lineage"):
        M5ThinkingBudgetEvaluationSummary.model_validate(
            lora.model_dump(mode="json") | {"adapter_sha256": None}
        )


def test_real_protocol_v2_evidence_authorizes_m5_3() -> None:
    raw = Path("reports/m5/raw")
    base_path = raw / "m5_thinking_budget_v2_base.json"
    seed42_path = raw / "m5_thinking_budget_v2_seed42.json"
    seed20260727_path = raw / "m5_thinking_budget_v2_seed20260727.json"
    source_gate_path = raw / "m5_format_repair_gate.json"
    base = M5ThinkingBudgetEvaluationSummary.model_validate_json(base_path.read_bytes())
    candidates = (
        M5ThinkingBudgetEvaluationSummary.model_validate_json(seed42_path.read_bytes()),
        M5ThinkingBudgetEvaluationSummary.model_validate_json(seed20260727_path.read_bytes()),
    )
    source_gate = M5FormatRepairGateResult.model_validate_json(source_gate_path.read_bytes())

    actual = evaluate_m5_thinking_budget_gate(
        base,
        candidates,
        source_gate,
        base_summary_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
        candidate_summary_sha256=(
            hashlib.sha256(seed42_path.read_bytes()).hexdigest(),
            hashlib.sha256(seed20260727_path.read_bytes()).hexdigest(),
        ),
        source_gate_sha256=hashlib.sha256(source_gate_path.read_bytes()).hexdigest(),
    )
    expected = M5ThinkingBudgetGateResult.model_validate_json(
        (raw / "m5_thinking_budget_v2_gate.json").read_bytes()
    )

    assert actual == expected
    assert actual.status == "passed"
    assert actual.m5_3_authorized
