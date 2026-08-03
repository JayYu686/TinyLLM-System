"""Versioned Qwen3 thinking-budget evaluation contracts for M5."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

EARLY_STOPPING_TEXT = (
    "\n\n Considering the limited time by the user, I have to give the solution "
    "based on the thinking directly now.\n</think>\n\n"
)


class M5ThinkingBudgetGenerationConfig(StrictSchema):
    """Qwen-official two-stage generation policy with a bounded final continuation."""

    batch_size: Literal[4]
    thinking_budget_tokens: Literal[1536]
    final_answer_max_new_tokens: Literal[128]
    nonthinking_max_new_tokens: Literal[128]
    thinking_do_sample: Literal[True]
    temperature: float = Field(gt=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: Literal[20]
    repetition_penalty: float = Field(ge=1.0, le=2.0)
    nonthinking_do_sample: Literal[False]
    base_seed: Literal[20260726]
    early_stopping_text: Literal[
        "\n\n Considering the limited time by the user, I have to give the solution "
        "based on the thinking directly now.\n</think>\n\n"
    ]

    @model_validator(mode="after")
    def validate_sampling(self) -> M5ThinkingBudgetGenerationConfig:
        """Keep Qwen's recommended Thinking sampler frozen."""

        if (self.temperature, self.top_p, self.repetition_penalty) != (0.6, 0.95, 1.0):
            raise ValueError("Thinking Budget sampler differs from the Qwen protocol")
        return self


class M5ThinkingBudgetEvaluationConfig(StrictSchema):
    """Frozen M5-only protocol v2; it never consumes the M6 release set."""

    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["m5-thinking-budget-v2"]
    suite_version: Literal["m5-reasoning-dev-v1-53ddf557"]
    expected_items: Literal[200]
    task_config_sha256: str = Field(pattern=SHA256_PATTERN)
    model_repository: Literal["Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"]
    base_revision: Literal[
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "b968826d9c46dd6066d109eabc6255188de91218",
    ]
    attention_architecture: Literal["gqa"]
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    nonthinking_template_id: Literal["qwen3-chatml-nonthinking-generation-v1"]
    compare_modes_separately: Literal[True]
    consume_m6_frozen_results: Literal[False]
    controlled_format_min_basis_points: Literal[9900]
    max_forced_close_basis_points: Literal[1000]
    min_thinking_score_basis_points: Literal[9000]
    nonthinking_regression_tolerance_basis_points: Literal[200]
    generation: M5ThinkingBudgetGenerationConfig

    @model_validator(mode="after")
    def validate_model_identity(self) -> M5ThinkingBudgetEvaluationConfig:
        """Bind each reviewed repository to its immutable Revision."""

        pairs = {
            "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
            "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
        }
        if pairs[self.model_repository] != self.base_revision:
            raise ValueError("Thinking Budget repository and Revision differ")
        return self


class M5ThinkingBudgetItemResult(StrictSchema):
    """One private result with model output and controller intervention separated."""

    schema_version: Literal["1.0"] = "1.0"
    item_id: str = Field(pattern=r"^m5-reasoning:dev:[a-z0-9][a-z0-9._-]{2,95}$")
    mode: Literal["thinking", "nonthinking"]
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    response: str = Field(max_length=131072)
    response_sha256: str = Field(pattern=SHA256_PATTERN)
    first_pass_response: str = Field(max_length=131072)
    continuation_response: str = Field(max_length=32768)
    controller_injected_text: str = Field(max_length=512)
    controller_action: Literal[
        "not_applicable",
        "natural_complete",
        "natural_close_continue",
        "forced_close_continue",
    ]
    prompt_tokens: int = Field(gt=0)
    first_pass_tokens: int = Field(ge=0, le=1536)
    continuation_tokens: int = Field(ge=0, le=128)
    injected_tokens: int = Field(ge=0, le=128)
    generated_tokens: int = Field(ge=0, le=1664)
    finish_reason: Literal["eos", "length"]
    natural_thinking_closed: bool
    budget_forced_close: bool
    format_valid: bool
    final_json_valid: bool
    final_answer_correct: bool
    visible_reasoning_leakage: bool

    @model_validator(mode="after")
    def validate_item(self) -> M5ThinkingBudgetItemResult:
        """Bind hashes, token accounting, mode semantics, and controller disclosure."""

        if hashlib.sha256(self.response.encode()).hexdigest() != self.response_sha256:
            raise ValueError("thinking-budget response hash does not match content")
        if self.generated_tokens != self.first_pass_tokens + self.continuation_tokens:
            raise ValueError("thinking-budget generated Token accounting differs")
        if self.final_answer_correct and (not self.format_valid or not self.final_json_valid):
            raise ValueError("correct thinking-budget answer requires valid format and JSON")
        if self.mode == "nonthinking":
            if (
                self.controller_action != "not_applicable"
                or self.natural_thinking_closed
                or self.budget_forced_close
                or self.controller_injected_text
                or self.injected_tokens
            ):
                raise ValueError("Non-thinking result cannot claim Thinking controller activity")
        elif self.controller_action == "forced_close_continue":
            if (
                not self.budget_forced_close
                or self.natural_thinking_closed
                or self.controller_injected_text != EARLY_STOPPING_TEXT
                or self.injected_tokens == 0
            ):
                raise ValueError("forced-close result must disclose the official injected text")
        elif (
            self.budget_forced_close
            or self.controller_injected_text
            or self.injected_tokens
            or not self.natural_thinking_closed
        ):
            raise ValueError("natural Thinking result cannot claim controller injection")
        return self


class M5ThinkingBudgetModeSummary(StrictSchema):
    """Content-free aggregate for one controlled evaluation mode."""

    mode: Literal["thinking", "nonthinking"]
    evaluated_items: Literal[200]
    format_valid_items: int = Field(ge=0, le=200)
    final_json_valid_items: int = Field(ge=0, le=200)
    final_answer_correct_items: int = Field(ge=0, le=200)
    visible_reasoning_leakage_items: int = Field(ge=0, le=200)
    natural_thinking_closed_items: int = Field(ge=0, le=200)
    budget_forced_close_items: int = Field(ge=0, le=200)
    format_valid_basis_points: int = Field(ge=0, le=10000)
    final_answer_score_basis_points: int = Field(ge=0, le=10000)
    natural_close_basis_points: int = Field(ge=0, le=10000)
    forced_close_basis_points: int = Field(ge=0, le=10000)
    generated_tokens: int = Field(ge=0)
    injected_tokens: int = Field(ge=0)
    length_limited_items: int = Field(ge=0, le=200)

    @model_validator(mode="after")
    def validate_summary(self) -> M5ThinkingBudgetModeSummary:
        """Require exact basis-point accounting and mutually exclusive close paths."""

        pairs = (
            (self.format_valid_basis_points, self.format_valid_items),
            (self.final_answer_score_basis_points, self.final_answer_correct_items),
            (self.natural_close_basis_points, self.natural_thinking_closed_items),
            (self.forced_close_basis_points, self.budget_forced_close_items),
        )
        if any(basis_points != count * 50 for basis_points, count in pairs):
            raise ValueError("thinking-budget basis points do not match counts")
        if self.mode == "thinking":
            if (
                self.natural_thinking_closed_items + self.budget_forced_close_items
                != self.evaluated_items
                or self.visible_reasoning_leakage_items
            ):
                raise ValueError("Thinking close paths must cover all items without leakage")
        elif (
            self.natural_thinking_closed_items
            or self.budget_forced_close_items
            or self.natural_close_basis_points
            or self.forced_close_basis_points
            or self.injected_tokens
        ):
            raise ValueError("Non-thinking summary cannot contain controller activity")
        return self


class M5ThinkingBudgetEvaluationSummary(StrictSchema):
    """Path-free public result for one protocol-v2 Base or Candidate evaluation."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded"]
    evaluation_id: str = Field(min_length=1, max_length=180)
    protocol_version: Literal["m5-thinking-budget-v2"]
    model_kind: Literal["base", "ablation_candidate", "lora_candidate"]
    training_run_id: str | None = Field(default=None, min_length=1, max_length=180)
    training_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    thinking_fraction_basis_points: Literal[0, 3000, 5000] | None = None
    model_revision: Literal[
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "b968826d9c46dd6066d109eabc6255188de91218",
    ]
    adaptation: Literal["full", "lora"] = "full"
    adapter_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    attention_architecture: Literal["gqa"]
    suite_version: Literal["m5-reasoning-dev-v1-53ddf557"]
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    physical_gpu_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1, max_length=128)
    duration_seconds: float = Field(gt=0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    thinking: M5ThinkingBudgetModeSummary
    nonthinking: M5ThinkingBudgetModeSummary
    raw_results_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_lineage(self) -> M5ThinkingBudgetEvaluationSummary:
        """Keep Base and trained Candidate identities disjoint."""

        candidate_fields = (
            self.training_run_id,
            self.training_seed,
            self.thinking_fraction_basis_points,
        )
        if self.model_kind == "base" and any(value is not None for value in candidate_fields):
            raise ValueError("thinking-budget Base cannot claim training lineage")
        if self.model_kind == "ablation_candidate" and any(
            value is None for value in candidate_fields
        ):
            raise ValueError("thinking-budget Candidate requires complete training lineage")
        if self.model_kind == "lora_candidate":
            if (
                any(value is None for value in candidate_fields)
                or self.adaptation != "lora"
                or self.adapter_sha256 is None
                or self.model_revision != "b968826d9c46dd6066d109eabc6255188de91218"
            ):
                raise ValueError("thinking-budget LoRA Candidate requires Adapter lineage")
        elif self.adaptation != "full" or self.adapter_sha256 is not None:
            raise ValueError("non-LoRA Thinking Budget result cannot claim an Adapter")
        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("reserved memory cannot be below allocated memory")
        return self


class M5ThinkingBudgetGateResult(StrictSchema):
    """Two-seed quality gate that authorizes the formal M5.3 training stage."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["passed", "rejected"]
    protocol_version: Literal["m5-thinking-budget-v2"]
    base_evaluation_id: str = Field(min_length=1, max_length=180)
    candidate_evaluation_ids: tuple[str, str]
    evaluation_config_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluation_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_summary_sha256: tuple[str, str]
    source_format_repair_gate_sha256: str = Field(pattern=SHA256_PATTERN)
    mixture_version: Literal["m5-format-repair-mixture-v1-1396b60b"]
    mixture_manifest_sha256: Literal[
        "2467b5dce0d909b865b73219d2f608bdbc9c6fcc1bb09b93c5ebea8a7b60bd0e"
    ]
    training_run_ids: tuple[str, str]
    training_seeds: tuple[Literal[42], Literal[20260727]]
    selected_thinking_fraction_basis_points: Literal[3000]
    base_nonthinking_score_basis_points: int = Field(ge=0, le=10_000)
    controlled_format_basis_points: tuple[int, int]
    forced_close_basis_points: tuple[int, int]
    thinking_scores_basis_points: tuple[int, int]
    nonthinking_scores_basis_points: tuple[int, int]
    controlled_format_gate_passed: bool
    forced_close_gate_passed: bool
    thinking_score_gate_passed: bool
    nonthinking_regression_gate_passed: bool
    m5_3_authorized: bool
    gate_reason: Literal[
        "all_protocol_v2_gates_passed",
        "controlled_format_gate_failed",
        "forced_close_gate_failed",
        "thinking_score_gate_failed",
        "nonthinking_regression_gate_failed",
        "multiple_gates_failed",
    ]

    @model_validator(mode="after")
    def validate_gate(self) -> M5ThinkingBudgetGateResult:
        """Recompute every threshold and bind authorization to the AND gate."""

        expected = (
            all(value >= 9900 for value in self.controlled_format_basis_points),
            all(value <= 1000 for value in self.forced_close_basis_points),
            all(value >= 9000 for value in self.thinking_scores_basis_points),
            all(
                value >= self.base_nonthinking_score_basis_points - 200
                for value in self.nonthinking_scores_basis_points
            ),
        )
        declared = (
            self.controlled_format_gate_passed,
            self.forced_close_gate_passed,
            self.thinking_score_gate_passed,
            self.nonthinking_regression_gate_passed,
        )
        if declared != expected:
            raise ValueError("Thinking Budget gate booleans differ from frozen thresholds")
        all_passed = all(expected)
        if self.status == "passed":
            if (
                not all_passed
                or not self.m5_3_authorized
                or self.gate_reason != "all_protocol_v2_gates_passed"
            ):
                raise ValueError("passed Thinking Budget gate must authorize M5.3")
        elif (
            all_passed or self.m5_3_authorized or self.gate_reason == "all_protocol_v2_gates_passed"
        ):
            raise ValueError("rejected Thinking Budget gate cannot authorize M5.3")
        return self
