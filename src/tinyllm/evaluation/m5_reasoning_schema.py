"""Strict M5.2 dual-mode Reasoning Dev evaluation and selection contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN


class M5ReasoningGenerationConfig(StrictSchema):
    """Frozen, mode-specific Qwen3 generation policy for M5-only development selection."""

    batch_size: Literal[4]
    thinking_max_new_tokens: Literal[896]
    nonthinking_max_new_tokens: Literal[128]
    thinking_do_sample: Literal[True]
    temperature: float = Field(gt=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: Literal[20]
    repetition_penalty: float = Field(ge=1.0, le=2.0)
    nonthinking_do_sample: Literal[False]
    base_seed: Literal[20260726]

    @model_validator(mode="after")
    def validate_sampling(self) -> M5ReasoningGenerationConfig:
        """Reject silent drift from the preregistered Thinking sampler."""

        if (self.temperature, self.top_p, self.repetition_penalty) != (0.6, 0.95, 1.0):
            raise ValueError("M5 Reasoning sampling parameters differ from the frozen protocol")
        return self


class M5ReasoningEvaluationConfig(StrictSchema):
    """Frozen 200-item dual-mode M5 Dev protocol that cannot consume M6 metrics."""

    schema_version: Literal["1.0"] = "1.0"
    suite_version: Literal[
        "m5-reasoning-dev-v1-3eb153c2",
        "m5-reasoning-dev-v1-53ddf557",
    ]
    expected_items: Literal[200]
    task_config_sha256: str = Field(pattern=SHA256_PATTERN)
    model_repository: Literal["Qwen/Qwen3-0.6B"]
    base_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    attention_architecture: Literal["gqa"]
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    nonthinking_template_id: Literal["qwen3-chatml-nonthinking-generation-v1"]
    compare_modes_separately: Literal[True]
    consume_m6_frozen_results: Literal[False]
    generation: M5ReasoningGenerationConfig


class M5ReasoningItemResult(StrictSchema):
    """One private response and deterministic M5 Dev score."""

    schema_version: Literal["1.0"] = "1.0"
    item_id: str = Field(pattern=r"^m5-reasoning:dev:[a-z0-9][a-z0-9._-]{2,95}$")
    mode: Literal["thinking", "nonthinking"]
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    response: str = Field(max_length=32768)
    response_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_tokens: int = Field(gt=0)
    generated_tokens: int = Field(ge=0)
    finish_reason: Literal["eos", "length"]
    format_valid: bool
    final_json_valid: bool
    final_answer_correct: bool
    visible_reasoning_leakage: bool

    @model_validator(mode="after")
    def validate_score_transitions(self) -> M5ReasoningItemResult:
        """Require correctness to imply valid structure and JSON."""

        if self.final_answer_correct and (not self.format_valid or not self.final_json_valid):
            raise ValueError("correct M5 answer requires valid format and JSON")
        if self.mode == "thinking" and self.visible_reasoning_leakage:
            raise ValueError("Thinking mode cannot label its requested trace as leakage")
        return self


class M5ModeSummary(StrictSchema):
    """Content-free aggregate for one evaluated Qwen3 mode."""

    mode: Literal["thinking", "nonthinking"]
    evaluated_items: Literal[200]
    format_valid_items: int = Field(ge=0, le=200)
    final_json_valid_items: int = Field(ge=0, le=200)
    final_answer_correct_items: int = Field(ge=0, le=200)
    visible_reasoning_leakage_items: int = Field(ge=0, le=200)
    format_valid_basis_points: int = Field(ge=0, le=10_000)
    final_answer_score_basis_points: int = Field(ge=0, le=10_000)
    generated_tokens: int = Field(ge=0)
    length_limited_items: int = Field(ge=0, le=200)

    @model_validator(mode="after")
    def validate_aggregates(self) -> M5ModeSummary:
        """Bind basis-point metrics and Non-thinking leakage semantics to counts."""

        if self.format_valid_basis_points != self.format_valid_items * 50:
            raise ValueError("M5 format basis points do not match count")
        if self.final_answer_score_basis_points != self.final_answer_correct_items * 50:
            raise ValueError("M5 answer basis points do not match count")
        if self.mode == "thinking" and self.visible_reasoning_leakage_items != 0:
            raise ValueError("Thinking traces are not leakage")
        return self


class M5ReasoningEvaluationSummary(StrictSchema):
    """Path-free public summary for one Base or ablation candidate evaluation."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded"]
    evaluation_id: str = Field(min_length=1, max_length=180)
    model_kind: Literal["base", "ablation_candidate"]
    training_run_id: str | None = Field(default=None, min_length=1, max_length=180)
    training_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    thinking_fraction_basis_points: Literal[0, 3000, 5000] | None = None
    model_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    attention_architecture: Literal["gqa"]
    suite_version: Literal[
        "m5-reasoning-dev-v1-3eb153c2",
        "m5-reasoning-dev-v1-53ddf557",
    ]
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    physical_gpu_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1, max_length=128)
    duration_seconds: float = Field(gt=0.0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    thinking: M5ModeSummary
    nonthinking: M5ModeSummary
    raw_results_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_model_kind(self) -> M5ReasoningEvaluationSummary:
        """Keep Base identity separate from trained Candidate lineage."""

        candidate_fields = (
            self.training_run_id,
            self.training_seed,
            self.thinking_fraction_basis_points,
        )
        if self.model_kind == "base" and any(value is not None for value in candidate_fields):
            raise ValueError("Base M5 evaluation cannot claim training lineage")
        if self.model_kind == "ablation_candidate" and any(
            value is None for value in candidate_fields
        ):
            raise ValueError("Candidate M5 evaluation requires complete training lineage")
        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("M5 evaluation reserved memory cannot be below allocated memory")
        return self


class M5AblationArmSummary(StrictSchema):
    """Two-seed aggregate and preregistered gates for one Thinking ratio."""

    thinking_fraction_basis_points: Literal[0, 3000, 5000]
    training_run_ids: tuple[str, str]
    training_seeds: tuple[int, int]
    nonthinking_scores_basis_points: tuple[int, int]
    thinking_format_basis_points: tuple[int, int]
    thinking_scores_basis_points: tuple[int, int]
    nonthinking_regression_gate_passed: bool
    thinking_format_gate_passed: bool
    mean_thinking_score_basis_points: int = Field(ge=0, le=10_000)


class M5AblationSelection(StrictSchema):
    """Deterministic result of the frozen M5.2 ratio-selection policy."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["selected", "no_eligible_arm"]
    base_evaluation_id: str = Field(min_length=1)
    base_nonthinking_score_basis_points: int = Field(ge=0, le=10_000)
    arms: tuple[M5AblationArmSummary, M5AblationArmSummary, M5AblationArmSummary]
    selected_thinking_fraction_basis_points: Literal[0, 3000, 5000] | None = None
    selection_reason: Literal[
        "highest_thinking_score",
        "lower_ratio_within_one_percentage_point",
        "no_arm_passed_preregistered_gates",
    ]

    @model_validator(mode="after")
    def validate_selection(self) -> M5AblationSelection:
        """Bind status and selected ratio to the declared reason."""

        ratios = tuple(item.thinking_fraction_basis_points for item in self.arms)
        if ratios != (0, 3000, 5000):
            raise ValueError("M5 ablation arms must be ordered 0/30/50")
        if self.status == "selected" and self.selected_thinking_fraction_basis_points is None:
            raise ValueError("selected M5 ablation must name a ratio")
        if self.status == "no_eligible_arm" and (
            self.selected_thinking_fraction_basis_points is not None
            or self.selection_reason != "no_arm_passed_preregistered_gates"
        ):
            raise ValueError("ineligible M5 ablation cannot claim a selected ratio")
        return self
