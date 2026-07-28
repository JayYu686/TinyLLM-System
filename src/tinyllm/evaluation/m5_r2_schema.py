"""Strict contracts for the M5.2-R2 counterfactual length diagnostic."""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal

from pydantic import Field, model_validator

from tinyllm.data.reasoning_schema import ReasoningLanguage, ReasoningTaskFamily
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M5R2Threshold = Literal[1024, 1280, 1536]
M5R2Count = Annotated[int, Field(ge=0, le=200)]
M5R2SHA256 = Annotated[str, Field(pattern=SHA256_PATTERN)]


class M5R2RepetitionDistribution(StrictSchema):
    """Content-free token-length and repetition distribution for one slice."""

    items: int = Field(gt=0, le=200)
    generated_tokens_min: int = Field(gt=0, le=896)
    generated_tokens_p50: float = Field(gt=0, le=896)
    generated_tokens_p90: int = Field(gt=0, le=896)
    generated_tokens_max: int = Field(gt=0, le=896)
    unique_token_ratio_mean_basis_points: int = Field(ge=0, le=10_000)
    repeated_8gram_ratio_mean_basis_points: int = Field(ge=0, le=10_000)
    max_identical_line_hash_repetitions: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_distribution(self) -> M5R2RepetitionDistribution:
        """Require ordered length statistics."""

        if not (
            self.generated_tokens_min
            <= self.generated_tokens_p50
            <= self.generated_tokens_p90
            <= self.generated_tokens_max
        ):
            raise ValueError("M5 R2 token-length distribution is not ordered")
        return self


class M5R2OfflineSeedAnalysis(StrictSchema):
    """One Seed's content-free D1 analysis of failures and matched successes."""

    training_seed: Literal[42, 20260727]
    source_evaluation_id: str = Field(min_length=1, max_length=180)
    source_raw_results_sha256: str = Field(pattern=SHA256_PATTERN)
    invalid_format_items: int = Field(gt=0, le=200)
    task_family_counts: dict[ReasoningTaskFamily, M5R2Count]
    language_counts: dict[ReasoningLanguage, M5R2Count]
    finish_reason_counts: dict[Literal["eos", "length"], M5R2Count]
    failure_distribution: M5R2RepetitionDistribution
    matched_valid_items: int = Field(gt=0, le=200)
    matched_valid_distribution: M5R2RepetitionDistribution

    @model_validator(mode="after")
    def validate_counts(self) -> M5R2OfflineSeedAnalysis:
        """Bind public slice counts to the analyzed sample groups."""

        if (
            sum(self.task_family_counts.values()) != self.invalid_format_items
            or sum(self.language_counts.values()) != self.invalid_format_items
            or sum(self.finish_reason_counts.values()) != self.invalid_format_items
            or self.failure_distribution.items != self.invalid_format_items
            or self.matched_valid_distribution.items != self.matched_valid_items
            or set(self.finish_reason_counts) != {"eos", "length"}
        ):
            raise ValueError("M5 R2 offline analysis counts are inconsistent")
        return self


class M5R2OfflineAnalysis(StrictSchema):
    """Public two-Seed D1 result without response or task-level content."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded"]
    analysis_version: Literal["m5-r2-offline-analysis-v1"]
    suite_version: Literal["m5-reasoning-dev-v1-53ddf557"]
    evaluation_config_sha256: Literal[
        "3735a34e70c06059fbc09f62d02fabc296fd97e79a45d60f3d00dede21108d51"
    ]
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    training_seeds: tuple[Literal[42], Literal[20260727]]
    slices: tuple[M5R2OfflineSeedAnalysis, M5R2OfflineSeedAnalysis]

    @model_validator(mode="after")
    def validate_seed_order(self) -> M5R2OfflineAnalysis:
        """Require the fixed R1 Seeds in deterministic order."""

        if (
            self.training_seeds != (42, 20260727)
            or tuple(item.training_seed for item in self.slices) != self.training_seeds
        ):
            raise ValueError("M5 R2 offline analysis requires ordered fixed Seeds")
        return self


class M5R2ReplayConfig(StrictSchema):
    """Frozen replay and decision policy for the R2 diagnostic."""

    schema_version: Literal["1.0"]
    diagnostic_version: Literal["m5-r2-length-replay-v1"]
    source_suite_version: Literal["m5-reasoning-dev-v1-53ddf557"]
    source_evaluation_config_sha256: Literal[
        "3735a34e70c06059fbc09f62d02fabc296fd97e79a45d60f3d00dede21108d51"
    ]
    source_mixture_version: Literal["m5-format-repair-mixture-v1-1396b60b"]
    source_mixture_manifest_sha256: Literal[
        "2467b5dce0d909b865b73219d2f608bdbc9c6fcc1bb09b93c5ebea8a7b60bd0e"
    ]
    seed42_training_run_id: Literal["20260727T075422Z-m5-format-repair-r1-seed42-7c825907-1c02"]
    seed42_source_evaluation_id: Literal[
        "20260727T090313Z-m5-reasoning-dev-ablation_candidate-3707f186"
    ]
    seed42_source_raw_results_sha256: Literal[
        "87e478b92c3992fa4f1196d05e32686291f2f2a4b559777e559fbc80988bd50d"
    ]
    seed42_model_export_sha256: Literal[
        "46d5ca599bec9cfcad12a3ed001fcb59b4646232d873377d7554219d7ce34f45"
    ]
    seed20260727_training_run_id: Literal[
        "20260727T075432Z-m5-format-repair-r1-seed20260727-59c6d0e9-3af2"
    ]
    seed20260727_source_evaluation_id: Literal[
        "20260727T090251Z-m5-reasoning-dev-ablation_candidate-e70ef06d"
    ]
    seed20260727_source_raw_results_sha256: Literal[
        "1e1fd1d43d170ddad4752114222c0655e1c814cebc23f8f3208575548c6b8cd7"
    ]
    seed20260727_model_export_sha256: Literal[
        "e39a14ee2cbb15d45b312ea1edc4adf298f9c27692b324630264ea0d483720f4"
    ]
    model_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    attention_architecture: Literal["gqa"]
    thinking_batch_size: Literal[4]
    thinking_base_seed: Literal[20260726]
    original_max_new_tokens: Literal[896]
    diagnostic_max_new_tokens: Literal[1536]
    score_thresholds: tuple[Literal[1024], Literal[1280], Literal[1536]]
    formal_candidate_max_new_tokens: Literal[1280]
    format_gate_basis_points: Literal[9900]
    require_896_response_match: Literal[True]
    require_1536_prefix_match: Literal[True]
    consume_m6_frozen_results: Literal[False]

    @model_validator(mode="after")
    def validate_thresholds(self) -> M5R2ReplayConfig:
        """Keep diagnostic thresholds ordered above the frozen source limit."""

        if self.score_thresholds != (1024, 1280, 1536):
            raise ValueError("M5 R2 score thresholds must be ordered 1024/1280/1536")
        if not (
            self.original_max_new_tokens
            < self.score_thresholds[0]
            < self.score_thresholds[1]
            < self.score_thresholds[2]
            == self.diagnostic_max_new_tokens
        ):
            raise ValueError("M5 R2 replay limits are inconsistent")
        return self


class M5R2ThresholdItemResult(StrictSchema):
    """Private score for one original failure at one counterfactual limit."""

    max_new_tokens: M5R2Threshold
    response: str = Field(min_length=1)
    response_sha256: str = Field(pattern=SHA256_PATTERN)
    generated_tokens: int = Field(gt=0, le=1536)
    finish_reason: Literal["eos", "length"]
    format_valid: bool
    final_json_valid: bool
    final_answer_correct: bool
    closing_tag_end_token: int | None = Field(default=None, gt=0, le=1536)

    @model_validator(mode="after")
    def validate_result(self) -> M5R2ThresholdItemResult:
        """Bind private response content, threshold, and scored state."""

        if hashlib.sha256(self.response.encode()).hexdigest() != self.response_sha256:
            raise ValueError("M5 R2 private response hash does not match content")
        if self.generated_tokens > self.max_new_tokens:
            raise ValueError("M5 R2 generated tokens exceed the scored threshold")
        if self.finish_reason == "length" and self.generated_tokens != self.max_new_tokens:
            raise ValueError("length-limited M5 R2 result must consume its threshold")
        if self.final_answer_correct and (not self.format_valid or not self.final_json_valid):
            raise ValueError("correct M5 R2 answer requires valid format and JSON")
        if self.format_valid and self.closing_tag_end_token is None:
            raise ValueError("valid M5 R2 Thinking format requires a closing-tag position")
        return self


class M5R2ReplayItemResult(StrictSchema):
    """Private replay evidence for one item in a replayed four-item Batch."""

    schema_version: Literal["1.0"] = "1.0"
    item_id: str = Field(min_length=1, max_length=160)
    task_family: ReasoningTaskFamily
    language: ReasoningLanguage
    batch_offset: int = Field(ge=0, lt=200)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    source_response_sha256: str = Field(pattern=SHA256_PATTERN)
    source_generated_tokens: int = Field(gt=0, le=896)
    source_finish_reason: Literal["eos", "length"]
    source_format_valid: bool
    source_final_json_valid: bool
    source_final_answer_correct: bool
    replay_896_response_sha256: str = Field(pattern=SHA256_PATTERN)
    replay_896_generated_tokens: int = Field(gt=0, le=896)
    replay_896_finish_reason: Literal["eos", "length"]
    replay_896_exact: Literal[True]
    replay_1536_prefix_tokens_compared: int = Field(gt=0, le=896)
    replay_1536_prefix_exact: Literal[True]
    thresholds: (
        tuple[
            M5R2ThresholdItemResult,
            M5R2ThresholdItemResult,
            M5R2ThresholdItemResult,
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_replay_evidence(self) -> M5R2ReplayItemResult:
        """Bind exact replay fields and limit scoring to original failures."""

        if self.batch_offset % 4:
            raise ValueError("M5 R2 Batch offset must be divisible by four")
        if (
            self.replay_896_response_sha256 != self.source_response_sha256
            or self.replay_896_generated_tokens != self.source_generated_tokens
            or self.replay_896_finish_reason != self.source_finish_reason
        ):
            raise ValueError("M5 R2 896 replay evidence differs from source")
        if self.replay_1536_prefix_tokens_compared != self.source_generated_tokens:
            raise ValueError("M5 R2 prefix evidence does not cover the full source generation")
        if self.source_final_answer_correct and (
            not self.source_format_valid or not self.source_final_json_valid
        ):
            raise ValueError("correct M5 R2 source answer requires valid format and JSON")
        is_original_failure = not self.source_format_valid
        if is_original_failure:
            if (
                self.source_generated_tokens != 896
                or self.source_finish_reason != "length"
                or self.thresholds is None
            ):
                raise ValueError("M5 R2 original failure must be length-limited and rescored")
            if tuple(item.max_new_tokens for item in self.thresholds) != (
                1024,
                1280,
                1536,
            ):
                raise ValueError("M5 R2 private thresholds must be 1024/1280/1536")
        elif self.thresholds is not None:
            raise ValueError("M5 R2 source successes must not be counterfactually rescored")
        return self


class M5R2ThresholdSummary(StrictSchema):
    """Content-free projected metrics for one counterfactual limit."""

    max_new_tokens: M5R2Threshold
    original_failed_items: int = Field(gt=0, le=200)
    recovered_format_items: int = Field(ge=0, le=200)
    recovered_final_json_items: int = Field(ge=0, le=200)
    recovered_correct_items: int = Field(ge=0, le=200)
    unresolved_format_items: int = Field(ge=0, le=200)
    projected_format_valid_items: int = Field(ge=0, le=200)
    projected_format_basis_points: int = Field(ge=0, le=10_000)
    projected_correct_items: int = Field(ge=0, le=200)
    projected_score_basis_points: int = Field(ge=0, le=10_000)
    closing_tag_end_token_min: int | None = Field(default=None, gt=0, le=1536)
    closing_tag_end_token_max: int | None = Field(default=None, gt=0, le=1536)

    @model_validator(mode="after")
    def validate_projection(self) -> M5R2ThresholdSummary:
        """Bind recovery counts and basis-point projections."""

        if self.recovered_format_items + self.unresolved_format_items != self.original_failed_items:
            raise ValueError("M5 R2 recovered and unresolved counts differ from source")
        if not (
            self.recovered_correct_items
            <= self.recovered_final_json_items
            <= self.recovered_format_items
        ):
            raise ValueError("M5 R2 recovered score counts are inconsistent")
        if self.projected_format_basis_points != self.projected_format_valid_items * 50:
            raise ValueError("M5 R2 projected format basis points differ from count")
        if self.projected_score_basis_points != self.projected_correct_items * 50:
            raise ValueError("M5 R2 projected score basis points differ from count")
        positions = (
            self.closing_tag_end_token_min,
            self.closing_tag_end_token_max,
        )
        if self.recovered_format_items == 0 and positions != (None, None):
            raise ValueError("M5 R2 empty recovery cannot claim closing positions")
        if self.recovered_format_items > 0 and (
            positions[0] is None
            or positions[1] is None
            or positions[0] > positions[1]
            or positions[1] > self.max_new_tokens
        ):
            raise ValueError("M5 R2 recovered closing positions are invalid")
        return self


class M5R2ReplaySummary(StrictSchema):
    """Public, path-free result for one Candidate's R2 replay."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded"]
    diagnostic_id: str = Field(min_length=1, max_length=180)
    diagnostic_version: Literal["m5-r2-length-replay-v1"]
    diagnostic_config_sha256: str = Field(pattern=SHA256_PATTERN)
    source_evaluation_id: str = Field(min_length=1, max_length=180)
    source_raw_results_sha256: str = Field(pattern=SHA256_PATTERN)
    training_run_id: str = Field(min_length=1, max_length=180)
    training_seed: Literal[42, 20260727]
    mixture_version: Literal["m5-format-repair-mixture-v1-1396b60b"]
    mixture_manifest_sha256: Literal[
        "2467b5dce0d909b865b73219d2f608bdbc9c6fcc1bb09b93c5ebea8a7b60bd0e"
    ]
    model_export_sha256: str = Field(pattern=SHA256_PATTERN)
    model_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    attention_architecture: Literal["gqa"]
    suite_version: Literal["m5-reasoning-dev-v1-53ddf557"]
    evaluation_config_sha256: Literal[
        "3735a34e70c06059fbc09f62d02fabc296fd97e79a45d60f3d00dede21108d51"
    ]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    physical_gpu_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1, max_length=128)
    duration_seconds: float = Field(gt=0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    source_format_valid_items: int = Field(ge=0, le=200)
    source_correct_items: int = Field(ge=0, le=200)
    original_failed_items: int = Field(gt=0, le=200)
    replayed_batches: int = Field(gt=0, le=50)
    replayed_items: int = Field(gt=0, le=200)
    replay_896_exact_items: int = Field(gt=0, le=200)
    replay_1536_prefix_exact_items: int = Field(gt=0, le=200)
    thresholds: tuple[
        M5R2ThresholdSummary,
        M5R2ThresholdSummary,
        M5R2ThresholdSummary,
    ]
    raw_results_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_summary(self) -> M5R2ReplaySummary:
        """Require complete replay proof and ordered projected metrics."""

        if self.source_format_valid_items + self.original_failed_items != 200:
            raise ValueError("M5 R2 source format counts must total 200")
        if self.replayed_items != self.replayed_batches * 4:
            raise ValueError("M5 R2 replayed items must equal four per Batch")
        if (
            self.replay_896_exact_items != self.replayed_items
            or self.replay_1536_prefix_exact_items != self.replayed_items
        ):
            raise ValueError("M5 R2 public summary requires exact replay for every Batch item")
        if tuple(item.max_new_tokens for item in self.thresholds) != (
            1024,
            1280,
            1536,
        ):
            raise ValueError("M5 R2 public thresholds must be 1024/1280/1536")
        for item in self.thresholds:
            if (
                item.original_failed_items != self.original_failed_items
                or item.projected_format_valid_items
                != self.source_format_valid_items + item.recovered_format_items
                or item.projected_correct_items
                != self.source_correct_items + item.recovered_correct_items
            ):
                raise ValueError("M5 R2 projected metrics differ from source and recovery")
        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("M5 R2 reserved memory cannot be below allocated memory")
        return self


class M5R2SeedProjection(StrictSchema):
    """One Seed's projected format score at the selected diagnostic limit."""

    training_seed: Literal[42, 20260727]
    source_evaluation_id: str = Field(min_length=1, max_length=180)
    projected_format_basis_points: int = Field(ge=0, le=10_000)
    projected_score_basis_points: int = Field(ge=0, le=10_000)
    unresolved_format_items: int = Field(ge=0, le=200)


class M5R2DiagnosticDecision(StrictSchema):
    """Combined two-seed conclusion without changing the formal protocol."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal[
        "supports_eval_protocol_revision",
        "tradeoff_review_required",
        "length_ceiling_insufficient",
    ]
    diagnostic_version: Literal["m5-r2-length-replay-v1"]
    summary_sha256: tuple[M5R2SHA256, M5R2SHA256]
    training_seeds: tuple[Literal[42], Literal[20260727]]
    selected_max_new_tokens: M5R2Threshold | None = None
    projections: tuple[M5R2SeedProjection, M5R2SeedProjection]
    formal_protocol_changed: Literal[False] = False
    decision_reason: Literal[
        "both_seeds_pass_at_or_below_conditionally_approved_limit",
        "both_seeds_pass_only_at_tradeoff_limit",
        "at_least_one_seed_fails_at_diagnostic_limit",
    ]

    @model_validator(mode="after")
    def validate_decision(self) -> M5R2DiagnosticDecision:
        """Bind decision status to the approved 1280 and diagnostic 1536 limits."""

        if self.training_seeds != (42, 20260727):
            raise ValueError("M5 R2 decision requires ordered fixed Seeds")
        if tuple(item.training_seed for item in self.projections) != self.training_seeds:
            raise ValueError("M5 R2 projection Seeds differ from decision")
        valid = (
            (
                self.status == "supports_eval_protocol_revision"
                and self.selected_max_new_tokens in {1024, 1280}
                and self.decision_reason
                == "both_seeds_pass_at_or_below_conditionally_approved_limit"
            )
            or (
                self.status == "tradeoff_review_required"
                and self.selected_max_new_tokens == 1536
                and self.decision_reason == "both_seeds_pass_only_at_tradeoff_limit"
            )
            or (
                self.status == "length_ceiling_insufficient"
                and self.selected_max_new_tokens is None
                and self.decision_reason == "at_least_one_seed_fails_at_diagnostic_limit"
            )
        )
        if not valid:
            raise ValueError("M5 R2 status, limit, and reason are inconsistent")
        return self
