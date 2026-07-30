"""Strict contracts for the M5.2-R3 Teacher-source strategy review."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.m5_r3_schema import M5R3TargetFamily
from tinyllm.data.reasoning_schema import ReasoningLanguage
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M5R3TeacherStrategy = Literal[
    "single_stage_prompt_control",
    "higher_generation_ceiling_only",
    "two_stage_solve_compress",
    "deterministic_rule_trace",
]


class M5R3P1TeacherStage(StrictSchema):
    """Pinned Qwen3-8B identity shared by the P1 solver and compressor."""

    repository: Literal["Qwen/Qwen3-8B"]
    revision: Literal["b968826d9c46dd6066d109eabc6255188de91218"]
    attention_architecture: Literal["gqa"]
    trust_remote_code: Literal[False]
    local_files_only: Literal[True]
    dtype: Literal["bfloat16"]


class M5R3P1SolverStage(M5R3P1TeacherStage):
    """Native-Thinking correctness stage used before compression."""

    mode: Literal["thinking"]
    do_sample: Literal[True]
    temperature: float
    top_p: float
    top_k: Literal[20]
    repetition_penalty: float
    candidate_count: Literal[1]
    max_new_tokens: Literal[896]
    base_seed: Literal[20260804]

    @model_validator(mode="after")
    def validate_sampling(self) -> M5R3P1SolverStage:
        """Freeze the solver sampling distribution used by P1."""

        if (self.temperature, self.top_p, self.repetition_penalty) != (0.6, 0.95, 1.0):
            raise ValueError("M5 R3 P1 solver sampling differs")
        return self


class M5R3P1CompressorStage(M5R3P1TeacherStage):
    """Greedy Non-thinking stage producing a constrained rationale envelope."""

    mode: Literal["nonthinking"]
    do_sample: Literal[False]
    candidate_count: Literal[1]
    max_new_tokens: Literal[256]
    base_seed: Literal[20260805]
    output_protocol: Literal["m5-r3-compressed-rationale-json-v1"]


class M5R3P1TracePolicy(StrictSchema):
    """Acceptance policy for model-distilled P1 visible rationales."""

    max_reasoning_tokens: Literal[192]
    max_repeated_8gram_basis_points: Literal[500]
    max_identical_line_hash_repetitions: Literal[1]
    require_unique_normalized_trace: Literal[True]
    require_exact_evidence_anchor: Literal[True]
    reject_other_labels: Literal[True]
    max_training_sequence_tokens: Literal[1024]


class M5R3P1Gate(StrictSchema):
    """Frozen per-family and per-language P1 feasibility gate."""

    accepted_per_family: dict[M5R3TargetFamily, Literal[14]]
    accepted_languages_per_family: dict[
        M5R3TargetFamily,
        dict[ReasoningLanguage, int],
    ]

    @model_validator(mode="after")
    def validate_gate(self) -> M5R3P1Gate:
        """Keep P1 directly comparable to the rejected P0 experiments."""

        expected_families = {"config": 14, "log_diagnosis": 14}
        expected_languages = {
            "config": {"en": 10, "zh": 4},
            "log_diagnosis": {"en": 10, "zh": 4},
        }
        if (
            self.accepted_per_family != expected_families
            or list(self.accepted_per_family) != ["config", "log_diagnosis"]
            or self.accepted_languages_per_family != expected_languages
            or any(
                list(counts) != ["en", "zh"]
                for counts in self.accepted_languages_per_family.values()
            )
        ):
            raise ValueError("M5 R3 P1 Gate must require 14 traces and 10/4 languages")
        return self


class M5R3RuleBaselinePolicy(StrictSchema):
    """Control-only deterministic trace policy with no training authorization."""

    source_kind: Literal["deterministic_rule_trace"]
    required_structural_passes: Literal[40]
    training_source_authorized: Literal[False]


class M5R3P1PilotConfig(StrictSchema):
    """Bounded 40-task interface selected by the source-strategy review."""

    pilot_version: Literal["m5-r3-p1-two-stage-v1"]
    task_seed: Literal[20260803]
    target_families: tuple[Literal["config"], Literal["log_diagnosis"]]
    tasks_per_family: Literal[20]
    language_counts_per_family: dict[ReasoningLanguage, int]
    evidence_variants_per_label: Literal[6]
    solver: M5R3P1SolverStage
    compressor: M5R3P1CompressorStage
    trace_policy: M5R3P1TracePolicy
    gate: M5R3P1Gate
    controlled_baseline_policy: M5R3RuleBaselinePolicy

    @field_validator("target_families", mode="before")
    @classmethod
    def normalize_yaml_families(cls, value: object) -> object:
        """Convert the YAML sequence to its immutable runtime form."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_scope(self) -> M5R3P1PilotConfig:
        """Freeze task balance and keep stage seeds in separate domains."""

        if (
            self.target_families != ("config", "log_diagnosis")
            or self.language_counts_per_family != {"en": 14, "zh": 6}
            or list(self.language_counts_per_family) != ["en", "zh"]
            or len(
                {
                    self.task_seed,
                    self.solver.base_seed,
                    self.compressor.base_seed,
                }
            )
            != 3
        ):
            raise ValueError("M5 R3 P1 scope or seed domains differ")
        return self


class M5R3TeacherSourceStrategyConfig(StrictSchema):
    """Immutable review inputs and the next bounded experiment interface."""

    schema_version: Literal["1.0"]
    review_version: Literal["m5-r3-teacher-source-strategy-v1"]
    parent_r2_decision_sha256: Literal[
        "04165538efce811240b4d4501b13f74151758af7373704c12d6df882e3044ed6"
    ]
    parent_p0_result_sha256: Literal[
        "5eff250ef4cde98d044c992a0aaf7e2eb75342faa9c377d265a25945a3d4388b"
    ]
    parent_p0_r1_result_sha256: Literal[
        "c59ab59fd048620e2b8de6a985a5a1deb877bb786d9b28b813531437b582c0b7"
    ]
    selected_strategy: Literal["two_stage_solve_compress"]
    controlled_baseline: Literal["deterministic_rule_trace"]
    pilot: M5R3P1PilotConfig
    formal_source_expansion_authorized: Literal[False]
    r3_mixture_authorized: Literal[False]
    r3_training_authorized: Literal[False]
    consume_m6_frozen_results: Literal[False]


class M5R3StrategyObservation(StrictSchema):
    """Content-free facts extracted from one completed parent experiment."""

    experiment: Literal["r2", "p0", "p0_r1"]
    status: Literal["length_ceiling_insufficient", "fail"]
    accepted_samples: int | None = Field(default=None, ge=0, le=40)
    accepted_per_family: tuple[int, int] | None = None
    accepted_languages: dict[ReasoningLanguage, int] | None = None
    reasoning_over_192_tokens: int | None = Field(default=None, ge=0)
    teacher_length_limit: int | None = Field(default=None, ge=0)
    projected_format_basis_points: tuple[int, int] | None = None
    unresolved_format_items: tuple[int, int] | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> M5R3StrategyObservation:
        """Keep R2 diagnostic evidence distinct from P0 quality evidence."""

        p0_fields = (
            self.accepted_samples,
            self.accepted_per_family,
            self.accepted_languages,
            self.reasoning_over_192_tokens,
            self.teacher_length_limit,
        )
        r2_fields = (self.projected_format_basis_points, self.unresolved_format_items)
        if self.experiment == "r2":
            if (
                self.status != "length_ceiling_insufficient"
                or any(value is not None for value in p0_fields)
                or any(value is None for value in r2_fields)
            ):
                raise ValueError("M5 R3 R2 observation shape differs")
        elif (
            self.status != "fail"
            or any(value is None for value in p0_fields)
            or any(value is not None for value in r2_fields)
        ):
            raise ValueError("M5 R3 P0 observation shape differs")
        if self.accepted_languages is not None and (
            list(self.accepted_languages) != ["en", "zh"]
            or sum(self.accepted_languages.values()) != self.accepted_samples
        ):
            raise ValueError("M5 R3 accepted-language counts differ")
        return self


class M5R3StrategyAlternative(StrictSchema):
    """One reviewed source strategy and its preregistered disposition."""

    strategy: M5R3TeacherStrategy
    disposition: Literal["selected_for_p1", "control_only", "rejected"]
    evidence_reason: Literal[
        "p0_and_p0_r1_failed_same_gate",
        "r2_1536_projection_failed_99_percent",
        "separates_correctness_from_length_control",
        "deterministic_control_not_formal_teacher_source",
    ]


class M5R3TeacherSourceStrategyReview(StrictSchema):
    """Path-free decision derived only from committed real evidence."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["two_stage_contract_authorized"]
    evidence_kind: Literal["deterministic_review_of_real_public_results"]
    quality_metric: Literal[False]
    review_version: Literal["m5-r3-teacher-source-strategy-v1"]
    review_config_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_r2_decision_sha256: Literal[
        "04165538efce811240b4d4501b13f74151758af7373704c12d6df882e3044ed6"
    ]
    parent_p0_result_sha256: Literal[
        "5eff250ef4cde98d044c992a0aaf7e2eb75342faa9c377d265a25945a3d4388b"
    ]
    parent_p0_r1_result_sha256: Literal[
        "c59ab59fd048620e2b8de6a985a5a1deb877bb786d9b28b813531437b582c0b7"
    ]
    observations: tuple[
        M5R3StrategyObservation,
        M5R3StrategyObservation,
        M5R3StrategyObservation,
    ]
    alternatives: tuple[
        M5R3StrategyAlternative,
        M5R3StrategyAlternative,
        M5R3StrategyAlternative,
        M5R3StrategyAlternative,
    ]
    selected_strategy: Literal["two_stage_solve_compress"]
    controlled_baseline: Literal["deterministic_rule_trace"]
    next_pilot_version: Literal["m5-r3-p1-two-stage-v1"]
    p1_contract_implementation_authorized: Literal[True]
    p1_gpu_pilot_authorized: Literal[False]
    formal_source_expansion_authorized: Literal[False]
    r3_mixture_authorized: Literal[False]
    r3_training_authorized: Literal[False]
    consumes_m6_frozen_results: Literal[False] = False
    decision_reason: Literal[
        "single_stage_length_control_failed_select_two_stage_with_rule_control"
    ]

    @model_validator(mode="after")
    def validate_decision(self) -> M5R3TeacherSourceStrategyReview:
        """Bind the selected strategy to ordered observations and alternatives."""

        expected_observations = (
            {
                "experiment": "r2",
                "status": "length_ceiling_insufficient",
                "accepted_samples": None,
                "accepted_per_family": None,
                "accepted_languages": None,
                "reasoning_over_192_tokens": None,
                "teacher_length_limit": None,
                "projected_format_basis_points": (9800, 9650),
                "unresolved_format_items": (4, 7),
            },
            {
                "experiment": "p0",
                "status": "fail",
                "accepted_samples": 10,
                "accepted_per_family": (5, 5),
                "accepted_languages": {"en": 9, "zh": 1},
                "reasoning_over_192_tokens": 52,
                "teacher_length_limit": 11,
                "projected_format_basis_points": None,
                "unresolved_format_items": None,
            },
            {
                "experiment": "p0_r1",
                "status": "fail",
                "accepted_samples": 12,
                "accepted_per_family": (4, 8),
                "accepted_languages": {"en": 10, "zh": 2},
                "reasoning_over_192_tokens": 46,
                "teacher_length_limit": 14,
                "projected_format_basis_points": None,
                "unresolved_format_items": None,
            },
        )
        if (
            tuple(item.experiment for item in self.observations) != ("r2", "p0", "p0_r1")
            or tuple(item.model_dump() for item in self.observations) != expected_observations
            or tuple(item.strategy for item in self.alternatives)
            != (
                "single_stage_prompt_control",
                "higher_generation_ceiling_only",
                "two_stage_solve_compress",
                "deterministic_rule_trace",
            )
            or tuple(item.disposition for item in self.alternatives)
            != ("rejected", "rejected", "selected_for_p1", "control_only")
            or tuple(item.evidence_reason for item in self.alternatives)
            != (
                "p0_and_p0_r1_failed_same_gate",
                "r2_1536_projection_failed_99_percent",
                "separates_correctness_from_length_control",
                "deterministic_control_not_formal_teacher_source",
            )
        ):
            raise ValueError("M5 R3 source-strategy decision order differs")
        return self
