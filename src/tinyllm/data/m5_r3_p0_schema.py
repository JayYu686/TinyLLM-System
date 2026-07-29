"""Strict contracts for the M5.2-R3-P0 targeted Teacher Pilot."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.m5_r3_schema import M5R3TargetFamily, M5R3TracePolicy
from tinyllm.data.reasoning_schema import (
    ReasoningLanguage,
    ReasoningTeacherIdentity,
    ReasoningVerifierIdentity,
)
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M5R3P0RejectionReason = Literal[
    "answer_mismatch",
    "duplicate_normalized_trace",
    "empty_final_answer",
    "empty_reasoning",
    "generation_runtime_error",
    "identical_line_repetition",
    "invalid_final_json",
    "missing_think_block",
    "multiple_think_blocks",
    "nested_think_tag",
    "no_candidate_passed",
    "reasoning_over_192_tokens",
    "repeated_8gram_over_500bp",
    "sequence_over_1024_tokens",
    "teacher_length_limit",
]
M5R3P0CandidateStatus = Literal["accepted", "rejected"]


class M5R3P0Sampling(StrictSchema):
    """Frozen native-Thinking sampling identity for the bounded P0 experiment."""

    do_sample: Literal[True]
    temperature: float
    top_p: float
    top_k: Literal[20]
    repetition_penalty: float
    candidate_count: Literal[2]
    max_new_tokens: Literal[384]
    base_seed: Literal[20260731]

    @model_validator(mode="after")
    def validate_sampling(self) -> M5R3P0Sampling:
        """Freeze the three floating-point sampling fields exactly."""

        if (self.temperature, self.top_p, self.repetition_penalty) != (0.6, 0.95, 1.0):
            raise ValueError("M5 R3 P0 sampling floats differ")
        return self


class M5R3P0Gate(StrictSchema):
    """Per-family and per-language acceptance gate for P0."""

    accepted_per_family: dict[M5R3TargetFamily, Literal[14]]
    accepted_languages_per_family: dict[
        M5R3TargetFamily,
        dict[ReasoningLanguage, int],
    ]

    @model_validator(mode="after")
    def validate_gate(self) -> M5R3P0Gate:
        """Freeze 14 accepted traces with 10/4 language coverage per family."""

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
            raise ValueError("M5 R3 P0 Gate must require 14 traces and 10/4 languages per family")
        return self


class M5R3P0Config(StrictSchema):
    """Complete immutable configuration for the 40-task P0 Teacher experiment."""

    schema_version: Literal["1.0"]
    pilot_version: Literal["m5-r3-p0-v1"]
    parent_source_audit_config_sha256: Literal[
        "a3bf415bfd2e950596ed338576bee3b675fffd1dc8c0b91c14da73af0c8a83a4"
    ]
    parent_source_audit_result_sha256: Literal[
        "538e008ecb9975f04f440d3de0807fc15929914dd59cb7a63a780aab492b66a6"
    ]
    historical_pilot_raw_sha256: Literal[
        "5e4e75df8a0843376d95a9e47e6d91c0d0456e5066d944e315e5b96173530411"
    ]
    reasoning_config_sha256: Literal[
        "d6aee88bf4a3922981465026f202dcffcc0da0294126fac27602eb2442df1a4b"
    ]
    tokenization_config_sha256: Literal[
        "f2c3e3fc05534344c6705befebf5761face41178fa6f3c2216f4c0cfcc90aacc"
    ]
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    task_seed: Literal[20260730]
    target_families: tuple[Literal["config"], Literal["log_diagnosis"]]
    tasks_per_family: Literal[20]
    language_counts_per_family: dict[ReasoningLanguage, int]
    evidence_variants_per_label: Literal[6]
    max_sequence_length: Literal[1024]
    teacher: ReasoningTeacherIdentity
    sampling: M5R3P0Sampling
    verifier: ReasoningVerifierIdentity
    trace_policy: M5R3TracePolicy
    gate: M5R3P0Gate
    consume_m6_frozen_results: Literal[False]

    @field_validator("target_families", mode="before")
    @classmethod
    def normalize_yaml_families(cls, value: object) -> object:
        """Convert the YAML list to the immutable runtime tuple."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_distribution(self) -> M5R3P0Config:
        """Keep P0 scope, languages, and teacher identities frozen."""

        if (
            self.target_families != ("config", "log_diagnosis")
            or self.language_counts_per_family != {"en": 14, "zh": 6}
            or list(self.language_counts_per_family) != ["en", "zh"]
            or self.teacher.revision != "b968826d9c46dd6066d109eabc6255188de91218"
            or self.teacher.attention_architecture != "gqa"
            or self.verifier.verifier_id != "m5-json-exact-v1"
        ):
            raise ValueError("M5 R3 P0 distribution or pinned identity differs")
        return self


class M5R3P0ContaminationReport(StrictSchema):
    """Content-free collision counts against frozen Dev and historical Pilot."""

    schema_version: Literal["1.0"] = "1.0"
    algorithm: Literal["m5-r3-exact-normalized-template-v1"]
    p0_tasks_sha256: str = Field(pattern=SHA256_PATTERN)
    p0_task_count: Literal[40]
    dev_task_count: Literal[200]
    historical_pilot_task_count: Literal[100]
    dev_exact_prompt_matches: int = Field(ge=0)
    dev_template_family_overlaps: int = Field(ge=0)
    historical_exact_prompt_matches: int = Field(ge=0)
    historical_normalized_prompt_matches: int = Field(ge=0)
    historical_template_family_overlaps: int = Field(ge=0)
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_status(self) -> M5R3P0ContaminationReport:
        """Derive pass/fail from all retained collision counts."""

        total = (
            self.dev_exact_prompt_matches
            + self.dev_template_family_overlaps
            + self.historical_exact_prompt_matches
            + self.historical_normalized_prompt_matches
            + self.historical_template_family_overlaps
        )
        if self.status != ("pass" if total == 0 else "fail"):
            raise ValueError("M5 R3 P0 contamination status does not match counts")
        return self


class M5R3P0CandidateAudit(StrictSchema):
    """Private candidate-level selection evidence without duplicating raw output."""

    schema_version: Literal["1.0"] = "1.0"
    task_id: str = Field(pattern=r"^m5-reasoning:pilot:r3p0-(config|log)-[a-z]{2}-\d{3}$")
    generation_id: str = Field(
        pattern=(r"^m5-reasoning:pilot:r3p0-(config|log)-[a-z]{2}-\d{3}:candidate-[01]$")
    )
    status: M5R3P0CandidateStatus
    rejection_reason: M5R3P0RejectionReason | None
    reasoning_tokens: int | None = Field(default=None, gt=0, le=384)
    repeated_8gram_basis_points: int | None = Field(default=None, ge=0, le=10_000)
    max_identical_line_hash_repetitions: int | None = Field(default=None, gt=0)
    normalized_trace_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    training_sequence_tokens: int | None = Field(default=None, gt=0)
    verification_id: str | None = Field(default=None, min_length=1, max_length=180)

    @model_validator(mode="after")
    def validate_candidate(self) -> M5R3P0CandidateAudit:
        """Require complete trace metrics only after parsing and verification."""

        metrics = (
            self.reasoning_tokens,
            self.repeated_8gram_basis_points,
            self.max_identical_line_hash_repetitions,
            self.normalized_trace_sha256,
            self.training_sequence_tokens,
            self.verification_id,
        )
        if self.status == "accepted":
            if self.rejection_reason is not None or any(value is None for value in metrics):
                raise ValueError("accepted M5 R3 P0 candidate requires metrics and no rejection")
        elif self.rejection_reason is None:
            raise ValueError("rejected M5 R3 P0 candidate requires a reason")
        return self


class M5R3P0FamilyResult(StrictSchema):
    """Public accepted-trace distribution for one targeted family."""

    task_family: M5R3TargetFamily
    input_tasks: Literal[20]
    input_language_counts: dict[ReasoningLanguage, int]
    accepted_items: int = Field(ge=0, le=20)
    accepted_language_counts: dict[ReasoningLanguage, int]
    reasoning_tokens_min: int | None = Field(default=None, gt=0, le=192)
    reasoning_tokens_p50: float | None = Field(default=None, gt=0, le=192)
    reasoning_tokens_p90: int | None = Field(default=None, gt=0, le=192)
    reasoning_tokens_max: int | None = Field(default=None, gt=0, le=192)
    repeated_8gram_mean_basis_points: int | None = Field(default=None, ge=0, le=500)
    gate_passed: bool

    @model_validator(mode="after")
    def validate_family(self) -> M5R3P0FamilyResult:
        """Bind counts, optional distributions, and the 14/10/4 Gate."""

        if (
            self.input_language_counts != {"en": 14, "zh": 6}
            or list(self.input_language_counts) != ["en", "zh"]
            or list(self.accepted_language_counts) != ["en", "zh"]
            or sum(self.accepted_language_counts.values()) != self.accepted_items
        ):
            raise ValueError("M5 R3 P0 family language counts are inconsistent")
        stats = (
            self.reasoning_tokens_min,
            self.reasoning_tokens_p50,
            self.reasoning_tokens_p90,
            self.reasoning_tokens_max,
            self.repeated_8gram_mean_basis_points,
        )
        if self.accepted_items == 0:
            if any(value is not None for value in stats):
                raise ValueError("empty M5 R3 P0 family cannot report trace statistics")
        else:
            if any(value is None for value in stats):
                raise ValueError("M5 R3 P0 trace statistics are missing or unordered")
            minimum = self.reasoning_tokens_min
            median = self.reasoning_tokens_p50
            p90 = self.reasoning_tokens_p90
            maximum = self.reasoning_tokens_max
            assert minimum is not None
            assert median is not None
            assert p90 is not None
            assert maximum is not None
            if not minimum <= median <= p90 <= maximum:
                raise ValueError("M5 R3 P0 trace statistics are missing or unordered")
        expected_gate = (
            self.accepted_items >= 14
            and self.accepted_language_counts["en"] >= 10
            and self.accepted_language_counts["zh"] >= 4
        )
        if self.gate_passed != expected_gate:
            raise ValueError("M5 R3 P0 family Gate does not match counts")
        return self


class M5R3P0Result(StrictSchema):
    """Public path-free result of one real 40-task P0 GPU experiment."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["pass", "fail"]
    pilot_version: Literal["m5-r3-p0-v1"]
    generated_at: datetime
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    model: ReasoningTeacherIdentity
    sampling: M5R3P0Sampling
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    physical_gpu_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1, max_length=128)
    torch_version: str = Field(min_length=1, max_length=64)
    transformers_version: str = Field(min_length=1, max_length=64)
    input_tasks: Literal[40]
    generation_attempts: int = Field(ge=1, le=80)
    accepted_samples: int = Field(ge=0, le=40)
    rejected_tasks: int = Field(ge=0, le=40)
    family_results: tuple[M5R3P0FamilyResult, M5R3P0FamilyResult]
    rejection_counts: dict[M5R3P0RejectionReason, int]
    contamination: M5R3P0ContaminationReport
    task_set_sha256: str = Field(pattern=SHA256_PATTERN)
    samples_sha256: str = Field(pattern=SHA256_PATTERN)
    duration_seconds: float = Field(gt=0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    raw_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    consumes_m6_frozen_results: Literal[False] = False

    @field_validator("generated_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        """Require timezone-aware UTC evidence."""

        offset = value.utcoffset()
        if value.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("M5 R3 P0 timestamp must use UTC")
        return value

    @field_validator("rejection_counts")
    @classmethod
    def validate_rejection_counts(
        cls,
        value: dict[M5R3P0RejectionReason, int],
    ) -> dict[M5R3P0RejectionReason, int]:
        """Require deterministic positive sparse rejection counts."""

        if list(value) != sorted(value) or any(count <= 0 for count in value.values()):
            raise ValueError("M5 R3 P0 rejection counts must be sorted and positive")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> M5R3P0Result:
        """Bind task accounting and pass status to both family Gates."""

        if (
            tuple(item.task_family for item in self.family_results) != ("config", "log_diagnosis")
            or self.accepted_samples != sum(item.accepted_items for item in self.family_results)
            or self.accepted_samples + self.rejected_tasks != self.input_tasks
            or self.peak_reserved_bytes < self.peak_allocated_bytes
        ):
            raise ValueError("M5 R3 P0 result accounting is inconsistent")
        expected = (
            "pass"
            if all(item.gate_passed for item in self.family_results)
            and self.contamination.status == "pass"
            and not self.git_dirty
            else "fail"
        )
        if self.status != expected:
            raise ValueError("M5 R3 P0 status does not match frozen Gates")
        return self
