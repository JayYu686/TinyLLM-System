"""Strict private and public contracts for the M5.2-R3 P1 two-stage pilot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.m5_r3_schema import M5R3TargetFamily
from tinyllm.data.m5_r3_source_strategy_schema import (
    M5R3P1CompressorStage,
    M5R3P1SolverStage,
)
from tinyllm.data.reasoning_schema import (
    ReasoningLanguage,
    ReasoningTask,
)
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M5R3P1Stage = Literal["solver", "compressor"]
M5R3P1GenerationStatus = Literal["succeeded", "failed"]
M5R3P1FinishReason = Literal["stop", "length", "error"]
M5R3P1RejectionReason = Literal[
    "compressor_answer_mismatch",
    "compressor_empty_reasoning",
    "compressor_invalid_json",
    "compressor_length_limit",
    "compressor_runtime_error",
    "duplicate_normalized_trace",
    "identical_line_repetition",
    "missing_evidence_anchor",
    "other_label_mentioned",
    "reasoning_over_192_tokens",
    "repeated_8gram_over_500bp",
    "sequence_over_1024_tokens",
    "solver_answer_mismatch",
    "solver_invalid_output",
    "solver_length_limit",
    "solver_runtime_error",
]


class M5R3P1TaskContext(StrictSchema):
    """Private P1 task plus frozen evidence required by the compressor verifier."""

    task: ReasoningTask
    evidence: str = Field(min_length=1, max_length=4096)
    evidence_anchor: str = Field(min_length=1, max_length=1024)
    label_key: Literal["issue", "root_cause"]
    allowed_labels: tuple[str, str, str, str]
    expected_label: str = Field(min_length=1, max_length=64)

    @field_validator("allowed_labels", mode="before")
    @classmethod
    def normalize_allowed_labels(cls, value: object) -> object:
        """Restore the immutable tuple after a JSON array round trip."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_context(self) -> M5R3P1TaskContext:
        """Bind task family, answer, evidence anchor, and closed labels."""

        if not self.task.id.startswith(
            ("m5-reasoning:pilot:r3p1-", "m5-reasoning:pilot:r3formal-")
        ):
            raise ValueError("M5 R3 two-stage task ID differs")
        expected_key = "issue" if self.task.task_family == "config" else "root_cause"
        try:
            decoded = json.loads(self.task.expected_answer_json)
        except json.JSONDecodeError as exc:
            raise ValueError("M5 R3 P1 expected answer is invalid") from exc

        def normalize(value: str) -> str:
            return " ".join(value.casefold().split())

        if (
            self.label_key != expected_key
            or tuple(sorted(self.allowed_labels)) != self.allowed_labels
            or self.expected_label not in self.allowed_labels
            or decoded != {self.label_key: self.expected_label}
            or normalize(self.evidence_anchor) not in normalize(self.evidence)
            or self.evidence not in self.task.prompt
        ):
            raise ValueError("M5 R3 P1 task context is inconsistent")
        return self


class M5R3P1StageGeneration(StrictSchema):
    """One private solver or compressor generation with explicit lineage."""

    schema_version: Literal["1.0"] = "1.0"
    generation_id: str = Field(
        pattern=(
            r"^m5-reasoning:pilot:r3(?:p1|formal)-"
            r"(config|log)-(en|zh)-\d{3}:(solver|compressor)$"
        )
    )
    task_id: str = Field(pattern=r"^m5-reasoning:pilot:r3(?:p1|formal)-(config|log)-(en|zh)-\d{3}$")
    stage: M5R3P1Stage
    seed: int = Field(ge=0, le=2**32 - 1)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    status: M5R3P1GenerationStatus
    finish_reason: M5R3P1FinishReason
    raw_output: str | None = Field(default=None, max_length=65536)
    raw_output_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    input_token_count: int = Field(ge=0)
    generated_token_count: int = Field(ge=0)
    error_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{2,63}$")

    @model_validator(mode="after")
    def validate_generation(self) -> M5R3P1StageGeneration:
        """Bind stage identity and success/failure output evidence."""

        if self.generation_id != f"{self.task_id}:{self.stage}":
            raise ValueError("M5 R3 P1 generation ID differs")
        if self.status == "succeeded":
            if (
                self.finish_reason == "error"
                or self.error_code is not None
                or self.raw_output is None
                or not self.raw_output.strip()
                or self.raw_output_sha256 is None
                or self.input_token_count == 0
                or self.generated_token_count == 0
            ):
                raise ValueError("successful M5 R3 P1 generation is incomplete")
            if hashlib.sha256(self.raw_output.encode()).hexdigest() != self.raw_output_sha256:
                raise ValueError("M5 R3 P1 output hash differs")
        elif (
            self.finish_reason != "error"
            or self.error_code is None
            or self.raw_output is not None
            or self.raw_output_sha256 is not None
            or self.input_token_count != 0
            or self.generated_token_count != 0
        ):
            raise ValueError("failed M5 R3 P1 generation is inconsistent")
        return self


class M5R3P1CompressedEnvelope(StrictSchema):
    """Strict model-generated rationale envelope returned by the compressor."""

    reasoning: str = Field(min_length=1, max_length=8192)
    final_answer: dict[str, str]

    @field_validator("reasoning")
    @classmethod
    def reject_blank_or_tags(cls, value: str) -> str:
        """Reject blank rationale and embedded ChatML/Thinking control text."""

        if not value.strip() or any(tag in value.casefold() for tag in ("<think", "</think", "<|")):
            raise ValueError("M5 R3 P1 compressed rationale contains control text")
        return value

    @field_validator("final_answer")
    @classmethod
    def require_single_string_field(cls, value: dict[str, str]) -> dict[str, str]:
        """Require one non-blank key/value pair."""

        if len(value) != 1 or any(not key or not item.strip() for key, item in value.items()):
            raise ValueError("M5 R3 P1 compressed final answer differs")
        return value


class M5R3P1CandidateAudit(StrictSchema):
    """Content-free final decision for one P1 task."""

    task_id: str = Field(pattern=r"^m5-reasoning:pilot:r3(?:p1|formal)-(config|log)-(en|zh)-\d{3}$")
    task_family: M5R3TargetFamily
    language: ReasoningLanguage
    status: Literal["accepted", "rejected"]
    rejection_reason: M5R3P1RejectionReason | None
    solver_output_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    compressor_output_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    reasoning_tokens: int | None = Field(default=None, gt=0, le=256)
    repeated_8gram_basis_points: int | None = Field(default=None, ge=0, le=10_000)
    max_identical_line_hash_repetitions: int | None = Field(default=None, gt=0)
    normalized_trace_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    evidence_anchor_matched: bool | None = None
    other_label_mentions: int | None = Field(default=None, ge=0, le=3)
    training_sequence_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_audit(self) -> M5R3P1CandidateAudit:
        """Require complete accepted metrics and an explicit rejected reason."""

        metrics = (
            self.solver_output_sha256,
            self.compressor_output_sha256,
            self.reasoning_tokens,
            self.repeated_8gram_basis_points,
            self.max_identical_line_hash_repetitions,
            self.normalized_trace_sha256,
            self.evidence_anchor_matched,
            self.other_label_mentions,
            self.training_sequence_tokens,
        )
        if self.status == "accepted":
            if (
                self.rejection_reason is not None
                or any(value is None for value in metrics)
                or self.evidence_anchor_matched is not True
                or self.other_label_mentions != 0
            ):
                raise ValueError("accepted M5 R3 P1 audit is incomplete")
        elif self.rejection_reason is None:
            raise ValueError("rejected M5 R3 P1 audit requires a reason")
        return self


class M5R3P1ContaminationReport(StrictSchema):
    """Path-free P1 collision counts against all frozen prior task sets."""

    schema_version: Literal["1.0"] = "1.0"
    algorithm: Literal["m5-r3-p1-exact-normalized-template-v1"]
    task_set_sha256: str = Field(pattern=SHA256_PATTERN)
    p1_task_count: Literal[40]
    dev_task_count: Literal[200]
    historical_pilot_task_count: Literal[100]
    parent_p0_task_count: Literal[40]
    parent_p0_r1_task_count: Literal[40]
    dev_exact_prompt_matches: int = Field(ge=0)
    dev_template_family_overlaps: int = Field(ge=0)
    historical_exact_prompt_matches: int = Field(ge=0)
    historical_normalized_prompt_matches: int = Field(ge=0)
    historical_template_family_overlaps: int = Field(ge=0)
    p0_exact_prompt_matches: int = Field(ge=0)
    p0_normalized_prompt_matches: int = Field(ge=0)
    p0_template_family_overlaps: int = Field(ge=0)
    p0_r1_exact_prompt_matches: int = Field(ge=0)
    p0_r1_normalized_prompt_matches: int = Field(ge=0)
    p0_r1_template_family_overlaps: int = Field(ge=0)
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_status(self) -> M5R3P1ContaminationReport:
        """Derive status from every collision count."""

        counts = tuple(
            value
            for key, value in self.model_dump().items()
            if key.endswith("_matches") or key.endswith("_overlaps")
        )
        if self.status != ("pass" if sum(counts) == 0 else "fail"):
            raise ValueError("M5 R3 P1 contamination status differs")
        return self


class M5R3P1FamilyResult(StrictSchema):
    """Public accepted distribution for one P1 family."""

    task_family: M5R3TargetFamily
    input_tasks: Literal[20]
    input_language_counts: dict[ReasoningLanguage, int]
    accepted_items: int = Field(ge=0, le=20)
    accepted_language_counts: dict[ReasoningLanguage, int]
    reasoning_tokens_min: int | None = Field(default=None, gt=0, le=192)
    reasoning_tokens_p50: float | None = Field(default=None, gt=0, le=192)
    reasoning_tokens_p90: int | None = Field(default=None, gt=0, le=192)
    reasoning_tokens_max: int | None = Field(default=None, gt=0, le=192)
    gate_passed: bool

    @model_validator(mode="after")
    def validate_family(self) -> M5R3P1FamilyResult:
        """Bind language counts, statistics, and the 14/10/4 Gate."""

        stats = (
            self.reasoning_tokens_min,
            self.reasoning_tokens_p50,
            self.reasoning_tokens_p90,
            self.reasoning_tokens_max,
        )
        if (
            self.input_language_counts != {"en": 14, "zh": 6}
            or list(self.input_language_counts) != ["en", "zh"]
            or list(self.accepted_language_counts) != ["en", "zh"]
            or sum(self.accepted_language_counts.values()) != self.accepted_items
            or (self.accepted_items == 0) != all(value is None for value in stats)
        ):
            raise ValueError("M5 R3 P1 family counts differ")
        if self.accepted_items:
            minimum, p50, p90, maximum = stats
            assert (
                minimum is not None and p50 is not None and p90 is not None and maximum is not None
            )
            if not minimum <= p50 <= p90 <= maximum:
                raise ValueError("M5 R3 P1 family statistics are unordered")
        expected_gate = (
            self.accepted_items >= 14
            and self.accepted_language_counts["en"] >= 10
            and self.accepted_language_counts["zh"] >= 4
        )
        if self.gate_passed != expected_gate:
            raise ValueError("M5 R3 P1 family Gate differs")
        return self


class M5R3P1ControlResult(StrictSchema):
    """Structural result for deterministic rule traces."""

    source_kind: Literal["deterministic_rule_trace"]
    input_tasks: Literal[40]
    structural_passes: int = Field(ge=0, le=40)
    reasoning_tokens_max: int = Field(gt=0, le=192)
    unique_trace_count: int = Field(ge=0, le=40)
    status: Literal["pass", "fail"]
    training_source_authorized: Literal[False]

    @model_validator(mode="after")
    def validate_control(self) -> M5R3P1ControlResult:
        """Require all 40 traces to pass for a control success."""

        expected = "pass" if self.structural_passes == self.unique_trace_count == 40 else "fail"
        if self.status != expected:
            raise ValueError("M5 R3 P1 control status differs")
        return self


class M5R3P1CPUSmoke(StrictSchema):
    """Synthetic content-free contract evidence authorizing only a real GPU Pilot."""

    schema_version: Literal["1.0"] = "1.0"
    evidence_kind: Literal["synthetic_cpu_contract_smoke"]
    model_generated: Literal[False]
    quality_metric: Literal[False]
    pilot_version: Literal["m5-r3-p1-two-stage-v1"]
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    task_set_sha256: str = Field(pattern=SHA256_PATTERN)
    samples_sha256: str = Field(pattern=SHA256_PATTERN)
    accepted_samples: Literal[40]
    family_results: tuple[M5R3P1FamilyResult, M5R3P1FamilyResult]
    control: M5R3P1ControlResult
    contamination: M5R3P1ContaminationReport
    tested_failure_paths: tuple[
        Literal["compressor_missing_evidence_anchor"],
        Literal["parent_task_contamination"],
        Literal["solver_lineage_drift"],
    ]
    p1_gpu_pilot_authorized: bool
    formal_source_expansion_authorized: Literal[False]
    r3_mixture_authorized: Literal[False]
    r3_training_authorized: Literal[False]
    consumes_m6_frozen_results: Literal[False] = False

    @model_validator(mode="after")
    def validate_smoke(self) -> M5R3P1CPUSmoke:
        """Authorize only P1 GPU execution after all synthetic contracts pass."""

        expected_authorized = (
            all(item.gate_passed for item in self.family_results)
            and self.control.status == "pass"
            and self.contamination.status == "pass"
            and self.tested_failure_paths
            == (
                "compressor_missing_evidence_anchor",
                "parent_task_contamination",
                "solver_lineage_drift",
            )
        )
        if (
            tuple(item.task_family for item in self.family_results) != ("config", "log_diagnosis")
            or self.p1_gpu_pilot_authorized != expected_authorized
        ):
            raise ValueError("M5 R3 P1 CPU Smoke authorization differs")
        return self


class M5R3P1Result(StrictSchema):
    """Public path-free result of one real two-stage P1 GPU Pilot."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["pass", "fail"]
    pilot_version: Literal["m5-r3-p1-two-stage-v1"]
    generated_at: datetime
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    solver: M5R3P1SolverStage
    compressor: M5R3P1CompressorStage
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    physical_gpu_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1, max_length=128)
    torch_version: str = Field(min_length=1, max_length=64)
    transformers_version: str = Field(min_length=1, max_length=64)
    teacher_tokenizers_version: str = Field(min_length=1, max_length=64)
    policy_tokenizers_version: Literal["0.21.4"]
    input_tasks: Literal[40]
    solver_attempts: int = Field(ge=1, le=40)
    compressor_attempts: int = Field(ge=0, le=40)
    accepted_samples: int = Field(ge=0, le=40)
    rejected_tasks: int = Field(ge=0, le=40)
    family_results: tuple[M5R3P1FamilyResult, M5R3P1FamilyResult]
    rejection_counts: dict[M5R3P1RejectionReason, int]
    control: M5R3P1ControlResult
    contamination: M5R3P1ContaminationReport
    task_set_sha256: str = Field(pattern=SHA256_PATTERN)
    samples_sha256: str = Field(pattern=SHA256_PATTERN)
    duration_seconds: float = Field(gt=0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    raw_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    formal_source_expansion_authorized: bool
    r3_mixture_authorized: Literal[False] = False
    r3_training_authorized: Literal[False] = False
    consumes_m6_frozen_results: Literal[False] = False

    @field_validator("generated_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        """Require an aware UTC timestamp."""

        offset = value.utcoffset()
        if value.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("M5 R3 P1 timestamp must use UTC")
        return value

    @field_validator("rejection_counts")
    @classmethod
    def validate_rejections(
        cls,
        value: dict[M5R3P1RejectionReason, int],
    ) -> dict[M5R3P1RejectionReason, int]:
        """Require sorted positive sparse counts."""

        if list(value) != sorted(value) or any(count <= 0 for count in value.values()):
            raise ValueError("M5 R3 P1 rejection counts differ")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> M5R3P1Result:
        """Bind accounting, Gate status, and expansion authorization."""

        passed = (
            all(item.gate_passed for item in self.family_results)
            and self.control.status == "pass"
            and self.contamination.status == "pass"
            and not self.git_dirty
        )
        if (
            tuple(item.task_family for item in self.family_results) != ("config", "log_diagnosis")
            or self.accepted_samples != sum(item.accepted_items for item in self.family_results)
            or self.accepted_samples + self.rejected_tasks != 40
            or sum(self.rejection_counts.values()) != self.rejected_tasks
            or self.peak_reserved_bytes < self.peak_allocated_bytes
            or self.status != ("pass" if passed else "fail")
            or self.formal_source_expansion_authorized != passed
        ):
            raise ValueError("M5 R3 P1 result accounting or authorization differs")
        return self
