"""Strict contracts for the M5.2-R3 240-to-160 formal source expansion."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.m5_r3_p1_schema import (
    M5R3P1RejectionReason,
    M5R3P1StageGeneration,
    M5R3P1TaskContext,
)
from tinyllm.data.m5_r3_schema import M5R3TargetFamily
from tinyllm.data.m5_r3_source_strategy_schema import M5R3P1TracePolicy
from tinyllm.data.reasoning_schema import ReasoningLanguage
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

CONTENT_REVIEW_RESULT_SHA256 = "ec42e7a3f62d5db7953677a75960e3c7e3bd6a328782e2353ea0130ddf4211ae"


class M5R3FormalTaskPolicy(StrictSchema):
    """Balanced deterministic 240-task source policy."""

    task_seed: Literal[20260808]
    target_families: tuple[Literal["config"], Literal["log_diagnosis"]]
    tasks_per_family: Literal[120]
    language_counts_per_family: dict[ReasoningLanguage, int]
    variants_per_label: Literal[30]

    @field_validator("target_families", mode="before")
    @classmethod
    def normalize_families(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_policy(self) -> M5R3FormalTaskPolicy:
        if (
            self.target_families != ("config", "log_diagnosis")
            or self.language_counts_per_family != {"en": 84, "zh": 36}
            or list(self.language_counts_per_family) != ["en", "zh"]
        ):
            raise ValueError("M5 R3 formal task policy differs")
        return self


class M5R3FormalTeacherStage(StrictSchema):
    """Pinned formal-source Teacher identity."""

    repository: Literal["Qwen/Qwen3-8B"]
    revision: Literal["b968826d9c46dd6066d109eabc6255188de91218"]
    attention_architecture: Literal["gqa"]
    trust_remote_code: Literal[False]
    local_files_only: Literal[True]
    dtype: Literal["bfloat16"]


class M5R3FormalSolver(M5R3FormalTeacherStage):
    """One concise Thinking solve attempt per formal task."""

    mode: Literal["thinking"]
    do_sample: Literal[True]
    temperature: float
    top_p: float
    top_k: Literal[20]
    repetition_penalty: float
    candidate_count: Literal[1]
    max_new_tokens: Literal[896]
    base_seed: Literal[20260809]
    prompt_protocol: Literal["m5-r3-concise-solver-v2"]

    @model_validator(mode="after")
    def validate_sampling(self) -> M5R3FormalSolver:
        if (self.temperature, self.top_p, self.repetition_penalty) != (0.6, 0.95, 1.0):
            raise ValueError("M5 R3 formal solver sampling differs")
        return self


class M5R3FormalCompressor(M5R3FormalTeacherStage):
    """Greedy isolated compressor for verified formal answers."""

    mode: Literal["nonthinking"]
    do_sample: Literal[False]
    candidate_count: Literal[1]
    max_new_tokens: Literal[256]
    base_seed: Literal[20260810]
    input_protocol: Literal["verified-evidence-answer-only-v1"]
    output_protocol: Literal["m5-r3-compressed-rationale-json-v2"]


class M5R3FormalSelectionPolicy(StrictSchema):
    """Frozen 160-sample stratified selection."""

    selected_per_family: dict[M5R3TargetFamily, int]
    selected_languages_per_family: dict[M5R3TargetFamily, dict[ReasoningLanguage, int]]
    stable_sort: tuple[
        Literal["reasoning_tokens"],
        Literal["repeated_8gram_basis_points"],
        Literal["sample_id"],
    ]
    max_source_reuse: Literal[4]

    @field_validator("stable_sort", mode="before")
    @classmethod
    def normalize_sort(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_selection(self) -> M5R3FormalSelectionPolicy:
        expected_languages = {
            "config": {"en": 56, "zh": 24},
            "log_diagnosis": {"en": 56, "zh": 24},
        }
        if (
            self.selected_per_family != {"config": 80, "log_diagnosis": 80}
            or self.selected_languages_per_family != expected_languages
            or self.stable_sort != ("reasoning_tokens", "repeated_8gram_basis_points", "sample_id")
        ):
            raise ValueError("M5 R3 formal selection policy differs")
        return self


class M5R3FormalSourceConfig(StrictSchema):
    """Complete formal source-expansion contract."""

    schema_version: Literal["1.0"]
    expansion_version: Literal["m5-r3-formal-source-v1"]
    parent_content_review_sha256: Literal[
        "ec42e7a3f62d5db7953677a75960e3c7e3bd6a328782e2353ea0130ddf4211ae"
    ]
    task_policy: M5R3FormalTaskPolicy
    solver: M5R3FormalSolver
    compressor: M5R3FormalCompressor
    trace_policy: M5R3P1TracePolicy
    selection: M5R3FormalSelectionPolicy
    r3_mixture_authorized: Literal[False]
    r3_training_authorized: Literal[False]
    consume_m6_frozen_results: Literal[False]


class M5R3FormalContaminationReport(StrictSchema):
    """Collision counts against every frozen M5 task source."""

    schema_version: Literal["1.0"] = "1.0"
    algorithm: Literal["m5-r3-formal-exact-normalized-template-v1"]
    task_set_sha256: str = Field(pattern=SHA256_PATTERN)
    formal_task_count: Literal[240]
    dev_task_count: Literal[200]
    historical_pilot_task_count: Literal[100]
    parent_p0_task_count: Literal[40]
    parent_p0_r1_task_count: Literal[40]
    parent_p1_task_count: Literal[40]
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
    p1_exact_prompt_matches: int = Field(ge=0)
    p1_normalized_prompt_matches: int = Field(ge=0)
    p1_template_family_overlaps: int = Field(ge=0)
    status: Literal["pass", "fail"]

    @model_validator(mode="after")
    def validate_status(self) -> M5R3FormalContaminationReport:
        counts = tuple(
            value
            for key, value in self.model_dump().items()
            if key.endswith("_matches") or key.endswith("_overlaps")
        )
        if self.status != ("pass" if sum(counts) == 0 else "fail"):
            raise ValueError("M5 R3 formal contamination status differs")
        return self


class M5R3FormalStratumResult(StrictSchema):
    """Acceptance and selection accounting for one family/language stratum."""

    task_family: M5R3TargetFamily
    language: ReasoningLanguage
    input_tasks: int = Field(ge=36, le=84)
    accepted_items: int = Field(ge=0, le=84)
    required_items: int = Field(ge=24, le=56)
    selected_items: int = Field(ge=0, le=56)
    gate_passed: bool

    @model_validator(mode="after")
    def validate_stratum(self) -> M5R3FormalStratumResult:
        expected = 56 if self.language == "en" else 24
        expected_input = 84 if self.language == "en" else 36
        if (
            self.required_items != expected
            or self.input_tasks != expected_input
            or self.selected_items != min(self.accepted_items, self.required_items)
            or self.gate_passed != (self.selected_items == self.required_items)
        ):
            raise ValueError("M5 R3 formal stratum accounting differs")
        return self


class M5R3FormalShardArtifact(StrictSchema):
    """Private independently reproducible generation shard."""

    schema_version: Literal["1.0"]
    expansion_version: Literal["m5-r3-formal-source-v1"]
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    shard_index: int = Field(ge=0, le=7)
    shard_count: int = Field(ge=1, le=8)
    physical_gpu_index: int = Field(ge=0, le=9)
    gpu_name: str = Field(min_length=1, max_length=128)
    torch_version: str = Field(min_length=1, max_length=64)
    transformers_version: str = Field(min_length=1, max_length=64)
    task_ids: tuple[str, ...] = Field(min_length=1, max_length=240)
    contexts: tuple[M5R3P1TaskContext, ...] = Field(min_length=1, max_length=240)
    generations: tuple[M5R3P1StageGeneration, ...] = Field(min_length=1, max_length=480)
    duration_seconds: float = Field(gt=0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)

    @field_validator("task_ids", "contexts", "generations", mode="before")
    @classmethod
    def normalize_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_shard(self) -> M5R3FormalShardArtifact:
        context_ids = tuple(item.task.id for item in self.contexts)
        solver_ids = tuple(item.task_id for item in self.generations if item.stage == "solver")
        if (
            self.shard_index >= self.shard_count
            or len(set(self.task_ids)) != len(self.task_ids)
            or context_ids != self.task_ids
            or solver_ids != self.task_ids
            or self.peak_reserved_bytes < self.peak_allocated_bytes
        ):
            raise ValueError("M5 R3 formal shard accounting differs")
        return self


class M5R3FormalCPUSmoke(StrictSchema):
    """Synthetic formal-source contract evidence."""

    schema_version: Literal["1.0"]
    evidence_kind: Literal["synthetic_cpu_contract_smoke"]
    model_generated: Literal[False]
    quality_metric: Literal[False]
    expansion_version: Literal["m5-r3-formal-source-v1"]
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    task_set_sha256: str = Field(pattern=SHA256_PATTERN)
    input_tasks: Literal[240]
    accepted_samples: Literal[240]
    selected_samples: Literal[160]
    stratum_results: tuple[
        M5R3FormalStratumResult,
        M5R3FormalStratumResult,
        M5R3FormalStratumResult,
        M5R3FormalStratumResult,
    ]
    contamination: M5R3FormalContaminationReport
    tested_failure_paths: tuple[str, ...] = Field(min_length=3)
    gpu_expansion_authorized: Literal[True]
    r3_mixture_authorized: Literal[False]
    r3_training_authorized: Literal[False]


class M5R3FormalSourceResult(StrictSchema):
    """Path-free public result for generation and 160-sample selection."""

    schema_version: Literal["1.0"]
    status: Literal["pass", "fail"]
    expansion_version: Literal["m5-r3-formal-source-v1"]
    generated_at: datetime
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_content_review_sha256: Literal[
        "ec42e7a3f62d5db7953677a75960e3c7e3bd6a328782e2353ea0130ddf4211ae"
    ]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    solver: M5R3FormalSolver
    compressor: M5R3FormalCompressor
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    physical_gpu_indices: tuple[int, ...] = Field(min_length=1, max_length=8)
    gpu_names: tuple[str, ...] = Field(min_length=1, max_length=8)
    torch_version: str = Field(min_length=1, max_length=64)
    transformers_version: str = Field(min_length=1, max_length=64)
    policy_tokenizers_version: Literal["0.21.4"]
    shard_count: int = Field(ge=1, le=8)
    shard_artifact_sha256s: tuple[str, ...] = Field(min_length=1, max_length=8)
    input_tasks: Literal[240]
    solver_attempts: Literal[240]
    compressor_attempts: int = Field(ge=0, le=240)
    accepted_samples: int = Field(ge=0, le=240)
    rejected_tasks: int = Field(ge=0, le=240)
    selected_samples: int = Field(ge=0, le=160)
    stratum_results: tuple[
        M5R3FormalStratumResult,
        M5R3FormalStratumResult,
        M5R3FormalStratumResult,
        M5R3FormalStratumResult,
    ]
    rejection_counts: dict[M5R3P1RejectionReason, int]
    contamination: M5R3FormalContaminationReport
    task_set_sha256: str = Field(pattern=SHA256_PATTERN)
    accepted_samples_sha256: str = Field(pattern=SHA256_PATTERN)
    selected_samples_sha256: str = Field(pattern=SHA256_PATTERN)
    duration_seconds: float = Field(gt=0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    raw_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    selected_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    formal_source_expansion_complete: bool
    r3_mixture_authorized: bool
    r3_training_authorized: Literal[False]
    consumes_m6_frozen_results: Literal[False] = False

    @field_validator("generated_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("M5 R3 formal timestamp must use UTC")
        return value

    @field_validator(
        "physical_gpu_indices",
        "gpu_names",
        "shard_artifact_sha256s",
        mode="before",
    )
    @classmethod
    def normalize_runtime_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("shard_artifact_sha256s")
    @classmethod
    def validate_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            len(item) != 64 or any(char not in "0123456789abcdef" for char in item)
            for item in value
        ):
            raise ValueError("M5 R3 formal shard hash differs")
        return value

    @field_validator("rejection_counts")
    @classmethod
    def validate_sparse_counts(
        cls, value: dict[M5R3P1RejectionReason, int]
    ) -> dict[M5R3P1RejectionReason, int]:
        if list(value) != sorted(value) or any(count <= 0 for count in value.values()):
            raise ValueError("M5 R3 formal rejection counts differ")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> M5R3FormalSourceResult:
        passed = (
            all(item.gate_passed for item in self.stratum_results)
            and self.selected_samples == 160
            and self.contamination.status == "pass"
        )
        if (
            tuple((item.task_family, item.language) for item in self.stratum_results)
            != (
                ("config", "en"),
                ("config", "zh"),
                ("log_diagnosis", "en"),
                ("log_diagnosis", "zh"),
            )
            or self.accepted_samples + self.rejected_tasks != self.input_tasks
            or sum(self.rejection_counts.values()) != self.rejected_tasks
            or self.shard_count != len(self.shard_artifact_sha256s)
            or self.shard_count != len(self.physical_gpu_indices)
            or self.shard_count != len(self.gpu_names)
            or self.peak_reserved_bytes < self.peak_allocated_bytes
            or self.status != ("pass" if passed else "fail")
            or self.formal_source_expansion_complete != passed
            or self.r3_mixture_authorized != passed
        ):
            raise ValueError("M5 R3 formal result accounting differs")
        return self
