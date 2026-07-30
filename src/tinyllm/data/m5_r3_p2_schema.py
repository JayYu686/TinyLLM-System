"""Strict contracts for the M5.2-R3 P2 fallback and input-isolation pilot."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.m5_r3_p1_schema import (
    M5R3P1ContaminationReport,
    M5R3P1ControlResult,
    M5R3P1FamilyResult,
    M5R3P1RejectionReason,
    M5R3P1StageGeneration,
)
from tinyllm.data.m5_r3_source_strategy_schema import M5R3P1Gate, M5R3P1TracePolicy
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

P1_RESULT_SHA256 = "c57b13d5a84a6b06450ad01ae7e9158ccc700736686575893b85aa27b92dfd95"
P1_GENERATION_SHA256 = "4d59d3d1d317ffc85cb1c5560bf14237b6455db30f30aeeed660f197e459e73e"
P1_TASK_SET_SHA256 = "ac2d020c5cf2653f67e061c3b536af2ceccc12602d668f42eead62e8812e836f"

M5R3P2FallbackReason = Literal[
    "solver_answer_mismatch",
    "solver_invalid_output",
    "solver_length_limit",
    "solver_runtime_error",
]


class M5R3P2TeacherStage(StrictSchema):
    """Pinned Qwen3-8B identity shared by both P2 stages."""

    repository: Literal["Qwen/Qwen3-8B"]
    revision: Literal["b968826d9c46dd6066d109eabc6255188de91218"]
    attention_architecture: Literal["gqa"]
    trust_remote_code: Literal[False]
    local_files_only: Literal[True]
    dtype: Literal["bfloat16"]


class M5R3P2FallbackSolver(M5R3P2TeacherStage):
    """One bounded second solver candidate for rejected P1 solver outputs."""

    mode: Literal["thinking"]
    do_sample: Literal[True]
    temperature: float
    top_p: float
    top_k: Literal[20]
    repetition_penalty: float
    candidate_count: Literal[1]
    max_new_tokens: Literal[896]
    base_seed: Literal[20260806]
    prompt_protocol: Literal["m5-r3-concise-solver-v2"]
    trigger_reasons: tuple[
        Literal["solver_answer_mismatch"],
        Literal["solver_invalid_output"],
        Literal["solver_length_limit"],
        Literal["solver_runtime_error"],
    ]

    @field_validator("trigger_reasons", mode="before")
    @classmethod
    def normalize_trigger_reasons(cls, value: object) -> object:
        """Convert YAML sequences to the frozen tuple representation."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_triggers(self) -> M5R3P2FallbackSolver:
        """Require the complete ordered P1 solver rejection set."""

        if (self.temperature, self.top_p, self.repetition_penalty) != (
            0.6,
            0.95,
            1.0,
        ) or self.trigger_reasons != (
            "solver_answer_mismatch",
            "solver_invalid_output",
            "solver_length_limit",
            "solver_runtime_error",
        ):
            raise ValueError("M5 R3 P2 fallback triggers differ")
        return self


class M5R3P2IsolatedCompressor(M5R3P2TeacherStage):
    """Greedy compressor that cannot see alternatives or raw solver reasoning."""

    mode: Literal["nonthinking"]
    do_sample: Literal[False]
    candidate_count: Literal[1]
    max_new_tokens: Literal[256]
    base_seed: Literal[20260807]
    input_protocol: Literal["verified-evidence-answer-only-v1"]
    output_protocol: Literal["m5-r3-compressed-rationale-json-v2"]


class M5R3P2Config(StrictSchema):
    """Immutable P2 protocol bound to the real rejected P1 evidence."""

    schema_version: Literal["1.0"]
    pilot_version: Literal["m5-r3-p2-fallback-isolated-v1"]
    parent_p1_result_sha256: Literal[
        "c57b13d5a84a6b06450ad01ae7e9158ccc700736686575893b85aa27b92dfd95"
    ]
    parent_p1_generation_artifact_sha256: Literal[
        "4d59d3d1d317ffc85cb1c5560bf14237b6455db30f30aeeed660f197e459e73e"
    ]
    task_set_sha256: Literal["ac2d020c5cf2653f67e061c3b536af2ceccc12602d668f42eead62e8812e836f"]
    fallback_solver: M5R3P2FallbackSolver
    isolated_compressor: M5R3P2IsolatedCompressor
    trace_policy: M5R3P1TracePolicy
    gate: M5R3P1Gate
    formal_source_expansion_authorized: Literal[False]
    r3_mixture_authorized: Literal[False]
    r3_training_authorized: Literal[False]
    consume_m6_frozen_results: Literal[False]

    @model_validator(mode="after")
    def validate_unchanged_gates(self) -> M5R3P2Config:
        """Keep P2 limited to the two observed generation failure mechanisms."""

        if (
            self.trace_policy.max_reasoning_tokens != 192
            or self.trace_policy.max_repeated_8gram_basis_points != 500
            or self.trace_policy.max_training_sequence_tokens != 1024
            or self.gate.accepted_per_family != {"config": 14, "log_diagnosis": 14}
        ):
            raise ValueError("M5 R3 P2 changed a frozen quality gate")
        return self


class M5R3P2GenerationDelta(StrictSchema):
    """Private P2-only GPU outputs linked to the complete P1 generation artifact."""

    schema_version: Literal["1.0"] = "1.0"
    pilot_version: Literal["m5-r3-p2-fallback-isolated-v1"]
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_p1_generation_artifact_sha256: Literal[
        "4d59d3d1d317ffc85cb1c5560bf14237b6455db30f30aeeed660f197e459e73e"
    ]
    task_set_sha256: Literal["ac2d020c5cf2653f67e061c3b536af2ceccc12602d668f42eead62e8812e836f"]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    physical_gpu_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1, max_length=128)
    torch_version: str = Field(min_length=1, max_length=64)
    transformers_version: str = Field(min_length=1, max_length=64)
    teacher_tokenizers_version: str = Field(min_length=1, max_length=64)
    duration_seconds: float = Field(gt=0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    fallback_solvers: tuple[M5R3P1StageGeneration, ...]
    isolated_compressors: tuple[M5R3P1StageGeneration, ...]

    @model_validator(mode="after")
    def validate_delta(self) -> M5R3P2GenerationDelta:
        """Require unique stage-correct P2 delta records."""

        fallback_ids = tuple(item.task_id for item in self.fallback_solvers)
        compressor_ids = tuple(item.task_id for item in self.isolated_compressors)
        if (
            self.peak_reserved_bytes < self.peak_allocated_bytes
            or not 1 <= len(fallback_ids) <= 40
            or len(fallback_ids) != len(set(fallback_ids))
            or len(compressor_ids) != len(set(compressor_ids))
            or any(item.stage != "solver" for item in self.fallback_solvers)
            or any(item.stage != "compressor" for item in self.isolated_compressors)
            or any(not item.startswith("m5-reasoning:pilot:r3p1-") for item in fallback_ids)
            or any(not item.startswith("m5-reasoning:pilot:r3p1-") for item in compressor_ids)
        ):
            raise ValueError("M5 R3 P2 generation delta differs")
        return self


class M5R3P2CPUSmoke(StrictSchema):
    """Synthetic contract evidence authorizing only a real P2 GPU run."""

    schema_version: Literal["1.0"] = "1.0"
    evidence_kind: Literal["synthetic_cpu_contract_smoke"]
    model_generated: Literal[False]
    quality_metric: Literal[False]
    pilot_version: Literal["m5-r3-p2-fallback-isolated-v1"]
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_p1_result_sha256: Literal[
        "c57b13d5a84a6b06450ad01ae7e9158ccc700736686575893b85aa27b92dfd95"
    ]
    task_set_sha256: Literal["ac2d020c5cf2653f67e061c3b536af2ceccc12602d668f42eead62e8812e836f"]
    fallback_solver_items: Literal[6]
    isolated_compressor_items: Literal[40]
    accepted_samples: Literal[40]
    family_results: tuple[M5R3P1FamilyResult, M5R3P1FamilyResult]
    control: M5R3P1ControlResult
    contamination: M5R3P1ContaminationReport
    tested_failure_paths: tuple[
        Literal["parent_generation_hash_drift"],
        Literal["fallback_seed_drift"],
        Literal["compressor_input_leakage"],
    ]
    p2_gpu_pilot_authorized: bool
    formal_source_expansion_authorized: Literal[False]
    r3_mixture_authorized: Literal[False]
    r3_training_authorized: Literal[False]
    consumes_m6_frozen_results: Literal[False] = False

    @model_validator(mode="after")
    def validate_smoke(self) -> M5R3P2CPUSmoke:
        """Authorize only GPU execution after every synthetic invariant passes."""

        expected = (
            all(item.gate_passed for item in self.family_results)
            and self.control.status == "pass"
            and self.contamination.status == "pass"
            and self.tested_failure_paths
            == (
                "parent_generation_hash_drift",
                "fallback_seed_drift",
                "compressor_input_leakage",
            )
        )
        if (
            tuple(item.task_family for item in self.family_results) != ("config", "log_diagnosis")
            or self.p2_gpu_pilot_authorized != expected
        ):
            raise ValueError("M5 R3 P2 CPU Smoke authorization differs")
        return self


class M5R3P2Result(StrictSchema):
    """Public path-free result of the real P2 fallback and isolation pilot."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["pass", "fail"]
    pilot_version: Literal["m5-r3-p2-fallback-isolated-v1"]
    generated_at: datetime
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_p1_result_sha256: Literal[
        "c57b13d5a84a6b06450ad01ae7e9158ccc700736686575893b85aa27b92dfd95"
    ]
    parent_p1_generation_artifact_sha256: Literal[
        "4d59d3d1d317ffc85cb1c5560bf14237b6455db30f30aeeed660f197e459e73e"
    ]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    fallback_solver: M5R3P2FallbackSolver
    isolated_compressor: M5R3P2IsolatedCompressor
    physical_gpu_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1, max_length=128)
    torch_version: str = Field(min_length=1, max_length=64)
    transformers_version: str = Field(min_length=1, max_length=64)
    teacher_tokenizers_version: str = Field(min_length=1, max_length=64)
    policy_tokenizers_version: Literal["0.21.4"]
    input_tasks: Literal[40]
    parent_solver_attempts: Literal[40]
    fallback_solver_attempts: int = Field(ge=1, le=40)
    isolated_compressor_attempts: int = Field(ge=0, le=40)
    accepted_samples: int = Field(ge=0, le=40)
    rejected_tasks: int = Field(ge=0, le=40)
    family_results: tuple[M5R3P1FamilyResult, M5R3P1FamilyResult]
    rejection_counts: dict[M5R3P1RejectionReason, int]
    fallback_trigger_counts: dict[M5R3P2FallbackReason, int]
    control: M5R3P1ControlResult
    contamination: M5R3P1ContaminationReport
    task_set_sha256: Literal["ac2d020c5cf2653f67e061c3b536af2ceccc12602d668f42eead62e8812e836f"]
    samples_sha256: str = Field(pattern=SHA256_PATTERN)
    duration_seconds: float = Field(gt=0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    generation_delta_sha256: str = Field(pattern=SHA256_PATTERN)
    raw_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    formal_source_expansion_authorized: bool
    r3_mixture_authorized: Literal[False]
    r3_training_authorized: Literal[False]
    consumes_m6_frozen_results: Literal[False] = False

    @field_validator("generated_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        """Require an aware UTC timestamp."""

        offset = value.utcoffset()
        if value.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("M5 R3 P2 timestamp must use UTC")
        return value

    @field_validator("rejection_counts", "fallback_trigger_counts")
    @classmethod
    def validate_sparse_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require sorted positive sparse public counts."""

        if list(value) != sorted(value) or any(count <= 0 for count in value.values()):
            raise ValueError("M5 R3 P2 sparse counts differ")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> M5R3P2Result:
        """Bind accounting, unchanged Gate status, and expansion authorization."""

        passed = (
            all(item.gate_passed for item in self.family_results)
            and self.control.status == "pass"
            and self.contamination.status == "pass"
        )
        if (
            tuple(item.task_family for item in self.family_results) != ("config", "log_diagnosis")
            or self.accepted_samples != sum(item.accepted_items for item in self.family_results)
            or self.accepted_samples + self.rejected_tasks != 40
            or sum(self.rejection_counts.values()) != self.rejected_tasks
            or sum(self.fallback_trigger_counts.values()) != self.fallback_solver_attempts
            or self.peak_reserved_bytes < self.peak_allocated_bytes
            or self.status != ("pass" if passed else "fail")
            or self.formal_source_expansion_authorized != passed
        ):
            raise ValueError("M5 R3 P2 result accounting or authorization differs")
        return self
