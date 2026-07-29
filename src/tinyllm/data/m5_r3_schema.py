"""Strict public contracts for the M5.2-R3 targeted trace-repair audit."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.reasoning_schema import ReasoningLanguage
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M5R3TargetFamily = Literal["config", "log_diagnosis"]
M5R3AuditStatus = Literal[
    "sufficient_for_targeted_mixture",
    "insufficient_requires_new_source",
]


class M5R3TracePolicy(StrictSchema):
    """Content and diversity constraints for one accepted targeted trace."""

    max_reasoning_tokens: Literal[192]
    max_repeated_8gram_basis_points: Literal[500]
    max_identical_line_hash_repetitions: Literal[1]
    require_unique_normalized_trace: Literal[True]


class M5R3SourceRequirement(StrictSchema):
    """Minimum distinct source examples required before building the R3 mixture."""

    selected_per_family: dict[M5R3TargetFamily, Literal[80]]
    selected_languages_per_family: dict[
        M5R3TargetFamily,
        dict[ReasoningLanguage, int],
    ]
    max_training_passes_per_source: Literal[4]

    @field_validator("selected_per_family")
    @classmethod
    def validate_family_order(
        cls,
        value: dict[M5R3TargetFamily, Literal[80]],
    ) -> dict[M5R3TargetFamily, Literal[80]]:
        """Require the fixed Config/Log family order and counts."""

        if value != {"config": 80, "log_diagnosis": 80} or list(value) != [
            "config",
            "log_diagnosis",
        ]:
            raise ValueError("M5 R3 requires 80 selected traces for Config and Log")
        return value

    @field_validator("selected_languages_per_family")
    @classmethod
    def validate_language_targets(
        cls,
        value: dict[M5R3TargetFamily, dict[ReasoningLanguage, int]],
    ) -> dict[M5R3TargetFamily, dict[ReasoningLanguage, int]]:
        """Freeze the 70/30 language split within both targeted families."""

        expected = {
            "config": {"en": 56, "zh": 24},
            "log_diagnosis": {"en": 56, "zh": 24},
        }
        if value != expected or any(list(counts) != ["en", "zh"] for counts in value.values()):
            raise ValueError("M5 R3 selected traces must use 56 English and 24 Chinese per family")
        return value


class M5R3SourceAuditConfig(StrictSchema):
    """Frozen source identity and selection policy for the R3 pre-training audit."""

    schema_version: Literal["1.0"]
    audit_version: Literal["m5-r3-source-audit-v1"]
    source_pilot_dataset_version: Literal["m5-reasoning-pilot-v1-b4db5ac8"]
    source_pilot_content_sha256: Literal[
        "b4db5ac8c9252b701c573de56a1f91629e3b1716e23223e5b2b979c5cc285684"
    ]
    source_raw_artifact_sha256: Literal[
        "5e4e75df8a0843376d95a9e47e6d91c0d0456e5066d944e315e5b96173530411"
    ]
    source_reasoning_config_sha256: Literal[
        "d6aee88bf4a3922981465026f202dcffcc0da0294126fac27602eb2442df1a4b"
    ]
    tokenization_config_sha256: Literal[
        "f2c3e3fc05534344c6705befebf5761face41178fa6f3c2216f4c0cfcc90aacc"
    ]
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    r2_decision_sha256: Literal["04165538efce811240b4d4501b13f74151758af7373704c12d6df882e3044ed6"]
    target_families: tuple[Literal["config"], Literal["log_diagnosis"]]
    trace_policy: M5R3TracePolicy
    source_requirement: M5R3SourceRequirement
    consume_m6_frozen_results: Literal[False]

    @field_validator("target_families", mode="before")
    @classmethod
    def normalize_yaml_target_families(cls, value: object) -> object:
        """Convert the YAML sequence to the immutable runtime representation."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_target_order(self) -> M5R3SourceAuditConfig:
        """Keep the diagnosis-derived family scope deterministic."""

        if self.target_families != ("config", "log_diagnosis"):
            raise ValueError("M5 R3 target families must be Config followed by Log")
        return self


class M5R3FamilySourceAudit(StrictSchema):
    """Content-free trace distribution and eligibility counts for one family."""

    task_family: M5R3TargetFamily
    source_items: int = Field(gt=0)
    source_language_counts: dict[ReasoningLanguage, int]
    reasoning_tokens_min: int = Field(gt=0)
    reasoning_tokens_p50: float = Field(gt=0)
    reasoning_tokens_p90: int = Field(gt=0)
    reasoning_tokens_max: int = Field(gt=0)
    repeated_8gram_ratio_mean_basis_points: int = Field(ge=0, le=10_000)
    normalized_unique_traces: int = Field(gt=0)
    eligible_items: int = Field(ge=0)
    eligible_language_counts: dict[ReasoningLanguage, int]
    exclusion_reason_counts: dict[
        Literal[
            "duplicate_normalized_trace",
            "identical_line_repetition",
            "reasoning_over_192_tokens",
            "repeated_8gram_over_500bp",
        ],
        int,
    ]

    @model_validator(mode="after")
    def validate_counts(self) -> M5R3FamilySourceAudit:
        """Bind language, uniqueness, eligibility, and length distributions."""

        if (
            list(self.source_language_counts) != ["en", "zh"]
            or sum(self.source_language_counts.values()) != self.source_items
            or list(self.eligible_language_counts) != ["en", "zh"]
            or sum(self.eligible_language_counts.values()) != self.eligible_items
            or self.eligible_items > self.source_items
            or self.normalized_unique_traces > self.source_items
            or not (
                self.reasoning_tokens_min
                <= self.reasoning_tokens_p50
                <= self.reasoning_tokens_p90
                <= self.reasoning_tokens_max
            )
        ):
            raise ValueError("M5 R3 family source audit counts are inconsistent")
        expected_reasons = [
            "duplicate_normalized_trace",
            "identical_line_repetition",
            "reasoning_over_192_tokens",
            "repeated_8gram_over_500bp",
        ]
        if list(self.exclusion_reason_counts) != expected_reasons:
            raise ValueError("M5 R3 exclusion reasons must be complete and sorted")
        return self


class M5R3SourceAudit(StrictSchema):
    """Public path-free decision on whether the existing Pilot can supply R3."""

    schema_version: Literal["1.0"] = "1.0"
    status: M5R3AuditStatus
    audit_version: Literal["m5-r3-source-audit-v1"]
    audit_config_sha256: str = Field(pattern=SHA256_PATTERN)
    source_pilot_dataset_version: Literal["m5-reasoning-pilot-v1-b4db5ac8"]
    source_pilot_content_sha256: Literal[
        "b4db5ac8c9252b701c573de56a1f91629e3b1716e23223e5b2b979c5cc285684"
    ]
    source_raw_artifact_sha256: Literal[
        "5e4e75df8a0843376d95a9e47e6d91c0d0456e5066d944e315e5b96173530411"
    ]
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    r2_decision_sha256: Literal["04165538efce811240b4d4501b13f74151758af7373704c12d6df882e3044ed6"]
    target_families: tuple[Literal["config"], Literal["log_diagnosis"]]
    family_audits: tuple[M5R3FamilySourceAudit, M5R3FamilySourceAudit]
    eligible_source_items: int = Field(ge=0)
    required_source_items: Literal[160]
    new_teacher_source_required: bool
    decision_reason: Literal[
        "existing_pilot_meets_targeted_source_gate",
        "existing_pilot_lacks_concise_diverse_config_log_traces",
    ]
    consumes_m6_frozen_results: Literal[False] = False

    @model_validator(mode="after")
    def validate_decision(self) -> M5R3SourceAudit:
        """Derive the public decision from per-family and language gates."""

        if tuple(item.task_family for item in self.family_audits) != self.target_families:
            raise ValueError("M5 R3 family audits must follow the fixed target order")
        if self.eligible_source_items != sum(item.eligible_items for item in self.family_audits):
            raise ValueError("M5 R3 eligible total does not match family audits")
        required_languages: dict[ReasoningLanguage, int] = {"en": 56, "zh": 24}
        sufficient = all(
            item.eligible_items >= 80
            and all(
                item.eligible_language_counts[language] >= count
                for language, count in required_languages.items()
            )
            for item in self.family_audits
        )
        expected_status: M5R3AuditStatus = (
            "sufficient_for_targeted_mixture" if sufficient else "insufficient_requires_new_source"
        )
        if (
            self.status != expected_status
            or self.new_teacher_source_required == sufficient
            or (
                self.decision_reason
                != (
                    "existing_pilot_meets_targeted_source_gate"
                    if sufficient
                    else "existing_pilot_lacks_concise_diverse_config_log_traces"
                )
            )
        ):
            raise ValueError("M5 R3 source decision does not match eligibility gates")
        return self
