"""Strict contracts for the corrected M5.2-R3 exact-token mixture."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.m5_mixture_schema import M5MixtureArtifactFile
from tinyllm.data.m5_r3_schema import M5R3TargetFamily
from tinyllm.data.reasoning_schema import ReasoningLanguage
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M5R3LabelQuotas = dict[
    M5R3TargetFamily,
    dict[ReasoningLanguage, dict[str, int]],
]

EXPECTED_M5_R3_LABEL_QUOTAS: M5R3LabelQuotas = {
    "config": {
        "en": {
            "forbidden_truncation": 15,
            "missing_checkpoint": 17,
            "unsupported_precision": 6,
            "world_size_mismatch": 18,
        },
        "zh": {
            "forbidden_truncation": 6,
            "missing_checkpoint": 5,
            "unsupported_precision": 9,
            "world_size_mismatch": 4,
        },
    },
    "log_diagnosis": {
        "en": {
            "collective_timeout": 14,
            "cuda_oom": 14,
            "disk_full": 14,
            "non_finite_gradient": 14,
        },
        "zh": {
            "collective_timeout": 6,
            "cuda_oom": 6,
            "disk_full": 6,
            "non_finite_gradient": 6,
        },
    },
}


class M5R3FormalSourceLineage(StrictSchema):
    """Immutable formal-source inputs consumed by the mixture builder."""

    result_sha256: str = Field(pattern=SHA256_PATTERN)
    raw_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    accepted_samples_sha256: str = Field(pattern=SHA256_PATTERN)


class M5R3PilotLineage(StrictSchema):
    """Immutable original Thinking Pilot used for the general stratum."""

    raw_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_version: Literal["m5-reasoning-pilot-v1-b4db5ac8"]
    content_sha256: Literal["b4db5ac8c9252b701c573de56a1f91629e3b1716e23223e5b2b979c5cc285684"]


class M5R3MixtureTokenBudget(StrictSchema):
    """Frozen exact supervised-token allocation."""

    total_supervised_tokens: Literal[1_000_000]
    nonthinking_supervised_tokens: Literal[700_000]
    general_thinking_supervised_tokens: Literal[150_000]
    targeted_thinking_supervised_tokens: Literal[150_000]


class M5R3MixtureSelectionPolicy(StrictSchema):
    """Outcome-independent family/language/label selection and exposure cap."""

    policy_id: Literal["family-language-label-balanced-v2"]
    quotas: M5R3LabelQuotas
    stable_sort: tuple[
        Literal["reasoning_tokens"],
        Literal["repeated_8gram_basis_points"],
        Literal["sample_id"],
    ]
    max_source_uses: Literal[30]

    @field_validator("stable_sort", mode="before")
    @classmethod
    def normalize_sort(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_selection(self) -> M5R3MixtureSelectionPolicy:
        if self.quotas != EXPECTED_M5_R3_LABEL_QUOTAS or self.stable_sort != (
            "reasoning_tokens",
            "repeated_8gram_basis_points",
            "sample_id",
        ):
            raise ValueError("M5 R3 mixture selection policy differs")
        if (
            sum(
                count
                for family in self.quotas.values()
                for language in family.values()
                for count in language.values()
            )
            != 160
        ):
            raise ValueError("M5 R3 mixture selection must contain 160 sources")
        return self


class M5R3MixtureConfig(StrictSchema):
    """Complete versioned correction to the infeasible four-use contract."""

    schema_version: Literal["1.0"]
    mixture_protocol: Literal["m5-r3-mixture-v2"]
    formal_source: M5R3FormalSourceLineage
    pilot: M5R3PilotLineage
    token_budget: M5R3MixtureTokenBudget
    selection: M5R3MixtureSelectionPolicy
    build_seed: Literal[20260731]
    consume_m6_frozen_results: Literal[False]
    r3_training_authorized: Literal[False]


class M5R3MixtureManifest(StrictSchema):
    """Content-addressed R3 mixture with exact strata and source-use evidence."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m5-r3-mixture"] = "m5-r3-mixture"
    mixture_version: str = Field(pattern=r"^m5-r3-mixture-v2-[0-9a-f]{8}$")
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_dataset_version: Literal["m2-sft-v1-f82ff32e"]
    parent_content_sha256: str = Field(pattern=SHA256_PATTERN)
    pilot_dataset_version: Literal["m5-reasoning-pilot-v1-b4db5ac8"]
    pilot_content_sha256: Literal[
        "b4db5ac8c9252b701c573de56a1f91629e3b1716e23223e5b2b979c5cc285684"
    ]
    formal_result_sha256: str = Field(pattern=SHA256_PATTERN)
    formal_raw_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    formal_accepted_samples_sha256: str = Field(pattern=SHA256_PATTERN)
    targeted_selection_policy_id: Literal["family-language-label-balanced-v2"]
    targeted_selection_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    nonthinking_template_id: Literal["qwen3-chatml-nonthinking-v1"]
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    sequence_length: Literal[1024]
    pad_token_id: Literal[151643]
    target_supervised_tokens: Literal[1_000_000]
    thinking_fraction_basis_points: Literal[3000]
    nonthinking_supervised_tokens: Literal[700_000]
    thinking_supervised_tokens: Literal[300_000]
    general_thinking_supervised_tokens: Literal[150_000]
    targeted_thinking_supervised_tokens: Literal[150_000]
    sequence_count: int = Field(gt=0)
    nonthinking_sequence_count: int = Field(gt=0)
    general_thinking_sequence_count: int = Field(gt=0)
    targeted_thinking_sequence_count: int = Field(gt=0)
    nonthinking_source_sequences: int = Field(gt=0)
    general_thinking_source_sequences: Literal[96]
    targeted_thinking_source_sequences: Literal[160]
    nonthinking_reuse_count: int = Field(ge=0)
    general_thinking_reuse_count: int = Field(ge=0)
    targeted_thinking_reuse_count: int = Field(ge=0)
    targeted_source_supervised_tokens_per_pass: int = Field(gt=0)
    targeted_source_use_min: int = Field(gt=0, le=30)
    targeted_source_use_max: int = Field(gt=0, le=30)
    partially_masked_sequences: int = Field(ge=0, le=3)
    targeted_source_family_counts: dict[M5R3TargetFamily, int]
    targeted_source_language_counts: dict[ReasoningLanguage, int]
    targeted_source_label_counts: dict[str, int]
    build_seed: Literal[20260731]
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact: M5MixtureArtifactFile
    r3_training_authorized: Literal[True]
    consume_m6_frozen_results: Literal[False]

    @model_validator(mode="after")
    def validate_counts_and_identity(self) -> M5R3MixtureManifest:
        if self.mixture_version != f"m5-r3-mixture-v2-{self.content_sha256[:8]}":
            raise ValueError("M5 R3 mixture version does not match content hash")
        if (
            self.nonthinking_supervised_tokens + self.thinking_supervised_tokens
            != self.target_supervised_tokens
            or self.general_thinking_supervised_tokens + self.targeted_thinking_supervised_tokens
            != self.thinking_supervised_tokens
        ):
            raise ValueError("M5 R3 supervised-token strata differ")
        if (
            self.nonthinking_sequence_count
            + self.general_thinking_sequence_count
            + self.targeted_thinking_sequence_count
            != self.sequence_count
        ):
            raise ValueError("M5 R3 mixture sequence strata differ")
        if self.targeted_source_family_counts != {"config": 80, "log_diagnosis": 80}:
            raise ValueError("M5 R3 targeted family counts differ")
        if self.targeted_source_language_counts != {"en": 112, "zh": 48}:
            raise ValueError("M5 R3 targeted language counts differ")
        expected_labels = {
            label: sum(
                language.get(label, 0)
                for family in EXPECTED_M5_R3_LABEL_QUOTAS.values()
                for language in family.values()
            )
            for label in sorted(
                {
                    label
                    for family in EXPECTED_M5_R3_LABEL_QUOTAS.values()
                    for language in family.values()
                    for label in language
                }
            )
        }
        if self.targeted_source_label_counts != expected_labels:
            raise ValueError("M5 R3 targeted label counts differ")
        if self.targeted_source_use_min > self.targeted_source_use_max:
            raise ValueError("M5 R3 source-use range is inverted")
        return self
