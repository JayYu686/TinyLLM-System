"""Strict manifests for deterministic M5.2 supervised-token mixtures."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tinyllm.data.reasoning_schema import ReasoningLanguage, ReasoningTaskFamily
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN


class M5MixtureArtifactFile(StrictSchema):
    """One immutable private mixture payload."""

    path: Literal["sequences.npz"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M5MixtureManifest(StrictSchema):
    """Content-addressed M5.2 mixture with exact supervised-token accounting."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m5-ablation-mixture"] = "m5-ablation-mixture"
    mixture_version: str = Field(pattern=r"^m5-ablation-mixture-v1-[0-9a-f]{8}$")
    parent_dataset_version: Literal["m2-sft-v1-f82ff32e"]
    parent_content_sha256: str = Field(pattern=SHA256_PATTERN)
    pilot_dataset_version: str = Field(pattern=r"^m5-reasoning-pilot-v1-[0-9a-f]{8}$")
    pilot_content_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    nonthinking_template_id: Literal["qwen3-chatml-nonthinking-v1"]
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    sequence_length: Literal[1024]
    pad_token_id: Literal[151643]
    target_supervised_tokens: Literal[1_000_000]
    thinking_fraction_basis_points: Literal[0, 3000, 5000]
    nonthinking_supervised_tokens: int = Field(ge=0)
    thinking_supervised_tokens: int = Field(ge=0)
    sequence_count: int = Field(gt=0)
    nonthinking_sequence_count: int = Field(ge=0)
    thinking_sequence_count: int = Field(ge=0)
    nonthinking_source_sequences: int = Field(gt=0)
    thinking_source_sequences: int = Field(gt=0)
    nonthinking_reuse_count: int = Field(ge=0)
    thinking_reuse_count: int = Field(ge=0)
    partially_masked_sequences: int = Field(ge=0, le=2)
    build_seed: int = Field(ge=0, le=2**32 - 1)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact: M5MixtureArtifactFile

    @model_validator(mode="after")
    def validate_counts_and_identity(self) -> M5MixtureManifest:
        """Bind the ratio, sequence totals, and content-addressed version."""

        if self.mixture_version != f"m5-ablation-mixture-v1-{self.content_sha256[:8]}":
            raise ValueError("mixture version does not match content hash")
        if self.nonthinking_supervised_tokens + self.thinking_supervised_tokens != 1_000_000:
            raise ValueError("mixture supervised-token counts must total exactly 1M")
        if self.thinking_supervised_tokens != self.thinking_fraction_basis_points * 100:
            raise ValueError("Thinking supervised-token count does not match ratio")
        if self.nonthinking_sequence_count + self.thinking_sequence_count != self.sequence_count:
            raise ValueError("mixture mode sequence counts do not equal total sequences")
        if self.thinking_fraction_basis_points == 0 and self.thinking_sequence_count != 0:
            raise ValueError("zero-Think mixture cannot contain Thinking sequences")
        if self.thinking_fraction_basis_points > 0 and self.thinking_sequence_count == 0:
            raise ValueError("Thinking mixture must contain Thinking sequences")
        return self


class M5FormatRepairMixtureManifest(StrictSchema):
    """Content-addressed R1 mixture with separately auditable repair supervision."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m5-format-repair-mixture"] = "m5-format-repair-mixture"
    mixture_version: str = Field(pattern=r"^m5-format-repair-mixture-v1-[0-9a-f]{8}$")
    parent_dataset_version: Literal["m2-sft-v1-f82ff32e"]
    parent_content_sha256: str = Field(pattern=SHA256_PATTERN)
    pilot_dataset_version: str = Field(pattern=r"^m5-reasoning-pilot-v1-[0-9a-f]{8}$")
    pilot_content_sha256: str = Field(pattern=SHA256_PATTERN)
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
    repair_thinking_supervised_tokens: Literal[150_000]
    sequence_count: int = Field(gt=0)
    nonthinking_sequence_count: int = Field(gt=0)
    thinking_sequence_count: int = Field(gt=0)
    general_thinking_sequence_count: int = Field(gt=0)
    repair_thinking_sequence_count: int = Field(gt=0)
    nonthinking_source_sequences: int = Field(gt=0)
    general_thinking_source_sequences: int = Field(gt=0)
    repair_thinking_source_sequences: Literal[40]
    nonthinking_reuse_count: int = Field(ge=0)
    general_thinking_reuse_count: int = Field(ge=0)
    repair_thinking_reuse_count: int = Field(ge=0)
    partially_masked_sequences: int = Field(ge=0, le=3)
    repair_policy_id: Literal["short-complete-balanced-v1"]
    repair_max_supervised_tokens: Literal[512]
    repair_source_family_counts: dict[ReasoningTaskFamily, int]
    repair_source_language_counts: dict[ReasoningLanguage, int]
    repair_source_sha256: str = Field(pattern=SHA256_PATTERN)
    build_seed: int = Field(ge=0, le=2**32 - 1)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact: M5MixtureArtifactFile

    @model_validator(mode="after")
    def validate_counts_and_identity(self) -> M5FormatRepairMixtureManifest:
        """Bind R1 policy, exact Token strata, and content-addressed version."""

        if self.mixture_version != (f"m5-format-repair-mixture-v1-{self.content_sha256[:8]}"):
            raise ValueError("format-repair mixture version does not match content hash")
        if self.nonthinking_supervised_tokens + self.thinking_supervised_tokens != 1_000_000:
            raise ValueError("format-repair supervised-token counts must total exactly 1M")
        if (
            self.general_thinking_supervised_tokens + self.repair_thinking_supervised_tokens
            != self.thinking_supervised_tokens
        ):
            raise ValueError("format-repair Thinking strata must total exactly 300K")
        if self.nonthinking_sequence_count + self.thinking_sequence_count != self.sequence_count:
            raise ValueError("format-repair mode counts do not equal total sequences")
        if (
            self.general_thinking_sequence_count + self.repair_thinking_sequence_count
            != self.thinking_sequence_count
        ):
            raise ValueError("format-repair Thinking sequence strata do not equal their total")
        if self.repair_source_family_counts != {
            "config": 8,
            "json": 8,
            "linux": 8,
            "log_diagnosis": 8,
            "python": 8,
        }:
            raise ValueError("format-repair sources must contain eight samples per family")
        if self.repair_source_language_counts != {"en": 25, "zh": 15}:
            raise ValueError("format-repair sources must contain 25 English and 15 Chinese samples")
        return self


class M5DualModeCorrectionMixtureManifest(StrictSchema):
    """M6 rejection remediation with Qwen3-aligned explicit mode context."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m5-dual-mode-correction-mixture"] = "m5-dual-mode-correction-mixture"
    mixture_version: str = Field(pattern=r"^m5-dual-mode-correction-mixture-v1-[0-9a-f]{8}$")
    parent_dataset_version: Literal["m2-sft-v1-f82ff32e"]
    parent_content_sha256: str = Field(pattern=SHA256_PATTERN)
    source_r3_mixture_version: Literal["m5-r3-mixture-v2-b47723e1"]
    source_r3_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_consumed_m6_results: Literal[False]
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    nonthinking_template_id: Literal["qwen3-chatml-nonthinking-sft-v2"]
    nonthinking_template_sha256: Literal[
        "fba6724bd16200356794105a2273bbd42e777c8311ef1760059c6f0766171ca2"
    ]
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    thinking_template_sha256: Literal[
        "4786143dbb7adb72a922d5efdcbe6596f2d65dcdc35d7bbf1b22830b795c2af9"
    ]
    sequence_length: Literal[1024]
    pad_token_id: Literal[151643]
    target_supervised_tokens: Literal[1_000_000]
    thinking_fraction_basis_points: Literal[3000]
    nonthinking_supervised_tokens: Literal[700_000]
    thinking_supervised_tokens: Literal[300_000]
    general_nonthinking_supervised_tokens: Literal[640_000]
    domain_nonthinking_supervised_tokens: Literal[60_000]
    domain_thinking_supervised_tokens: Literal[300_000]
    sequence_count: int = Field(gt=0)
    nonthinking_sequence_count: int = Field(gt=0)
    thinking_sequence_count: int = Field(gt=0)
    general_nonthinking_source_sequences: int = Field(gt=0)
    domain_source_pairs: int = Field(gt=0)
    general_nonthinking_reuse_count: int = Field(ge=0)
    domain_nonthinking_reuse_count: int = Field(ge=0)
    domain_thinking_reuse_count: int = Field(ge=0)
    partially_masked_sequences: int = Field(ge=0, le=3)
    build_seed: int = Field(ge=0, le=2**32 - 1)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact: M5MixtureArtifactFile

    @model_validator(mode="after")
    def validate_counts_and_identity(self) -> M5DualModeCorrectionMixtureManifest:
        """Bind exact token strata, paired sources, and content-addressed identity."""

        expected_version = f"m5-dual-mode-correction-mixture-v1-{self.content_sha256[:8]}"
        if self.mixture_version != expected_version:
            raise ValueError("dual-mode correction version does not match content hash")
        if (
            self.nonthinking_supervised_tokens + self.thinking_supervised_tokens
            != self.target_supervised_tokens
            or self.general_nonthinking_supervised_tokens
            + self.domain_nonthinking_supervised_tokens
            != self.nonthinking_supervised_tokens
            or self.domain_thinking_supervised_tokens != self.thinking_supervised_tokens
        ):
            raise ValueError("dual-mode correction supervised-token strata differ")
        if self.nonthinking_sequence_count + self.thinking_sequence_count != self.sequence_count:
            raise ValueError("dual-mode correction sequence counts differ")
        if self.domain_nonthinking_reuse_count > self.domain_source_pairs * 26:
            raise ValueError("dual-mode correction exceeds the paired Non-thinking reuse cap")
        return self


class M6GateRepairMixtureManifest(StrictSchema):
    """Independent authored remediation for evidence grounding and concise Thinking."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m6-gate-repair-mixture"] = "m6-gate-repair-mixture"
    mixture_version: str = Field(pattern=r"^m6-gate-repair-mixture-v1-[0-9a-f]{8}$")
    parent_dataset_version: Literal["m2-sft-v1-f82ff32e"]
    parent_content_sha256: str = Field(pattern=SHA256_PATTERN)
    diagnostic_protocol_version: Literal["m6-release-v2"]
    source_consumed_evaluation_content: Literal[False]
    evaluation_prompt_overlap_count: Literal[0]
    authored_source_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    nonthinking_template_id: Literal["qwen3-chatml-nonthinking-sft-v2"]
    nonthinking_template_sha256: Literal[
        "fba6724bd16200356794105a2273bbd42e777c8311ef1760059c6f0766171ca2"
    ]
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    thinking_template_sha256: Literal[
        "4786143dbb7adb72a922d5efdcbe6596f2d65dcdc35d7bbf1b22830b795c2af9"
    ]
    sequence_length: Literal[1024]
    pad_token_id: Literal[151643]
    target_supervised_tokens: Literal[1_000_000]
    thinking_fraction_basis_points: Literal[3000]
    nonthinking_supervised_tokens: Literal[700_000]
    thinking_supervised_tokens: Literal[300_000]
    general_nonthinking_supervised_tokens: Literal[400_000]
    domain_nonthinking_supervised_tokens: Literal[300_000]
    domain_thinking_supervised_tokens: Literal[300_000]
    sequence_count: int = Field(gt=0)
    nonthinking_sequence_count: int = Field(gt=0)
    thinking_sequence_count: int = Field(gt=0)
    general_nonthinking_source_sequences: int = Field(gt=0)
    authored_domain_source_pairs: int = Field(ge=600)
    authored_refusal_source_pairs: int = Field(ge=280)
    general_nonthinking_reuse_count: int = Field(ge=0)
    domain_nonthinking_reuse_count: int = Field(ge=0)
    domain_thinking_reuse_count: int = Field(ge=0)
    partially_masked_sequences: int = Field(ge=0, le=3)
    compact_reasoning_max_supervised_tokens: int = Field(gt=0, le=256)
    build_seed: int = Field(ge=0, le=2**32 - 1)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact: M5MixtureArtifactFile

    @model_validator(mode="after")
    def validate_counts_and_identity(self) -> M6GateRepairMixtureManifest:
        """Bind token strata and the content-addressed dataset identity."""

        if self.mixture_version != f"m6-gate-repair-mixture-v1-{self.content_sha256[:8]}":
            raise ValueError("M6 gate-repair version does not match content hash")
        if (
            self.nonthinking_supervised_tokens + self.thinking_supervised_tokens
            != self.target_supervised_tokens
            or self.general_nonthinking_supervised_tokens
            + self.domain_nonthinking_supervised_tokens
            != self.nonthinking_supervised_tokens
            or self.domain_thinking_supervised_tokens != self.thinking_supervised_tokens
        ):
            raise ValueError("M6 gate-repair supervised-token strata differ")
        if self.nonthinking_sequence_count + self.thinking_sequence_count != self.sequence_count:
            raise ValueError("M6 gate-repair sequence counts differ")
        return self


class M6GateReplayMixtureManifest(StrictSchema):
    """Continual-learning replay that preserves v2 gains while repairing its failures."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m6-gate-replay-mixture"] = "m6-gate-replay-mixture"
    mixture_version: str = Field(pattern=r"^m6-gate-replay-mixture-v1-[0-9a-f]{8}$")
    parent_dataset_version: Literal["m2-sft-v1-f82ff32e"]
    parent_content_sha256: str = Field(pattern=SHA256_PATTERN)
    diagnostic_protocol_version: Literal["m6-release-v2"]
    source_consumed_evaluation_content: Literal[False]
    evaluation_prompt_overlap_count: Literal[0]
    correction_mixture_version: Literal["m5-dual-mode-correction-mixture-v1-4bc342d4"]
    correction_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    repair_mixture_version: Literal["m6-gate-repair-mixture-v1-be2aa7fa"]
    repair_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    nonthinking_template_id: Literal["qwen3-chatml-nonthinking-sft-v2"]
    nonthinking_template_sha256: Literal[
        "fba6724bd16200356794105a2273bbd42e777c8311ef1760059c6f0766171ca2"
    ]
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    thinking_template_sha256: Literal[
        "4786143dbb7adb72a922d5efdcbe6596f2d65dcdc35d7bbf1b22830b795c2af9"
    ]
    sequence_length: Literal[1024]
    pad_token_id: Literal[151643]
    target_supervised_tokens: Literal[1_000_000]
    thinking_fraction_basis_points: Literal[3000]
    nonthinking_supervised_tokens: Literal[700_000]
    thinking_supervised_tokens: Literal[300_000]
    correction_nonthinking_supervised_tokens: Literal[400_000]
    repair_nonthinking_supervised_tokens: Literal[300_000]
    correction_thinking_supervised_tokens: Literal[150_000]
    repair_thinking_supervised_tokens: Literal[150_000]
    sequence_count: int = Field(gt=0)
    nonthinking_sequence_count: int = Field(gt=0)
    thinking_sequence_count: int = Field(gt=0)
    correction_nonthinking_source_sequences: int = Field(gt=0)
    repair_nonthinking_source_sequences: int = Field(gt=0)
    correction_thinking_source_sequences: int = Field(gt=0)
    repair_thinking_source_sequences: int = Field(gt=0)
    correction_nonthinking_reuse_count: int = Field(ge=0)
    repair_nonthinking_reuse_count: int = Field(ge=0)
    correction_thinking_reuse_count: int = Field(ge=0)
    repair_thinking_reuse_count: int = Field(ge=0)
    partially_masked_sequences: int = Field(ge=0, le=4)
    build_seed: int = Field(ge=0, le=2**32 - 1)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact: M5MixtureArtifactFile

    @model_validator(mode="after")
    def validate_counts_and_identity(self) -> M6GateReplayMixtureManifest:
        """Bind the replay ratios and content-addressed dataset identity."""

        if self.mixture_version != f"m6-gate-replay-mixture-v1-{self.content_sha256[:8]}":
            raise ValueError("M6 gate-replay version does not match content hash")
        if (
            self.correction_nonthinking_supervised_tokens
            + self.repair_nonthinking_supervised_tokens
            != self.nonthinking_supervised_tokens
            or self.correction_thinking_supervised_tokens + self.repair_thinking_supervised_tokens
            != self.thinking_supervised_tokens
            or self.nonthinking_supervised_tokens + self.thinking_supervised_tokens
            != self.target_supervised_tokens
        ):
            raise ValueError("M6 gate-replay supervised-token strata differ")
        if self.nonthinking_sequence_count + self.thinking_sequence_count != self.sequence_count:
            raise ValueError("M6 gate-replay sequence counts differ")
        return self


class M6DomainGeneralizationMixtureManifest(StrictSchema):
    """Broad seven-family remediation built without frozen evaluation content."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m6-domain-generalization-mixture"] = "m6-domain-generalization-mixture"
    mixture_version: str = Field(pattern=r"^m6-domain-generalization-mixture-v[12]-[0-9a-f]{8}$")
    parent_dataset_version: Literal["m2-sft-v1-f82ff32e"]
    parent_content_sha256: str = Field(pattern=SHA256_PATTERN)
    diagnostic_protocol_version: Literal["m6-release-v3"]
    source_consumed_evaluation_content: Literal[False]
    evaluation_prompt_overlap_count: Literal[0]
    authored_source_sha256: str = Field(pattern=SHA256_PATTERN)
    authored_source_tasks: Literal[900, 1500]
    authored_source_category_counts: dict[str, int]
    training_value_offsets: tuple[Literal[401], Literal[449], Literal[497]]
    refinement: Literal["json-object-and-evidence-refusal"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    nonthinking_template_id: Literal["qwen3-chatml-nonthinking-sft-v2"]
    nonthinking_template_sha256: Literal[
        "fba6724bd16200356794105a2273bbd42e777c8311ef1760059c6f0766171ca2"
    ]
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    thinking_template_sha256: Literal[
        "4786143dbb7adb72a922d5efdcbe6596f2d65dcdc35d7bbf1b22830b795c2af9"
    ]
    sequence_length: Literal[1024]
    pad_token_id: Literal[151643]
    target_supervised_tokens: Literal[1_000_000]
    thinking_fraction_basis_points: Literal[3000]
    nonthinking_supervised_tokens: Literal[700_000]
    thinking_supervised_tokens: Literal[300_000]
    general_nonthinking_supervised_tokens: Literal[250_000]
    domain_nonthinking_supervised_tokens: Literal[450_000]
    domain_thinking_supervised_tokens: Literal[300_000]
    sequence_count: int = Field(gt=0)
    nonthinking_sequence_count: int = Field(gt=0)
    thinking_sequence_count: int = Field(gt=0)
    general_nonthinking_source_sequences: int = Field(gt=0)
    domain_source_pairs: Literal[900, 1500]
    general_nonthinking_reuse_count: int = Field(ge=0)
    domain_nonthinking_reuse_count: int = Field(ge=0)
    domain_thinking_reuse_count: int = Field(ge=0)
    partially_masked_sequences: int = Field(ge=0, le=3)
    compact_reasoning_max_supervised_tokens: int = Field(gt=0, le=256)
    build_seed: int = Field(ge=0, le=2**32 - 1)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    artifact: M5MixtureArtifactFile

    @model_validator(mode="after")
    def validate_counts_and_identity(self) -> M6DomainGeneralizationMixtureManifest:
        version = 1 if self.refinement is None else 2
        expected_version = f"m6-domain-generalization-mixture-v{version}-{self.content_sha256[:8]}"
        if self.mixture_version != expected_version:
            raise ValueError("M6 domain-generalization version does not match content hash")
        expected_categories = (
            {
                "config": 120,
                "json": 120,
                "linux": 135,
                "logs": 135,
                "python": 150,
                "refusal": 120,
                "short_code": 120,
            }
            if self.refinement is None
            else {
                "config": 120,
                "json": 360,
                "linux": 135,
                "logs": 135,
                "python": 150,
                "refusal": 480,
                "short_code": 120,
            }
        )
        expected_tasks = 900 if self.refinement is None else 1500
        if (
            self.authored_source_tasks != expected_tasks
            or self.domain_source_pairs != expected_tasks
        ):
            raise ValueError("M6 domain-generalization source task count differs")
        if self.authored_source_category_counts != expected_categories:
            raise ValueError("M6 domain-generalization source categories differ")
        if (
            self.general_nonthinking_supervised_tokens + self.domain_nonthinking_supervised_tokens
            != self.nonthinking_supervised_tokens
            or self.domain_thinking_supervised_tokens != self.thinking_supervised_tokens
            or self.nonthinking_supervised_tokens + self.thinking_supervised_tokens
            != self.target_supervised_tokens
        ):
            raise ValueError("M6 domain-generalization supervised-token strata differ")
        if self.nonthinking_sequence_count + self.thinking_sequence_count != self.sequence_count:
            raise ValueError("M6 domain-generalization sequence counts differ")
        return self
