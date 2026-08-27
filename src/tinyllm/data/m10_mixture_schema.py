"""Strict contracts for the frozen M10 Agent SFT mixture."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.m10_agent_schema import M10SourceId
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M10MixtureLanguage = Literal["en", "zh"]
M10MixtureMode = Literal["nonthinking", "thinking"]


class M10FrozenInput(StrictSchema):
    """One immutable upstream selected by version and content identity."""

    version: str = Field(min_length=1, max_length=180)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    approval_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class M10FrozenInputs(StrictSchema):
    """The five frozen sources in their public contract order."""

    toolace: M10FrozenInput
    hermes_function_calling: M10FrozenInput
    tinyllm_devops: M10FrozenInput
    m6_domain_replay: M10FrozenInput
    m2_no_tool_replay: M10FrozenInput

    @model_validator(mode="after")
    def validate_approval_boundary(self) -> M10FrozenInputs:
        if self.tinyllm_devops.approval_sha256 is None:
            raise ValueError("M10 authored source requires immutable maintainer approval")
        if any(value.approval_sha256 is not None for key, value in self if key != "tinyllm_devops"):
            raise ValueError("only the M10 authored source may carry content approval")
        return self


class M10MixtureStratum(StrictSchema):
    """Exact supervised-token budget for one source/language/mode slice."""

    source_id: M10SourceId
    language: M10MixtureLanguage
    mode: M10MixtureMode
    supervised_tokens: int = Field(gt=0)


class M10FrozenDedupConfig(StrictSchema):
    """Frozen exact and MinHash near-duplicate policy."""

    exact_normalized: Literal[True]
    minhash_permutations: Literal[128]
    prompt_5gram_threshold_basis_points: Literal[8500]
    tool_schema_5gram_threshold_basis_points: Literal[8500]


class M10FrozenContaminationConfig(StrictSchema):
    """Frozen evaluation boundaries checked before token selection."""

    m9_dev_version: Literal["tinyllm-devops-agent-dev-v1-f958bcc6"]
    m9_release_version: Literal["tinyllm-devops-agent-release-v1-1ae9b75b"]
    bfcl_version: Literal["bfcl-v1.3-ea13468e-offline-core"]
    m6_domain_version: Literal["tinyllm-domain-thinking-boundary-audit-v1-b82cbca1"]
    fail_on_match: Literal[True]


class M10FrozenMixtureConfig(StrictSchema):
    """Exact build recipe for the one-million supervised-token logical epoch."""

    schema_version: Literal["1.0"]
    config_version: Literal["m10-agent-frozen-mixture-v1", "m10-agent-frozen-mixture-v2"]
    source_config_sha256: str = Field(pattern=SHA256_PATTERN)
    build_seed: Literal[20260822, 20260825, 20260827]
    dataset_name: Literal["tinyllm-agent-sft"]
    target_supervised_tokens: Literal[1_000_000]
    sequence_length: Literal[2048]
    tokenizer_id: Literal["Qwen/Qwen3-0.6B"]
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    inputs: M10FrozenInputs
    strata: tuple[
        M10MixtureStratum,
        M10MixtureStratum,
        M10MixtureStratum,
        M10MixtureStratum,
        M10MixtureStratum,
        M10MixtureStratum,
        M10MixtureStratum,
        M10MixtureStratum,
        M10MixtureStratum,
        M10MixtureStratum,
    ]
    deduplication: M10FrozenDedupConfig
    contamination: M10FrozenContaminationConfig

    @field_validator("strata", mode="before")
    @classmethod
    def freeze_strata(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_exact_budget(self) -> M10FrozenMixtureConfig:
        expected = (
            (
                ("toolace", "en", "nonthinking", 300_000),
                ("hermes_function_calling", "en", "nonthinking", 200_000),
                ("tinyllm_devops", "en", "nonthinking", 80_000),
                ("tinyllm_devops", "zh", "nonthinking", 120_000),
                ("m6_domain_replay", "en", "nonthinking", 70_000),
                ("m6_domain_replay", "en", "thinking", 30_000),
                ("m6_domain_replay", "zh", "nonthinking", 70_000),
                ("m6_domain_replay", "zh", "thinking", 30_000),
                ("m2_no_tool_replay", "en", "nonthinking", 20_000),
                ("m2_no_tool_replay", "zh", "nonthinking", 80_000),
            )
            if self.config_version == "m10-agent-frozen-mixture-v1"
            else (
                ("toolace", "en", "nonthinking", 200_000),
                ("hermes_function_calling", "en", "nonthinking", 100_000),
                ("tinyllm_devops", "en", "nonthinking", 280_000),
                ("tinyllm_devops", "zh", "nonthinking", 120_000),
                ("m6_domain_replay", "en", "nonthinking", 70_000),
                ("m6_domain_replay", "en", "thinking", 30_000),
                ("m6_domain_replay", "zh", "nonthinking", 70_000),
                ("m6_domain_replay", "zh", "thinking", 30_000),
                ("m2_no_tool_replay", "en", "nonthinking", 20_000),
                ("m2_no_tool_replay", "zh", "nonthinking", 80_000),
            )
        )
        observed = tuple(
            (item.source_id, item.language, item.mode, item.supervised_tokens)
            for item in self.strata
        )
        if observed != expected:
            raise ValueError("M10 source/language/mode strata differ from the frozen matrix")
        if sum(item.supervised_tokens for item in self.strata) != self.target_supervised_tokens:
            raise ValueError("M10 strata do not sum to the target supervised-token budget")
        language = {
            key: sum(item.supervised_tokens for item in self.strata if item.language == key)
            for key in ("en", "zh")
        }
        mode = {
            key: sum(item.supervised_tokens for item in self.strata if item.mode == key)
            for key in ("nonthinking", "thinking")
        }
        if language != {"en": 700_000, "zh": 300_000}:
            raise ValueError("M10 language matrix must remain exactly 70/30")
        if mode != {"nonthinking": 940_000, "thinking": 60_000}:
            raise ValueError("M10 dual-mode matrix must remain exactly 94/6")
        return self


class M10MixtureArtifact(StrictSchema):
    """One content-addressed private mixture file."""

    path: Literal["sequences.npz"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M10MixtureInputEvidence(StrictSchema):
    """Verified candidate and tokenization facts for one upstream source."""

    source_id: M10SourceId
    version: str = Field(min_length=1, max_length=180)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    input_candidates: int = Field(gt=0)
    accepted_candidates: int = Field(gt=0)
    duplicate_rejections: int = Field(ge=0)
    overlength_rejections: int = Field(ge=0)


class M10FrozenMixtureManifest(StrictSchema):
    """Immutable training-ready mixture identity and fail-closed evidence."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["tinyllm-agent-sft"] = "tinyllm-agent-sft"
    dataset_version: str = Field(pattern=r"^m10-agent-sft-v[12]-[0-9a-f]{8}$")
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    frozen_config_sha256: str = Field(pattern=SHA256_PATTERN)
    source_config_sha256: str = Field(pattern=SHA256_PATTERN)
    build_seed: Literal[20260822, 20260825, 20260827]
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    tokenizer_sha256: str = Field(pattern=SHA256_PATTERN)
    template_id: Literal["qwen3-agent-chatml-nonthinking-v1"]
    template_sha256: str = Field(pattern=SHA256_PATTERN)
    sequence_length: Literal[2048]
    sequence_count: int = Field(gt=0)
    target_supervised_tokens: Literal[1_000_000]
    source_supervised_tokens: dict[M10SourceId, int]
    language_supervised_tokens: dict[M10MixtureLanguage, int]
    mode_supervised_tokens: dict[M10MixtureMode, int]
    stratum_supervised_tokens: dict[str, int]
    reuse_counts: dict[str, int]
    partial_sequence_counts: dict[str, int]
    input_evidence: tuple[
        M10MixtureInputEvidence,
        M10MixtureInputEvidence,
        M10MixtureInputEvidence,
        M10MixtureInputEvidence,
        M10MixtureInputEvidence,
    ]
    exact_duplicate_drops: int = Field(ge=0)
    near_duplicate_drops: int = Field(ge=0)
    duplicate_report_sha256: str = Field(pattern=SHA256_PATTERN)
    contamination_report_sha256: str = Field(pattern=SHA256_PATTERN)
    contamination_status: Literal["pass"]
    artifact: M10MixtureArtifact
    status: Literal["committed"] = "committed"
    training_permitted: Literal[True] = True

    @field_validator("input_evidence", mode="before")
    @classmethod
    def freeze_evidence(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_manifest(self) -> M10FrozenMixtureManifest:
        repair = self.dataset_version.startswith("m10-agent-sft-v2-")
        expected_sources = {
            "toolace": 200_000 if repair else 300_000,
            "hermes_function_calling": 100_000 if repair else 200_000,
            "tinyllm_devops": 400_000 if repair else 200_000,
            "m6_domain_replay": 200_000,
            "m2_no_tool_replay": 100_000,
        }
        expected_version = f"m10-agent-sft-v{'2' if repair else '1'}-{self.content_sha256[:8]}"
        if self.dataset_version != expected_version:
            raise ValueError("M10 dataset version differs from its content hash")
        if self.source_supervised_tokens != expected_sources:
            raise ValueError("M10 source supervised-token counts differ")
        if self.language_supervised_tokens != {"en": 700_000, "zh": 300_000}:
            raise ValueError("M10 language supervised-token counts differ")
        if self.mode_supervised_tokens != {"nonthinking": 940_000, "thinking": 60_000}:
            raise ValueError("M10 mode supervised-token counts differ")
        if sum(self.stratum_supervised_tokens.values()) != self.target_supervised_tokens:
            raise ValueError("M10 stratum counts do not sum to target")
        if tuple(item.source_id for item in self.input_evidence) != tuple(expected_sources):
            raise ValueError("M10 input evidence order differs")
        if any(
            item.accepted_candidates + item.duplicate_rejections + item.overlength_rejections
            != item.input_candidates
            for item in self.input_evidence
        ):
            raise ValueError("M10 input candidate accounting differs")
        return self


class M10FrozenMixtureReport(StrictSchema):
    """Content-free public evidence for the final private mixture."""

    schema_version: Literal["1.0"] = "1.0"
    report_version: Literal["m10-frozen-mixture-v1", "m10-frozen-mixture-v2"] = (
        "m10-frozen-mixture-v1"
    )
    status: Literal["pass"]
    dataset_version: str = Field(pattern=r"^m10-agent-sft-v[12]-[0-9a-f]{8}$")
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    sequence_count: int = Field(gt=0)
    target_supervised_tokens: Literal[1_000_000]
    source_supervised_tokens: dict[M10SourceId, int]
    language_supervised_tokens: dict[M10MixtureLanguage, int]
    mode_supervised_tokens: dict[M10MixtureMode, int]
    overlength_rejections: dict[M10SourceId, int]
    exact_duplicate_drops: int = Field(ge=0)
    near_duplicate_drops: int = Field(ge=0)
    duplicate_report_sha256: str = Field(pattern=SHA256_PATTERN)
    contamination_report_sha256: str = Field(pattern=SHA256_PATTERN)
    training_permitted: Literal[True]
    private_artifacts_only: Literal[True] = True
    contains_source_or_evaluation_content: Literal[False] = False
