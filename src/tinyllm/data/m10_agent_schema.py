"""Strict public contracts for the M10 Agent training dataset."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M10SourceId = Literal[
    "toolace",
    "hermes_function_calling",
    "tinyllm_devops",
    "m6_domain_replay",
    "m2_no_tool_replay",
]
M10ExternalSourceId = Literal["toolace", "hermes_function_calling"]
M10SourceKind = Literal["external", "authored", "registered_replay"]
M10SourceReadiness = Literal["ready", "pending_build"]


class M10AgentArtifactSpec(StrictSchema):
    """One immutable source artifact selected from a pinned dataset revision."""

    filename: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M10AgentSourceSpec(StrictSchema):
    """Identity, license, readiness, and token share of one M10 source."""

    source_id: M10SourceId
    source_kind: M10SourceKind
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    license: Literal["apache-2.0", "mixed-allowlisted"]
    mixture_basis_points: int = Field(gt=0, le=10_000)
    readiness: M10SourceReadiness
    redistributable: bool
    artifacts: tuple[M10AgentArtifactSpec, ...] = ()
    license_evidence_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    content_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("artifacts", mode="before")
    @classmethod
    def normalize_yaml_artifacts(cls, value: object) -> object:
        """Convert YAML lists to the immutable runtime representation."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_source_identity(self) -> M10AgentSourceSpec:
        """Keep external, authored, and registered sources unambiguous."""

        if self.source_kind == "external":
            if self.source_id not in {"toolace", "hermes_function_calling"}:
                raise ValueError("M10 external source_id is invalid")
            if (
                not self.artifacts
                or self.license_evidence_sha256 is None
                or self.content_sha256 is not None
            ):
                raise ValueError("M10 external sources require artifacts, not a content hash")
            if self.readiness != "ready":
                raise ValueError("M10 pinned external sources must be ready")
        elif self.source_kind == "registered_replay":
            if self.source_id not in {"m6_domain_replay", "m2_no_tool_replay"}:
                raise ValueError("M10 replay source_id is invalid")
            if self.artifacts or self.content_sha256 is None or self.manifest_sha256 is None:
                raise ValueError("M10 replay sources require content and manifest hashes")
            if self.readiness != "ready":
                raise ValueError("M10 registered replay sources must be ready")
        else:
            if self.source_id != "tinyllm_devops":
                raise ValueError("M10 authored source_id is invalid")
            if self.artifacts:
                raise ValueError("M10 authored data cannot reference an external artifact")
            frozen = self.content_sha256 is not None and self.manifest_sha256 is not None
            if frozen != (self.readiness == "ready"):
                raise ValueError("M10 authored readiness must match its frozen hashes")
        return self


class M10AgentLanguageTarget(StrictSchema):
    """Language ratio measured over supervised tokens after filtering."""

    unit: Literal["supervised_tokens"]
    english_basis_points: Literal[7000]
    chinese_basis_points: Literal[3000]


class M10AgentSupervisionPolicy(StrictSchema):
    """Loss-mask and reasoning policy for canonical Agent conversations."""

    assistant_tool_calls: Literal[True]
    assistant_final_answers: Literal[True]
    mask_system_messages: Literal[True]
    mask_user_messages: Literal[True]
    mask_tool_results: Literal[True]
    primary_mode: Literal["nonthinking"]
    preserve_native_dual_mode_with_m6_replay: Literal[True]
    synthetic_cot_teacher_data: Literal[False]


class M10AgentDedupPolicy(StrictSchema):
    """Deterministic exact and near-duplicate controls."""

    exact_normalized_dedup: Literal[True]
    near_dedup_algorithm: Literal["minhash_5gram"]
    near_dedup_fields: tuple[Literal["prompt", "tool_schema"], Literal["prompt", "tool_schema"]]
    near_duplicate_threshold_basis_points: int = Field(ge=1, le=10_000)
    cross_source_before_split: Literal[True]

    @field_validator("near_dedup_fields", mode="before")
    @classmethod
    def normalize_yaml_fields(cls, value: object) -> object:
        """Keep ordered field identity stable across YAML and JSON."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_field_order(self) -> M10AgentDedupPolicy:
        if self.near_dedup_fields != ("prompt", "tool_schema"):
            raise ValueError("M10 near dedup fields must be prompt followed by tool_schema")
        return self


class M10AgentContaminationTarget(StrictSchema):
    """One evaluation boundary that training data must not overlap."""

    target_id: Literal["m9_dev", "m9_release", "bfcl_core", "m6_domain"]
    visibility: Literal["public", "sealed_private"]
    matching: tuple[Literal["exact", "minhash_5gram"], Literal["exact", "minhash_5gram"]]
    may_influence_generation_or_selection: Literal[False]

    @field_validator("matching", mode="before")
    @classmethod
    def normalize_yaml_matching(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class M10AgentContaminationPolicy(StrictSchema):
    """Leakage controls, including the sealed M9 Release boundary."""

    targets: tuple[
        M10AgentContaminationTarget,
        M10AgentContaminationTarget,
        M10AgentContaminationTarget,
        M10AgentContaminationTarget,
    ]
    fail_on_exact_match: Literal[True]
    fail_on_near_match: Literal[True]
    release_scan_output: Literal["content_free_counts_and_hashes_only"]

    @field_validator("targets", mode="before")
    @classmethod
    def normalize_yaml_targets(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_target_order(self) -> M10AgentContaminationPolicy:
        expected = ("m9_dev", "m9_release", "bfcl_core", "m6_domain")
        if tuple(item.target_id for item in self.targets) != expected:
            raise ValueError("M10 contamination targets must use the frozen order")
        release = self.targets[1]
        if release.visibility != "sealed_private":
            raise ValueError("M9 Release must remain sealed_private")
        return self


class M10AgentDataConfig(StrictSchema):
    """Preregistered M10 mixture contract; it is not trainable until frozen."""

    schema_version: Literal["1.0"]
    config_version: Literal["m10-agent-data-v1"]
    status: Literal["preregistered", "frozen"]
    training_permitted: bool
    dataset_name: Literal["tinyllm-agent-sft"]
    tokenizer_id: Literal["Qwen/Qwen3-0.6B"]
    tokenizer_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    sequence_length: Literal[2048]
    sources: tuple[
        M10AgentSourceSpec,
        M10AgentSourceSpec,
        M10AgentSourceSpec,
        M10AgentSourceSpec,
        M10AgentSourceSpec,
    ]
    language_target: M10AgentLanguageTarget
    supervision: M10AgentSupervisionPolicy
    deduplication: M10AgentDedupPolicy
    contamination: M10AgentContaminationPolicy

    @field_validator("sources", mode="before")
    @classmethod
    def normalize_yaml_sources(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_frozen_mixture(self) -> M10AgentDataConfig:
        expected_ids = (
            "toolace",
            "hermes_function_calling",
            "tinyllm_devops",
            "m6_domain_replay",
            "m2_no_tool_replay",
        )
        expected_weights = (3000, 2000, 2000, 2000, 1000)
        if tuple(item.source_id for item in self.sources) != expected_ids:
            raise ValueError("M10 sources must use the frozen order")
        if tuple(item.mixture_basis_points for item in self.sources) != expected_weights:
            raise ValueError("M10 source token shares differ from the preregistered mixture")
        ready = all(item.readiness == "ready" for item in self.sources)
        if self.training_permitted != ready or (self.status == "frozen") != ready:
            raise ValueError("M10 training is permitted only after every source is frozen")
        return self


class M10SourceRolePathCount(StrictSchema):
    """Content-free count of one source conversation role path."""

    role_path: str = Field(min_length=1, pattern=r"^[a-z]+(?:>[a-z]+)*$")
    rows: int = Field(gt=0)


class M10SourceRejectionCount(StrictSchema):
    """Primary, mutually exclusive rejection reason count."""

    reason: Literal[
        "invalid_row_shape",
        "invalid_role_path",
        "invalid_tool_schema",
        "malformed_tool_call",
    ]
    rows: int = Field(gt=0)


class M10ExternalSourceProfile(StrictSchema):
    """Path-free aggregate profile of one pinned external dataset."""

    source_id: M10ExternalSourceId
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    artifacts: tuple[M10AgentArtifactSpec, ...]
    source_rows: int = Field(gt=0)
    accepted_shape_rows: int = Field(ge=0)
    rejected_shape_rows: int = Field(ge=0)
    role_paths: tuple[M10SourceRolePathCount, ...]
    rejection_counts: tuple[M10SourceRejectionCount, ...]
    rows_with_tool_definitions: int = Field(ge=0)
    tool_definitions: int = Field(ge=0)
    tools_per_row_min: int = Field(ge=0)
    tools_per_row_max: int = Field(ge=0)
    tools_per_row_mean_milli: int = Field(ge=0)
    tool_call_candidate_rows: int = Field(ge=0)
    no_tool_candidate_rows: int = Field(ge=0)
    parsed_tool_calls: int = Field(ge=0)
    malformed_tool_calls: int = Field(ge=0)
    dict_to_object_normalizations: int = Field(ge=0)
    null_required_normalizations: int = Field(ge=0)
    tool_name_collision_rows: int = Field(ge=0)
    contains_source_content: Literal[False] = False

    @field_validator("artifacts", "role_paths", "rejection_counts", mode="before")
    @classmethod
    def normalize_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_profile_counts(self) -> M10ExternalSourceProfile:
        if self.accepted_shape_rows + self.rejected_shape_rows != self.source_rows:
            raise ValueError("M10 source acceptance counts do not sum to source_rows")
        if sum(item.rows for item in self.role_paths) != self.source_rows:
            raise ValueError("M10 role path counts do not sum to source_rows")
        if sum(item.rows for item in self.rejection_counts) != self.rejected_shape_rows:
            raise ValueError("M10 rejection counts do not sum to rejected rows")
        if self.tool_call_candidate_rows + self.no_tool_candidate_rows != self.source_rows:
            raise ValueError("M10 tool/no-tool candidate counts do not sum to source_rows")
        if not (self.tools_per_row_min <= self.tools_per_row_max):
            raise ValueError("M10 tool distribution bounds are inconsistent")
        return self


class M10ExternalSourceProfileReport(StrictSchema):
    """Deterministic aggregate report for the two selected external sources."""

    schema_version: Literal["1.0"] = "1.0"
    profile_version: Literal["m10-external-source-profile-v1"]
    data_config_sha256: str = Field(pattern=SHA256_PATTERN)
    profiles: tuple[M10ExternalSourceProfile, M10ExternalSourceProfile]
    total_source_rows: int = Field(gt=0)
    total_accepted_shape_rows: int = Field(ge=0)
    total_rejected_shape_rows: int = Field(ge=0)
    contains_source_content: Literal[False] = False

    @field_validator("profiles", mode="before")
    @classmethod
    def normalize_profiles(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_report(self) -> M10ExternalSourceProfileReport:
        if tuple(item.source_id for item in self.profiles) != (
            "toolace",
            "hermes_function_calling",
        ):
            raise ValueError("M10 external profiles must use the frozen order")
        if self.total_source_rows != sum(item.source_rows for item in self.profiles):
            raise ValueError("M10 total source rows differ from profiles")
        if self.total_accepted_shape_rows != sum(
            item.accepted_shape_rows for item in self.profiles
        ):
            raise ValueError("M10 total accepted rows differ from profiles")
        if self.total_rejected_shape_rows != sum(
            item.rejected_shape_rows for item in self.profiles
        ):
            raise ValueError("M10 total rejected rows differ from profiles")
        return self
