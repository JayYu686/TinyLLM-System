"""Public schemas for deterministic M2 normalization, deduplication, and splitting."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.schema import (
    DataSourceName,
    ImportedMessage,
    ImportedSampleMetadata,
)
from tinyllm.schemas.base import StrictSchema

DataSplit = Literal["train", "validation", "test"]
PipelineRejectionReason = Literal[
    "empty_after_normalization",
    "exact_duplicate",
    "forbidden_control_character",
    "message_too_long",
    "sample_too_long",
]
PIPELINE_REJECTION_REASONS = frozenset(
    {
        "empty_after_normalization",
        "exact_duplicate",
        "forbidden_control_character",
        "message_too_long",
        "sample_too_long",
    }
)


class NormalizationConfig(StrictSchema):
    """Conservative text rules that preserve internal code whitespace."""

    unicode_form: Literal["nfc"]
    normalize_line_endings: Literal[True]
    strip_bom: Literal[True]
    trim_outer_whitespace: Literal[True]
    reject_control_characters: Literal[True]
    max_message_chars: int = Field(gt=0)
    max_sample_chars: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_length_limits(self) -> NormalizationConfig:
        """A sample cannot be shorter than its largest permitted message."""

        if self.max_sample_chars < self.max_message_chars:
            raise ValueError("max sample characters must be >= max message characters")
        return self


class DeduplicationConfig(StrictSchema):
    """Exact-only M2 core deduplication and deterministic keep priority."""

    exact: Literal[True]
    near: Literal[False]
    source_priority: tuple[DataSourceName, ...]

    @field_validator("source_priority", mode="before")
    @classmethod
    def freeze_yaml_priority(cls, value: object) -> object:
        """Convert the natural YAML sequence representation to an immutable tuple."""

        return tuple(value) if isinstance(value, list) else value

    @field_validator("source_priority")
    @classmethod
    def validate_priority(cls, value: tuple[DataSourceName, ...]) -> tuple[DataSourceName, ...]:
        """Require each supported source exactly once."""

        if set(value) != {"oasst1", "commitpackft"} or len(value) != 2:
            raise ValueError("source priority must contain oasst1 and commitpackft exactly once")
        return value


class GroupedSplitConfig(StrictSchema):
    """Integer-basis-point split policy with a fixed explicit seed."""

    seed: int = Field(ge=0, le=2**32 - 1)
    train_basis_points: int = Field(ge=0, le=10_000)
    validation_basis_points: int = Field(ge=0, le=10_000)
    test_basis_points: int = Field(ge=0, le=10_000)

    @model_validator(mode="after")
    def validate_total(self) -> GroupedSplitConfig:
        """Require an exact 100% integer partition and a usable train split."""

        total = self.train_basis_points + self.validation_basis_points + self.test_basis_points
        if total != 10_000:
            raise ValueError("split basis points must sum to 10000")
        if self.train_basis_points == 0:
            raise ValueError("train split must be non-empty by policy")
        return self


class M2ProcessingConfig(StrictSchema):
    """Complete M2.2 deterministic processing contract."""

    schema_version: Literal["1.0"] = "1.0"
    normalization: NormalizationConfig
    deduplication: DeduplicationConfig
    split: GroupedSplitConfig


class ProcessedSample(StrictSchema):
    """Normalized, exact-deduplicated, and leakage-safe split sample."""

    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^(oasst1|commitpackft):[a-zA-Z0-9._-]+$")
    source: DataSourceName
    messages: tuple[ImportedMessage, ...] = Field(min_length=2)
    metadata: ImportedSampleMetadata
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    origin_sample_ids: tuple[str, ...] = Field(min_length=1)
    origin_record_sha256s: tuple[str, ...] = Field(min_length=1)
    group_keys: tuple[str, ...] = Field(min_length=1)
    component_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: DataSplit

    @field_validator("origin_sample_ids", "origin_record_sha256s", "group_keys")
    @classmethod
    def validate_sorted_identity_sets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require stable unique lineage and grouping identities."""

        if any(not item for item in value) or tuple(sorted(set(value))) != value:
            raise ValueError("lineage and group identities must be non-empty, unique, and sorted")
        return value

    @field_validator("origin_sample_ids")
    @classmethod
    def validate_origin_sample_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require every origin to use the public imported-sample identity format."""

        pattern = re.compile(r"^(oasst1|commitpackft):[a-zA-Z0-9._-]+$")
        if any(pattern.fullmatch(item) is None for item in value):
            raise ValueError("origin sample IDs must use a supported source prefix")
        return value

    @field_validator("group_keys")
    @classmethod
    def validate_group_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require source namespaces so equal raw IDs from different datasets do not collide."""

        invalid_group = any(
            not item.startswith(("oasst1:", "commitpackft:")) or item.endswith(":")
            for item in value
        )
        if invalid_group:
            raise ValueError("group keys must use a non-empty supported source namespace")
        return value

    @field_validator("origin_record_sha256s")
    @classmethod
    def validate_origin_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject malformed record lineage hashes."""

        invalid_hash = any(
            len(item) != 64 or any(char not in "0123456789abcdef" for char in item)
            for item in value
        )
        if invalid_hash:
            raise ValueError("origin record hashes must be lowercase SHA256 values")
        return value

    @model_validator(mode="after")
    def validate_content_and_identity(self) -> ProcessedSample:
        """Bind the retained identity and content hash to the actual normalized sample."""

        if not self.id.startswith(f"{self.source}:"):
            raise ValueError("processed sample ID must match its source")
        if self.id not in self.origin_sample_ids:
            raise ValueError("processed sample ID must be present in origin sample IDs")
        roles = [message.role for message in self.messages]
        offset = 1 if roles[0] == "system" else 0
        expected = [
            "user" if index % 2 == 0 else "assistant" for index in range(len(roles) - offset)
        ]
        if roles[offset:] != expected:
            raise ValueError("processed user and assistant messages must alternate")
        payload = json.dumps(
            [message.to_dict() for message in self.messages],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        if hashlib.sha256(payload).hexdigest() != self.content_sha256:
            raise ValueError("processed sample content hash does not match messages")
        return self


class PipelineRejectedRecord(StrictSchema):
    """Content-free audit record for one M2.2 rejected imported sample."""

    schema_version: Literal["1.0"] = "1.0"
    sample_id: str = Field(pattern=r"^(oasst1|commitpackft):[a-zA-Z0-9._-]+$")
    source: DataSourceName
    reason: PipelineRejectionReason
    message_index: int | None = Field(default=None, ge=0)
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    duplicate_of_sample_id: str | None = Field(
        default=None,
        pattern=r"^(oasst1|commitpackft):[a-zA-Z0-9._-]+$",
    )

    @model_validator(mode="after")
    def validate_duplicate_evidence(self) -> PipelineRejectedRecord:
        """Exact duplicates must identify both their content and retained sample."""

        if not self.sample_id.startswith(f"{self.source}:"):
            raise ValueError("rejected sample ID must match its source")
        if self.reason == "exact_duplicate":
            if self.content_sha256 is None or self.duplicate_of_sample_id is None:
                raise ValueError(
                    "exact duplicate rejection requires content hash and retained sample"
                )
            if self.duplicate_of_sample_id == self.sample_id:
                raise ValueError("exact duplicate cannot reference itself")
            if self.message_index is not None:
                raise ValueError("exact duplicate rejection cannot identify one message")
        elif self.duplicate_of_sample_id is not None:
            raise ValueError("only exact duplicates may reference a retained sample")
        message_reasons = {
            "empty_after_normalization",
            "forbidden_control_character",
            "message_too_long",
        }
        if self.reason in message_reasons and self.message_index is None:
            raise ValueError("message-level rejection requires a message index")
        if self.reason == "sample_too_long" and self.message_index is not None:
            raise ValueError("sample-level rejection cannot identify one message")
        return self


class DataProcessingManifest(StrictSchema):
    """Deterministic M2.2 lineage, counts, and content identities."""

    schema_version: Literal["1.0"] = "1.0"
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_samples: int = Field(ge=0)
    normalized_samples: int = Field(ge=0)
    output_samples: int = Field(ge=0)
    rejected_samples: int = Field(ge=0)
    normalization_rejections: int = Field(ge=0)
    exact_duplicates: int = Field(ge=0)
    component_count: int = Field(ge=0)
    rejection_counts: dict[str, int]
    split_counts: dict[str, int]
    split_sha256s: dict[str, str]

    @field_validator("rejection_counts")
    @classmethod
    def validate_rejection_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require sorted positive rejection counts."""

        if any(not key or count <= 0 for key, count in value.items()):
            raise ValueError("rejection counts must use non-empty keys and positive values")
        if not set(value).issubset(PIPELINE_REJECTION_REASONS):
            raise ValueError("rejection counts contain an unknown reason")
        if list(value) != sorted(value):
            raise ValueError("rejection count keys must be sorted")
        return value

    @field_validator("split_counts")
    @classmethod
    def validate_split_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require all three split names even when a small fixture yields zero."""

        if list(value) != ["test", "train", "validation"] or any(
            count < 0 for count in value.values()
        ):
            raise ValueError("split counts must contain sorted test/train/validation keys")
        return value

    @field_validator("split_sha256s")
    @classmethod
    def validate_split_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        """Require a valid deterministic hash for every split, including empty ones."""

        if list(value) != ["test", "train", "validation"]:
            raise ValueError("split hashes must contain sorted test/train/validation keys")
        if any(
            len(item) != 64 or any(char not in "0123456789abcdef" for char in item)
            for item in value.values()
        ):
            raise ValueError("split hashes must be lowercase SHA256 values")
        return value

    @model_validator(mode="after")
    def validate_totals(self) -> DataProcessingManifest:
        """Keep every pipeline transition and summary total consistent."""

        if self.output_samples + self.rejected_samples != self.input_samples:
            raise ValueError("output and rejected samples must equal input samples")
        if self.output_samples + self.exact_duplicates != self.normalized_samples:
            raise ValueError("output and exact duplicates must equal normalized samples")
        if self.normalization_rejections + self.exact_duplicates != self.rejected_samples:
            raise ValueError("normalization and duplicate rejections must equal rejected samples")
        if sum(self.rejection_counts.values()) != self.rejected_samples:
            raise ValueError("rejection counts must equal rejected samples")
        if sum(self.split_counts.values()) != self.output_samples:
            raise ValueError("split counts must equal output samples")
        if self.component_count > self.output_samples:
            raise ValueError("component count cannot exceed output samples")
        if (self.output_samples == 0) != (self.component_count == 0):
            raise ValueError("component count must be zero exactly when output is empty")
        return self
