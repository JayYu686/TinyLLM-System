"""Versioned schemas for the M2 dataset import boundary."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema

DataSourceName = Literal["oasst1", "commitpackft"]
MessageRole = Literal["system", "user", "assistant"]
RejectionReason = Literal[
    "deleted",
    "duplicate_source_id",
    "empty_content",
    "empty_instruction",
    "invalid_conversation",
    "malformed_row",
    "missing_parent",
    "missing_repository",
    "not_python",
    "not_ready",
    "review_not_positive",
    "unsupported_language",
    "unsupported_license",
    "unsupported_role",
]


class DatasetSource(StrictSchema):
    """Immutable identity and public metadata for one upstream snapshot."""

    schema_version: Literal["1.0"] = "1.0"
    name: DataSourceName
    dataset_id: str = Field(min_length=1, max_length=128)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_card_url: str = Field(pattern=r"^https://huggingface\.co/datasets/")
    dataset_card_license: str = Field(min_length=1, max_length=64)
    dataset_card_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ImportedMessage(StrictSchema):
    """One validated message before normalization and tokenization."""

    role: MessageRole
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        """Reject whitespace-only messages without silently normalizing them."""

        if not value.strip():
            raise ValueError("message content cannot be blank")
        return value


class ImportedSampleMetadata(StrictSchema):
    """Lineage and grouping metadata retained through the M2 pipeline."""

    language: str = Field(pattern=r"^[a-z][a-z0-9-]{1,15}$")
    category: Literal["conversation", "code_edit"]
    license: str = Field(min_length=1, max_length=64)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_record_id: str = Field(min_length=1, max_length=512)
    group_ids: tuple[str, ...] = Field(min_length=1)
    raw_record_sha256s: tuple[str, ...] = Field(min_length=1)

    @field_validator("group_ids")
    @classmethod
    def validate_group_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require canonical unique group identities for deterministic splitting."""

        if any(not item.strip() for item in value):
            raise ValueError("group IDs cannot be blank")
        if tuple(sorted(set(value))) != value:
            raise ValueError("group IDs must be unique and sorted")
        return value

    @field_validator("raw_record_sha256s")
    @classmethod
    def validate_record_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require one valid hash for every source record used by the sample."""

        invalid_hash = any(
            len(item) != 64 or any(char not in "0123456789abcdef" for char in item)
            for item in value
        )
        if invalid_hash:
            raise ValueError("source record hashes must be lowercase SHA256 values")
        return value


class ImportedSample(StrictSchema):
    """A licensed, structurally valid sample that is not yet trainable."""

    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^(oasst1|commitpackft):[a-zA-Z0-9._-]+$")
    source: DataSourceName
    messages: tuple[ImportedMessage, ...] = Field(min_length=2)
    metadata: ImportedSampleMetadata

    @model_validator(mode="after")
    def validate_conversation(self) -> ImportedSample:
        """Require a user-first alternating conversation ending in an answer."""

        roles = [message.role for message in self.messages]
        offset = 1 if roles[0] == "system" else 0
        conversational_roles = roles[offset:]
        if not conversational_roles or conversational_roles[0] != "user":
            raise ValueError("sample must start with a user message")
        if conversational_roles[-1] != "assistant":
            raise ValueError("sample must end with an assistant message")
        expected = [
            "user" if index % 2 == 0 else "assistant" for index in range(len(roles) - offset)
        ]
        if conversational_roles != expected:
            raise ValueError("user and assistant messages must alternate")
        return self


class RejectedRecord(StrictSchema):
    """Privacy-preserving evidence for one rejected candidate or malformed row."""

    schema_version: Literal["1.0"] = "1.0"
    source: DataSourceName
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_record_id: str = Field(min_length=1, max_length=512)
    raw_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: RejectionReason
    field: str | None = Field(default=None, max_length=128)


class DataImportManifest(StrictSchema):
    """Deterministic summary and lineage for one source import operation."""

    schema_version: Literal["1.0"] = "1.0"
    source: DatasetSource
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_rows: int = Field(ge=0)
    candidate_samples: int = Field(ge=0)
    accepted_samples: int = Field(ge=0)
    rejected_samples: int = Field(ge=0)
    rejection_counts: dict[str, int]
    license_counts: dict[str, int]

    @field_validator("rejection_counts", "license_counts")
    @classmethod
    def validate_count_mapping(cls, value: dict[str, int]) -> dict[str, int]:
        """Reject blank keys, non-positive counts, and unstable insertion order."""

        if any(not key or count <= 0 for key, count in value.items()):
            raise ValueError("summary counts must use non-empty keys and positive values")
        if list(value) != sorted(value):
            raise ValueError("summary count keys must be sorted")
        return value

    @model_validator(mode="after")
    def validate_totals(self) -> DataImportManifest:
        """Keep candidate and reason totals internally consistent."""

        if self.accepted_samples + self.rejected_samples != self.candidate_samples:
            raise ValueError("accepted and rejected samples must equal candidate samples")
        if sum(self.rejection_counts.values()) != self.rejected_samples:
            raise ValueError("rejection counts must equal rejected samples")
        if sum(self.license_counts.values()) != self.accepted_samples:
            raise ValueError("license counts must equal accepted samples")
        return self
