"""Strict M2.3c immutable dataset registration and shard schemas."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.processing_schema import DataSplit
from tinyllm.schemas.base import StrictSchema

DatasetFileRole = Literal["lineage", "rejections", "shard_array", "shard_metadata"]


class DatasetArtifactFile(StrictSchema):
    """One integrity-checked file below an immutable dataset version directory."""

    path: PurePosixPath
    role: DatasetFileRole
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: PurePosixPath) -> PurePosixPath:
        """Reject absolute, traversing, reserved, or platform-dependent paths."""

        if value.is_absolute() or not value.parts or ".." in value.parts:
            raise ValueError("dataset artifact path must be safe and relative")
        if "\\" in str(value):
            raise ValueError("dataset artifact path must use POSIX separators")
        if value.name in {"manifest.json", "registration.json", "COMMITTED"}:
            raise ValueError("dataset artifact path cannot use a reserved root filename")
        return value


class DatasetShardPack(StrictSchema):
    """Metadata required to reconstruct and validate one Pack from shard arrays."""

    pack_id: str = Field(pattern=r"^(train|validation|test)-[0-9a-f]{16}$")
    pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_sequence_length: int = Field(gt=1)
    sample_ids: tuple[str, ...] = Field(min_length=1)
    sample_token_counts: tuple[int, ...] = Field(min_length=1)
    token_count: int = Field(gt=0)
    supervised_token_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_boundaries(self) -> DatasetShardPack:
        """Keep persisted Pack boundaries internally consistent."""

        if len(self.sample_ids) != len(self.sample_token_counts):
            raise ValueError("shard Pack sample IDs and counts must have equal lengths")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("shard Pack sample IDs must be unique")
        if any(count <= 0 for count in self.sample_token_counts):
            raise ValueError("shard Pack sample token counts must be positive")
        if sum(self.sample_token_counts) != self.token_count:
            raise ValueError("shard Pack boundaries must sum to token count")
        if self.token_count > self.max_sequence_length:
            raise ValueError("shard Pack exceeds maximum sequence length")
        if self.supervised_token_count > self.token_count:
            raise ValueError("shard Pack supervised count exceeds token count")
        return self


class DatasetShardMetadata(StrictSchema):
    """Deterministic description of one NumPy shard and its Pack offsets."""

    schema_version: Literal["1.0"] = "1.0"
    storage_format: Literal["numpy-sharded-v1"]
    split: DataSplit
    shard_index: int = Field(ge=0)
    token_count: int = Field(gt=0)
    pack_count: int = Field(gt=0)
    packs: tuple[DatasetShardPack, ...] = Field(min_length=1)
    array_dtypes: dict[str, str]

    @field_validator("array_dtypes")
    @classmethod
    def validate_dtypes(cls, value: dict[str, str]) -> dict[str, str]:
        """Freeze portable, non-pickle array names and little-endian dtypes."""

        expected = {
            "input_ids": "<i4",
            "labels": "<i4",
            "pack_offsets": "<i8",
            "position_ids": "<u2",
            "segment_ids": "<u2",
        }
        if value != expected or list(value) != sorted(expected):
            raise ValueError("dataset shard array dtypes do not match numpy-sharded-v1")
        return value

    @model_validator(mode="after")
    def validate_shard(self) -> DatasetShardMetadata:
        """Bind split, order, and aggregate counts to Pack metadata."""

        if self.pack_count != len(self.packs):
            raise ValueError("dataset shard Pack count does not match metadata")
        if self.token_count != sum(pack.token_count for pack in self.packs):
            raise ValueError("dataset shard token count does not match Packs")
        if tuple(sorted(pack.pack_id for pack in self.packs)) != tuple(
            pack.pack_id for pack in self.packs
        ):
            raise ValueError("dataset shard Packs must be sorted by Pack ID")
        if any(not pack.pack_id.startswith(f"{self.split}-") for pack in self.packs):
            raise ValueError("dataset shard Pack split does not match shard split")
        return self


class DatasetRegistration(StrictSchema):
    """Private storage/environment record for one immutable Dataset Version."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m2-sft"]
    dataset_version: str = Field(pattern=r"^m2-sft-v1-[0-9a-f]{8}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    storage_format: Literal["numpy-sharded-v1"]
    shard_token_limit: int = Field(ge=1024)
    registered_at: datetime
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    python_version: str = Field(min_length=1, max_length=64)
    numpy_version: str = Field(min_length=1, max_length=64)
    packed_sequences: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    files: tuple[DatasetArtifactFile, ...] = Field(min_length=1)

    @field_validator("registered_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        """Require an explicit UTC registration timestamp."""

        offset = value.utcoffset()
        if value.tzinfo is None or offset is None:
            raise ValueError("registration timestamp must be timezone-aware")
        if offset.total_seconds() != 0:
            raise ValueError("registration timestamp must use UTC")
        return value

    @field_validator("files")
    @classmethod
    def validate_files(
        cls, value: tuple[DatasetArtifactFile, ...]
    ) -> tuple[DatasetArtifactFile, ...]:
        """Require unique, sorted paths so the Registration is reproducible."""

        paths = tuple(str(item.path) for item in value)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("registered dataset files must have unique sorted paths")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> DatasetRegistration:
        """Bind public version suffix and minimum storage coverage to content identity."""

        if self.dataset_version != f"m2-sft-v1-{self.content_sha256[:8]}":
            raise ValueError("registered dataset version does not match content hash")
        roles = {item.role for item in self.files}
        if not {"lineage", "rejections", "shard_array", "shard_metadata"}.issubset(roles):
            raise ValueError("registration does not cover every required artifact role")
        return self


class DatasetCommitMarker(StrictSchema):
    """Last-written proof that Manifest, Registration, and all files were committed."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m2-sft"]
    dataset_version: str = Field(pattern=r"^m2-sft-v1-[0-9a-f]{8}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_version(self) -> DatasetCommitMarker:
        """Bind the marker to the content-derived Dataset Version."""

        if self.dataset_version != f"m2-sft-v1-{self.content_sha256[:8]}":
            raise ValueError("commit marker version does not match content hash")
        return self


class RegisteredDatasetSummary(StrictSchema):
    """Path-free stable JSON result for prepare and inspect commands."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["ok"]
    operation: Literal["prepare", "inspect"]
    created: bool | None
    verified: Literal[True]
    dataset_name: Literal["m2-sft"]
    dataset_version: str = Field(pattern=r"^m2-sft-v1-[0-9a-f]{8}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    storage_format: Literal["numpy-sharded-v1"]
    registered_at: datetime
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    source_rows: dict[str, int]
    imported_samples: dict[str, int]
    processed_samples: int
    tokenized_samples: int
    balanced_samples: int
    packed_sequences: int
    total_tokens: int
    total_supervised_tokens: int
    rejection_counts: dict[str, int]
    registered_files: int
    registered_bytes: int

    @field_validator("source_rows", "imported_samples")
    @classmethod
    def validate_source_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require both sources in stable order with non-negative counts."""

        if list(value) != ["commitpackft", "oasst1"] or any(count < 0 for count in value.values()):
            raise ValueError("source count mappings must contain commitpackft and oasst1")
        return value

    @field_validator("rejection_counts")
    @classmethod
    def validate_rejections(cls, value: dict[str, int]) -> dict[str, int]:
        """Require deterministic positive sparse rejection counts."""

        if list(value) != sorted(value) or any(
            not key or count <= 0 for key, count in value.items()
        ):
            raise ValueError("rejection counts must use sorted keys and positive counts")
        return value

    @model_validator(mode="after")
    def validate_summary(self) -> RegisteredDatasetSummary:
        """Keep inspect/prepare-specific fields and non-negative totals coherent."""

        numeric = (
            self.processed_samples,
            self.tokenized_samples,
            self.balanced_samples,
            self.packed_sequences,
            self.total_tokens,
            self.total_supervised_tokens,
            self.registered_files,
            self.registered_bytes,
        )
        if any(value < 0 for value in numeric):
            raise ValueError("registered dataset summary totals must be non-negative")
        if (self.operation == "prepare") != (self.created is not None):
            raise ValueError("only prepare summary may include creation status")
        return self
