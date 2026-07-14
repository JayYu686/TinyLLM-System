"""Strict M2.3b balancing, packing, and final Dataset Manifest schemas."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.processing_schema import DataSplit
from tinyllm.data.schema import DataSourceName
from tinyllm.data.tokenization_schema import ChatTemplateIdentity, TokenizerIdentity
from tinyllm.schemas.base import StrictSchema


class TokenBalanceConfig(StrictSchema):
    """Frozen Train-only source/language token targets."""

    selection_seed: int = Field(ge=0, le=2**32 - 1)
    apply_to_split: Literal["train"]
    oasst1_zh_basis_points: int = Field(ge=0, le=10_000)
    oasst1_en_basis_points: int = Field(ge=0, le=10_000)
    commitpackft_en_basis_points: int = Field(ge=0, le=10_000)
    tolerance_basis_points: int = Field(ge=0, le=10_000)
    require_all_strata: Literal[True]

    @model_validator(mode="after")
    def validate_targets(self) -> TokenBalanceConfig:
        """Require a complete non-zero 100% target partition."""

        targets = (
            self.oasst1_zh_basis_points,
            self.oasst1_en_basis_points,
            self.commitpackft_en_basis_points,
        )
        if sum(targets) != 10_000:
            raise ValueError("balance target basis points must sum to 10000")
        if any(target == 0 for target in targets):
            raise ValueError("all required balance targets must be non-zero")
        return self

    def targets(self) -> dict[str, int]:
        """Return stable public Stratum names and integer targets."""

        return {
            "commitpackft:en": self.commitpackft_en_basis_points,
            "oasst1:en": self.oasst1_en_basis_points,
            "oasst1:zh": self.oasst1_zh_basis_points,
        }


class SequencePackingConfig(StrictSchema):
    """Split-local boundary-aware deterministic packing policy."""

    algorithm: Literal["best-fit-decreasing-v1"]
    max_sequence_length: int = Field(gt=1)
    split_local: Literal[True]
    reset_position_ids: Literal[True]
    emit_segment_ids: Literal[True]
    pad_to_max_length: Literal[False]
    add_separator: Literal[False]


class M2PackingConfig(StrictSchema):
    """Complete formal M2.3b build configuration."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m2-sft"]
    balance: TokenBalanceConfig
    packing: SequencePackingConfig


class BalanceRejectedRecord(StrictSchema):
    """Content-free evidence for a deterministic Train downsampling decision."""

    schema_version: Literal["1.0"] = "1.0"
    sample_id: str = Field(pattern=r"^(oasst1|commitpackft):[a-zA-Z0-9._-]+$")
    source: DataSourceName
    split: Literal["train"]
    language: Literal["en", "zh"]
    stratum: Literal["oasst1:zh", "oasst1:en", "commitpackft:en"]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_count: int = Field(gt=0)
    reason: Literal["balance_downsampled"]

    @model_validator(mode="after")
    def validate_stratum(self) -> BalanceRejectedRecord:
        """Bind source/language metadata to the stable Stratum name."""

        if not self.sample_id.startswith(f"{self.source}:"):
            raise ValueError("balance rejection sample ID must match source")
        if self.stratum != f"{self.source}:{self.language}":
            raise ValueError("balance rejection Stratum must match source and language")
        return self


class PackedSequence(StrictSchema):
    """One unpadded split-local pack with explicit sample boundaries."""

    schema_version: Literal["1.0"] = "1.0"
    pack_id: str = Field(pattern=r"^(train|validation|test)-[0-9a-f]{16}$")
    pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: DataSplit
    max_sequence_length: int = Field(gt=1)
    sample_ids: tuple[str, ...] = Field(min_length=1)
    sample_token_counts: tuple[int, ...] = Field(min_length=1)
    input_ids: tuple[int, ...] = Field(min_length=1)
    labels: tuple[int, ...] = Field(min_length=1)
    position_ids: tuple[int, ...] = Field(min_length=1)
    segment_ids: tuple[int, ...] = Field(min_length=1)
    token_count: int = Field(gt=0)
    supervised_token_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_pack(self) -> PackedSequence:
        """Validate boundaries, label mask, reset positions, capacity, and content identity."""

        arrays = (self.input_ids, self.labels, self.position_ids, self.segment_ids)
        if any(len(array) != self.token_count for array in arrays):
            raise ValueError("all packed token arrays must match token count")
        if len(self.sample_ids) != len(self.sample_token_counts):
            raise ValueError("sample IDs and sample token counts must have equal lengths")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("a sample cannot appear twice in one pack")
        if sum(self.sample_token_counts) != self.token_count:
            raise ValueError("sample token counts must sum to packed token count")
        if self.token_count > self.max_sequence_length:
            raise ValueError("packed token count exceeds maximum sequence length")
        if any(token_id < 0 for token_id in self.input_ids):
            raise ValueError("packed input IDs must be non-negative")

        supervised = 0
        cursor = 0
        for segment_id, sample_tokens in enumerate(self.sample_token_counts):
            end = cursor + sample_tokens
            if self.segment_ids[cursor:end] != (segment_id,) * sample_tokens:
                raise ValueError("segment IDs do not match sample boundaries")
            if self.position_ids[cursor:end] != tuple(range(sample_tokens)):
                raise ValueError("position IDs must reset at every sample boundary")
            for token_id, label in zip(
                self.input_ids[cursor:end], self.labels[cursor:end], strict=True
            ):
                if label != -100 and label != token_id:
                    raise ValueError("packed labels must be masked or equal input IDs")
                if label != -100:
                    supervised += 1
            cursor = end
        if supervised != self.supervised_token_count:
            raise ValueError("packed supervised token count does not match labels")

        payload = {
            "input_ids": self.input_ids,
            "labels": self.labels,
            "max_sequence_length": self.max_sequence_length,
            "position_ids": self.position_ids,
            "sample_ids": self.sample_ids,
            "sample_token_counts": self.sample_token_counts,
            "segment_ids": self.segment_ids,
            "split": self.split,
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        expected_hash = hashlib.sha256(encoded).hexdigest()
        if self.pack_sha256 != expected_hash:
            raise ValueError("pack SHA256 does not match packed content")
        if self.pack_id != f"{self.split}-{expected_hash[:16]}":
            raise ValueError("pack ID does not match split and content hash")
        return self


class SourceDatasetLineage(StrictSchema):
    """Minimal deterministic upstream identity carried into the final manifest."""

    source: DataSourceName
    dataset_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_card_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    import_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class M2DatasetManifest(StrictSchema):
    """Content-addressed, timestamp-free final M2 dataset build identity."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m2-sft"]
    dataset_version: str = Field(pattern=r"^m2-sft-v1-[0-9a-f]{8}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_lineage: tuple[SourceDatasetLineage, SourceDatasetLineage]
    processing_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processing_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processing_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer: TokenizerIdentity
    template: ChatTemplateIdentity
    packing_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_sequence_length: int = Field(gt=1)
    input_processed_samples: int = Field(ge=0)
    tokenized_samples: int = Field(ge=0)
    tokenization_rejections: int = Field(ge=0)
    balanced_samples: int = Field(ge=0)
    balance_rejections: int = Field(ge=0)
    packed_sequences: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    total_supervised_tokens: int = Field(ge=0)
    packing_capacity_tokens: int = Field(ge=0)
    packing_efficiency_basis_points: int = Field(ge=0, le=10_000)
    split_sample_counts: dict[str, int]
    split_pack_counts: dict[str, int]
    split_token_counts: dict[str, int]
    split_supervised_token_counts: dict[str, int]
    split_sha256s: dict[str, str]
    source_token_counts: dict[str, int]
    language_token_counts: dict[str, int]
    license_sample_counts: dict[str, int]
    train_stratum_token_counts: dict[str, int]
    train_stratum_basis_points: dict[str, int]
    rejection_counts: dict[str, int]

    @field_validator(
        "split_sample_counts",
        "split_pack_counts",
        "split_token_counts",
        "split_supervised_token_counts",
    )
    @classmethod
    def validate_split_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require stable non-negative counts for every split."""

        if list(value) != ["test", "train", "validation"] or any(
            count < 0 for count in value.values()
        ):
            raise ValueError("split count mappings must contain sorted test/train/validation keys")
        return value

    @field_validator("split_sha256s")
    @classmethod
    def validate_split_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        """Require one valid content hash for every split."""

        if list(value) != ["test", "train", "validation"]:
            raise ValueError("split hashes must contain sorted test/train/validation keys")
        if any(
            len(item) != 64 or any(char not in "0123456789abcdef" for char in item)
            for item in value.values()
        ):
            raise ValueError("split hashes must be lowercase SHA256 values")
        return value

    @field_validator("source_token_counts")
    @classmethod
    def validate_source_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require both pinned sources in stable order."""

        if list(value) != ["commitpackft", "oasst1"] or any(count < 0 for count in value.values()):
            raise ValueError("source token counts must contain commitpackft and oasst1")
        return value

    @field_validator("language_token_counts")
    @classmethod
    def validate_language_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require both M2 target languages in stable order."""

        if list(value) != ["en", "zh"] or any(count < 0 for count in value.values()):
            raise ValueError("language token counts must contain sorted en and zh keys")
        return value

    @field_validator(
        "train_stratum_token_counts",
        "train_stratum_basis_points",
    )
    @classmethod
    def validate_stratum_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require every frozen Train stratum in stable order."""

        expected = ["commitpackft:en", "oasst1:en", "oasst1:zh"]
        if list(value) != expected or any(count < 0 for count in value.values()):
            raise ValueError("Train Stratum mappings must contain all sorted target strata")
        return value

    @field_validator("license_sample_counts", "rejection_counts")
    @classmethod
    def validate_summary_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Reject unstable, blank, or non-positive sparse summary entries."""

        if list(value) != sorted(value):
            raise ValueError("summary count keys must be sorted")
        if any(not key or count <= 0 for key, count in value.items()):
            raise ValueError("summary counts must use non-empty keys and positive values")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> M2DatasetManifest:
        """Cross-check every stage count and content-derived version."""

        if self.dataset_version != f"m2-sft-v1-{self.content_sha256[:8]}":
            raise ValueError("dataset version does not match content hash")
        if [source.source for source in self.source_lineage] != ["commitpackft", "oasst1"]:
            raise ValueError("source lineage must contain sorted commitpackft and oasst1 entries")
        if self.tokenized_samples + self.tokenization_rejections != self.input_processed_samples:
            raise ValueError("tokenized and rejected counts must equal processed input count")
        if self.balanced_samples + self.balance_rejections != self.tokenized_samples:
            raise ValueError("balanced and downsampled counts must equal tokenized count")
        if sum(self.split_sample_counts.values()) != self.balanced_samples:
            raise ValueError("split sample counts must equal balanced samples")
        if sum(self.split_pack_counts.values()) != self.packed_sequences:
            raise ValueError("split pack counts must equal packed sequence count")
        if sum(self.split_token_counts.values()) != self.total_tokens:
            raise ValueError("split token counts must equal total tokens")
        if sum(self.split_supervised_token_counts.values()) != self.total_supervised_tokens:
            raise ValueError("split supervised counts must equal total supervised tokens")
        if sum(self.source_token_counts.values()) != self.total_tokens:
            raise ValueError("source token counts must equal total tokens")
        if sum(self.language_token_counts.values()) != self.total_tokens:
            raise ValueError("language token counts must equal total tokens")
        if sum(self.license_sample_counts.values()) != self.balanced_samples:
            raise ValueError("license sample counts must equal balanced samples")
        train_tokens = self.split_token_counts["train"]
        if sum(self.train_stratum_token_counts.values()) != train_tokens:
            raise ValueError("Train Stratum token counts must equal Train split tokens")
        expected_basis_points = {
            stratum: count * 10_000 // train_tokens if train_tokens else 0
            for stratum, count in self.train_stratum_token_counts.items()
        }
        if self.train_stratum_basis_points != expected_basis_points:
            raise ValueError("Train Stratum basis points do not match Train token counts")
        expected_capacity = self.packed_sequences * self.max_sequence_length
        if self.packing_capacity_tokens != expected_capacity:
            raise ValueError("packing capacity does not match pack count and sequence length")
        expected_efficiency = (
            self.total_tokens * 10_000 // expected_capacity if expected_capacity else 0
        )
        if self.packing_efficiency_basis_points != expected_efficiency:
            raise ValueError("packing efficiency does not match observed tokens and capacity")
        if sum(self.rejection_counts.values()) != (
            self.tokenization_rejections + self.balance_rejections
        ):
            raise ValueError("rejection counts do not match stage rejection totals")
        return self
