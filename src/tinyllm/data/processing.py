"""Deterministic M2.2 normalization, exact deduplication, and grouped splitting."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from tinyllm.data.processing_schema import (
    DataProcessingManifest,
    DataSplit,
    M2ProcessingConfig,
    PipelineRejectedRecord,
    PipelineRejectionReason,
    ProcessedSample,
)
from tinyllm.data.schema import ImportedMessage, ImportedSample


class DataProcessingError(ValueError):
    """Raised when input identity or the M2.2 YAML contract is invalid."""


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Processed samples, content-free rejections, and deterministic evidence."""

    manifest: DataProcessingManifest
    samples: tuple[ProcessedSample, ...]
    rejected: tuple[PipelineRejectedRecord, ...]


@dataclass(frozen=True, slots=True)
class _NormalizedSample:
    sample: ImportedSample
    messages: tuple[ImportedMessage, ...]
    content_key: str
    content_sha256: str
    group_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DeduplicatedSample:
    normalized: _NormalizedSample
    origin_sample_ids: tuple[str, ...]
    origin_record_sha256s: tuple[str, ...]
    group_keys: tuple[str, ...]


class _DisjointSet:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self._parent.setdefault(value, value)

    def find(self, value: str) -> str:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self._parent[second] = first

    def values(self) -> tuple[str, ...]:
        return tuple(sorted(self._parent))


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sequence_hash(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = _canonical_json(value).encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def load_m2_processing_config(path: Path) -> M2ProcessingConfig:
    """Load a strict M2.2 YAML configuration without allowing implicit defaults in files."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise DataProcessingError("data processing config must use a .yaml or .yml extension")
    try:
        decoded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DataProcessingError(f"cannot read data processing config: {path}") from exc
    except yaml.YAMLError as exc:
        raise DataProcessingError(f"invalid YAML in data processing config: {path}") from exc
    try:
        return M2ProcessingConfig.model_validate(decoded)
    except ValidationError as exc:
        messages: list[str] = []
        for error in exc.errors(include_url=False, include_context=False):
            location = ".".join(str(part) for part in error["loc"])
            messages.append(f"{location}: {error['msg']}" if location else str(error["msg"]))
        raise DataProcessingError("; ".join(messages)) from exc


def _normalize_text(value: str, config: M2ProcessingConfig) -> str:
    policy = config.normalization
    normalized = unicodedata.normalize("NFC", value)
    if policy.normalize_line_endings:
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if policy.strip_bom:
        normalized = normalized.removeprefix("\ufeff")
    if policy.trim_outer_whitespace:
        normalized = normalized.strip()
    return normalized


def _has_forbidden_control(value: str) -> bool:
    return any(
        character not in {"\n", "\t"} and unicodedata.category(character) == "Cc"
        for character in value
    )


def _pipeline_rejection(
    sample: ImportedSample,
    reason: PipelineRejectionReason,
    *,
    message_index: int | None = None,
    content_sha256: str | None = None,
    duplicate_of_sample_id: str | None = None,
) -> PipelineRejectedRecord:
    return PipelineRejectedRecord(
        sample_id=sample.id,
        source=sample.source,
        reason=reason,
        message_index=message_index,
        content_sha256=content_sha256,
        duplicate_of_sample_id=duplicate_of_sample_id,
    )


def _normalize_sample(
    sample: ImportedSample, config: M2ProcessingConfig
) -> tuple[_NormalizedSample | None, PipelineRejectedRecord | None]:
    messages: list[ImportedMessage] = []
    for index, message in enumerate(sample.messages):
        content = _normalize_text(message.content, config)
        if not content:
            return None, _pipeline_rejection(
                sample, "empty_after_normalization", message_index=index
            )
        if config.normalization.reject_control_characters and _has_forbidden_control(content):
            return None, _pipeline_rejection(
                sample, "forbidden_control_character", message_index=index
            )
        if len(content) > config.normalization.max_message_chars:
            return None, _pipeline_rejection(sample, "message_too_long", message_index=index)
        messages.append(ImportedMessage(role=message.role, content=content))

    if sum(len(message.content) for message in messages) > config.normalization.max_sample_chars:
        return None, _pipeline_rejection(sample, "sample_too_long")

    message_tuple = tuple(messages)
    content_payload = [message.to_dict() for message in message_tuple]
    content_key = _canonical_json(content_payload)
    group_keys = tuple(
        sorted(f"{sample.source}:{group_id}" for group_id in sample.metadata.group_ids)
    )
    return (
        _NormalizedSample(
            sample=sample,
            messages=message_tuple,
            content_key=content_key,
            content_sha256=hashlib.sha256(content_key.encode("utf-8")).hexdigest(),
            group_keys=group_keys,
        ),
        None,
    )


def _deduplicate(
    normalized: list[_NormalizedSample],
    config: M2ProcessingConfig,
) -> tuple[list[_DeduplicatedSample], list[PipelineRejectedRecord]]:
    priority = {source: index for index, source in enumerate(config.deduplication.source_priority)}
    by_content: dict[str, list[_NormalizedSample]] = defaultdict(list)
    for sample in normalized:
        by_content[sample.content_key].append(sample)

    kept: list[_DeduplicatedSample] = []
    rejected: list[PipelineRejectedRecord] = []
    for content_key in sorted(by_content):
        candidates = sorted(
            by_content[content_key],
            key=lambda item: (priority[item.sample.source], item.sample.id),
        )
        winner = candidates[0]
        origin_sample_ids = tuple(sorted(item.sample.id for item in candidates))
        origin_record_hashes = tuple(
            sorted(
                {
                    record_hash
                    for item in candidates
                    for record_hash in item.sample.metadata.raw_record_sha256s
                }
            )
        )
        group_keys = tuple(
            sorted({group_key for item in candidates for group_key in item.group_keys})
        )
        kept.append(
            _DeduplicatedSample(
                normalized=winner,
                origin_sample_ids=origin_sample_ids,
                origin_record_sha256s=origin_record_hashes,
                group_keys=group_keys,
            )
        )
        for duplicate in candidates[1:]:
            rejected.append(
                _pipeline_rejection(
                    duplicate.sample,
                    "exact_duplicate",
                    content_sha256=winner.content_sha256,
                    duplicate_of_sample_id=winner.sample.id,
                )
            )
    kept.sort(key=lambda item: item.normalized.sample.id)
    return kept, rejected


def _component_assignments(
    samples: list[_DeduplicatedSample],
) -> tuple[dict[str, str], int]:
    groups = _DisjointSet()
    for sample in samples:
        for group_key in sample.group_keys:
            groups.add(group_key)
        first = sample.group_keys[0]
        for group_key in sample.group_keys[1:]:
            groups.union(first, group_key)

    members: dict[str, list[str]] = defaultdict(list)
    for group_key in groups.values():
        members[groups.find(group_key)].append(group_key)
    component_by_root = {
        root: _content_hash(sorted(component_members))
        for root, component_members in members.items()
    }
    assignment = {
        group_key: component_by_root[groups.find(group_key)] for group_key in groups.values()
    }
    return assignment, len(component_by_root)


def _split_for_component(component_id: str, config: M2ProcessingConfig) -> DataSplit:
    payload = f"{config.split.seed}\0{component_id}".encode()
    bucket = int(hashlib.sha256(payload).hexdigest()[:16], 16) % 10_000
    if bucket < config.split.train_basis_points:
        return "train"
    validation_end = config.split.train_basis_points + config.split.validation_basis_points
    if bucket < validation_end:
        return "validation"
    return "test"


def _validate_unique_sample_ids(samples: tuple[ImportedSample, ...]) -> None:
    counts = Counter(sample.id for sample in samples)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        raise DataProcessingError(f"duplicate imported sample ID: {duplicates[0]}")


def process_imported_samples(
    samples: Iterable[ImportedSample],
    *,
    config: M2ProcessingConfig,
) -> ProcessingResult:
    """Normalize, exact-deduplicate, and group-split imported samples deterministically."""

    materialized = tuple(samples)
    _validate_unique_sample_ids(materialized)
    ordered_input = tuple(sorted(materialized, key=lambda sample: sample.id))
    input_sha256 = _sequence_hash(sample.to_dict() for sample in ordered_input)
    config_sha256 = _content_hash(config.to_dict())

    normalized: list[_NormalizedSample] = []
    rejected: list[PipelineRejectedRecord] = []
    for sample in ordered_input:
        normalized_sample, rejection = _normalize_sample(sample, config)
        if rejection is not None:
            rejected.append(rejection)
        elif normalized_sample is not None:
            normalized.append(normalized_sample)

    deduplicated, duplicate_rejections = _deduplicate(normalized, config)
    rejected.extend(duplicate_rejections)
    component_by_group, component_count = _component_assignments(deduplicated)

    processed: list[ProcessedSample] = []
    for item in deduplicated:
        component_id = component_by_group[item.group_keys[0]]
        sample = item.normalized.sample
        processed.append(
            ProcessedSample(
                id=sample.id,
                source=sample.source,
                messages=item.normalized.messages,
                metadata=sample.metadata,
                content_sha256=item.normalized.content_sha256,
                origin_sample_ids=item.origin_sample_ids,
                origin_record_sha256s=item.origin_record_sha256s,
                group_keys=item.group_keys,
                component_id=component_id,
                split=_split_for_component(component_id, config),
            )
        )
    processed.sort(key=lambda sample: sample.id)
    rejected.sort(key=lambda record: (record.sample_id, record.reason))

    split_names: tuple[DataSplit, ...] = ("test", "train", "validation")
    split_counts = {
        split: sum(sample.split == split for sample in processed) for split in split_names
    }
    split_sha256s = {
        split: _sequence_hash(sample.to_dict() for sample in processed if sample.split == split)
        for split in split_names
    }
    rejection_counts = Counter(record.reason for record in rejected)
    normalization_rejections = len(rejected) - len(duplicate_rejections)
    manifest = DataProcessingManifest(
        input_sha256=input_sha256,
        config_sha256=config_sha256,
        output_sha256=_sequence_hash(sample.to_dict() for sample in processed),
        input_samples=len(ordered_input),
        normalized_samples=len(normalized),
        output_samples=len(processed),
        rejected_samples=len(rejected),
        normalization_rejections=normalization_rejections,
        exact_duplicates=len(duplicate_rejections),
        component_count=component_count,
        rejection_counts=dict(sorted(rejection_counts.items())),
        split_counts=cast(dict[str, int], split_counts),
        split_sha256s=cast(dict[str, str], split_sha256s),
    )
    return ProcessingResult(
        manifest=manifest,
        samples=tuple(processed),
        rejected=tuple(rejected),
    )
