"""Atomic immutable registry for verified M2 NumPy-sharded datasets."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import TypeVar, cast

import numpy as np
from pydantic import ValidationError

from tinyllm.data.acquisition import M2AcquisitionManifest
from tinyllm.data.packing import DatasetBuild
from tinyllm.data.packing_schema import (
    M2DatasetManifest,
    M2PackingConfig,
    PackedSequence,
    SourceDatasetLineage,
)
from tinyllm.data.processing_schema import (
    DataProcessingManifest,
    DataSplit,
    PipelineRejectedRecord,
)
from tinyllm.data.registry_schema import (
    DatasetArtifactFile,
    DatasetCommitMarker,
    DatasetRegistration,
    DatasetShardMetadata,
    DatasetShardPack,
    RegisteredDatasetSummary,
)
from tinyllm.data.schema import DataImportManifest, RejectedRecord
from tinyllm.data.tokenization_schema import M2TokenizationConfig
from tinyllm.schemas.base import StrictSchema

DEFAULT_SHARD_TOKEN_LIMIT = 4_194_304
_SPLITS: tuple[DataSplit, ...] = ("test", "train", "validation")
_ARRAY_DTYPES = {
    "input_ids": "<i4",
    "labels": "<i4",
    "pack_offsets": "<i8",
    "position_ids": "<u2",
    "segment_ids": "<u2",
}
_DATASET_VERSION_PATTERN = re.compile(r"^m2-sft-v1-[0-9a-f]{8}$")
_SchemaT = TypeVar("_SchemaT", bound=StrictSchema)


class DatasetRegistryErrorCode(StrEnum):
    """Stable immutable-dataset failure classes."""

    NOT_FOUND = "DATASET_NOT_FOUND"
    INCOMPLETE = "DATASET_INCOMPLETE"
    CORRUPT = "DATASET_CORRUPT"
    CONFLICT = "DATASET_CONFLICT"
    WRITE_FAILED = "DATASET_WRITE_FAILED"
    INVALID_INPUT = "DATASET_INVALID_INPUT"


class DatasetRegistryError(RuntimeError):
    """Dataset Registry failure with a stable public error code."""

    def __init__(self, code: DatasetRegistryErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class DatasetLineage:
    """Deterministic stage records and content-free rejections persisted with Packs."""

    acquisition_manifest: M2AcquisitionManifest
    source_manifests: tuple[DataImportManifest, DataImportManifest]
    processing_manifest: DataProcessingManifest
    tokenization_config: M2TokenizationConfig
    packing_config: M2PackingConfig
    oasst1_rejected: tuple[RejectedRecord, ...]
    commitpackft_rejected: tuple[RejectedRecord, ...]
    processing_rejected: tuple[PipelineRejectedRecord, ...]


@dataclass(frozen=True, slots=True)
class DatasetRegistrationResult:
    """An opened committed version and whether this call atomically created it."""

    dataset: RegisteredDataset
    created: bool


@dataclass(frozen=True, slots=True)
class RegisteredDataset:
    """A fully hash-verified committed dataset directory."""

    root: Path
    manifest: M2DatasetManifest
    registration: DatasetRegistration
    marker: DatasetCommitMarker

    def iter_packs(self) -> Iterator[PackedSequence]:
        """Reconstruct validated Packs from safe NumPy arrays in deterministic order."""

        metadata_files = tuple(
            item for item in self.registration.files if item.role == "shard_metadata"
        )
        observed_packs = 0
        observed_tokens = 0
        for artifact in metadata_files:
            metadata_path = _safe_registered_path(self.root, artifact.path)
            metadata = _read_schema(metadata_path, DatasetShardMetadata)
            shard_root = metadata_path.parent
            arrays = {
                name: _load_array(shard_root / f"{name}.npy", dtype)
                for name, dtype in _ARRAY_DTYPES.items()
            }
            offsets = arrays["pack_offsets"]
            if offsets.shape != (metadata.pack_count + 1,):
                raise DatasetRegistryError(
                    DatasetRegistryErrorCode.CORRUPT,
                    "dataset shard Pack offsets have an invalid shape",
                )
            if int(offsets[0]) != 0 or int(offsets[-1]) != metadata.token_count:
                raise DatasetRegistryError(
                    DatasetRegistryErrorCode.CORRUPT,
                    "dataset shard Pack offsets do not cover token arrays",
                )
            if np.any(offsets[1:] <= offsets[:-1]):
                raise DatasetRegistryError(
                    DatasetRegistryErrorCode.CORRUPT,
                    "dataset shard Pack offsets are not strictly increasing",
                )
            for name in ("input_ids", "labels", "position_ids", "segment_ids"):
                if arrays[name].shape != (metadata.token_count,):
                    raise DatasetRegistryError(
                        DatasetRegistryErrorCode.CORRUPT,
                        "dataset shard token array has an invalid shape",
                    )
            for index, pack_metadata in enumerate(metadata.packs):
                start, end = int(offsets[index]), int(offsets[index + 1])
                if end - start != pack_metadata.token_count:
                    raise DatasetRegistryError(
                        DatasetRegistryErrorCode.CORRUPT,
                        "dataset shard Pack offset does not match metadata",
                    )
                try:
                    pack = PackedSequence(
                        pack_id=pack_metadata.pack_id,
                        pack_sha256=pack_metadata.pack_sha256,
                        split=metadata.split,
                        max_sequence_length=pack_metadata.max_sequence_length,
                        sample_ids=pack_metadata.sample_ids,
                        sample_token_counts=pack_metadata.sample_token_counts,
                        input_ids=tuple(int(item) for item in arrays["input_ids"][start:end]),
                        labels=tuple(int(item) for item in arrays["labels"][start:end]),
                        position_ids=tuple(int(item) for item in arrays["position_ids"][start:end]),
                        segment_ids=tuple(int(item) for item in arrays["segment_ids"][start:end]),
                        token_count=pack_metadata.token_count,
                        supervised_token_count=pack_metadata.supervised_token_count,
                    )
                except ValidationError as exc:
                    raise DatasetRegistryError(
                        DatasetRegistryErrorCode.CORRUPT,
                        "dataset shard Pack failed content validation",
                    ) from exc
                observed_packs += 1
                observed_tokens += pack.token_count
                yield pack
        if observed_packs != self.manifest.packed_sequences:
            raise DatasetRegistryError(
                DatasetRegistryErrorCode.CORRUPT,
                "registered shard Pack total does not match Manifest",
            )
        if observed_tokens != self.manifest.total_tokens:
            raise DatasetRegistryError(
                DatasetRegistryErrorCode.CORRUPT,
                "registered shard Token total does not match Manifest",
            )


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_durable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_schema(path: Path, value: StrictSchema) -> None:
    _write_durable(path, _json_bytes(value.to_dict()))


def _write_jsonl(path: Path, values: Iterable[StrictSchema]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for value in values:
            handle.write(_canonical_json_bytes(value.to_dict()) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_schema(path: Path, schema: type[_SchemaT]) -> _SchemaT:
    try:
        return schema.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.CORRUPT,
            f"registered {path.name} failed Schema validation",
        ) from exc


def _safe_registered_path(root: Path, relative: PurePosixPath) -> Path:
    path = root.joinpath(*relative.parts)
    try:
        if not path.resolve(strict=False).is_relative_to(root.resolve(strict=True)):
            raise DatasetRegistryError(
                DatasetRegistryErrorCode.CORRUPT,
                "registered dataset file escapes version directory",
            )
    except OSError as exc:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.CORRUPT,
            "cannot resolve registered dataset file",
        ) from exc
    if path.is_symlink():
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.CORRUPT,
            "registered dataset cannot contain symbolic links",
        )
    return path


def _artifact_file(root: Path, path: Path, role: str) -> DatasetArtifactFile:
    relative = PurePosixPath(path.relative_to(root).as_posix())
    return DatasetArtifactFile.model_validate(
        {
            "path": relative,
            "role": role,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    )


def _split_shards(
    packs: tuple[PackedSequence, ...],
    *,
    token_limit: int,
) -> tuple[tuple[PackedSequence, ...], ...]:
    shards: list[tuple[PackedSequence, ...]] = []
    current: list[PackedSequence] = []
    current_tokens = 0
    for pack in packs:
        if current and current_tokens + pack.token_count > token_limit:
            shards.append(tuple(current))
            current = []
            current_tokens = 0
        current.append(pack)
        current_tokens += pack.token_count
    if current:
        shards.append(tuple(current))
    return tuple(shards)


def _write_numpy_array(path: Path, values: Iterable[int], *, count: int, dtype: str) -> None:
    array = np.fromiter(values, dtype=np.dtype(dtype), count=count)
    if array.shape != (count,):
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INVALID_INPUT,
            "dataset shard array count does not match Packs",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _write_shards(
    root: Path,
    packs: tuple[PackedSequence, ...],
    *,
    token_limit: int,
) -> list[DatasetArtifactFile]:
    files: list[DatasetArtifactFile] = []
    for split in _SPLITS:
        split_packs = tuple(
            sorted((pack for pack in packs if pack.split == split), key=lambda p: p.pack_id)
        )
        for shard_index, shard_packs in enumerate(
            _split_shards(split_packs, token_limit=token_limit)
        ):
            shard_root = root / "shards" / f"{split}-{shard_index:05d}"
            token_count = sum(pack.token_count for pack in shard_packs)
            if token_count > token_limit:
                raise DatasetRegistryError(
                    DatasetRegistryErrorCode.INVALID_INPUT,
                    "dataset shard exceeds configured Token limit",
                )
            array_values = {
                "input_ids": chain.from_iterable(pack.input_ids for pack in shard_packs),
                "labels": chain.from_iterable(pack.labels for pack in shard_packs),
                "position_ids": chain.from_iterable(pack.position_ids for pack in shard_packs),
                "segment_ids": chain.from_iterable(pack.segment_ids for pack in shard_packs),
            }
            for name, values in array_values.items():
                path = shard_root / f"{name}.npy"
                _write_numpy_array(
                    path,
                    values,
                    count=token_count,
                    dtype=_ARRAY_DTYPES[name],
                )
                files.append(_artifact_file(root, path, "shard_array"))
            offsets = [0]
            for pack in shard_packs:
                offsets.append(offsets[-1] + pack.token_count)
            offsets_path = shard_root / "pack_offsets.npy"
            _write_numpy_array(
                offsets_path,
                offsets,
                count=len(offsets),
                dtype=_ARRAY_DTYPES["pack_offsets"],
            )
            files.append(_artifact_file(root, offsets_path, "shard_array"))

            metadata = DatasetShardMetadata(
                storage_format="numpy-sharded-v1",
                split=split,
                shard_index=shard_index,
                token_count=token_count,
                pack_count=len(shard_packs),
                packs=tuple(
                    DatasetShardPack(
                        pack_id=pack.pack_id,
                        pack_sha256=pack.pack_sha256,
                        max_sequence_length=pack.max_sequence_length,
                        sample_ids=pack.sample_ids,
                        sample_token_counts=pack.sample_token_counts,
                        token_count=pack.token_count,
                        supervised_token_count=pack.supervised_token_count,
                    )
                    for pack in shard_packs
                ),
                array_dtypes=dict(sorted(_ARRAY_DTYPES.items())),
            )
            metadata_path = shard_root / "metadata.json"
            _write_schema(metadata_path, metadata)
            files.append(_artifact_file(root, metadata_path, "shard_metadata"))
    return files


def _write_lineage_and_rejections(
    root: Path,
    build: DatasetBuild,
    lineage: DatasetLineage,
) -> list[DatasetArtifactFile]:
    source_manifests = sorted(lineage.source_manifests, key=lambda item: item.source.name)
    lineage_values: tuple[tuple[str, StrictSchema], ...] = (
        ("lineage/acquisition.json", lineage.acquisition_manifest),
        ("lineage/source-commitpackft.json", source_manifests[0]),
        ("lineage/source-oasst1.json", source_manifests[1]),
        ("lineage/processing.json", lineage.processing_manifest),
        ("lineage/tokenization-config.json", lineage.tokenization_config),
        ("lineage/packing-config.json", lineage.packing_config),
    )
    files: list[DatasetArtifactFile] = []
    for relative, value in lineage_values:
        path = root / relative
        _write_schema(path, value)
        files.append(_artifact_file(root, path, "lineage"))

    rejection_values: tuple[tuple[str, Iterable[StrictSchema]], ...] = (
        ("rejections/import-oasst1.jsonl", lineage.oasst1_rejected),
        ("rejections/import-commitpackft.jsonl", lineage.commitpackft_rejected),
        ("rejections/processing.jsonl", lineage.processing_rejected),
        ("rejections/tokenization.jsonl", build.tokenization_rejected),
        ("rejections/balance.jsonl", build.balance_rejected),
    )
    for relative, values in rejection_values:
        path = root / relative
        _write_jsonl(path, values)
        files.append(_artifact_file(root, path, "rejections"))
    return files


def _validate_lineage(build: DatasetBuild, lineage: DatasetLineage) -> None:
    ordered_sources = tuple(sorted(lineage.source_manifests, key=lambda item: item.source.name))
    expected_sources = tuple(
        SourceDatasetLineage(
            source=manifest.source.name,
            dataset_id=manifest.source.dataset_id,
            revision=manifest.source.revision,
            dataset_card_sha256=manifest.source.dataset_card_sha256,
            input_sha256=manifest.input_sha256,
            import_config_sha256=manifest.config_sha256,
        )
        for manifest in ordered_sources
    )
    if expected_sources != build.manifest.source_lineage:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INVALID_INPUT,
            "registration source lineage does not match Dataset Manifest",
        )
    processing = lineage.processing_manifest
    if (
        processing.config_sha256 != build.manifest.processing_config_sha256
        or processing.input_sha256 != build.manifest.processing_input_sha256
        or processing.output_sha256 != build.manifest.processing_output_sha256
        or processing.output_samples != build.manifest.input_processed_samples
    ):
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INVALID_INPUT,
            "registration processing lineage does not match Dataset Manifest",
        )
    if (
        lineage.tokenization_config.tokenizer != build.manifest.tokenizer
        or lineage.tokenization_config.template != build.manifest.template
    ):
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INVALID_INPUT,
            "registration Tokenizer lineage does not match Dataset Manifest",
        )
    packing_hash = hashlib.sha256(
        _canonical_json_bytes(lineage.packing_config.to_dict())
    ).hexdigest()
    if packing_hash != build.manifest.packing_config_sha256:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INVALID_INPUT,
            "registration Packing config does not match Dataset Manifest",
        )
    rejection_sets = {
        "oasst1": lineage.oasst1_rejected,
        "commitpackft": lineage.commitpackft_rejected,
    }
    for manifest in ordered_sources:
        records = rejection_sets[manifest.source.name]
        if (
            len(records) != manifest.rejected_samples
            or dict(sorted(Counter(record.reason for record in records).items()))
            != manifest.rejection_counts
        ):
            raise DatasetRegistryError(
                DatasetRegistryErrorCode.INVALID_INPUT,
                "registration import rejections do not match source Manifest",
            )
    if (
        len(lineage.processing_rejected) != processing.rejected_samples
        or dict(sorted(Counter(record.reason for record in lineage.processing_rejected).items()))
        != processing.rejection_counts
    ):
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INVALID_INPUT,
            "registration processing rejections do not match processing Manifest",
        )


def _load_array(path: Path, dtype: str) -> np.ndarray:
    if not path.is_file() or path.is_symlink():
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.CORRUPT,
            "registered NumPy shard array is missing",
        )
    try:
        array = cast(np.ndarray, np.load(path, allow_pickle=False, mmap_mode="r"))
    except (OSError, ValueError) as exc:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.CORRUPT,
            "registered NumPy shard array cannot be loaded safely",
        ) from exc
    if array.ndim != 1 or array.dtype.str != dtype:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.CORRUPT,
            "registered NumPy shard dtype or rank is invalid",
        )
    return array


def _validate_registered_files(root: Path, registration: DatasetRegistration) -> None:
    expected = {str(item.path): item for item in registration.files}
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise DatasetRegistryError(
                DatasetRegistryErrorCode.CORRUPT,
                "registered dataset cannot contain symbolic links",
            )
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in {"manifest.json", "registration.json", "COMMITTED"}:
                actual.add(relative)
    if actual != set(expected):
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.CORRUPT,
            "registered dataset file inventory does not match Registration",
        )
    for relative, record in expected.items():
        path = _safe_registered_path(root, PurePosixPath(relative))
        if not path.is_file() or path.stat().st_size != record.size_bytes:
            raise DatasetRegistryError(
                DatasetRegistryErrorCode.CORRUPT,
                "registered dataset file size mismatch",
            )
        if _sha256_file(path) != record.sha256:
            raise DatasetRegistryError(
                DatasetRegistryErrorCode.CORRUPT,
                "registered dataset file SHA256 mismatch",
            )


def open_registered_dataset(
    *,
    artifact_root: Path,
    dataset_version: str,
) -> RegisteredDataset:
    """Open only after validating completion, Schemas, inventory, and every file hash."""

    if not artifact_root.is_absolute():
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INVALID_INPUT,
            "Artifact Root must be absolute",
        )
    if _DATASET_VERSION_PATTERN.fullmatch(dataset_version) is None:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INVALID_INPUT,
            "Dataset Version has an invalid format",
        )
    dataset_parent = artifact_root / "datasets" / "m2-sft"
    root = dataset_parent / dataset_version
    try:
        if not root.resolve(strict=False).is_relative_to(dataset_parent.resolve(strict=False)):
            raise DatasetRegistryError(
                DatasetRegistryErrorCode.INVALID_INPUT,
                "Dataset Version escapes Registry root",
            )
    except OSError as exc:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INVALID_INPUT,
            "cannot resolve Dataset Registry path",
        ) from exc
    if not root.is_dir() or root.is_symlink():
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.NOT_FOUND,
            "registered Dataset Version does not exist",
        )
    marker_path = root / "COMMITTED"
    if not marker_path.is_file() or marker_path.is_symlink():
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INCOMPLETE,
            "registered Dataset Version has no valid completion marker",
        )
    manifest_path = root / "manifest.json"
    registration_path = root / "registration.json"
    manifest = _read_schema(manifest_path, M2DatasetManifest)
    registration = _read_schema(registration_path, DatasetRegistration)
    marker = _read_schema(marker_path, DatasetCommitMarker)
    manifest_sha256 = _sha256_file(manifest_path)
    registration_sha256 = _sha256_file(registration_path)
    if marker.manifest_sha256 != manifest_sha256 or marker.registration_sha256 != (
        registration_sha256
    ):
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.CORRUPT,
            "dataset completion marker hash mismatch",
        )
    if manifest.dataset_version != dataset_version or registration.dataset_version != (
        dataset_version
    ):
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.CONFLICT,
            "dataset directory version does not match Manifest or Registration",
        )
    identities = {
        manifest.content_sha256,
        registration.content_sha256,
        marker.content_sha256,
    }
    if len(identities) != 1 or marker.dataset_version != dataset_version:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.CONFLICT,
            "dataset content identity is inconsistent",
        )
    if registration.manifest_sha256 != manifest_sha256:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.CORRUPT,
            "Registration Manifest hash mismatch",
        )
    if registration.packed_sequences != manifest.packed_sequences or (
        registration.total_tokens != manifest.total_tokens
    ):
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.CONFLICT,
            "Registration counts do not match Manifest",
        )
    _validate_registered_files(root, registration)
    return RegisteredDataset(
        root=root,
        manifest=manifest,
        registration=registration,
        marker=marker,
    )


def summarize_registered_dataset(
    dataset: RegisteredDataset,
    *,
    operation: str,
    created: bool | None,
) -> RegisteredDatasetSummary:
    """Build path-free stable CLI evidence from already verified lineage files."""

    sources = tuple(
        _read_schema(
            dataset.root / f"lineage/source-{source}.json",
            DataImportManifest,
        )
        for source in ("commitpackft", "oasst1")
    )
    processing = _read_schema(dataset.root / "lineage/processing.json", DataProcessingManifest)
    rejection_counts: dict[str, int] = {}
    for source in sources:
        for reason, count in source.rejection_counts.items():
            rejection_counts[f"import.{source.source.name}.{reason}"] = count
    for reason, count in processing.rejection_counts.items():
        rejection_counts[f"processing.{reason}"] = count
    for reason, count in dataset.manifest.rejection_counts.items():
        stage = "balance" if reason == "balance_downsampled" else "tokenization"
        rejection_counts[f"{stage}.{reason}"] = count
    registered_bytes = sum(item.size_bytes for item in dataset.registration.files) + sum(
        (dataset.root / name).stat().st_size
        for name in ("manifest.json", "registration.json", "COMMITTED")
    )
    return RegisteredDatasetSummary.model_validate(
        {
            "status": "ok",
            "operation": operation,
            "created": created,
            "verified": True,
            "dataset_name": dataset.manifest.dataset_name,
            "dataset_version": dataset.manifest.dataset_version,
            "content_sha256": dataset.manifest.content_sha256,
            "storage_format": dataset.registration.storage_format,
            "registered_at": dataset.registration.registered_at,
            "git_commit": dataset.registration.git_commit,
            "git_dirty": dataset.registration.git_dirty,
            "source_rows": {source.source.name: source.source_rows for source in sources},
            "imported_samples": {source.source.name: source.accepted_samples for source in sources},
            "processed_samples": processing.output_samples,
            "tokenized_samples": dataset.manifest.tokenized_samples,
            "balanced_samples": dataset.manifest.balanced_samples,
            "packed_sequences": dataset.manifest.packed_sequences,
            "total_tokens": dataset.manifest.total_tokens,
            "total_supervised_tokens": dataset.manifest.total_supervised_tokens,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "registered_files": len(dataset.registration.files) + 3,
            "registered_bytes": registered_bytes,
        }
    )


@contextmanager
def _registry_lock(dataset_parent: Path) -> Iterator[None]:
    lock_path = dataset_parent / ".registry.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.WRITE_FAILED,
            "cannot acquire Dataset Registry writer lock",
        ) from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _register_dataset_locked(
    build: DatasetBuild,
    *,
    artifact_root: Path,
    lineage: DatasetLineage,
    git_commit: str,
    git_dirty: bool,
    shard_token_limit: int = DEFAULT_SHARD_TOKEN_LIMIT,
    registered_at: datetime | None = None,
) -> DatasetRegistrationResult:
    """Write, hash, validate, and atomically publish one immutable Dataset Version."""

    if not artifact_root.is_absolute():
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INVALID_INPUT,
            "Artifact Root must be absolute",
        )
    if shard_token_limit < build.manifest.max_sequence_length:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INVALID_INPUT,
            "shard Token limit must fit one maximum-length Pack",
        )
    _validate_lineage(build, lineage)
    dataset_parent = artifact_root / "datasets" / build.manifest.dataset_name
    destination = dataset_parent / build.manifest.dataset_version
    if destination.exists() or destination.is_symlink():
        existing = open_registered_dataset(
            artifact_root=artifact_root,
            dataset_version=build.manifest.dataset_version,
        )
        if existing.manifest != build.manifest:
            raise DatasetRegistryError(
                DatasetRegistryErrorCode.CONFLICT,
                "existing Dataset Version has different content",
            )
        return DatasetRegistrationResult(dataset=existing, created=False)

    try:
        dataset_parent.mkdir(parents=True, exist_ok=True)
        required_bytes = max(16 * 1024 * 1024, build.manifest.total_tokens * 14)
        if shutil.disk_usage(dataset_parent).free < required_bytes:
            raise DatasetRegistryError(
                DatasetRegistryErrorCode.WRITE_FAILED,
                "insufficient free space for registered dataset",
            )
    except OSError as exc:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.WRITE_FAILED,
            "cannot prepare Dataset Registry directory",
        ) from exc

    temporary = dataset_parent / f".{build.manifest.dataset_version}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        manifest_path = temporary / "manifest.json"
        _write_schema(manifest_path, build.manifest)
        manifest_sha256 = _sha256_file(manifest_path)
        files = _write_lineage_and_rejections(temporary, build, lineage)
        files.extend(
            _write_shards(
                temporary,
                build.packs,
                token_limit=shard_token_limit,
            )
        )
        registration = DatasetRegistration(
            dataset_name=build.manifest.dataset_name,
            dataset_version=build.manifest.dataset_version,
            content_sha256=build.manifest.content_sha256,
            manifest_sha256=manifest_sha256,
            storage_format="numpy-sharded-v1",
            shard_token_limit=shard_token_limit,
            registered_at=registered_at or datetime.now(UTC),
            git_commit=git_commit,
            git_dirty=git_dirty,
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            packed_sequences=build.manifest.packed_sequences,
            total_tokens=build.manifest.total_tokens,
            files=tuple(sorted(files, key=lambda item: str(item.path))),
        )
        registration_path = temporary / "registration.json"
        _write_schema(registration_path, registration)
        marker = DatasetCommitMarker(
            dataset_name=build.manifest.dataset_name,
            dataset_version=build.manifest.dataset_version,
            content_sha256=build.manifest.content_sha256,
            manifest_sha256=manifest_sha256,
            registration_sha256=_sha256_file(registration_path),
        )
        _write_schema(temporary / "COMMITTED", marker)
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(temporary)
        os.rename(temporary, destination)
        _fsync_directory(dataset_parent)
    except DatasetRegistryError:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except (OSError, ValidationError, ValueError, OverflowError) as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.WRITE_FAILED,
            "cannot atomically register Dataset Version",
        ) from exc

    opened = open_registered_dataset(
        artifact_root=artifact_root,
        dataset_version=build.manifest.dataset_version,
    )
    return DatasetRegistrationResult(dataset=opened, created=True)


def register_dataset(
    build: DatasetBuild,
    *,
    artifact_root: Path,
    lineage: DatasetLineage,
    git_commit: str,
    git_dirty: bool,
    shard_token_limit: int = DEFAULT_SHARD_TOKEN_LIMIT,
    registered_at: datetime | None = None,
) -> DatasetRegistrationResult:
    """Serialize writers so the atomic final Rename can never replace a competing version."""

    if not artifact_root.is_absolute():
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.INVALID_INPUT,
            "Artifact Root must be absolute",
        )
    dataset_parent = artifact_root / "datasets" / build.manifest.dataset_name
    try:
        dataset_parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatasetRegistryError(
            DatasetRegistryErrorCode.WRITE_FAILED,
            "cannot prepare Dataset Registry directory",
        ) from exc
    with _registry_lock(dataset_parent):
        return _register_dataset_locked(
            build,
            artifact_root=artifact_root,
            lineage=lineage,
            git_commit=git_commit,
            git_dirty=git_dirty,
            shard_token_limit=shard_token_limit,
            registered_at=registered_at,
        )
