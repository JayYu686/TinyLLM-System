"""Deterministic M2.3b Train balancing and boundary-aware sequence packing."""

from __future__ import annotations

import hashlib
import heapq
import json
from bisect import bisect_left, insort
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import ValidationError

from tinyllm.data.packing_schema import (
    BalanceRejectedRecord,
    M2DatasetManifest,
    M2PackingConfig,
    PackedSequence,
    SourceDatasetLineage,
)
from tinyllm.data.processing_schema import DataProcessingManifest, DataSplit
from tinyllm.data.schema import DataImportManifest
from tinyllm.data.tokenization import TokenizationBatch
from tinyllm.data.tokenization_schema import (
    M2TokenizationConfig,
    TokenizationRejectedRecord,
    TokenizedSample,
)
from tinyllm.schemas.base import StrictSchema

_SPLITS: tuple[DataSplit, ...] = ("test", "train", "validation")
_STRATA = ("commitpackft:en", "oasst1:en", "oasst1:zh")


class PackingError(ValueError):
    """Raised when balancing, lineage, or packing violates the frozen M2 contract."""


@dataclass(frozen=True, slots=True)
class DatasetBuild:
    """Final in-memory packs, rejections, and content-addressed manifest."""

    manifest: M2DatasetManifest
    packs: tuple[PackedSequence, ...]
    selected_samples: tuple[TokenizedSample, ...]
    tokenization_rejected: tuple[TokenizationRejectedRecord, ...]
    balance_rejected: tuple[BalanceRejectedRecord, ...]


@dataclass(slots=True)
class _PackBin:
    """Mutable internal bin used only while applying Best-Fit Decreasing."""

    samples: list[TokenizedSample] = field(default_factory=list)
    token_count: int = 0


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _dataset_content_hash(
    *,
    balance_rejected: tuple[BalanceRejectedRecord, ...],
    packing_config: M2PackingConfig,
    packs: tuple[PackedSequence, ...],
    processing_manifest: DataProcessingManifest,
    source_manifests: tuple[DataImportManifest, ...],
    tokenization_config: M2TokenizationConfig,
    tokenization_rejected: tuple[TokenizationRejectedRecord, ...],
) -> str:
    """Stream the canonical v1 payload without materializing every Pack as one JSON value."""

    digest = hashlib.sha256()

    def update(value: str) -> None:
        digest.update(value.encode("utf-8"))

    def scalar(value: object) -> None:
        digest.update(_canonical_json(value))

    def sequence(values: Iterable[StrictSchema]) -> None:
        update("[")
        for index, value in enumerate(values):
            if index:
                update(",")
            scalar(value.to_dict())
        update("]")

    ordered_sources = tuple(sorted(source_manifests, key=lambda item: item.source.name))
    update("{")
    update('"balance_rejected":')
    sequence(balance_rejected)
    update(',"packing_config":')
    scalar(packing_config.to_dict())
    update(',"packs":')
    sequence(packs)
    update(',"processing_manifest":')
    scalar(processing_manifest.to_dict())
    update(',"source_manifests":')
    sequence(ordered_sources)
    update(',"tokenization_config":')
    scalar(tokenization_config.to_dict())
    update(',"tokenization_rejected":')
    sequence(tokenization_rejected)
    update("}")
    return digest.hexdigest()


def _sequence_hash(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = _canonical_json(value)
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def _validation_messages(exc: ValidationError) -> str:
    messages: list[str] = []
    for error in exc.errors(include_url=False, include_context=False):
        location = ".".join(str(part) for part in error["loc"])
        messages.append(f"{location}: {error['msg']}" if location else str(error["msg"]))
    return "; ".join(messages)


def load_m2_packing_config(path: Path) -> M2PackingConfig:
    """Load the strict formal M2.3b YAML contract."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise PackingError("packing config must use a .yaml or .yml extension")
    try:
        decoded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PackingError(f"cannot read packing config: {path}") from exc
    except yaml.YAMLError as exc:
        raise PackingError(f"invalid YAML in packing config: {path}") from exc
    try:
        return M2PackingConfig.model_validate(decoded)
    except ValidationError as exc:
        raise PackingError(_validation_messages(exc)) from exc


def _stratum(sample: TokenizedSample) -> str:
    if sample.source == "commitpackft" and sample.language == "en":
        return "commitpackft:en"
    if sample.source == "oasst1" and sample.language in {"en", "zh"}:
        return f"oasst1:{sample.language}"
    raise PackingError(
        f"sample {sample.id} is outside the frozen M2 source/language strata: "
        f"{sample.source}:{sample.language}"
    )


def _selection_key(sample: TokenizedSample, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{sample.id}".encode()).hexdigest()
    return digest, sample.id


def _closest_prefix(
    samples: list[TokenizedSample],
    *,
    target_tokens: int,
    seed: int,
) -> tuple[TokenizedSample, ...]:
    ordered = sorted(samples, key=lambda sample: _selection_key(sample, seed))
    best_count = 1
    cumulative = 0
    best_distance: int | None = None
    for index, sample in enumerate(ordered, start=1):
        cumulative += sample.token_count
        distance = abs(cumulative - target_tokens)
        if best_distance is None or distance < best_distance:
            best_count = index
            best_distance = distance
        if cumulative >= target_tokens:
            break
    return tuple(ordered[:best_count])


def _balance_train_samples(
    samples: tuple[TokenizedSample, ...],
    *,
    config: M2PackingConfig,
) -> tuple[tuple[TokenizedSample, ...], tuple[BalanceRejectedRecord, ...]]:
    by_stratum: dict[str, list[TokenizedSample]] = defaultdict(list)
    retained_non_train: list[TokenizedSample] = []
    for sample in samples:
        stratum = _stratum(sample)
        if sample.split == config.balance.apply_to_split:
            by_stratum[stratum].append(sample)
        else:
            retained_non_train.append(sample)

    missing = [stratum for stratum in _STRATA if not by_stratum[stratum]]
    if missing and config.balance.require_all_strata:
        raise PackingError("Train balance is missing required Stratum: " + ", ".join(missing))

    targets = config.balance.targets()
    available_tokens = {
        stratum: sum(sample.token_count for sample in by_stratum[stratum]) for stratum in _STRATA
    }
    common_scale = min(
        available_tokens[stratum] * 10_000 // targets[stratum] for stratum in _STRATA
    )
    selected_train: list[TokenizedSample] = []
    for stratum in _STRATA:
        target_tokens = max(1, common_scale * targets[stratum] // 10_000)
        selected_train.extend(
            _closest_prefix(
                by_stratum[stratum],
                target_tokens=target_tokens,
                seed=config.balance.selection_seed,
            )
        )

    selected_ids = {sample.id for sample in selected_train}
    selected_tokens = Counter({_stratum(sample): 0 for sample in selected_train})
    for sample in selected_train:
        selected_tokens[_stratum(sample)] += sample.token_count
    total_train_tokens = sum(selected_tokens.values())
    if total_train_tokens == 0:
        raise PackingError("Train balance selected zero tokens")
    for stratum in _STRATA:
        deviation_numerator = abs(
            selected_tokens[stratum] * 10_000 - targets[stratum] * total_train_tokens
        )
        if deviation_numerator > config.balance.tolerance_basis_points * total_train_tokens:
            actual = selected_tokens[stratum] * 10_000 // total_train_tokens
            raise PackingError(
                f"Train Stratum {stratum} is outside tolerance: "
                f"target={targets[stratum]}bp actual={actual}bp"
            )

    rejected = tuple(
        sorted(
            (
                BalanceRejectedRecord(
                    sample_id=sample.id,
                    source=sample.source,
                    split="train",
                    language=cast(Literal["en", "zh"], sample.language),
                    stratum=cast(
                        Literal["oasst1:zh", "oasst1:en", "commitpackft:en"],
                        _stratum(sample),
                    ),
                    content_sha256=sample.content_sha256,
                    token_count=sample.token_count,
                    reason="balance_downsampled",
                )
                for sample in samples
                if sample.split == "train" and sample.id not in selected_ids
            ),
            key=lambda record: record.sample_id,
        )
    )
    selected = tuple(sorted((*retained_non_train, *selected_train), key=lambda sample: sample.id))
    return selected, rejected


def _pack_payload(
    *,
    split: DataSplit,
    max_sequence_length: int,
    samples: tuple[TokenizedSample, ...],
) -> dict[str, object]:
    input_ids: list[int] = []
    labels: list[int] = []
    position_ids: list[int] = []
    segment_ids: list[int] = []
    for segment_id, sample in enumerate(samples):
        input_ids.extend(sample.input_ids)
        labels.extend(sample.labels)
        position_ids.extend(range(sample.token_count))
        segment_ids.extend([segment_id] * sample.token_count)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "max_sequence_length": max_sequence_length,
        "position_ids": position_ids,
        "sample_ids": [sample.id for sample in samples],
        "sample_token_counts": [sample.token_count for sample in samples],
        "segment_ids": segment_ids,
        "split": split,
    }


def _pack_split(
    samples: tuple[TokenizedSample, ...],
    *,
    split: DataSplit,
    max_sequence_length: int,
) -> tuple[PackedSequence, ...]:
    bins: list[_PackBin] = []
    bins_by_remaining: dict[int, list[int]] = {}
    available_remaining: list[int] = []

    def add_available_bin(remaining: int, bin_index: int) -> None:
        if remaining <= 0:
            return
        indices = bins_by_remaining.setdefault(remaining, [])
        if not indices:
            insort(available_remaining, remaining)
        heapq.heappush(indices, bin_index)

    for sample in sorted(samples, key=lambda item: (-item.token_count, item.id)):
        if sample.token_count > max_sequence_length:
            raise PackingError(f"sample {sample.id} exceeds packing sequence length")
        remaining_index = bisect_left(available_remaining, sample.token_count)
        if remaining_index < len(available_remaining):
            remaining = available_remaining[remaining_index]
            indices = bins_by_remaining[remaining]
            bin_index = heapq.heappop(indices)
            if not indices:
                del bins_by_remaining[remaining]
                available_remaining.pop(remaining_index)
            target = bins[bin_index]
        else:
            target = _PackBin()
            bins.append(target)
            bin_index = len(bins) - 1
        target.samples.append(sample)
        target.token_count += sample.token_count
        add_available_bin(max_sequence_length - target.token_count, bin_index)

    packs: list[PackedSequence] = []
    for current in bins:
        packed_samples = tuple(current.samples)
        payload = _pack_payload(
            split=split,
            max_sequence_length=max_sequence_length,
            samples=packed_samples,
        )
        pack_sha256 = _content_hash(payload)
        packs.append(
            PackedSequence(
                pack_id=f"{split}-{pack_sha256[:16]}",
                pack_sha256=pack_sha256,
                split=split,
                max_sequence_length=max_sequence_length,
                sample_ids=tuple(sample.id for sample in packed_samples),
                sample_token_counts=tuple(sample.token_count for sample in packed_samples),
                input_ids=tuple(cast(list[int], payload["input_ids"])),
                labels=tuple(cast(list[int], payload["labels"])),
                position_ids=tuple(cast(list[int], payload["position_ids"])),
                segment_ids=tuple(cast(list[int], payload["segment_ids"])),
                token_count=current.token_count,
                supervised_token_count=sum(
                    sample.supervised_token_count for sample in packed_samples
                ),
            )
        )
    return tuple(sorted(packs, key=lambda pack: pack.pack_id))


def pack_tokenized_samples(
    samples: Iterable[TokenizedSample],
    *,
    config: M2PackingConfig,
) -> tuple[PackedSequence, ...]:
    """Pack each split independently and retain explicit intra-pack sample boundaries."""

    materialized = tuple(samples)
    ids = [sample.id for sample in materialized]
    if len(ids) != len(set(ids)):
        raise PackingError("tokenized sample IDs must be unique before packing")
    expected_maximum = config.packing.max_sequence_length
    mismatched = sorted(
        sample.id for sample in materialized if sample.max_sequence_length != expected_maximum
    )
    if mismatched:
        raise PackingError(
            f"sample maximum sequence length does not match packing config: {mismatched[0]}"
        )
    packs = [
        pack
        for split in _SPLITS
        for pack in _pack_split(
            tuple(sample for sample in materialized if sample.split == split),
            split=split,
            max_sequence_length=expected_maximum,
        )
    ]
    packed_ids = [sample_id for pack in packs for sample_id in pack.sample_ids]
    if sorted(packed_ids) != sorted(ids):
        raise PackingError("packing must include every selected sample exactly once")
    return tuple(packs)


def _source_lineage(
    source_manifests: tuple[DataImportManifest, ...],
) -> tuple[SourceDatasetLineage, SourceDatasetLineage]:
    if len(source_manifests) != 2:
        raise PackingError("final dataset requires exactly two source import manifests")
    ordered = tuple(sorted(source_manifests, key=lambda manifest: manifest.source.name))
    if [manifest.source.name for manifest in ordered] != ["commitpackft", "oasst1"]:
        raise PackingError("source manifests must contain commitpackft and oasst1 exactly once")
    lineage = tuple(
        SourceDatasetLineage(
            source=manifest.source.name,
            dataset_id=manifest.source.dataset_id,
            revision=manifest.source.revision,
            dataset_card_sha256=manifest.source.dataset_card_sha256,
            input_sha256=manifest.input_sha256,
            import_config_sha256=manifest.config_sha256,
        )
        for manifest in ordered
    )
    return cast(tuple[SourceDatasetLineage, SourceDatasetLineage], lineage)


def build_m2_dataset(
    tokenization: TokenizationBatch,
    *,
    tokenization_config: M2TokenizationConfig,
    packing_config: M2PackingConfig,
    processing_manifest: DataProcessingManifest,
    source_manifests: tuple[DataImportManifest, ...],
) -> DatasetBuild:
    """Balance, pack, and summarize one deterministic content-addressed M2 build."""

    if tokenization_config.max_sequence_length != packing_config.packing.max_sequence_length:
        raise PackingError("tokenization and packing maximum sequence lengths must match")
    if sum(manifest.accepted_samples for manifest in source_manifests) != (
        processing_manifest.input_samples
    ):
        raise PackingError("source accepted counts must equal processing input count")
    if len(tokenization.samples) + len(tokenization.rejected) != (
        processing_manifest.output_samples
    ):
        raise PackingError("tokenization outcomes must equal processing output count")
    tokenization_split_counts = {
        split: sum(sample.split == split for sample in tokenization.samples)
        + sum(record.split == split for record in tokenization.rejected)
        for split in _SPLITS
    }
    if tokenization_split_counts != processing_manifest.split_counts:
        raise PackingError("tokenization split outcomes must match processing split counts")
    tokenized_ids = [sample.id for sample in tokenization.samples]
    rejected_ids = [record.sample_id for record in tokenization.rejected]
    if len(tokenized_ids + rejected_ids) != len(set(tokenized_ids + rejected_ids)):
        raise PackingError("tokenization accepted and rejected sample IDs must be unique")
    for sample in tokenization.samples:
        if sample.max_sequence_length != tokenization_config.max_sequence_length:
            raise PackingError(
                f"sample {sample.id} maximum sequence length does not match Tokenizer config"
            )
        if sample.tokenizer_sha256 != tokenization_config.tokenizer.tokenizer_sha256:
            raise PackingError(f"sample {sample.id} tokenizer identity does not match config")
        if sample.template_sha256 != tokenization_config.template.template_sha256:
            raise PackingError(f"sample {sample.id} template identity does not match config")
    for record in tokenization.rejected:
        if record.max_sequence_length != tokenization_config.max_sequence_length:
            raise PackingError(
                f"rejection {record.sample_id} maximum sequence length does not match "
                "Tokenizer config"
            )

    selected, balance_rejected = _balance_train_samples(
        tokenization.samples,
        config=packing_config,
    )
    packs = pack_tokenized_samples(selected, config=packing_config)

    split_sample_counts = {
        split: sum(sample.split == split for sample in selected) for split in _SPLITS
    }
    split_pack_counts = {split: sum(pack.split == split for pack in packs) for split in _SPLITS}
    split_token_counts = {
        split: sum(pack.token_count for pack in packs if pack.split == split) for split in _SPLITS
    }
    split_supervised_counts = {
        split: sum(pack.supervised_token_count for pack in packs if pack.split == split)
        for split in _SPLITS
    }
    split_hashes = {
        split: _sequence_hash(pack.to_dict() for pack in packs if pack.split == split)
        for split in _SPLITS
    }
    source_token_counts = {
        source: sum(sample.token_count for sample in selected if sample.source == source)
        for source in ("commitpackft", "oasst1")
    }
    language_token_counts = {
        language: sum(sample.token_count for sample in selected if sample.language == language)
        for language in ("en", "zh")
    }
    license_sample_counts = Counter(sample.license for sample in selected)
    train_stratum_tokens = {
        stratum: sum(
            sample.token_count
            for sample in selected
            if sample.split == "train" and _stratum(sample) == stratum
        )
        for stratum in _STRATA
    }
    train_tokens = sum(train_stratum_tokens.values())
    train_stratum_basis_points = {
        stratum: count * 10_000 // train_tokens if train_tokens else 0
        for stratum, count in train_stratum_tokens.items()
    }
    rejection_counts: Counter[str] = Counter(record.reason for record in tokenization.rejected)
    rejection_counts.update(record.reason for record in balance_rejected)
    total_tokens = sum(pack.token_count for pack in packs)
    total_supervised = sum(pack.supervised_token_count for pack in packs)
    capacity = len(packs) * packing_config.packing.max_sequence_length

    content_sha256 = _dataset_content_hash(
        balance_rejected=balance_rejected,
        packing_config=packing_config,
        packs=packs,
        processing_manifest=processing_manifest,
        source_manifests=source_manifests,
        tokenization_config=tokenization_config,
        tokenization_rejected=tokenization.rejected,
    )
    manifest = M2DatasetManifest(
        dataset_name=packing_config.dataset_name,
        dataset_version=f"m2-sft-v1-{content_sha256[:8]}",
        content_sha256=content_sha256,
        source_lineage=_source_lineage(source_manifests),
        processing_config_sha256=processing_manifest.config_sha256,
        processing_input_sha256=processing_manifest.input_sha256,
        processing_output_sha256=processing_manifest.output_sha256,
        tokenizer=tokenization_config.tokenizer,
        template=tokenization_config.template,
        packing_config_sha256=_content_hash(packing_config.to_dict()),
        max_sequence_length=packing_config.packing.max_sequence_length,
        input_processed_samples=processing_manifest.output_samples,
        tokenized_samples=len(tokenization.samples),
        tokenization_rejections=len(tokenization.rejected),
        balanced_samples=len(selected),
        balance_rejections=len(balance_rejected),
        packed_sequences=len(packs),
        total_tokens=total_tokens,
        total_supervised_tokens=total_supervised,
        packing_capacity_tokens=capacity,
        packing_efficiency_basis_points=(total_tokens * 10_000 // capacity if capacity else 0),
        split_sample_counts=cast(dict[str, int], split_sample_counts),
        split_pack_counts=cast(dict[str, int], split_pack_counts),
        split_token_counts=cast(dict[str, int], split_token_counts),
        split_supervised_token_counts=cast(dict[str, int], split_supervised_counts),
        split_sha256s=cast(dict[str, str], split_hashes),
        source_token_counts=source_token_counts,
        language_token_counts=language_token_counts,
        license_sample_counts=dict(sorted(license_sample_counts.items())),
        train_stratum_token_counts=train_stratum_tokens,
        train_stratum_basis_points=train_stratum_basis_points,
        rejection_counts=dict(sorted(rejection_counts.items())),
    )
    return DatasetBuild(
        manifest=manifest,
        packs=packs,
        selected_samples=selected,
        tokenization_rejected=tokenization.rejected,
        balance_rejected=balance_rejected,
    )
