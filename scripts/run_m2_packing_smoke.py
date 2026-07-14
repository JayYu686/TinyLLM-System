#!/usr/bin/env python3
"""Reproduce the public synthetic M2.3b balancing and packing smoke."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from tinyllm.data import (
    COMMITPACKFT_SOURCE,
    OASST1_SOURCE,
    DataImportManifest,
    DataProcessingManifest,
    TokenizationBatch,
    TokenizedSample,
    build_m2_dataset,
    load_m2_packing_config,
    load_m2_tokenization_config,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sequence_hash(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = _canonical_json(value)
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def _sample(
    source: Literal["oasst1", "commitpackft"],
    suffix: str,
    *,
    language: Literal["en", "zh"],
    split: Literal["train", "validation", "test"],
    token_count: int,
    token_offset: int,
    tokenizer_sha256: str,
    template_sha256: str,
) -> TokenizedSample:
    sample_id = f"{source}:{suffix}"
    input_ids = tuple(range(token_offset, token_offset + token_count))
    supervised_tokens = 10
    return TokenizedSample(
        id=sample_id,
        source=source,
        split=split,
        component_id=_hash({"component": sample_id}),
        group_keys=(f"{source}:group-{suffix}",),
        origin_sample_ids=(sample_id,),
        origin_record_sha256s=(_hash({"record": sample_id}),),
        language=language,
        license="apache-2.0" if source == "oasst1" else "mit",
        content_sha256=_hash({"content": sample_id}),
        rendered_sha256=_hash({"rendered": sample_id}),
        tokenizer_sha256=tokenizer_sha256,
        template_sha256=template_sha256,
        max_sequence_length=1024,
        input_ids=input_ids,
        labels=(-100,) * (token_count - supervised_tokens) + input_ids[-supervised_tokens:],
        token_count=token_count,
        supervised_token_count=supervised_tokens,
    )


def _synthetic_samples(project_root: Path) -> tuple[TokenizedSample, ...]:
    config = load_m2_tokenization_config(project_root / "configs/data/m2_tokenization.yaml")
    samples: list[TokenizedSample] = []
    strata: tuple[tuple[Literal["oasst1", "commitpackft"], Literal["en", "zh"]], ...] = (
        ("oasst1", "zh"),
        ("oasst1", "en"),
        ("commitpackft", "en"),
    )
    for source, language in strata:
        for index in range(5):
            samples.append(
                _sample(
                    source,
                    f"train-{language}-{index}",
                    language=language,
                    split="train",
                    token_count=200,
                    token_offset=1000 + len(samples) * 250,
                    tokenizer_sha256=config.tokenizer.tokenizer_sha256,
                    template_sha256=config.template.template_sha256,
                )
            )
    samples.extend(
        (
            _sample(
                "oasst1",
                "validation-en",
                language="en",
                split="validation",
                token_count=150,
                token_offset=5000,
                tokenizer_sha256=config.tokenizer.tokenizer_sha256,
                template_sha256=config.template.template_sha256,
            ),
            _sample(
                "commitpackft",
                "test-en",
                language="en",
                split="test",
                token_count=150,
                token_offset=5200,
                tokenizer_sha256=config.tokenizer.tokenizer_sha256,
                template_sha256=config.template.template_sha256,
            ),
        )
    )
    return tuple(samples)


def _source_manifest(
    source: Literal["oasst1", "commitpackft"],
    samples: tuple[TokenizedSample, ...],
) -> DataImportManifest:
    source_samples = tuple(sample for sample in samples if sample.source == source)
    descriptor = OASST1_SOURCE if source == "oasst1" else COMMITPACKFT_SOURCE
    license_name = "apache-2.0" if source == "oasst1" else "mit"
    return DataImportManifest(
        source=descriptor,
        input_sha256=_sequence_hash(sample.id for sample in source_samples),
        config_sha256=_hash({"fixture": "m2-packing-smoke", "source": source}),
        source_rows=len(source_samples),
        candidate_samples=len(source_samples),
        accepted_samples=len(source_samples),
        rejected_samples=0,
        rejection_counts={},
        license_counts={license_name: len(source_samples)},
    )


def _processing_manifest(samples: tuple[TokenizedSample, ...]) -> DataProcessingManifest:
    ordered = tuple(sorted(samples, key=lambda sample: sample.id))
    split_counts = {
        split: sum(sample.split == split for sample in ordered)
        for split in ("test", "train", "validation")
    }
    return DataProcessingManifest(
        input_sha256=_sequence_hash(sample.id for sample in ordered),
        config_sha256=_hash({"fixture": "m2-packing-smoke-processing"}),
        output_sha256=_sequence_hash(sample.to_dict() for sample in ordered),
        input_samples=len(ordered),
        normalized_samples=len(ordered),
        output_samples=len(ordered),
        rejected_samples=0,
        normalization_rejections=0,
        exact_duplicates=0,
        component_count=len(ordered),
        rejection_counts={},
        split_counts=split_counts,
        split_sha256s={
            split: _sequence_hash(sample.to_dict() for sample in ordered if sample.split == split)
            for split in ("test", "train", "validation")
        },
    )


def run_smoke(project_root: Path) -> dict[str, object]:
    """Return stable public evidence for synthetic balancing and packing."""

    samples = _synthetic_samples(project_root)
    tokenization_config = load_m2_tokenization_config(
        project_root / "configs/data/m2_tokenization.yaml"
    )
    packing_config = load_m2_packing_config(project_root / "configs/data/m2_packing.yaml")
    processing_manifest = _processing_manifest(samples)
    source_manifests = (
        _source_manifest("oasst1", samples),
        _source_manifest("commitpackft", samples),
    )
    result = build_m2_dataset(
        TokenizationBatch(samples=samples, rejected=()),
        tokenization_config=tokenization_config,
        packing_config=packing_config,
        processing_manifest=processing_manifest,
        source_manifests=source_manifests,
    )
    rebuilt = build_m2_dataset(
        TokenizationBatch(samples=tuple(reversed(samples)), rejected=()),
        tokenization_config=tokenization_config,
        packing_config=packing_config,
        processing_manifest=processing_manifest,
        source_manifests=source_manifests,
    )
    if result != rebuilt:
        raise RuntimeError("M2 packing rebuild changed after input order reversal")

    selected_ids = [sample_id for pack in result.packs for sample_id in pack.sample_ids]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("a selected sample appeared in more than one pack")
    pack_summaries = [
        {
            "pack_id": pack.pack_id,
            "pack_sha256": pack.pack_sha256,
            "split": pack.split,
            "sample_ids": list(pack.sample_ids),
            "sample_token_counts": list(pack.sample_token_counts),
            "token_count": pack.token_count,
            "supervised_token_count": pack.supervised_token_count,
            "position_resets_verified": all(
                pack.position_ids[
                    sum(pack.sample_token_counts[:index]) : sum(
                        pack.sample_token_counts[: index + 1]
                    )
                ]
                == tuple(range(sample_tokens))
                for index, sample_tokens in enumerate(pack.sample_token_counts)
            ),
            "segment_boundaries_verified": all(
                pack.segment_ids[
                    sum(pack.sample_token_counts[:index]) : sum(
                        pack.sample_token_counts[: index + 1]
                    )
                ]
                == (index,) * sample_tokens
                for index, sample_tokens in enumerate(pack.sample_token_counts)
            ),
        }
        for pack in result.packs
    ]
    return {
        "status": "pass",
        "scope": "public-synthetic-token-arrays",
        "input_samples": len(samples),
        "input_sha256": _sequence_hash(sample.to_dict() for sample in samples),
        "rebuild_after_input_reversal": "identical",
        "manifest": result.manifest.to_dict(),
        "balance_rejected": [record.to_dict() for record in result.balance_rejected],
        "packs": pack_summaries,
    }


def main() -> int:
    """Run the smoke and print stable JSON to standard output."""

    project_root = Path(__file__).resolve().parents[1]
    print(json.dumps(run_smoke(project_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
