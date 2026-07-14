#!/usr/bin/env python3
"""Reproduce the public synthetic M2.3c immutable Registry smoke."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_m2_packing_smoke import (
    _processing_manifest,
    _source_manifest,
    _synthetic_samples,
)
from tinyllm.data import (
    M2_ACQUISITION_MANIFEST,
    DatasetLineage,
    DatasetRegistryError,
    TokenizationBatch,
    build_m2_dataset,
    load_m2_packing_config,
    load_m2_tokenization_config,
    open_registered_dataset,
    register_dataset,
    summarize_registered_dataset,
)


def run_smoke(project_root: Path) -> dict[str, object]:
    """Register, reopen, rebuild, and corrupt a synthetic version without leaking paths."""

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
    build = build_m2_dataset(
        TokenizationBatch(samples=samples, rejected=()),
        tokenization_config=tokenization_config,
        packing_config=packing_config,
        processing_manifest=processing_manifest,
        source_manifests=source_manifests,
    )
    lineage = DatasetLineage(
        acquisition_manifest=M2_ACQUISITION_MANIFEST,
        source_manifests=source_manifests,
        processing_manifest=processing_manifest,
        tokenization_config=tokenization_config,
        packing_config=packing_config,
        oasst1_rejected=(),
        commitpackft_rejected=(),
        processing_rejected=(),
    )
    with TemporaryDirectory(prefix="tinyllm-m2-registry-smoke-") as temporary:
        artifact_root = Path(temporary)
        first = register_dataset(
            build,
            artifact_root=artifact_root,
            lineage=lineage,
            git_commit="a" * 40,
            git_dirty=False,
            shard_token_limit=1024,
            registered_at=datetime(2026, 7, 14, tzinfo=UTC),
        )
        reopened = open_registered_dataset(
            artifact_root=artifact_root,
            dataset_version=build.manifest.dataset_version,
        )
        reconstructed = tuple(reopened.iter_packs())
        second = register_dataset(
            build,
            artifact_root=artifact_root,
            lineage=lineage,
            git_commit="b" * 40,
            git_dirty=True,
            shard_token_limit=1024,
            registered_at=datetime(2027, 1, 1, tzinfo=UTC),
        )
        summary = summarize_registered_dataset(
            first.dataset,
            operation="prepare",
            created=first.created,
        )
        array_path = next((reopened.root / "shards").rglob("input_ids.npy"))
        with array_path.open("ab") as handle:
            handle.write(b"corruption-smoke")
        try:
            open_registered_dataset(
                artifact_root=artifact_root,
                dataset_version=build.manifest.dataset_version,
            )
        except DatasetRegistryError as exc:
            corruption_code = str(exc.code)
        else:
            raise RuntimeError("corrupted Registry unexpectedly passed integrity validation")

    return {
        "status": "pass",
        "scope": "public-synthetic-token-arrays",
        "dataset_version": summary.dataset_version,
        "content_sha256": summary.content_sha256,
        "first_registration_created": first.created,
        "second_registration_created": second.created,
        "reconstructed_packs_identical": reconstructed == build.packs,
        "registered_files": summary.registered_files,
        "storage_format": summary.storage_format,
        "packed_sequences": summary.packed_sequences,
        "total_tokens": summary.total_tokens,
        "corruption_refusal_code": corruption_code,
    }


def main() -> int:
    """Run the smoke and print stable JSON to standard output."""

    project_root = Path(__file__).resolve().parents[1]
    print(json.dumps(run_smoke(project_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
