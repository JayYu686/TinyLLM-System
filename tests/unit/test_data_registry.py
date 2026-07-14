from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from tests.unit.test_data_packing import (
    balanced_samples,
    build,
    processing_manifest,
    source_manifest,
)
from tinyllm.data import (
    M2_ACQUISITION_MANIFEST,
    DataProcessingManifest,
    DatasetArtifactFile,
    DatasetBuild,
    DatasetLineage,
    DatasetRegistrationResult,
    DatasetRegistryError,
    DatasetRegistryErrorCode,
    load_m2_packing_config,
    load_m2_tokenization_config,
    open_registered_dataset,
    register_dataset,
    summarize_registered_dataset,
)

PACKING_CONFIG = Path("configs/data/m2_packing.yaml")
TOKENIZATION_CONFIG = Path("configs/data/m2_tokenization.yaml")
REGISTERED_AT = datetime(2026, 7, 14, tzinfo=UTC)


def lineage() -> DatasetLineage:
    samples = balanced_samples()
    return DatasetLineage(
        acquisition_manifest=M2_ACQUISITION_MANIFEST,
        source_manifests=(
            source_manifest("oasst1", accepted=6),
            source_manifest("commitpackft", accepted=4),
        ),
        processing_manifest=processing_manifest(samples),
        tokenization_config=load_m2_tokenization_config(TOKENIZATION_CONFIG),
        packing_config=load_m2_packing_config(PACKING_CONFIG),
        oasst1_rejected=(),
        commitpackft_rejected=(),
        processing_rejected=(),
    )


def register(tmp_path: Path) -> tuple[DatasetBuild, DatasetRegistrationResult]:
    dataset_build = build(balanced_samples())
    result = register_dataset(
        dataset_build,
        artifact_root=tmp_path,
        lineage=lineage(),
        git_commit="a" * 40,
        git_dirty=False,
        registered_at=REGISTERED_AT,
    )
    return dataset_build, result


def test_registry_atomically_round_trips_and_is_idempotent(tmp_path: Path) -> None:
    dataset_build, result = register(tmp_path)

    assert result.created is True
    assert result.dataset.root.is_dir()
    assert (result.dataset.root / "COMMITTED").is_file()
    assert tuple(result.dataset.iter_packs()) == dataset_build.packs
    assert not list(result.dataset.root.parent.glob(".*.tmp-*"))

    reopened = open_registered_dataset(
        artifact_root=tmp_path,
        dataset_version=dataset_build.manifest.dataset_version,
    )
    assert reopened.manifest == dataset_build.manifest
    second = register_dataset(
        dataset_build,
        artifact_root=tmp_path,
        lineage=lineage(),
        git_commit="b" * 40,
        git_dirty=True,
        registered_at=datetime(2027, 1, 1, tzinfo=UTC),
    )
    assert second.created is False
    assert second.dataset.registration.git_commit == "a" * 40


def test_registry_summary_is_path_free_and_factual(tmp_path: Path) -> None:
    dataset_build, result = register(tmp_path)

    summary = summarize_registered_dataset(
        result.dataset,
        operation="prepare",
        created=True,
    )

    assert summary.dataset_version == dataset_build.manifest.dataset_version
    assert summary.source_rows == {"commitpackft": 4, "oasst1": 6}
    assert summary.imported_samples == {"commitpackft": 4, "oasst1": 6}
    assert summary.total_tokens == 100
    assert summary.verified is True
    assert summary.registered_files == len(result.dataset.registration.files) + 3
    assert summary.registered_bytes > 0
    assert str(tmp_path) not in summary.model_dump_json()


@pytest.mark.parametrize("mutation", ["missing-marker", "unknown-file", "bad-array", "symlink"])
def test_registry_refuses_incomplete_corrupt_or_unknown_content(
    tmp_path: Path,
    mutation: str,
) -> None:
    dataset_build, result = register(tmp_path)
    root = result.dataset.root
    if mutation == "missing-marker":
        (root / "COMMITTED").unlink()
        expected = DatasetRegistryErrorCode.INCOMPLETE
    elif mutation == "unknown-file":
        (root / "rogue.txt").write_text("unexpected", encoding="utf-8")
        expected = DatasetRegistryErrorCode.CORRUPT
    elif mutation == "bad-array":
        array = next((root / "shards").rglob("input_ids.npy"))
        with array.open("ab") as handle:
            handle.write(b"corrupt")
        expected = DatasetRegistryErrorCode.CORRUPT
    else:
        (root / "rogue-link").symlink_to(root / "manifest.json")
        expected = DatasetRegistryErrorCode.CORRUPT

    with pytest.raises(DatasetRegistryError) as error:
        open_registered_dataset(
            artifact_root=tmp_path,
            dataset_version=dataset_build.manifest.dataset_version,
        )
    assert error.value.code == expected


def test_registry_refuses_partial_existing_version_without_overwrite(tmp_path: Path) -> None:
    dataset_build = build(balanced_samples())
    destination = tmp_path / "datasets" / "m2-sft" / dataset_build.manifest.dataset_version
    destination.mkdir(parents=True)
    (destination / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DatasetRegistryError) as error:
        register_dataset(
            dataset_build,
            artifact_root=tmp_path,
            lineage=lineage(),
            git_commit="a" * 40,
            git_dirty=False,
            registered_at=REGISTERED_AT,
        )

    assert error.value.code == DatasetRegistryErrorCode.INCOMPLETE
    assert (destination / "manifest.json").read_text(encoding="utf-8") == "{}"


def test_registry_cleans_temporary_directory_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_build = build(balanced_samples())

    def fail_write(*_args: object, **_kwargs: object) -> list[DatasetArtifactFile]:
        raise OSError("simulated disk failure")

    monkeypatch.setattr("tinyllm.data.registry._write_shards", fail_write)
    with pytest.raises(DatasetRegistryError) as error:
        register_dataset(
            dataset_build,
            artifact_root=tmp_path,
            lineage=lineage(),
            git_commit="a" * 40,
            git_dirty=False,
            registered_at=REGISTERED_AT,
        )

    assert error.value.code == DatasetRegistryErrorCode.WRITE_FAILED
    parent = tmp_path / "datasets" / "m2-sft"
    assert not list(parent.glob(".*.tmp-*"))
    assert not (parent / dataset_build.manifest.dataset_version).exists()


def test_registry_refuses_lineage_drift_and_invalid_shard_limit(tmp_path: Path) -> None:
    dataset_build = build(balanced_samples())
    valid_lineage = lineage()
    drifted_processing = DataProcessingManifest.model_validate(
        {**valid_lineage.processing_manifest.model_dump(), "config_sha256": "f" * 64}
    )
    drifted_lineage = DatasetLineage(
        acquisition_manifest=valid_lineage.acquisition_manifest,
        source_manifests=valid_lineage.source_manifests,
        processing_manifest=drifted_processing,
        tokenization_config=valid_lineage.tokenization_config,
        packing_config=valid_lineage.packing_config,
        oasst1_rejected=(),
        commitpackft_rejected=(),
        processing_rejected=(),
    )

    with pytest.raises(DatasetRegistryError, match="processing lineage"):
        register_dataset(
            dataset_build,
            artifact_root=tmp_path,
            lineage=drifted_lineage,
            git_commit="a" * 40,
            git_dirty=False,
            registered_at=REGISTERED_AT,
        )
    with pytest.raises(DatasetRegistryError, match="shard Token limit"):
        register_dataset(
            dataset_build,
            artifact_root=tmp_path,
            lineage=valid_lineage,
            git_commit="a" * 40,
            git_dirty=False,
            shard_token_limit=10,
            registered_at=REGISTERED_AT,
        )


def test_dataset_artifact_file_schema_refuses_traversal_and_reserved_paths() -> None:
    with pytest.raises(ValidationError, match="safe and relative"):
        DatasetArtifactFile(
            path=PurePosixPath("../escape"),
            role="lineage",
            size_bytes=1,
            sha256="a" * 64,
        )
    with pytest.raises(ValidationError, match="reserved"):
        DatasetArtifactFile(
            path=PurePosixPath("manifest.json"),
            role="lineage",
            size_bytes=1,
            sha256="a" * 64,
        )
