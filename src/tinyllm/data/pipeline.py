"""End-to-end fixed M2 preparation pipeline from pinned artifacts to Registry."""

from __future__ import annotations

import gc
from pathlib import Path

from tinyllm.data.acquisition import (
    M2_ACQUISITION_MANIFEST,
    acquire_m2_artifacts,
    iter_jsonl_records,
)
from tinyllm.data.importers import import_commitpackft, import_oasst1
from tinyllm.data.packing import build_m2_dataset, load_m2_packing_config
from tinyllm.data.processing import load_m2_processing_config, process_imported_samples
from tinyllm.data.registry import (
    DatasetLineage,
    register_dataset,
    summarize_registered_dataset,
)
from tinyllm.data.registry_schema import RegisteredDatasetSummary
from tinyllm.data.tokenization import (
    TokenizersBackend,
    load_m2_tokenization_config,
    tokenize_processed_samples,
)
from tinyllm.lineage.git import read_git_identity


def prepare_m2_dataset(
    *,
    project_root: Path,
    artifact_root: Path,
    processing_config_path: Path,
    tokenization_config_path: Path,
    packing_config_path: Path,
    offline: bool = False,
) -> RegisteredDatasetSummary:
    """Execute every fixed M2 stage and atomically register the verified result."""

    artifacts = acquire_m2_artifacts(cache_root=artifact_root / "cache", offline=offline)
    processing_config = load_m2_processing_config(processing_config_path)
    tokenization_config = load_m2_tokenization_config(tokenization_config_path)
    packing_config = load_m2_packing_config(packing_config_path)

    oasst1 = import_oasst1(iter_jsonl_records(artifacts.oasst1_jsonl, compression="gzip"))
    commitpackft = import_commitpackft(
        iter_jsonl_records(artifacts.commitpackft_jsonl, compression="none")
    )
    source_manifests = (oasst1.manifest, commitpackft.manifest)
    oasst1_rejected = oasst1.rejected
    commitpackft_rejected = commitpackft.rejected
    processing = process_imported_samples(
        (*oasst1.samples, *commitpackft.samples),
        config=processing_config,
    )
    del oasst1, commitpackft
    gc.collect()

    backend = TokenizersBackend.from_files(
        artifacts.tokenizer_json,
        artifacts.tokenizer_config_json,
        tokenization_config.tokenizer,
    )
    tokenization = tokenize_processed_samples(
        processing.samples,
        backend=backend,
        config=tokenization_config,
    )
    processing_manifest = processing.manifest
    processing_rejected = processing.rejected
    del processing, backend
    gc.collect()

    build = build_m2_dataset(
        tokenization,
        tokenization_config=tokenization_config,
        packing_config=packing_config,
        processing_manifest=processing_manifest,
        source_manifests=source_manifests,
    )
    del tokenization
    gc.collect()

    git_commit, git_dirty = read_git_identity(project_root)
    registration = register_dataset(
        build,
        artifact_root=artifact_root,
        lineage=DatasetLineage(
            acquisition_manifest=M2_ACQUISITION_MANIFEST,
            source_manifests=source_manifests,
            processing_manifest=processing_manifest,
            tokenization_config=tokenization_config,
            packing_config=packing_config,
            oasst1_rejected=oasst1_rejected,
            commitpackft_rejected=commitpackft_rejected,
            processing_rejected=processing_rejected,
        ),
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    return summarize_registered_dataset(
        registration.dataset,
        operation="prepare",
        created=registration.created,
    )
