from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import tinyllm.data.pipeline as pipeline_module
from tests.unit.test_data_packing import (
    balanced_samples,
    processing_manifest,
    source_manifest,
)
from tests.unit.test_data_registry import lineage, register
from tinyllm.data import (
    AcquiredM2Artifacts,
    ImportResult,
    ProcessingResult,
    TokenizationBatch,
    load_m2_packing_config,
    load_m2_processing_config,
    load_m2_tokenization_config,
    prepare_m2_dataset,
    summarize_registered_dataset,
)


def test_prepare_pipeline_wires_every_fixed_stage_and_returns_registry_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = balanced_samples()
    dataset_build, registration = register(tmp_path)
    expected = summarize_registered_dataset(
        registration.dataset,
        operation="prepare",
        created=True,
    )
    acquired = AcquiredM2Artifacts(
        oasst1_jsonl=tmp_path / "oasst.jsonl.gz",
        commitpackft_jsonl=tmp_path / "commit.jsonl",
        tokenizer_json=tmp_path / "tokenizer.json",
        tokenizer_config_json=tmp_path / "tokenizer_config.json",
    )
    oasst_manifest = source_manifest("oasst1", accepted=6)
    commit_manifest = source_manifest("commitpackft", accepted=4)
    processing = ProcessingResult(
        manifest=processing_manifest(samples),
        samples=(),
        rejected=(),
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(pipeline_module, "acquire_m2_artifacts", lambda **_kwargs: acquired)
    monkeypatch.setattr(
        pipeline_module,
        "load_m2_processing_config",
        lambda _path: load_m2_processing_config(Path("configs/data/m2_processing.yaml")),
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_m2_tokenization_config",
        lambda _path: load_m2_tokenization_config(Path("configs/data/m2_tokenization.yaml")),
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_m2_packing_config",
        lambda _path: load_m2_packing_config(Path("configs/data/m2_packing.yaml")),
    )
    monkeypatch.setattr(pipeline_module, "iter_jsonl_records", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        pipeline_module,
        "import_oasst1",
        lambda _rows: ImportResult(manifest=oasst_manifest, samples=(), rejected=()),
    )
    monkeypatch.setattr(
        pipeline_module,
        "import_commitpackft",
        lambda _rows: ImportResult(manifest=commit_manifest, samples=(), rejected=()),
    )
    monkeypatch.setattr(
        pipeline_module,
        "process_imported_samples",
        lambda *_args, **_kwargs: processing,
    )
    monkeypatch.setattr(
        "tinyllm.data.pipeline.TokenizersBackend.from_files",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "tokenize_processed_samples",
        lambda *_args, **_kwargs: TokenizationBatch(samples=samples, rejected=()),
    )
    monkeypatch.setattr(
        pipeline_module,
        "build_m2_dataset",
        lambda *_args, **_kwargs: dataset_build,
    )
    monkeypatch.setattr(pipeline_module, "read_git_identity", lambda _root: ("a" * 40, False))

    def fake_register(*_args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return registration

    monkeypatch.setattr(pipeline_module, "register_dataset", fake_register)

    summary = prepare_m2_dataset(
        project_root=Path.cwd(),
        artifact_root=tmp_path,
        processing_config_path=Path("unused-processing.yaml"),
        tokenization_config_path=Path("unused-tokenization.yaml"),
        packing_config_path=Path("unused-packing.yaml"),
        offline=True,
    )

    assert summary == expected
    assert captured["artifact_root"] == tmp_path
    assert captured["git_commit"] == "a" * 40
    assert captured["git_dirty"] is False
    assert captured["lineage"] == lineage()
