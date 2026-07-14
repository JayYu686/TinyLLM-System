from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from tinyllm.data import (
    DataImportManifest,
    DataProcessingManifest,
    import_commitpackft,
    import_oasst1,
    load_m2_processing_config,
    process_imported_samples,
)


def _fixture(name: str) -> list[dict[str, Any]]:
    path = Path("tests/fixtures/data") / name
    return cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.integration
def test_pinned_import_contract_smoke_is_deterministic_and_license_filtered() -> None:
    """Cross the fixture, importer, lineage, and public-schema boundaries."""

    oasst = import_oasst1(_fixture("oasst1.synthetic.json"))
    commitpack = import_commitpackft(_fixture("commitpackft.synthetic.json"))

    assert oasst.manifest.accepted_samples == 1
    assert commitpack.manifest.accepted_samples == 1
    assert commitpack.manifest.rejection_counts == {
        "not_python": 1,
        "unsupported_license": 1,
    }
    assert all(sample.metadata.license != "unknown" for sample in commitpack.samples)
    assert DataImportManifest.model_validate_json(oasst.manifest.model_dump_json()) == (
        oasst.manifest
    )
    assert DataImportManifest.model_validate_json(commitpack.manifest.model_dump_json()) == (
        commitpack.manifest
    )
    assert import_oasst1(_fixture("oasst1.synthetic.json")).manifest == oasst.manifest
    assert import_commitpackft(_fixture("commitpackft.synthetic.json")).manifest == (
        commitpack.manifest
    )


@pytest.mark.integration
def test_import_to_grouped_split_smoke_has_reproducible_content_identity() -> None:
    """Cross both importers into M2.2 without treating the result as trainable data."""

    imported = (
        *import_oasst1(_fixture("oasst1.synthetic.json")).samples,
        *import_commitpackft(_fixture("commitpackft.synthetic.json")).samples,
    )
    config = load_m2_processing_config(Path("configs/data/m2_processing.yaml"))

    first = process_imported_samples(imported, config=config)
    second = process_imported_samples(reversed(imported), config=config)

    assert first == second
    assert first.manifest.input_samples == 2
    assert first.manifest.output_samples == 2
    assert first.manifest.exact_duplicates == 0
    assert first.manifest.component_count == 2
    assert DataProcessingManifest.model_validate_json(first.manifest.model_dump_json()) == (
        first.manifest
    )
    assert all(sample.origin_sample_ids == (sample.id,) for sample in first.samples)

    committed = json.loads(
        Path("reports/m2/raw/deterministic_pipeline_smoke.json").read_text(encoding="utf-8")
    )
    assert (
        committed["oasst_import_manifest"]
        == import_oasst1(_fixture("oasst1.synthetic.json")).manifest.to_dict()
    )
    assert (
        committed["commitpackft_import_manifest"]
        == import_commitpackft(_fixture("commitpackft.synthetic.json")).manifest.to_dict()
    )
    assert committed["processing_manifest"] == first.manifest.to_dict()
    assert committed["processed_samples"] == [
        {
            "id": sample.id,
            "source": sample.source,
            "split": sample.split,
            "component_id": sample.component_id,
            "content_sha256": sample.content_sha256,
            "group_keys": list(sample.group_keys),
        }
        for sample in first.samples
    ]
