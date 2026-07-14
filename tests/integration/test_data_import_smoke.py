from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from tinyllm.data import DataImportManifest, import_commitpackft, import_oasst1


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
