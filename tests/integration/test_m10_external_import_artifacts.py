from __future__ import annotations

import json
from pathlib import Path

from tinyllm.data.m10_canonical_schema import M10ExternalImportReport


def test_public_m10_external_import_report_matches_formal_private_build() -> None:
    path = Path("reports/m10/raw/m10_external_canonical_import.json")
    report = M10ExternalImportReport.model_validate_json(path.read_bytes())

    assert report.status == "pass"
    assert report.total_source_rows == 13_193
    assert report.total_accepted_rows == 12_592
    assert report.total_rejected_rows == 601
    toolace, hermes = report.sources
    assert toolace.import_version == "m10-toolace-canonical-v1-5ff7e195"
    assert toolace.accepted_rows == 10_770
    assert toolace.rejection_counts == {
        "invalid_row_shape": 5,
        "invalid_tool_schema": 524,
        "malformed_tool_call": 1,
    }
    assert toolace.language_counts == {"en": 10_763, "zh": 7}
    assert hermes.import_version == "m10-hermes-canonical-v1-fb8b61ba"
    assert hermes.accepted_rows == 1_822
    assert hermes.rejection_counts == {
        "invalid_role_path": 10,
        "invalid_tool_schema": 61,
    }
    assert hermes.language_counts == {"en": 1_822}

    serialized = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)
    assert "/data/" not in serialized
    assert "/home/" not in serialized
    assert "conversations" not in serialized
    assert "tool_response" not in serialized
