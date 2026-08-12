from __future__ import annotations

import json
from pathlib import Path

from tinyllm.lineage import RunIndexListResult, RunIndexRebuildResult

EVIDENCE = Path("reports/m6/raw/m6_v7_acceptance.json")
REPORT = Path("reports/m6/m6_acceptance.md")


def test_m6_acceptance_evidence_records_real_candidate_gate() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert evidence["schema_version"] == "1.0"
    assert evidence["status"] == "Candidate"
    assert evidence["production_eligible"] is False
    assert evidence["checks"] == {"passed": 11, "total": 11}
    assert evidence["human_review"]["approved_judgments"] == 160
    assert evidence["human_review"]["status"] == "complete"
    thinking = evidence["modes"]["thinking"]
    nonthinking = evidence["modes"]["nonthinking"]
    assert thinking["candidate_score_basis_points"] - thinking["base_score_basis_points"] == 734
    assert (
        nonthinking["candidate_score_basis_points"] - nonthinking["base_score_basis_points"] == 1834
    )
    assert thinking["bootstrap_95_lower_basis_points"] > 0
    assert nonthinking["bootstrap_95_lower_basis_points"] > 0
    assert (
        evidence["general"]["candidate_basis_points"] - evidence["general"]["base_basis_points"]
        == 268
    )
    assert evidence["run_index"]["indexed_runs"] == 57


def test_m6_chinese_report_binds_public_identity_and_boundary() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    report = REPORT.read_text(encoding="utf-8")

    for value in (
        evidence["model_version"],
        evidence["comparison_sha256"],
        evidence["suite_content_sha256"],
        evidence["candidate"]["model_artifact_sha256"],
    ):
        assert value in report
    assert "M6 状态：`COMPLETE`" in report
    assert "production_eligible=false" in report


def test_run_index_result_schemas_reject_inconsistent_counts() -> None:
    try:
        RunIndexRebuildResult(
            index_sha256="a" * 64,
            source_tree_sha256="b" * 64,
            source_manifests=2,
            indexed_runs=1,
        )
    except ValueError as exc:
        assert "every source manifest" in str(exc)
    else:
        raise AssertionError("incomplete index projection was accepted")

    try:
        RunIndexListResult(limit=1, returned_runs=1, runs=())
    except ValueError as exc:
        assert "returned_runs" in str(exc)
    else:
        raise AssertionError("inconsistent list count was accepted")
