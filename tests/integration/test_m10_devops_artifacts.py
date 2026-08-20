from __future__ import annotations

import json
from pathlib import Path

from tinyllm.data.m10_devops import (
    build_devops_samples,
    build_manifest,
    scan_authored_duplicates,
)
from tinyllm.data.m10_devops_schema import M10DevOpsBuildReport


def test_public_m10_devops_build_evidence_matches_deterministic_source() -> None:
    report_path = Path("reports/m10/raw/m10_devops_training_build.json")
    report = M10DevOpsBuildReport.model_validate_json(report_path.read_bytes())
    samples = build_devops_samples()
    manifest = build_manifest(samples)
    duplicate = scan_authored_duplicates(samples)

    assert report.dataset_version == manifest.dataset_version
    assert report.items_sha256 == manifest.items_sha256
    assert report.content_sha256 == manifest.content_sha256
    assert report.category_counts == manifest.category_counts
    assert report.language_counts == manifest.language_counts
    assert report.duplicate_report_sha256 == duplicate.report_sha256
    assert report.duplicate_status == "pass"
    assert report.contamination_status == "pass"
    assert report.review_status == "pending"
    assert report.training_permitted is False

    serialized = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)
    assert "/home/" not in serialized
    assert "/data/" not in serialized
    assert "m9-devops-agent-release" not in serialized
