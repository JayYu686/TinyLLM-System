from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tinyllm.data.m10_devops import (
    build_devops_samples,
    build_manifest,
    scan_authored_duplicates,
)
from tinyllm.data.m10_devops_review import render_json
from tinyllm.data.m10_devops_schema import (
    M10DevOpsBuildReport,
    M10DevOpsContentReviewResult,
)


def test_public_m10_devops_build_evidence_matches_deterministic_source() -> None:
    report_path = Path("reports/m10/raw/m10_devops_training_build.json")
    report = M10DevOpsBuildReport.model_validate_json(report_path.read_bytes())
    samples = build_devops_samples()
    manifest = build_manifest(samples, review_status="approved")
    duplicate = scan_authored_duplicates(samples)

    assert report.dataset_version == manifest.dataset_version
    assert report.items_sha256 == manifest.items_sha256
    assert report.content_sha256 == manifest.content_sha256
    assert report.category_counts == manifest.category_counts
    assert report.language_counts == manifest.language_counts
    assert report.duplicate_report_sha256 == duplicate.report_sha256
    assert report.duplicate_status == "pass"
    assert report.contamination_status == "pass"
    assert report.review_status == "approved"
    assert report.training_permitted is True
    assert report.status == "ready"
    assert report.manifest_sha256 == hashlib.sha256(render_json(manifest)).hexdigest()

    review = M10DevOpsContentReviewResult.model_validate_json(
        Path("reports/m10/raw/m10_devops_content_review.json").read_bytes()
    )
    assert review.source_dataset_version == report.dataset_version
    assert review.source_items_sha256 == report.items_sha256
    assert review.source_content_sha256 == report.content_sha256
    assert review.approved_manifest_sha256 == report.manifest_sha256
    assert review.reviewed_items == review.passed_items == 80
    assert review.authored_source_authorized is True
    assert review.full_m10_mixture_authorized is False
    assert review.m10_training_authorized is False

    serialized = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)
    assert "/home/" not in serialized
    assert "/data/" not in serialized
    assert "m9-devops-agent-release" not in serialized
