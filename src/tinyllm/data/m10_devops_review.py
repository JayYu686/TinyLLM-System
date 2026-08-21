"""Finalize the maintainer review of the immutable M10 DevOps source."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from tinyllm.data.m10_devops import (
    build_manifest,
    build_public_report,
    load_dataset,
    render_review_packet,
)
from tinyllm.data.m10_devops_schema import (
    M10DevOpsBuildReport,
    M10DevOpsContaminationReport,
    M10DevOpsContentReviewResult,
    M10DevOpsDatasetManifest,
    M10DevOpsDuplicateReport,
)
from tinyllm.schemas.base import StrictSchema


class M10DevOpsReviewError(ValueError):
    """Raised when review evidence does not bind to the frozen source."""


def render_json(value: StrictSchema | dict[str, object]) -> bytes:
    """Render public and private review evidence deterministically."""

    if isinstance(value, StrictSchema):
        value = value.to_dict()
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    """Return one lowercase SHA256 digest."""

    return hashlib.sha256(value).hexdigest()


def finalize_m10_devops_content_review(
    *,
    dataset_dir: Path,
    review_packet_path: Path,
    reviewed_at: datetime | None = None,
) -> tuple[
    M10DevOpsContentReviewResult,
    M10DevOpsDatasetManifest,
    M10DevOpsBuildReport,
]:
    """Approve exactly the deterministic 80-item packet without mutating source facts."""

    try:
        pending_manifest, samples = load_dataset(dataset_dir)
        if pending_manifest.review_status != "pending":
            raise M10DevOpsReviewError("M10 DevOps source is not pending review")
        review_packet = review_packet_path.read_bytes()
        expected_packet = render_review_packet(samples, pending_manifest).encode()
        if review_packet != expected_packet:
            raise M10DevOpsReviewError("M10 DevOps review packet differs from the source")

        duplicate = M10DevOpsDuplicateReport.model_validate_json(
            (dataset_dir / "duplicate-report.json").read_bytes()
        )
        contamination = M10DevOpsContaminationReport.model_validate_json(
            (dataset_dir / "contamination-report.json").read_bytes()
        )
        if duplicate.status != "pass" or contamination.status != "pass":
            raise M10DevOpsReviewError("M10 DevOps machine gates are not passing")
        if (
            contamination.source_dataset_version != pending_manifest.dataset_version
            or contamination.source_content_sha256 != pending_manifest.content_sha256
        ):
            raise M10DevOpsReviewError("M10 DevOps contamination lineage differs")

        approved_manifest = build_manifest(samples, review_status="approved")
        if (
            approved_manifest.dataset_version != pending_manifest.dataset_version
            or approved_manifest.items_sha256 != pending_manifest.items_sha256
            or approved_manifest.content_sha256 != pending_manifest.content_sha256
        ):
            raise M10DevOpsReviewError("M10 DevOps approved identity differs from pending source")

        timestamp = datetime.now(UTC) if reviewed_at is None else reviewed_at
        result = M10DevOpsContentReviewResult(
            reviewed_at=timestamp,
            source_dataset_version=pending_manifest.dataset_version,
            source_pending_manifest_sha256=sha256_bytes(
                (dataset_dir / "manifest.json").read_bytes()
            ),
            source_items_sha256=pending_manifest.items_sha256,
            source_content_sha256=pending_manifest.content_sha256,
            source_review_packet_sha256=sha256_bytes(review_packet),
            source_duplicate_report_sha256=duplicate.report_sha256,
            source_contamination_report_sha256=contamination.report_sha256,
            approved_manifest_sha256=sha256_bytes(render_json(approved_manifest)),
            category_counts={
                "single_tool": 10,
                "no_tool": 10,
                "wrong_tool_irrelevance": 10,
                "missing_argument_clarification": 10,
                "sequential_multi_step": 10,
                "parallel_independent_tools": 10,
                "tool_failure_recovery": 10,
                "grounding_approval_security": 10,
            },
            language_counts={"en": 40, "zh": 40},
        )
        return (
            result,
            approved_manifest,
            build_public_report(approved_manifest, duplicate, contamination),
        )
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        raise M10DevOpsReviewError("M10 DevOps review evidence is invalid") from exc
