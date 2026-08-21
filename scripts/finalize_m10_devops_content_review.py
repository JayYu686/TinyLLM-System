#!/usr/bin/env python3
"""Finalize the 80-item M10 DevOps maintainer content review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from tinyllm.data.m10_devops_review import (
    M10DevOpsReviewError,
    finalize_m10_devops_content_review,
    render_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--review-packet", type=Path, required=True)
    parser.add_argument("--approval-dir", type=Path, required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument(
        "--public-review",
        type=Path,
        default=Path("reports/m10/raw/m10_devops_content_review.json"),
    )
    parser.add_argument(
        "--public-build",
        type=Path,
        default=Path("reports/m10/raw/m10_devops_training_build.json"),
    )
    parser.add_argument(
        "--maintainer-confirmed",
        action="store_true",
        help="Required assertion that the maintainer approved all 80 sampled trajectories.",
    )
    return parser


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise M10DevOpsReviewError("reviewed-at must be an ISO 8601 timestamp") from exc


def _atomic_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _commit_approval_dir(path: Path, *, approval: bytes, approved_manifest: bytes) -> None:
    files = {
        "approval.json": approval,
        "approved-manifest.json": approved_manifest,
    }
    commit = render_json(
        {
            "schema_version": "1.0",
            "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
        }
    )
    expected = {**files, "COMMITTED.json": commit}
    if path.exists():
        if path.is_dir() and all(
            (path / name).is_file() and (path / name).read_bytes() == payload
            for name, payload in expected.items()
        ):
            return
        raise M10DevOpsReviewError("M10 DevOps approval directory already differs")

    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for name, payload in expected.items():
            (staging / name).write_bytes(payload)
        staging.rename(path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    args = _parser().parse_args()
    try:
        if not args.maintainer_confirmed:
            raise M10DevOpsReviewError("explicit maintainer confirmation is required")
        result, approved_manifest, public_build = finalize_m10_devops_content_review(
            dataset_dir=args.dataset_dir,
            review_packet_path=args.review_packet,
            reviewed_at=_parse_timestamp(args.reviewed_at),
        )
        approval = render_json(result)
        approved = render_json(approved_manifest)
        _commit_approval_dir(args.approval_dir, approval=approval, approved_manifest=approved)
        _atomic_file(args.public_review, approval)
        _atomic_file(args.public_build, render_json(public_build))
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": result.status,
                    "dataset_version": result.source_dataset_version,
                    "reviewed_items": result.reviewed_items,
                    "passed_items": result.passed_items,
                    "authored_source_authorized": result.authored_source_authorized,
                    "full_m10_mixture_authorized": result.full_m10_mixture_authorized,
                    "m10_training_authorized": result.m10_training_authorized,
                    "approval_sha256": hashlib.sha256(approval).hexdigest(),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (M10DevOpsReviewError, OSError, ValueError) as exc:
        print(json.dumps({"schema_version": "1.0", "status": "error", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
