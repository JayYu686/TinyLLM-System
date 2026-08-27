#!/usr/bin/env python3
"""Build and validate the private M10.5 DevOps repair training source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.data.m10_devops import (
    M10DevOpsDataError,
    build_manifest,
    build_public_report,
    load_bfcl_target,
    load_m6_domain_target,
    load_m9_target,
    render_review_packet,
    scan_authored_duplicates,
    scan_contamination,
    write_dataset,
)
from tinyllm.data.m10_repair import (
    build_repair_samples,
    build_repair_v3_samples,
    validate_repair_samples,
    validate_repair_v3_samples,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--m9-dev-dir", type=Path, default=Path("evals/agent/dev/v1"))
    parser.add_argument("--m9-release-dir", type=Path, required=True)
    parser.add_argument("--bfcl-data-root", type=Path, required=True)
    parser.add_argument("--m6-domain-dir", type=Path, default=Path("evals/domain/v7"))
    parser.add_argument(
        "--public-report",
        type=Path,
        default=Path("reports/m10/raw/m10_repair_training_build.json"),
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("reports/m10/raw/m10_repair_content_quality.json"),
    )
    return parser


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.source_version == "v3":
            samples = build_repair_v3_samples()
            quality = validate_repair_v3_samples(samples)
        else:
            samples = build_repair_samples()
            quality = validate_repair_samples(samples)
        manifest = build_manifest(samples, review_status="pending")
        duplicate_report = scan_authored_duplicates(samples)
        targets = (
            load_m9_target(args.m9_dev_dir, target_id="m9_dev"),
            load_m9_target(args.m9_release_dir, target_id="m9_release"),
            load_bfcl_target(args.bfcl_data_root),
            load_m6_domain_target(args.m6_domain_dir),
        )
        contamination = scan_contamination(samples, manifest, targets)
        dataset_dir = write_dataset(
            args.artifact_root / "datasets/m10-agent/devops",
            samples,
            manifest,
            duplicate_report,
            contamination,
        )
        review_dir = args.artifact_root / "reviews" / manifest.dataset_version
        review_packet = review_dir / "review_packet.md"
        _atomic_text(review_packet, render_review_packet(samples, manifest))
        public = build_public_report(manifest, duplicate_report, contamination)
        _atomic_text(
            args.public_report,
            json.dumps(public.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        _atomic_text(
            args.quality_report,
            json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        result = {
            "schema_version": "1.0",
            "status": public.status,
            "dataset_version": manifest.dataset_version,
            "dataset_dir": str(dataset_dir),
            "review_packet": str(review_packet),
            "public_report": str(args.public_report),
            "quality_report": str(args.quality_report),
            "duplicate_status": duplicate_report.status,
            "contamination_status": contamination.status,
            "training_permitted": public.training_permitted,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0 if duplicate_report.status == contamination.status == "pass" else 6
    except M10DevOpsDataError as exc:
        print(json.dumps({"schema_version": "1.0", "status": "failed", "error": str(exc)}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
