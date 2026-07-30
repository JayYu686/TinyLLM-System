#!/usr/bin/env python3
"""Finalize all private M5.2-R3 P2 maintainer judgments into a public summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from tinyllm.data.m5_r3_review import (
    M5R3ContentReviewError,
    finalize_m5_r3_content_review,
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed maintainer-review finalizer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-result",
        type=Path,
        default=Path("reports/m5/raw/m5_r3_p2.json"),
    )
    parser.add_argument("--private-raw", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maintainer-confirmed",
        action="store_true",
        help="Required explicit assertion that the maintainer reviewed all 33 items.",
    )
    return parser


def main() -> int:
    """Write one path-free review summary, refusing drafts and overwrite."""

    args = build_parser().parse_args()
    try:
        if not args.maintainer_confirmed:
            raise M5R3ContentReviewError("explicit maintainer confirmation is required")
        if args.output.exists():
            raise M5R3ContentReviewError("M5 R3 content-review output already exists")
        result = finalize_m5_r3_content_review(
            public_result_path=args.public_result,
            private_raw_path=args.private_raw,
            judgments_path=args.judgments,
        )
        _atomic_json(args.output, result.to_dict())
        print(result.model_dump_json())
        return 0 if result.status == "approved" else 6
    except (M5R3ContentReviewError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
