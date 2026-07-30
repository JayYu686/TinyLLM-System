#!/usr/bin/env python3
"""Review M5.2-R3 Teacher-source alternatives from committed real evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from tinyllm.data.m5_r3_source_strategy import (
    M5R3SourceStrategyError,
    review_m5_r3_teacher_source_strategy,
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
    """Build the deterministic review interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/m5_r3_teacher_source_strategy.yaml"),
    )
    parser.add_argument(
        "--r2-decision",
        type=Path,
        default=Path("reports/m5/raw/m5_r2_length_diagnostic.json"),
    )
    parser.add_argument(
        "--p0-result",
        type=Path,
        default=Path("reports/m5/raw/m5_r3_p0.json"),
    )
    parser.add_argument(
        "--p0-r1-result",
        type=Path,
        default=Path("reports/m5/raw/m5_r3_p0_r1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Write one path-free review, refusing overwrite and input drift."""

    args = build_parser().parse_args()
    try:
        if args.output.exists():
            raise M5R3SourceStrategyError("M5 R3 source-strategy output already exists")
        result = review_m5_r3_teacher_source_strategy(
            config_path=args.config,
            r2_decision_path=args.r2_decision,
            p0_result_path=args.p0_result,
            p0_r1_result_path=args.p0_r1_result,
        )
        _atomic_json(args.output, result.to_dict())
        print(result.model_dump_json())
        return 0
    except (M5R3SourceStrategyError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
