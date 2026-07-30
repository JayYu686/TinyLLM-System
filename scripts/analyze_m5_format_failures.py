#!/usr/bin/env python3
"""Convert four private M5.2 result files into a redacted R1 failure analysis."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from tinyllm.evaluation.m5_format_analysis import analyze_m5_format_failures
from tinyllm.evaluation.m5_reasoning import M5ReasoningEvaluationError


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    """Build the private-input, public-output analysis interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reasoning-config",
        type=Path,
        default=Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml"),
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat exactly four times for the 30%/50% two-Seed evaluations.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Validate private results and atomically publish content-free aggregates."""

    args = build_parser().parse_args()
    try:
        analysis = analyze_m5_format_failures(
            evaluation_directories=tuple(args.evaluation_dir),
            reasoning_config_path=args.reasoning_config,
        )
    except M5ReasoningEvaluationError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 6
    _atomic_json(args.output, analysis.to_dict())
    print(analysis.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
