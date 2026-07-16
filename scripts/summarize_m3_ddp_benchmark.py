#!/usr/bin/env python3
"""Validate private M3 benchmark evidence and build the strict matrix summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tinyllm.benchmark import build_m3_matrix_summary, load_benchmark_evidence


def build_parser() -> argparse.ArgumentParser:
    """Build the offline evidence aggregation interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Load, validate, aggregate, and atomically publish one private summary."""

    args = build_parser().parse_args()
    if args.output.exists():
        print("M3 benchmark summary failed: output already exists", file=sys.stderr)
        return 2
    if not args.output.parent.is_dir():
        print("M3 benchmark summary failed: output parent does not exist", file=sys.stderr)
        return 2
    try:
        runs = load_benchmark_evidence(args.evidence_root)
        summary = build_m3_matrix_summary(runs)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"M3 benchmark summary failed: {exc}", file=sys.stderr)
        return 4
    print(summary.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
