#!/usr/bin/env python3
"""Combine two exact R2 Seed replays into one public length diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

from tinyllm.evaluation.m5_r2_diagnostic import (
    M5R2DiagnosticError,
    load_m5_r2_replay_config,
    load_m5_r2_summary,
    select_m5_r2_diagnostic,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed two-Seed R2 decision interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/m5_r2_length_replay.yaml"),
    )
    parser.add_argument("--seed42-summary", type=Path, required=True)
    parser.add_argument("--seed20260727-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Validate both summaries, persist the conclusion, and expose gate status."""

    args = build_parser().parse_args()
    try:
        if args.output.exists():
            raise M5R2DiagnosticError("M5 R2 public decision output already exists")
        config = load_m5_r2_replay_config(args.config)
        paths = (args.seed42_summary, args.seed20260727_summary)
        summaries = tuple(load_m5_r2_summary(path) for path in paths)
        decision = select_m5_r2_diagnostic(
            summaries,
            summary_sha256=tuple(_sha256_file(path) for path in paths),
            format_gate_basis_points=config.format_gate_basis_points,
            formal_candidate_max_new_tokens=config.formal_candidate_max_new_tokens,
        )
        _atomic_json(args.output, decision.to_dict())
        print(decision.model_dump_json())
        return 6 if decision.status == "length_ceiling_insufficient" else 0
    except (M5R2DiagnosticError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
