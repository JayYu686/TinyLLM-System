#!/usr/bin/env python3
"""Audit whether the existing private Pilot can supply M5.2-R3 targeted traces."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from tinyllm.data.m5_r3_audit import M5R3AuditError, audit_m5_r3_sources


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    """Build the private-input and path-free public-output audit interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/m5_r3_targeted_repair.yaml"),
    )
    parser.add_argument("--raw-pilot-artifact", type=Path, required=True)
    parser.add_argument(
        "--reasoning-config",
        type=Path,
        default=Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml"),
    )
    parser.add_argument(
        "--tokenization-config",
        type=Path,
        default=Path("configs/data/m2_tokenization.yaml"),
    )
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument(
        "--r2-decision",
        type=Path,
        default=Path("reports/m5/raw/m5_r2_length_diagnostic.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Run the CPU-only source audit and return 6 when new Teacher data is required."""

    args = build_parser().parse_args()
    try:
        if args.output.exists():
            raise M5R3AuditError("M5 R3 audit output already exists")
        result = audit_m5_r3_sources(
            config_path=args.config,
            raw_pilot_artifact=args.raw_pilot_artifact,
            reasoning_config_path=args.reasoning_config,
            tokenization_config_path=args.tokenization_config,
            tokenizer_dir=args.tokenizer_dir,
            r2_decision_path=args.r2_decision,
        )
        _atomic_json(args.output, result.to_dict())
        print(result.model_dump_json())
        return 0 if not result.new_teacher_source_required else 6
    except (M5R3AuditError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
