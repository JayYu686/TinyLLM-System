#!/usr/bin/env python3
"""Verify and profile the pinned external M10 Agent training sources."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from tinyllm.data.m10_agent import M10AgentDataError, profile_m10_external_sources


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit private-input and content-free-output interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/m10_agent.yaml"),
    )
    parser.add_argument("--toolace-artifact", type=Path, required=True)
    parser.add_argument("--hermes-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Write an immutable aggregate profile, failing closed on source drift."""

    args = build_parser().parse_args()
    try:
        if args.output.exists():
            raise M10AgentDataError("M10 source profile output already exists")
        report = profile_m10_external_sources(
            config_path=args.config,
            toolace_artifact=args.toolace_artifact,
            hermes_artifact=args.hermes_artifact,
        )
        _atomic_json(args.output, report.to_dict())
        print(report.model_dump_json())
        return 0
    except (M10AgentDataError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
