#!/usr/bin/env python3
"""Freeze the authorized 50M-token M5.3 dual-mode dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.data.m5_formal import M5FormalDatasetError, build_m5_formal_dataset
from tinyllm.lineage import read_git_identity


def main() -> int:
    """Build or revalidate the formal data under one clean Git identity."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--authorization-gate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-seed", type=int, default=20260731)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    git_commit, git_dirty = read_git_identity(project_root)
    try:
        if git_dirty:
            raise M5FormalDatasetError("formal M5 data requires a clean Git worktree")
        manifest = build_m5_formal_dataset(
            source_root=args.source_root,
            authorization_gate_path=args.authorization_gate,
            output_root=args.output_root,
            build_seed=args.build_seed,
            git_commit=git_commit,
        )
    except (M5FormalDatasetError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "succeeded", "manifest": manifest.to_dict()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
