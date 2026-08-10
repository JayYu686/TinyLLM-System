#!/usr/bin/env python3
"""Build the immutable 1M-token Qwen3 dual-mode correction mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.data import M5MixtureError, build_m5_dual_mode_correction_mixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-r3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-seed", type=int, default=20260810)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = build_m5_dual_mode_correction_mixture(
            artifact_root=args.artifact_root,
            source_r3_root=args.source_r3_root,
            output_root=args.output_root,
            build_seed=args.build_seed,
        )
    except (M5MixtureError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 3
    print(json.dumps({"status": "succeeded", "manifest": manifest.to_dict()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
