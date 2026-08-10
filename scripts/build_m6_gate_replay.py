#!/usr/bin/env python3
"""Build the immutable 1M-token M6 continual-learning replay mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.data import M5MixtureError, build_m6_gate_replay_mixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-seed", type=int, default=20260812)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = build_m6_gate_replay_mixture(
            correction_root=args.correction_root,
            repair_root=args.repair_root,
            output_root=args.output_root,
            build_seed=args.build_seed,
        )
    except (M5MixtureError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 3
    print(json.dumps({"status": "succeeded", "manifest": manifest.to_dict()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
