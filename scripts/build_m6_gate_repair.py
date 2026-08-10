#!/usr/bin/env python3
"""Build the immutable 1M-token M6 R2 gate-repair mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.data import M5MixtureError, build_m6_gate_repair_mixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-seed", type=int, default=20260811)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = build_m6_gate_repair_mixture(
            artifact_root=args.artifact_root,
            tokenizer_config_path=args.tokenizer_config,
            model_dir=args.model_dir,
            project_root=args.project_root,
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
