#!/usr/bin/env python3
"""Build one private, exact-1M-supervised-token M5.2 ablation mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.data import M5MixtureError, build_m5_ablation_mixture


def build_parser() -> argparse.ArgumentParser:
    """Build the private M5.2 mixture interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
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
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--thinking-fraction", type=float, required=True)
    parser.add_argument("--build-seed", type=int, default=20260725)
    return parser


def main() -> int:
    """Build, reopen, and print the content-addressed private Manifest."""

    args = build_parser().parse_args()
    try:
        manifest = build_m5_ablation_mixture(
            artifact_root=args.artifact_root,
            raw_pilot_artifact=args.raw_pilot_artifact,
            reasoning_config_path=args.reasoning_config,
            tokenizer_config_path=args.tokenization_config,
            model_dir=args.model_dir,
            output_root=args.output_root,
            thinking_fraction=args.thinking_fraction,
            build_seed=args.build_seed,
        )
    except (M5MixtureError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(manifest.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
