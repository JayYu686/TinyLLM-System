#!/usr/bin/env python3
"""Build the private exact-token M5.2-R3 label-aware mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.data import M5R3MixtureError, build_m5_r3_mixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--raw-pilot-artifact", type=Path, required=True)
    parser.add_argument("--formal-result", type=Path, required=True)
    parser.add_argument("--formal-raw-artifact", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/m5_r3_mixture_v2.yaml"),
    )
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = build_m5_r3_mixture(
            artifact_root=args.artifact_root,
            raw_pilot_artifact=args.raw_pilot_artifact,
            formal_result_path=args.formal_result,
            formal_raw_artifact=args.formal_raw_artifact,
            config_path=args.config,
            reasoning_config_path=args.reasoning_config,
            tokenizer_config_path=args.tokenization_config,
            model_dir=args.model_dir,
            output_root=args.output_root,
        )
    except (M5R3MixtureError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(manifest.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
