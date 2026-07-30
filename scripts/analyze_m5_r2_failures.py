#!/usr/bin/env python3
"""Generate the content-free M5.2-R2 D1 analysis from both private R1 results."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from tinyllm.evaluation.m5_r2_diagnostic import M5R2DiagnosticError
from tinyllm.evaluation.m5_r2_offline import analyze_m5_r2_failures


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    """Build the private-input, public-output D1 interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/m5_r2_length_replay.yaml"),
    )
    parser.add_argument(
        "--reasoning-config",
        type=Path,
        default=Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml"),
    )
    parser.add_argument("--seed42-evaluation", type=Path, required=True)
    parser.add_argument("--seed20260727-evaluation", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Run the local-tokenizer analysis without loading a model or GPU."""

    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    args = build_parser().parse_args()
    try:
        if args.output.exists():
            raise M5R2DiagnosticError("M5 R2 offline analysis output already exists")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        result = analyze_m5_r2_failures(
            evaluation_directories=(
                args.seed42_evaluation,
                args.seed20260727_evaluation,
            ),
            reasoning_config_path=args.reasoning_config,
            replay_config_path=args.config,
            tokenizer=tokenizer,
        )
        _atomic_json(args.output, result.to_dict())
        print(result.model_dump_json())
        return 0
    except (M5R2DiagnosticError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
