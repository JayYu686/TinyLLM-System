#!/usr/bin/env python3
"""Run one formal M3 DDP benchmark repetition on explicit idle GPUs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from tinyllm.benchmark.config import BenchmarkProfile
from tinyllm.benchmark.schema import BenchmarkGroup
from tinyllm.benchmark.supervisor import exit_code_for_error, run_formal_benchmark
from tinyllm.training.smoke_preflight import parse_gpu_indices


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed formal benchmark interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("strong", "weak"), required=True)
    parser.add_argument(
        "--group",
        choices=("standard", "same_numa", "cross_numa"),
        default="standard",
    )
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--gpu-indices", type=parse_gpu_indices, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser


def main() -> int:
    """Execute the formal supervisor and print stable JSON."""

    args = build_parser().parse_args()
    try:
        result = run_formal_benchmark(
            config_path=args.config,
            output_root=args.output_root,
            evidence_dir=args.evidence_dir,
            profile=cast(BenchmarkProfile, args.profile),
            group=cast(BenchmarkGroup, args.group),
            repeat=args.repeat,
            gpu_indices=args.gpu_indices,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return exit_code_for_error(exc)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
