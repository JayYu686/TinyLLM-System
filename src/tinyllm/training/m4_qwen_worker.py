"""torchrun worker entry point for the formal M4.3 Qwen3-8B gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tinyllm.training.checkpoint import CheckpointError
from tinyllm.training.m4_qwen import run_m4_qwen


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded worker interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--export-final", action="store_true")
    return parser


def main() -> int:
    """Run one M4.3 phase and preserve stable failure categories."""

    args = build_parser().parse_args()
    try:
        result = run_m4_qwen(
            config_path=args.config,
            artifact_root=args.artifact_root,
            model_dir=args.model_dir,
            output_root=args.output_root,
            resume_run=args.resume_run,
            stop_after_step=args.stop_after_step,
            probe_only=args.probe_only,
            export_final=args.export_final,
        )
    except CheckpointError as exc:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error": {
                            "code": exc.code,
                            "message": str(exc),
                            "context": exc.context,
                        },
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        return 5
    except Exception as exc:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        return 4
    if result is not None:
        print(result.model_dump_json(indent=2), flush=True)
    return 143 if args.stop_after_step is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
