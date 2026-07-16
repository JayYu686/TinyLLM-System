"""torchrun worker entry point for the M4.1 FSDP2 correctness gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tinyllm.training.fsdp2 import run_fsdp2_correctness


def main() -> int:
    """Parse the worker interface and emit one Rank-zero JSON result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_fsdp2_correctness(
            config_path=args.config,
            output_root=args.output_root,
        )
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
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
