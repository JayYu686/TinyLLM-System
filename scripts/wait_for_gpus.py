#!/usr/bin/env python3
"""Wait for a safe GPU subset, then exec one command without a shell."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tinyllm.training.smoke_preflight import (
    MAX_MEMORY_USED_MIB,
    MAX_TEMPERATURE_C,
    MAX_UTILIZATION_PERCENT,
    GpuPreflight,
    inspect_gpus,
    parse_gpu_indices,
)


def select_gpus(
    rows: tuple[GpuPreflight, ...],
    *,
    count: int,
    max_memory_used_mib: int = MAX_MEMORY_USED_MIB,
) -> tuple[int, ...] | None:
    """Select the first requested-order subset satisfying formal Preflight."""

    safe = tuple(
        row["index"]
        for row in rows
        if row["memory_used_mib"] <= max_memory_used_mib
        and row["utilization_percent"] <= MAX_UTILIZATION_PERCENT
        and row["temperature_c"] <= MAX_TEMPERATURE_C
    )
    return safe[:count] if len(safe) >= count else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-gpus", type=parse_gpu_indices, required=True)
    parser.add_argument("--count", type=int, choices=range(1, 11), required=True)
    parser.add_argument(
        "--max-memory-used-mib",
        type=int,
        choices=(MAX_MEMORY_USED_MIB, 3072),
        default=MAX_MEMORY_USED_MIB,
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=604_800)
    parser.add_argument("--prerequisite-path", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.command or args.command[0] != "--" or len(args.command) == 1:
        raise SystemExit("command must follow --")
    if args.count > len(args.candidate_gpus):
        raise SystemExit("count exceeds candidate GPU count")
    if args.poll_seconds < 5 or args.timeout_seconds <= 0:
        raise SystemExit("invalid wait interval or timeout")
    started = time.monotonic()
    consecutive_preflight_errors = 0
    while True:
        prerequisite_ready = args.prerequisite_path is None or args.prerequisite_path.is_file()
        try:
            rows = inspect_gpus(args.candidate_gpus) if prerequisite_ready else ()
        except RuntimeError:
            consecutive_preflight_errors += 1
            rows = ()
            print(
                json.dumps(
                    {
                        "status": "preflight_retry",
                        "reason": "nvidia_smi_preflight_failed",
                        "consecutive_errors": consecutive_preflight_errors,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
        else:
            consecutive_preflight_errors = 0
        selected = (
            select_gpus(
                rows,
                count=args.count,
                max_memory_used_mib=args.max_memory_used_mib,
            )
            if rows
            else None
        )
        if selected is not None:
            replacements = {
                "{gpu_index}": str(selected[0]),
                "{gpu_indices}": ",".join(str(index) for index in selected),
            }
            command = [replacements.get(value, value) for value in args.command[1:]]
            executable = str(Path(command[0]).resolve()) if "/" in command[0] else command[0]
            print(
                json.dumps({"status": "selected", "gpu_indices": selected, "command": command[0]}),
                flush=True,
            )
            os.execvpe(executable, command, os.environ.copy())
        if time.monotonic() - started >= args.timeout_seconds:
            print(json.dumps({"status": "timeout", "candidate_gpus": args.candidate_gpus}))
            return 3
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
