#!/usr/bin/env python3
"""Run the CPU-only, network-free M4 dependency compatibility gate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from tinyllm.training.m4_dependencies import run_m4_dependency_smoke


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    """Parse the stable script interface, run the gate, and emit JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON evidence")
    args = parser.parse_args()
    result = run_m4_dependency_smoke()
    rendered = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        _atomic_write(args.output, rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
