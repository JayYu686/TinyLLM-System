#!/usr/bin/env python3
"""Build the public M9 Dev split and private sealed Release split."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from tinyllm.agent_eval.suite import build_manifest, build_tasks, check_suite, write_suite


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the committed Dev split.")
    parser.add_argument(
        "--release-root",
        type=Path,
        default=None,
        help="Private output root. Required unless --check is used.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dev_root = project_root / "evals" / "agent" / "dev" / "v1"
    dev_tasks = build_tasks("dev")
    if args.check:
        stale = check_suite(dev_root, dev_tasks)
        if stale:
            parser.error("stale M9 Dev suite files: " + ", ".join(stale))
        print(json.dumps(build_manifest(dev_tasks).to_dict(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.release_root is None or not args.release_root.is_absolute():
        parser.error("--release-root must be an absolute private Artifact Store path")
    dev_manifest = write_suite(dev_root, dev_tasks)
    release_tasks = build_tasks("release")
    release_manifest = build_manifest(release_tasks)
    release_root = args.release_root / release_manifest.suite_version
    write_suite(release_root, release_tasks)
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "succeeded",
                "dev": dev_manifest.to_dict(),
                "release": release_manifest.to_dict(),
                "release_root": str(release_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
