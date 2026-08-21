#!/usr/bin/env python3
"""Build both pinned M10 external sources into private canonical artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from tinyllm.data.m10_agent import M10AgentDataError
from tinyllm.data.m10_canonical import (
    M10CanonicalImportError,
    build_external_import_report,
    import_external_source,
    write_external_import,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/m10_agent.yaml"))
    parser.add_argument("--toolace-artifact", type=Path, required=True)
    parser.add_argument("--hermes-artifact", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--public-report",
        type=Path,
        default=Path("reports/m10/raw/m10_external_canonical_import.json"),
    )
    return parser


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = build_parser().parse_args()
    try:
        toolace = import_external_source(
            config_path=args.config,
            source_id="toolace",
            artifact_path=args.toolace_artifact,
        )
        hermes = import_external_source(
            config_path=args.config,
            source_id="hermes_function_calling",
            artifact_path=args.hermes_artifact,
        )
        toolace_dir = write_external_import(
            args.artifact_root / "datasets/m10-agent/external/toolace", toolace
        )
        hermes_dir = write_external_import(
            args.artifact_root / "datasets/m10-agent/external/hermes", hermes
        )
        report = build_external_import_report((toolace, hermes))
        _atomic_json(args.public_report, report.to_dict())
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": report.status,
                    "toolace_version": toolace.manifest.import_version,
                    "toolace_dir": str(toolace_dir),
                    "hermes_version": hermes.manifest.import_version,
                    "hermes_dir": str(hermes_dir),
                    "accepted_rows": report.total_accepted_rows,
                    "rejected_rows": report.total_rejected_rows,
                    "public_report": str(args.public_report),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0 if report.status == "pass" else 6
    except (M10CanonicalImportError, M10AgentDataError, OSError, ValueError) as exc:
        print(
            json.dumps({"schema_version": "1.0", "status": "error", "error": str(exc)}),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
