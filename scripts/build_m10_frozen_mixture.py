#!/usr/bin/env python3
"""Build and verify the private exact-token M10 Agent SFT mixture."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from tinyllm.data.m10_mixture import (
    M10MixtureError,
    build_frozen_mixture,
    build_public_report,
    write_frozen_mixture,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/m10_agent_frozen.yaml"))
    parser.add_argument("--source-config", type=Path, default=Path("configs/data/m10_agent.yaml"))
    parser.add_argument(
        "--tokenizer-config", type=Path, default=Path("configs/data/m2_tokenization.yaml")
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--m9-dev-dir", type=Path, default=Path("evals/agent/dev/v1"))
    parser.add_argument("--m9-release-dir", type=Path, required=True)
    parser.add_argument("--bfcl-data-root", type=Path, required=True)
    parser.add_argument("--m6-domain-dir", type=Path, default=Path("evals/domain/v7"))
    parser.add_argument(
        "--public-report",
        type=Path,
        default=Path("reports/m10/raw/m10_frozen_mixture.json"),
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
    args = _parser().parse_args()
    try:
        build = build_frozen_mixture(
            config_path=args.config,
            source_config_path=args.source_config,
            tokenizer_config_path=args.tokenizer_config,
            model_dir=args.model_dir,
            artifact_root=args.artifact_root,
            m9_dev_dir=args.m9_dev_dir,
            m9_release_dir=args.m9_release_dir,
            bfcl_data_root=args.bfcl_data_root,
            m6_domain_dir=args.m6_domain_dir,
        )
        manifest = write_frozen_mixture(args.artifact_root / "datasets/m10-agent/frozen", build)
        report = build_public_report(manifest)
        _atomic_json(args.public_report, report.to_dict())
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": report.status,
                    "dataset_version": report.dataset_version,
                    "sequence_count": report.sequence_count,
                    "supervised_tokens": report.target_supervised_tokens,
                    "training_permitted": report.training_permitted,
                    "public_report": str(args.public_report),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except M10MixtureError as exc:
        print(
            json.dumps(
                {"schema_version": "1.0", "status": "failed", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
