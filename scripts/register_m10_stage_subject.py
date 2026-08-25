#!/usr/bin/env python3
"""Register one hash-verified M10 1M/5M Full-SFT evaluation stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.deployment.m10_stage import register_m10_stage_evaluation_subject


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("/data/yujielun/tinyllm"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--stage-tokens", type=int, choices=(1_000_000, 5_000_000), default=5_000_000
    )
    args = parser.parse_args()

    record, record_sha256 = register_m10_stage_evaluation_subject(
        artifact_root=args.artifact_root,
        source_run=args.source_run,
        stage_tokens=args.stage_tokens,
    )
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "registered",
                "subject_id": record.subject_id,
                "evaluation_subject_sha256": record_sha256,
                "model_artifact_sha256": record.model_artifact_sha256,
                "source_checkpoint_id": record.model.training_checkpoint_id,
                "source_stage_tokens": record.model.training_tokens,
                "production_eligible": record.production_eligible,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
