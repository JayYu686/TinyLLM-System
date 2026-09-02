#!/usr/bin/env python3
"""Register one Dev-selected M10 LoRA checkpoint as an immutable evaluation subject."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from tinyllm.deployment.m10_lora_stage import (
    M10LoRAStageRegistrationError,
    register_m10_lora_checkpoint_evaluation_subject,
)


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint-export-directory", type=Path, required=True)
    parser.add_argument("--historical-subject-id", required=True)
    args = parser.parse_args()
    try:
        record, record_sha256 = register_m10_lora_checkpoint_evaluation_subject(
            artifact_root=args.artifact_root,
            source_run=args.source_run,
            checkpoint_export_directory=args.checkpoint_export_directory,
            historical_subject_id=args.historical_subject_id,
        )
    except M10LoRAStageRegistrationError as exc:
        parser.exit(5, f"registration failed: {exc}\n")
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "succeeded",
                "subject_id": record.subject_id,
                "evaluation_subject_sha256": record_sha256,
                "model_artifact_sha256": record.effective_artifact_sha256,
                "checkpoint_export_evidence_sha256": (record.checkpoint_export_evidence_sha256),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
