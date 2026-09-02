#!/usr/bin/env python3
"""Register an inference-calibrated M10 Agent LoRA subject."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from tinyllm.deployment.m10_lora_stage import (
    M10LoRAStageRegistrationError,
    register_m10_lora_calibrated_evaluation_subject,
)


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-subject-id", required=True)
    parser.add_argument("--calibrated-adapter-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        record, record_sha256 = register_m10_lora_calibrated_evaluation_subject(
            artifact_root=args.artifact_root,
            source_subject_id=args.source_subject_id,
            calibrated_adapter_dir=args.calibrated_adapter_dir,
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
                "adapter_calibration_evidence_sha256": (record.adapter_calibration_evidence_sha256),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
