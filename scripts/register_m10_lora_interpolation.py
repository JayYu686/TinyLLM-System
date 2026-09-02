#!/usr/bin/env python3
"""Create and register one same-Run M10 LoRA checkpoint interpolation."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, cast

from tinyllm.deployment.m10_lora_stage import (
    M10LoRAStageRegistrationError,
    create_m10_lora_interpolated_adapter,
    register_m10_lora_interpolated_evaluation_subject,
)


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--early-subject-id", required=True)
    parser.add_argument("--late-subject-id", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--late-weight-basis-points",
        type=int,
        choices=(2500, 5000, 7500),
        required=True,
    )
    args = parser.parse_args()
    try:
        evidence = create_m10_lora_interpolated_adapter(
            artifact_root=args.artifact_root,
            early_subject_id=args.early_subject_id,
            late_subject_id=args.late_subject_id,
            output_directory=args.output_directory,
            late_weight_basis_points=cast(Literal[2500, 5000, 7500], args.late_weight_basis_points),
        )
        record, record_sha256 = register_m10_lora_interpolated_evaluation_subject(
            artifact_root=args.artifact_root,
            interpolated_adapter_dir=args.output_directory,
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
                "adapter_interpolation_evidence_sha256": (
                    record.adapter_interpolation_evidence_sha256
                ),
                "early_weight_basis_points": evidence.early_weight_basis_points,
                "late_weight_basis_points": evidence.late_weight_basis_points,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
