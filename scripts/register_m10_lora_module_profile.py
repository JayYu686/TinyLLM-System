#!/usr/bin/env python3
"""Create and register one fixed M10 LoRA module-strength profile."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, cast

from tinyllm.deployment.m10_lora_stage import (
    M10LoRAStageRegistrationError,
    create_m10_lora_module_profile_adapter,
    register_m10_lora_module_profile_evaluation_subject,
)

ModuleProfile = Literal[
    "attention_full_mlp_eighth",
    "mlp_full_attention_eighth",
    "qv_full_rest_eighth",
]


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-subject-id", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=(
            "attention_full_mlp_eighth",
            "mlp_full_attention_eighth",
            "qv_full_rest_eighth",
        ),
        required=True,
    )
    args = parser.parse_args()
    try:
        evidence = create_m10_lora_module_profile_adapter(
            artifact_root=args.artifact_root,
            source_subject_id=args.source_subject_id,
            output_directory=args.output_directory,
            profile=cast(ModuleProfile, args.profile),
        )
        record, record_sha256 = register_m10_lora_module_profile_evaluation_subject(
            artifact_root=args.artifact_root,
            module_profile_adapter_dir=args.output_directory,
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
                "adapter_module_profile_evidence_sha256": (
                    record.adapter_module_profile_evidence_sha256
                ),
                "profile": evidence.profile,
                "module_relative_scale_basis_points": (evidence.module_relative_scale_basis_points),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
