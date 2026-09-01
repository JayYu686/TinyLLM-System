#!/usr/bin/env python3
"""Register the immutable DevOps-Adapter/Base-fallback M10 evaluation subject."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.deployment.m10_lora_stage import register_m10_lora_routed_evaluation_subject


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-subject-id", required=True)
    args = parser.parse_args()
    record, record_sha256 = register_m10_lora_routed_evaluation_subject(
        artifact_root=args.artifact_root,
        source_subject_id=args.source_subject_id,
    )
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "registered",
                "subject_id": record.subject_id,
                "evaluation_subject_sha256": record_sha256,
                "model_artifact_sha256": record.effective_artifact_sha256,
                "routing_policy_sha256": record.adapter_routing_policy.policy_sha256
                if record.adapter_routing_policy is not None
                else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
