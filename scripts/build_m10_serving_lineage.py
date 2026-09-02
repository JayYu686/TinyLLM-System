#!/usr/bin/env python3
"""Build exact-model Serving lineage evidence for the M10 final gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.agent_eval.m10_gate import assemble_m10_serving_lineage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate-subject", required=True)
    parser.add_argument("--platform-gate", type=Path, required=True)
    parser.add_argument("--dev-summary", type=Path, required=True)
    parser.add_argument("--release-summary", type=Path, required=True)
    parser.add_argument("--bfcl-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence = assemble_m10_serving_lineage(
        artifact_root=args.artifact_root,
        candidate_subject_id=args.candidate_subject,
        platform_gate_path=args.platform_gate,
        dev_summary_path=args.dev_summary,
        release_summary_path=args.release_summary,
        bfcl_summary_path=args.bfcl_summary,
        output_path=args.output,
    )
    print(json.dumps(evidence.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
