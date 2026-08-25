#!/usr/bin/env python3
"""Assemble paired Agent Dev and M6 evidence into the M10 continuation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.training.m10_stage_gate import assemble_m10_stage_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("/data/yujielun/tinyllm"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--candidate-subject", required=True)
    parser.add_argument("--parent-agent-summary", type=Path, required=True)
    parser.add_argument("--candidate-agent-summary", type=Path, required=True)
    parser.add_argument("--parent-m6-summary", type=Path, required=True)
    parser.add_argument("--candidate-m6-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    m6_evidence, gate = assemble_m10_stage_gate(
        artifact_root=args.artifact_root,
        source_run=args.source_run,
        candidate_subject_id=args.candidate_subject,
        parent_agent_summary_path=args.parent_agent_summary,
        candidate_agent_summary_path=args.candidate_agent_summary,
        parent_m6_summary_path=args.parent_m6_summary,
        candidate_m6_summary_path=args.candidate_m6_summary,
        output_directory=args.output_dir,
    )
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "decision": gate.decision,
                "agent_dev_improvement_basis_points": gate.agent_dev_improvement_basis_points,
                "m6_regression_basis_points": gate.m6_regression_basis_points,
                "candidate_m6_aggregate_basis_points": (
                    m6_evidence.candidate_aggregate_basis_points
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if gate.decision == "accepted" else 6


if __name__ == "__main__":
    raise SystemExit(main())
