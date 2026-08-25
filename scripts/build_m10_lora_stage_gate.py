#!/usr/bin/env python3
"""Build a lineage-checked M10 Agent LoRA continuation Gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinyllm.training.m10_lora_gate import assemble_m10_lora_stage_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--candidate-subject-id", required=True)
    parser.add_argument("--parent-agent-summary", type=Path, required=True)
    parser.add_argument("--candidate-agent-summary", type=Path, required=True)
    parser.add_argument("--parent-m6-summary", type=Path)
    parser.add_argument("--candidate-m6-summary", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    _, gate = assemble_m10_lora_stage_gate(
        artifact_root=args.artifact_root,
        source_run=args.source_run,
        candidate_subject_id=args.candidate_subject_id,
        parent_agent_summary_path=args.parent_agent_summary,
        candidate_agent_summary_path=args.candidate_agent_summary,
        parent_m6_summary_path=args.parent_m6_summary,
        candidate_m6_summary_path=args.candidate_m6_summary,
        output_directory=args.output_directory,
    )
    print(gate.model_dump_json())
    return 0 if gate.decision == "accepted" else 6


if __name__ == "__main__":
    raise SystemExit(main())
