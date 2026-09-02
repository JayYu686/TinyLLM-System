#!/usr/bin/env python3
"""Assemble the lineage-checked final M10 Agent model gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.agent_eval.m10_gate import assemble_m10_agent_model_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--candidate-items", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--parent-items", type=Path, required=True)
    parser.add_argument("--candidate-bfcl", type=Path, required=True)
    parser.add_argument("--parent-bfcl", type=Path, required=True)
    parser.add_argument("--m6-evidence", type=Path, required=True)
    parser.add_argument("--serving-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate = assemble_m10_agent_model_gate(
        candidate_summary_path=args.candidate_summary,
        candidate_items_path=args.candidate_items,
        parent_summary_path=args.parent_summary,
        parent_items_path=args.parent_items,
        candidate_bfcl_path=args.candidate_bfcl,
        parent_bfcl_path=args.parent_bfcl,
        m6_evidence_path=args.m6_evidence,
        serving_evidence_path=args.serving_evidence,
        output_path=args.output,
    )
    print(json.dumps(gate.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0 if gate.decision == "accepted" else 6


if __name__ == "__main__":
    raise SystemExit(main())
