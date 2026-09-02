#!/usr/bin/env python3
"""Recover aggregate evidence after all Agent task results were persisted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyllm.agent_eval.finalize import AgentEvalFinalizeError, finalize_agent_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        summary = finalize_agent_evaluation(
            suite_directory=args.suite,
            output_directory=args.output,
            artifact_root=args.artifact_root,
            project_root=args.project_root,
        )
    except (AgentEvalFinalizeError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"schema_version": "1.0", "status": "error", "error": str(exc)}))
        return 6
    print(summary.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
