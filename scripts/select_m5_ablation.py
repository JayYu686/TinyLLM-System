#!/usr/bin/env python3
"""Apply the preregistered M5.2 selection policy to one Base and six Candidate summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from tinyllm.evaluation.m5_reasoning import M5ReasoningEvaluationError, select_m5_ablation
from tinyllm.evaluation.m5_reasoning_schema import M5ReasoningEvaluationSummary


def _load(path: Path) -> M5ReasoningEvaluationSummary:
    try:
        return M5ReasoningEvaluationSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise M5ReasoningEvaluationError("M5 evaluation summary is invalid") from exc


def main() -> int:
    """Load seven summaries, select deterministically, and write public evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        selection = select_m5_ablation(
            _load(args.base),
            tuple(_load(path) for path in args.candidate),
        )
    except M5ReasoningEvaluationError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(selection.model_dump_json())
    return 0 if selection.status == "selected" else 6


if __name__ == "__main__":
    raise SystemExit(main())
