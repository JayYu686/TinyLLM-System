#!/usr/bin/env python3
"""Apply the frozen M5.2-R1 two-seed format-reliability gate."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from pydantic import ValidationError

from tinyllm.evaluation import (
    M5ReasoningEvaluationError,
    M5ReasoningEvaluationSummary,
    evaluate_m5_format_repair_gate,
)
from tinyllm.training.m5_ablation_schema import M5AblationRunResult


def _load_evaluation(path: Path) -> M5ReasoningEvaluationSummary:
    try:
        return M5ReasoningEvaluationSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise M5ReasoningEvaluationError("M5 R1 evaluation summary is invalid") from exc


def _load_training(path: Path) -> M5AblationRunResult:
    try:
        return M5AblationRunResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise M5ReasoningEvaluationError("M5 R1 training result is invalid") from exc


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    """Validate R1 lineage, apply both gates, and write public evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--training-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        gate = evaluate_m5_format_repair_gate(
            _load_evaluation(args.base),
            tuple(_load_evaluation(path) for path in args.candidate),
            tuple(_load_training(path) for path in args.training_result),
        )
    except M5ReasoningEvaluationError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 6
    _atomic_json(args.output, gate.to_dict())
    print(gate.model_dump_json())
    return 0 if gate.status == "passed" else 6


if __name__ == "__main__":
    raise SystemExit(main())
