#!/usr/bin/env python3
"""Apply the frozen M5 Thinking Budget v2 gate to Base and two R1 Seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from tinyllm.evaluation import M5FormatRepairGateResult
from tinyllm.evaluation.m5_thinking_budget import (
    M5ThinkingBudgetError,
    evaluate_m5_thinking_budget_gate,
)
from tinyllm.evaluation.m5_thinking_budget_schema import (
    M5ThinkingBudgetEvaluationSummary,
)


def _load_summary(path: Path) -> tuple[M5ThinkingBudgetEvaluationSummary, str]:
    try:
        payload = path.read_bytes()
        return (
            M5ThinkingBudgetEvaluationSummary.model_validate_json(payload),
            hashlib.sha256(payload).hexdigest(),
        )
    except (OSError, ValidationError) as exc:
        raise M5ThinkingBudgetError("Thinking Budget evaluation summary is invalid") from exc


def _load_source_gate(path: Path) -> tuple[M5FormatRepairGateResult, str]:
    try:
        payload = path.read_bytes()
        return (
            M5FormatRepairGateResult.model_validate_json(payload),
            hashlib.sha256(payload).hexdigest(),
        )
    except (OSError, ValidationError) as exc:
        raise M5ThinkingBudgetError("source M5 format-repair gate is invalid") from exc


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    """Validate all lineage and write one content-free protocol-v2 decision."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        base, base_sha256 = _load_summary(args.base)
        loaded_candidates = [_load_summary(path) for path in args.candidate]
        loaded_candidates.sort(key=lambda value: int(value[0].training_seed or -1))
        source_gate, source_gate_sha256 = _load_source_gate(args.source_gate)
        gate = evaluate_m5_thinking_budget_gate(
            base,
            tuple(value[0] for value in loaded_candidates),
            source_gate,
            base_summary_sha256=base_sha256,
            candidate_summary_sha256=cast(
                tuple[str, str], tuple(value[1] for value in loaded_candidates)
            ),
            source_gate_sha256=source_gate_sha256,
        )
    except M5ThinkingBudgetError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 6
    _atomic_json(args.output, gate.to_dict())
    print(gate.model_dump_json())
    return 0 if gate.status == "passed" else 6


if __name__ == "__main__":
    raise SystemExit(main())
