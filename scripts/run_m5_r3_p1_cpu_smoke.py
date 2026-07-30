#!/usr/bin/env python3
"""Exercise the M5.2-R3 P1 two-stage contracts without model generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, cast

from tinyllm.data import (
    ReasoningTask,
    generate_reasoning_dev_tasks,
    load_m5_reasoning_data_config,
)
from tinyllm.data.m5_r3_p0 import generate_m5_r3_p0_tasks, load_m5_r3_p0_config
from tinyllm.data.m5_r3_p1 import (
    M5R3P1Error,
    build_m5_r3_p1_compressor_prompt,
    build_m5_r3_p1_dataset,
    check_m5_r3_p1_contamination,
    generate_m5_r3_p1_contexts,
    m5_r3_p1_stage_seed,
)
from tinyllm.data.m5_r3_p1_schema import (
    M5R3P1CPUSmoke,
    M5R3P1StageGeneration,
)
from tinyllm.data.m5_r3_source_strategy import (
    load_m5_r3_teacher_source_strategy_config,
    m5_r3_teacher_source_strategy_config_sha256,
)
from tinyllm.data.reasoning_schema import canonical_json
from tinyllm.data.tokenization import OffsetTokenizer, TokenEncoding


class _WhitespaceTokenizer:
    def encode(self, text: str) -> TokenEncoding:
        tokens = tuple(text.split())
        return TokenEncoding(
            ids=tuple(range(len(tokens))),
            offsets=tuple((index, index + 1) for index in range(len(tokens))),
        )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _historical_tasks(count: int = 100) -> tuple[ReasoningTask, ...]:
    tasks: list[ReasoningTask] = []
    for index in range(count):
        prompt = f"synthetic historical CPU fixture {index}"
        answer = '{"issue":"missing_checkpoint"}'
        tasks.append(
            ReasoningTask(
                id=f"m5-reasoning:pilot:p1cpu-historical-{index:03d}",
                split="pilot_train",
                task_family="config",
                language="en",
                template_family="pilot.config.p1cpu-historical.v1",
                prompt=prompt,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                expected_answer_json=answer,
                expected_answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
            )
        )
    return tuple(tasks)


def _synthetic_generations(
    *,
    missing_anchor: bool = False,
) -> tuple[M5R3P1StageGeneration, ...]:
    config = load_m5_r3_teacher_source_strategy_config(
        Path("configs/data/m5_r3_teacher_source_strategy.yaml")
    )
    records: list[M5R3P1StageGeneration] = []
    for index, context in enumerate(generate_m5_r3_p1_contexts(config)):
        solver_reasoning = f"synthetic solver fixture {context.task.id}"
        solver_output = f"<think>{solver_reasoning}</think>\n\n{context.task.expected_answer_json}"
        records.append(
            M5R3P1StageGeneration(
                generation_id=f"{context.task.id}:solver",
                task_id=context.task.id,
                stage="solver",
                seed=m5_r3_p1_stage_seed(config.pilot.solver.base_seed, index),
                prompt_sha256=context.task.prompt_sha256,
                status="succeeded",
                finish_reason="stop",
                raw_output=solver_output,
                raw_output_sha256=hashlib.sha256(solver_output.encode()).hexdigest(),
                input_token_count=64,
                generated_token_count=32,
            )
        )
        reasoning = (
            "synthetic rationale without evidence"
            if missing_anchor
            else f"{context.evidence_anchor} directly supports {context.expected_label}."
        )
        output = canonical_json(
            {
                "final_answer": json.loads(context.task.expected_answer_json),
                "reasoning": reasoning,
            }
        )
        compressor_prompt = build_m5_r3_p1_compressor_prompt(
            context,
            solver_reasoning=solver_reasoning,
            verified_final_answer=context.task.expected_answer_json,
        )
        records.append(
            M5R3P1StageGeneration(
                generation_id=f"{context.task.id}:compressor",
                task_id=context.task.id,
                stage="compressor",
                seed=m5_r3_p1_stage_seed(config.pilot.compressor.base_seed, index),
                prompt_sha256=hashlib.sha256(compressor_prompt.encode()).hexdigest(),
                status="succeeded",
                finish_reason="stop",
                raw_output=output,
                raw_output_sha256=hashlib.sha256(output.encode()).hexdigest(),
                input_token_count=128,
                generated_token_count=32,
            )
        )
    return tuple(records)


def run_smoke() -> M5R3P1CPUSmoke:
    """Build a passing synthetic dataset and exercise three blocking failures."""

    config = load_m5_r3_teacher_source_strategy_config(
        Path("configs/data/m5_r3_teacher_source_strategy.yaml")
    )
    reasoning = load_m5_reasoning_data_config(
        Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml")
    )
    contexts = generate_m5_r3_p1_contexts(config)
    dev = generate_reasoning_dev_tasks(reasoning)
    historical = _historical_tasks()
    p0 = generate_m5_r3_p0_tasks(load_m5_r3_p0_config(Path("configs/data/m5_r3_p0.yaml")))
    p0_r1 = generate_m5_r3_p0_tasks(load_m5_r3_p0_config(Path("configs/data/m5_r3_p0_r1.yaml")))
    tokenizer = cast(OffsetTokenizer, cast(Any, _WhitespaceTokenizer()))
    build = build_m5_r3_p1_dataset(
        contexts,
        _synthetic_generations(),
        config=config,
        dev_tasks=dev,
        historical_tasks=historical,
        p0_tasks=p0,
        p0_r1_tasks=p0_r1,
        tokenizer=tokenizer,
    )
    missing_anchor = build_m5_r3_p1_dataset(
        contexts,
        _synthetic_generations(missing_anchor=True),
        config=config,
        dev_tasks=dev,
        historical_tasks=historical,
        p0_tasks=p0,
        p0_r1_tasks=p0_r1,
        tokenizer=tokenizer,
    )
    if missing_anchor.rejection_counts != {"missing_evidence_anchor": 40}:
        raise M5R3P1Error("M5 R3 P1 missing-anchor failure path did not reject")
    drifted = list(_synthetic_generations())
    drifted[0] = drifted[0].model_copy(update={"seed": 1})
    try:
        build_m5_r3_p1_dataset(
            contexts,
            drifted,
            config=config,
            dev_tasks=dev,
            historical_tasks=historical,
            p0_tasks=p0,
            p0_r1_tasks=p0_r1,
            tokenizer=tokenizer,
        )
    except M5R3P1Error as exc:
        if "solver lineage differs" not in str(exc):
            raise
    else:
        raise M5R3P1Error("M5 R3 P1 solver lineage drift was accepted")
    collision = check_m5_r3_p1_contamination(
        contexts,
        dev_tasks=dev,
        historical_tasks=historical,
        p0_tasks=(contexts[0].task, *p0[1:]),
        p0_r1_tasks=p0_r1,
    )
    if collision.status != "fail":
        raise M5R3P1Error("M5 R3 P1 parent collision was accepted")
    return M5R3P1CPUSmoke(
        evidence_kind="synthetic_cpu_contract_smoke",
        model_generated=False,
        quality_metric=False,
        pilot_version=config.pilot.pilot_version,
        config_sha256=m5_r3_teacher_source_strategy_config_sha256(config),
        task_set_sha256=build.task_set_sha256,
        samples_sha256=build.samples_sha256,
        accepted_samples=40,
        family_results=build.family_results,
        control=build.control,
        contamination=build.contamination,
        tested_failure_paths=(
            "compressor_missing_evidence_anchor",
            "parent_task_contamination",
            "solver_lineage_drift",
        ),
        p1_gpu_pilot_authorized=True,
        formal_source_expansion_authorized=False,
        r3_mixture_authorized=False,
        r3_training_authorized=False,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the deterministic CPU Smoke interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Write one immutable path-free CPU Smoke artifact."""

    args = build_parser().parse_args()
    try:
        if args.output.exists():
            raise M5R3P1Error("M5 R3 P1 CPU Smoke output already exists")
        result = run_smoke()
        _atomic_json(args.output, result.to_dict())
        print(result.model_dump_json())
        return 0
    except (M5R3P1Error, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
