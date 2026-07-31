#!/usr/bin/env python3
"""Exercise the M5.2-R3 formal 240-to-160 source contract without model generation."""

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
from tinyllm.data.m5_r3_formal import (
    M5R3FormalSourceError,
    build_m5_r3_formal_source,
    check_m5_r3_formal_contamination,
    generate_m5_r3_formal_contexts,
    load_m5_r3_formal_source_config,
    m5_r3_formal_source_config_sha256,
)
from tinyllm.data.m5_r3_formal_schema import M5R3FormalCPUSmoke
from tinyllm.data.m5_r3_p0 import generate_m5_r3_p0_tasks, load_m5_r3_p0_config
from tinyllm.data.m5_r3_p1 import (
    generate_m5_r3_p1_contexts,
    m5_r3_p1_stage_seed,
)
from tinyllm.data.m5_r3_p1_schema import M5R3P1StageGeneration
from tinyllm.data.m5_r3_p2 import (
    build_m5_r3_p2_fallback_solver_prompt,
    build_m5_r3_p2_isolated_compressor_prompt,
)
from tinyllm.data.m5_r3_source_strategy import (
    load_m5_r3_teacher_source_strategy_config,
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


def _historical_tasks() -> tuple[ReasoningTask, ...]:
    tasks: list[ReasoningTask] = []
    for index in range(100):
        prompt = f"formal CPU historical fixture {index}"
        answer = '{"issue":"missing_checkpoint"}'
        tasks.append(
            ReasoningTask(
                id=f"m5-reasoning:pilot:formal-cpu-historical-{index:03d}",
                split="pilot_train",
                task_family="config",
                language="en",
                template_family="pilot.config.formal-cpu-historical.v1",
                prompt=prompt,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                expected_answer_json=answer,
                expected_answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
            )
        )
    return tuple(tasks)


def _parents() -> tuple[
    tuple[ReasoningTask, ...],
    tuple[ReasoningTask, ...],
    tuple[ReasoningTask, ...],
    tuple[ReasoningTask, ...],
    tuple[ReasoningTask, ...],
]:
    reasoning = load_m5_reasoning_data_config(
        Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml")
    )
    strategy = load_m5_r3_teacher_source_strategy_config(
        Path("configs/data/m5_r3_teacher_source_strategy.yaml")
    )
    return (
        generate_reasoning_dev_tasks(reasoning),
        _historical_tasks(),
        generate_m5_r3_p0_tasks(load_m5_r3_p0_config(Path("configs/data/m5_r3_p0.yaml"))),
        generate_m5_r3_p0_tasks(load_m5_r3_p0_config(Path("configs/data/m5_r3_p0_r1.yaml"))),
        tuple(item.task for item in generate_m5_r3_p1_contexts(strategy)),
    )


def _synthetic_generations(
    *,
    invalid_task_ids: frozenset[str] = frozenset(),
) -> tuple[M5R3P1StageGeneration, ...]:
    config = load_m5_r3_formal_source_config(Path("configs/data/m5_r3_formal_source.yaml"))
    records: list[M5R3P1StageGeneration] = []
    for index, context in enumerate(generate_m5_r3_formal_contexts(config)):
        solver_prompt = build_m5_r3_p2_fallback_solver_prompt(context)
        solver_output = (
            f"<think>synthetic formal solver {context.task.id}</think>\n\n"
            f"{context.task.expected_answer_json}"
        )
        records.append(
            M5R3P1StageGeneration(
                generation_id=f"{context.task.id}:solver",
                task_id=context.task.id,
                stage="solver",
                seed=m5_r3_p1_stage_seed(config.solver.base_seed, index),
                prompt_sha256=hashlib.sha256(solver_prompt.encode()).hexdigest(),
                status="succeeded",
                finish_reason="stop",
                raw_output=solver_output,
                raw_output_sha256=hashlib.sha256(solver_output.encode()).hexdigest(),
                input_token_count=64,
                generated_token_count=32,
            )
        )
        compressor_prompt = build_m5_r3_p2_isolated_compressor_prompt(
            context,
            "",
            context.task.expected_answer_json,
        )
        output = (
            "{invalid"
            if context.task.id in invalid_task_ids
            else canonical_json(
                {
                    "final_answer": json.loads(context.task.expected_answer_json),
                    "reasoning": (
                        f"{context.evidence_anchor} directly supports {context.expected_label}."
                    ),
                }
            )
        )
        records.append(
            M5R3P1StageGeneration(
                generation_id=f"{context.task.id}:compressor",
                task_id=context.task.id,
                stage="compressor",
                seed=m5_r3_p1_stage_seed(config.compressor.base_seed, index),
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


def run_smoke() -> M5R3FormalCPUSmoke:
    """Build the passing contract and exercise three fail-closed paths."""

    config = load_m5_r3_formal_source_config(Path("configs/data/m5_r3_formal_source.yaml"))
    contexts = generate_m5_r3_formal_contexts(config)
    dev, historical, p0, p0_r1, p1 = _parents()
    tokenizer = cast(OffsetTokenizer, cast(Any, _WhitespaceTokenizer()))
    build = build_m5_r3_formal_source(
        contexts,
        _synthetic_generations(),
        config=config,
        dev_tasks=dev,
        historical_tasks=historical,
        p0_tasks=p0,
        p0_r1_tasks=p0_r1,
        p1_tasks=p1,
        tokenizer=tokenizer,
    )
    config_zh = tuple(
        item.task.id
        for item in contexts
        if item.task.task_family == "config" and item.task.language == "zh"
    )
    insufficient = build_m5_r3_formal_source(
        contexts,
        _synthetic_generations(invalid_task_ids=frozenset(config_zh[:13])),
        config=config,
        dev_tasks=dev,
        historical_tasks=historical,
        p0_tasks=p0,
        p0_r1_tasks=p0_r1,
        p1_tasks=p1,
        tokenizer=tokenizer,
    )
    if insufficient.stratum_results[1].gate_passed:
        raise M5R3FormalSourceError("M5 R3 formal insufficient stratum was accepted")

    drifted = list(_synthetic_generations())
    drifted[0] = drifted[0].model_copy(update={"seed": 1})
    try:
        build_m5_r3_formal_source(
            contexts,
            drifted,
            config=config,
            dev_tasks=dev,
            historical_tasks=historical,
            p0_tasks=p0,
            p0_r1_tasks=p0_r1,
            p1_tasks=p1,
            tokenizer=tokenizer,
        )
    except ValueError as exc:
        if "solver lineage differs" not in str(exc):
            raise
    else:
        raise M5R3FormalSourceError("M5 R3 formal solver lineage drift was accepted")

    collision = check_m5_r3_formal_contamination(
        contexts,
        dev_tasks=dev,
        historical_tasks=historical,
        p0_tasks=p0,
        p0_r1_tasks=p0_r1,
        p1_tasks=(contexts[0].task, *p1[1:]),
    )
    if collision.status != "fail":
        raise M5R3FormalSourceError("M5 R3 formal parent collision was accepted")

    return M5R3FormalCPUSmoke(
        schema_version="1.0",
        evidence_kind="synthetic_cpu_contract_smoke",
        model_generated=False,
        quality_metric=False,
        expansion_version=config.expansion_version,
        config_sha256=m5_r3_formal_source_config_sha256(config),
        task_set_sha256=build.task_set_sha256,
        input_tasks=240,
        accepted_samples=240,
        selected_samples=160,
        stratum_results=build.stratum_results,
        contamination=build.contamination,
        tested_failure_paths=(
            "insufficient_language_stratum",
            "parent_task_contamination",
            "solver_lineage_drift",
        ),
        gpu_expansion_authorized=True,
        r3_mixture_authorized=False,
        r3_training_authorized=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.output.exists():
            raise M5R3FormalSourceError("M5 R3 formal CPU Smoke output already exists")
        result = run_smoke()
        _atomic_json(args.output, result.to_dict())
        print(result.model_dump_json())
        return 0
    except (M5R3FormalSourceError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
