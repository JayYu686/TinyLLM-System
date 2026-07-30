#!/usr/bin/env python3
"""Exercise P2 parent fallback and compressor isolation without model generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from tinyllm.data import (
    ReasoningTask,
    generate_reasoning_dev_tasks,
    load_m5_reasoning_data_config,
)
from tinyllm.data.m5_r3_p0 import generate_m5_r3_p0_tasks, load_m5_r3_p0_config
from tinyllm.data.m5_r3_p1 import (
    M5R3P1Error,
    build_m5_r3_p1_dataset,
    generate_m5_r3_p1_contexts,
    m5_r3_p1_stage_seed,
)
from tinyllm.data.m5_r3_p1_schema import (
    M5R3P1FinishReason,
    M5R3P1Stage,
    M5R3P1StageGeneration,
)
from tinyllm.data.m5_r3_p2 import (
    M5R3P2Error,
    build_m5_r3_p2_fallback_solver_prompt,
    build_m5_r3_p2_isolated_compressor_prompt,
    load_m5_r3_p2_config,
    m5_r3_p2_config_sha256,
    select_m5_r3_p2_generations,
)
from tinyllm.data.m5_r3_p2_schema import M5R3P2Config, M5R3P2CPUSmoke
from tinyllm.data.m5_r3_source_strategy import (
    load_m5_r3_teacher_source_strategy_config,
)
from tinyllm.data.reasoning_schema import canonical_json
from tinyllm.data.tokenization import OffsetTokenizer, TokenEncoding

_FALLBACK_IDS = {
    "m5-reasoning:pilot:r3p1-config-en-002",
    "m5-reasoning:pilot:r3p1-config-en-004",
    "m5-reasoning:pilot:r3p1-config-en-008",
    "m5-reasoning:pilot:r3p1-config-en-010",
    "m5-reasoning:pilot:r3p1-config-en-011",
    "m5-reasoning:pilot:r3p1-config-zh-016",
}


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
        prompt = f"synthetic P2 historical CPU fixture {index}"
        answer = '{"issue":"missing_checkpoint"}'
        tasks.append(
            ReasoningTask(
                id=f"m5-reasoning:pilot:p2cpu-historical-{index:03d}",
                split="pilot_train",
                task_family="config",
                language="en",
                template_family="pilot.config.p2cpu-historical.v1",
                prompt=prompt,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                expected_answer_json=answer,
                expected_answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
            )
        )
    return tuple(tasks)


def _generation(
    *,
    task_id: str,
    stage: M5R3P1Stage,
    seed: int,
    prompt: str,
    output: str,
    finish_reason: M5R3P1FinishReason = "stop",
) -> M5R3P1StageGeneration:
    return M5R3P1StageGeneration(
        generation_id=f"{task_id}:{stage}",
        task_id=task_id,
        stage=stage,
        seed=seed,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        status="succeeded",
        finish_reason=finish_reason,
        raw_output=output,
        raw_output_sha256=hashlib.sha256(output.encode()).hexdigest(),
        input_token_count=64,
        generated_token_count=32,
    )


def _synthetic_generations() -> tuple[
    tuple[M5R3P1StageGeneration, ...],
    tuple[M5R3P1StageGeneration, ...],
    tuple[M5R3P1StageGeneration, ...],
]:
    p1 = load_m5_r3_teacher_source_strategy_config(
        Path("configs/data/m5_r3_teacher_source_strategy.yaml")
    )
    p2 = load_m5_r3_p2_config(Path("configs/data/m5_r3_p2.yaml"))
    parents: list[M5R3P1StageGeneration] = []
    fallbacks: list[M5R3P1StageGeneration] = []
    compressors: list[M5R3P1StageGeneration] = []
    for index, context in enumerate(generate_m5_r3_p1_contexts(p1)):
        solver_output = (
            f"<think>synthetic direct evidence {context.expected_label}</think>\n\n"
            f"{context.task.expected_answer_json}"
        )
        parents.append(
            _generation(
                task_id=context.task.id,
                stage="solver",
                seed=m5_r3_p1_stage_seed(p1.pilot.solver.base_seed, index),
                prompt=context.task.prompt,
                output=solver_output,
                finish_reason=("length" if context.task.id in _FALLBACK_IDS else "stop"),
            )
        )
        if context.task.id in _FALLBACK_IDS:
            fallbacks.append(
                _generation(
                    task_id=context.task.id,
                    stage="solver",
                    seed=m5_r3_p1_stage_seed(p2.fallback_solver.base_seed, index),
                    prompt=build_m5_r3_p2_fallback_solver_prompt(context),
                    output=solver_output,
                )
            )
        reasoning = f"{context.evidence_anchor} directly supports {context.expected_label}."
        compressor_output = canonical_json(
            {
                "reasoning": reasoning,
                "final_answer": json.loads(context.task.expected_answer_json),
            }
        )
        compressors.append(
            _generation(
                task_id=context.task.id,
                stage="compressor",
                seed=m5_r3_p1_stage_seed(
                    p2.isolated_compressor.base_seed,
                    index,
                ),
                prompt=build_m5_r3_p2_isolated_compressor_prompt(
                    context,
                    "synthetic private solver trace",
                    context.task.expected_answer_json,
                ),
                output=compressor_output,
            )
        )
    return tuple(parents), tuple(fallbacks), tuple(compressors)


def run_smoke() -> M5R3P2CPUSmoke:
    """Build a passing P2 fixture and exercise its three new failure boundaries."""

    p1 = load_m5_r3_teacher_source_strategy_config(
        Path("configs/data/m5_r3_teacher_source_strategy.yaml")
    )
    p2 = load_m5_r3_p2_config(Path("configs/data/m5_r3_p2.yaml"))
    reasoning = load_m5_reasoning_data_config(
        Path("configs/data/m5_reasoning_label_vocabulary_v2.yaml")
    )
    contexts = generate_m5_r3_p1_contexts(p1)
    parents, fallbacks, compressors = _synthetic_generations()
    selection = select_m5_r3_p2_generations(
        contexts,
        parents,
        fallbacks,
        compressors,
        p1_config=p1,
        p2_config=p2,
    )
    p0 = generate_m5_r3_p0_tasks(load_m5_r3_p0_config(Path("configs/data/m5_r3_p0.yaml")))
    p0_r1 = generate_m5_r3_p0_tasks(load_m5_r3_p0_config(Path("configs/data/m5_r3_p0_r1.yaml")))
    tokenizer = cast(OffsetTokenizer, cast(Any, _WhitespaceTokenizer()))
    build = build_m5_r3_p1_dataset(
        contexts,
        selection.generations,
        config=p1,
        dev_tasks=generate_reasoning_dev_tasks(reasoning),
        historical_tasks=_historical_tasks(),
        p0_tasks=p0,
        p0_r1_tasks=p0_r1,
        tokenizer=tokenizer,
        expected_stage_seeds=selection.expected_stage_seeds,
        expected_stage_prompt_sha256=selection.expected_stage_prompt_sha256,
        compressor_prompt_builder=build_m5_r3_p2_isolated_compressor_prompt,
    )
    drifted = list(fallbacks)
    drifted[0] = drifted[0].model_copy(update={"seed": 1})
    try:
        select_m5_r3_p2_generations(
            contexts,
            parents,
            drifted,
            compressors,
            p1_config=p1,
            p2_config=p2,
        )
    except M5R3P2Error as exc:
        if "fallback solver lineage differs" not in str(exc):
            raise
    else:
        raise M5R3P2Error("M5 R3 P2 accepted a fallback seed drift")
    marker = "PRIVATE_SOLVER_CONTENT_MUST_NOT_LEAK"
    for context in contexts:
        prompt = build_m5_r3_p2_isolated_compressor_prompt(
            context,
            marker,
            context.task.expected_answer_json,
        )
        if marker in prompt or any(
            label in prompt for label in context.allowed_labels if label != context.expected_label
        ):
            raise M5R3P2Error("M5 R3 P2 compressor input isolation failed")
    drifted_config = p2.to_dict()
    drifted_config["parent_p1_generation_artifact_sha256"] = "0" * 64
    try:
        M5R3P2Config.model_validate(drifted_config)
    except ValidationError:
        pass
    else:
        raise M5R3P2Error("M5 R3 P2 accepted a parent generation hash drift")
    return M5R3P2CPUSmoke(
        evidence_kind="synthetic_cpu_contract_smoke",
        model_generated=False,
        quality_metric=False,
        pilot_version=p2.pilot_version,
        config_sha256=m5_r3_p2_config_sha256(p2),
        parent_p1_result_sha256=p2.parent_p1_result_sha256,
        task_set_sha256=cast(Any, build.task_set_sha256),
        fallback_solver_items=6,
        isolated_compressor_items=40,
        accepted_samples=40,
        family_results=build.family_results,
        control=build.control,
        contamination=build.contamination,
        tested_failure_paths=(
            "parent_generation_hash_drift",
            "fallback_seed_drift",
            "compressor_input_leakage",
        ),
        p2_gpu_pilot_authorized=True,
        formal_source_expansion_authorized=False,
        r3_mixture_authorized=False,
        r3_training_authorized=False,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the deterministic P2 CPU Smoke interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Write one immutable path-free P2 CPU Smoke artifact."""

    args = build_parser().parse_args()
    try:
        if args.output.exists():
            raise M5R3P2Error("M5 R3 P2 CPU Smoke output already exists")
        result = run_smoke()
        _atomic_json(args.output, result.to_dict())
        print(result.model_dump_json())
        return 0
    except (M5R3P1Error, M5R3P2Error, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
