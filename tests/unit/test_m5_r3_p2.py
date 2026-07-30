from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from tinyllm.data.m5_r3_p0 import generate_m5_r3_p0_tasks, load_m5_r3_p0_config
from tinyllm.data.m5_r3_p1 import (
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
from tinyllm.data.m5_r3_source_strategy import (
    load_m5_r3_teacher_source_strategy_config,
)
from tinyllm.data.reasoning_schema import ReasoningTask
from tinyllm.data.tokenization import OffsetTokenizer, TokenEncoding

P1_CONFIG = Path("configs/data/m5_r3_teacher_source_strategy.yaml")
P2_CONFIG = Path("configs/data/m5_r3_p2.yaml")
_FALLBACK_IDS = {
    "m5-reasoning:pilot:r3p1-config-en-002",
    "m5-reasoning:pilot:r3p1-config-en-004",
    "m5-reasoning:pilot:r3p1-config-en-008",
    "m5-reasoning:pilot:r3p1-config-en-010",
    "m5-reasoning:pilot:r3p1-config-en-011",
    "m5-reasoning:pilot:r3p1-config-zh-016",
}


class _WordTokenizer:
    def encode(self, text: str) -> TokenEncoding:
        words = tuple(text.split())
        return TokenEncoding(
            ids=tuple(range(len(words))),
            offsets=tuple((index, index + 1) for index in range(len(words))),
        )


def _frozen_tasks(count: int, *, split: str) -> tuple[ReasoningTask, ...]:
    result: list[ReasoningTask] = []
    for index in range(count):
        prompt = f"{split} P2 fixture {index}"
        answer = '{"issue":"missing_checkpoint"}'
        result.append(
            ReasoningTask(
                id=f"m5-reasoning:{'dev' if split == 'dev' else 'pilot'}:p2-{index:03d}",
                split="reasoning_dev" if split == "dev" else "pilot_train",
                task_family="config",
                language="en",
                template_family=(
                    "dev.config.p2-fixture.v1" if split == "dev" else "pilot.config.p2-fixture.v1"
                ),
                prompt=prompt,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                expected_answer_json=answer,
                expected_answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
            )
        )
    return tuple(result)


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
    p1 = load_m5_r3_teacher_source_strategy_config(P1_CONFIG)
    p2 = load_m5_r3_p2_config(P2_CONFIG)
    contexts = generate_m5_r3_p1_contexts(p1)
    parents: list[M5R3P1StageGeneration] = []
    fallbacks: list[M5R3P1StageGeneration] = []
    compressors: list[M5R3P1StageGeneration] = []
    for index, context in enumerate(contexts):
        task_id = context.task.id
        solver_output = (
            f"<think>direct evidence for {context.expected_label}</think>\n\n"
            f"{context.task.expected_answer_json}"
        )
        parents.append(
            _generation(
                task_id=task_id,
                stage="solver",
                seed=m5_r3_p1_stage_seed(p1.pilot.solver.base_seed, index),
                prompt=context.task.prompt,
                output=solver_output,
                finish_reason="length" if task_id in _FALLBACK_IDS else "stop",
            )
        )
        if task_id in _FALLBACK_IDS:
            fallbacks.append(
                _generation(
                    task_id=task_id,
                    stage="solver",
                    seed=m5_r3_p1_stage_seed(p2.fallback_solver.base_seed, index),
                    prompt=build_m5_r3_p2_fallback_solver_prompt(context),
                    output=solver_output,
                )
            )
        reasoning = f"{context.evidence_anchor} directly supports {context.expected_label}."
        compressor_output = json.dumps(
            {
                "reasoning": reasoning,
                "final_answer": json.loads(context.task.expected_answer_json),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        compressors.append(
            _generation(
                task_id=task_id,
                stage="compressor",
                seed=m5_r3_p1_stage_seed(p2.isolated_compressor.base_seed, index),
                prompt=build_m5_r3_p2_isolated_compressor_prompt(
                    context,
                    "private solver reasoning must not leak",
                    context.task.expected_answer_json,
                ),
                output=compressor_output,
            )
        )
    return tuple(parents), tuple(fallbacks), tuple(compressors)


def test_p2_config_is_parent_bound_and_keeps_frozen_gates() -> None:
    config = load_m5_r3_p2_config(P2_CONFIG)

    assert config.parent_p1_result_sha256 == (
        "c57b13d5a84a6b06450ad01ae7e9158ccc700736686575893b85aa27b92dfd95"
    )
    assert config.fallback_solver.max_new_tokens == 896
    assert config.trace_policy.max_reasoning_tokens == 192
    assert config.gate.accepted_per_family == {
        "config": 14,
        "log_diagnosis": 14,
    }
    assert len(m5_r3_p2_config_sha256(config)) == 64


def test_p2_isolated_compressor_cannot_receive_solver_or_alternative_labels() -> None:
    p1 = load_m5_r3_teacher_source_strategy_config(P1_CONFIG)
    context = generate_m5_r3_p1_contexts(p1)[0]
    marker = "PRIVATE_SOLVER_MARKER"
    prompt = build_m5_r3_p2_isolated_compressor_prompt(
        context,
        marker,
        context.task.expected_answer_json,
    )

    assert marker not in prompt
    assert context.evidence_anchor in prompt
    assert context.expected_label in prompt
    assert all(
        label not in prompt for label in context.allowed_labels if label != context.expected_label
    )


def test_p2_synthetic_parent_fallback_and_isolated_compressor_pass() -> None:
    p1 = load_m5_r3_teacher_source_strategy_config(P1_CONFIG)
    p2 = load_m5_r3_p2_config(P2_CONFIG)
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
    parent_p0 = load_m5_r3_p0_config(Path("configs/data/m5_r3_p0.yaml"))
    parent_r1 = load_m5_r3_p0_config(Path("configs/data/m5_r3_p0_r1.yaml"))
    build = build_m5_r3_p1_dataset(
        contexts,
        selection.generations,
        config=p1,
        dev_tasks=_frozen_tasks(200, split="dev"),
        historical_tasks=_frozen_tasks(100, split="historical"),
        p0_tasks=generate_m5_r3_p0_tasks(parent_p0),
        p0_r1_tasks=generate_m5_r3_p0_tasks(parent_r1),
        tokenizer=cast(OffsetTokenizer, cast(Any, _WordTokenizer())),
        expected_stage_seeds=selection.expected_stage_seeds,
        expected_stage_prompt_sha256=selection.expected_stage_prompt_sha256,
        compressor_prompt_builder=build_m5_r3_p2_isolated_compressor_prompt,
    )

    assert selection.fallback_task_ids == tuple(sorted(_FALLBACK_IDS))
    assert selection.fallback_trigger_counts == {"solver_length_limit": 6}
    assert len(build.samples) == 40
    assert all(item.gate_passed for item in build.family_results)


def test_p2_rejects_missing_required_fallback() -> None:
    p1 = load_m5_r3_teacher_source_strategy_config(P1_CONFIG)
    p2 = load_m5_r3_p2_config(P2_CONFIG)
    contexts = generate_m5_r3_p1_contexts(p1)
    parents, fallbacks, compressors = _synthetic_generations()

    with pytest.raises(M5R3P2Error, match="required fallback solver is missing"):
        select_m5_r3_p2_generations(
            contexts,
            parents,
            fallbacks[1:],
            compressors,
            p1_config=p1,
            p2_config=p2,
        )
