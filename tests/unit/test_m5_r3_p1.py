from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from scripts.run_m5_r3_p1 import (
    M5R3P1EnvironmentError,
    _verify_model_directory,
    build_parser,
)
from tinyllm.data.m5_r3_p0 import generate_m5_r3_p0_tasks, load_m5_r3_p0_config
from tinyllm.data.m5_r3_p1 import (
    M5R3P1Build,
    M5R3P1Error,
    build_m5_r3_p1_compressor_prompt,
    build_m5_r3_p1_dataset,
    check_m5_r3_p1_contamination,
    generate_m5_r3_p1_contexts,
    m5_r3_p1_stage_seed,
)
from tinyllm.data.m5_r3_p1_schema import M5R3P1StageGeneration, M5R3P1TaskContext
from tinyllm.data.m5_r3_source_strategy import (
    load_m5_r3_teacher_source_strategy_config,
)
from tinyllm.data.reasoning_schema import ReasoningTask
from tinyllm.data.tokenization import OffsetTokenizer, TokenEncoding

CONFIG = Path("configs/data/m5_r3_teacher_source_strategy.yaml")


class _WordTokenizer:
    def encode(self, text: str) -> TokenEncoding:
        words = tuple(text.split())
        return TokenEncoding(
            ids=tuple(range(len(words))),
            offsets=tuple((index, index + 1) for index in range(len(words))),
        )


def _frozen_tasks(count: int, *, split: str) -> tuple[ReasoningTask, ...]:
    tasks: list[ReasoningTask] = []
    for index in range(count):
        prompt = f"{split} fixture prompt {index}"
        answer = '{"issue":"missing_checkpoint"}'
        tasks.append(
            ReasoningTask(
                id=f"m5-reasoning:{'dev' if split == 'dev' else 'pilot'}:{split}-{index:03d}",
                split="reasoning_dev" if split == "dev" else "pilot_train",
                task_family="config",
                language="en",
                template_family=(
                    "dev.config.fixture.v1"
                    if split == "dev"
                    else "pilot.config.historical-fixture.v1"
                ),
                prompt=prompt,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                expected_answer_json=answer,
                expected_answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
            )
        )
    return tuple(tasks)


def _parent_tasks() -> tuple[tuple[ReasoningTask, ...], tuple[ReasoningTask, ...]]:
    p0 = load_m5_r3_p0_config(Path("configs/data/m5_r3_p0.yaml"))
    r1 = load_m5_r3_p0_config(Path("configs/data/m5_r3_p0_r1.yaml"))
    return generate_m5_r3_p0_tasks(p0), generate_m5_r3_p0_tasks(r1)


def _generations(*, reasoning_override: str | None = None) -> tuple[M5R3P1StageGeneration, ...]:
    config = load_m5_r3_teacher_source_strategy_config(CONFIG)
    records: list[M5R3P1StageGeneration] = []
    for index, context in enumerate(generate_m5_r3_p1_contexts(config)):
        solver_output = (
            f"<think>solve {context.task.id}</think>\n\n{context.task.expected_answer_json}"
        )
        solver = M5R3P1StageGeneration(
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
        reasoning = (
            reasoning_override
            if reasoning_override is not None
            else f"{context.evidence_anchor} directly supports {context.expected_label}."
        )
        compressor_output = json.dumps(
            {
                "reasoning": reasoning,
                "final_answer": json.loads(context.task.expected_answer_json),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        compressor_prompt = build_m5_r3_p1_compressor_prompt(
            context,
            solver_reasoning=f"solve {context.task.id}",
            verified_final_answer=context.task.expected_answer_json,
        )
        compressor = M5R3P1StageGeneration(
            generation_id=f"{context.task.id}:compressor",
            task_id=context.task.id,
            stage="compressor",
            seed=m5_r3_p1_stage_seed(config.pilot.compressor.base_seed, index),
            prompt_sha256=hashlib.sha256(compressor_prompt.encode()).hexdigest(),
            status="succeeded",
            finish_reason="stop",
            raw_output=compressor_output,
            raw_output_sha256=hashlib.sha256(compressor_output.encode()).hexdigest(),
            input_token_count=128,
            generated_token_count=32,
        )
        records.extend((solver, compressor))
    return tuple(records)


def _build(generations: tuple[M5R3P1StageGeneration, ...]) -> M5R3P1Build:
    config = load_m5_r3_teacher_source_strategy_config(CONFIG)
    p0, r1 = _parent_tasks()
    return build_m5_r3_p1_dataset(
        generate_m5_r3_p1_contexts(config),
        generations,
        config=config,
        dev_tasks=_frozen_tasks(200, split="dev"),
        historical_tasks=_frozen_tasks(100, split="historical"),
        p0_tasks=p0,
        p0_r1_tasks=r1,
        tokenizer=cast(OffsetTokenizer, cast(Any, _WordTokenizer())),
    )


def test_p1_tasks_are_balanced_deterministic_and_disjoint() -> None:
    config = load_m5_r3_teacher_source_strategy_config(CONFIG)
    contexts = generate_m5_r3_p1_contexts(config)
    p0, r1 = _parent_tasks()
    contamination = check_m5_r3_p1_contamination(
        contexts,
        dev_tasks=_frozen_tasks(200, split="dev"),
        historical_tasks=_frozen_tasks(100, split="historical"),
        p0_tasks=p0,
        p0_r1_tasks=r1,
    )

    assert len(contexts) == 40
    assert sum(item.task.task_family == "config" for item in contexts) == 20
    assert sum(item.task.language == "en" for item in contexts) == 28
    assert sum(item.task.language == "zh" for item in contexts) == 12
    assert all(item.evidence in item.task.prompt for item in contexts)
    assert contamination.status == "pass"
    assert contamination.task_set_sha256 == (
        "7aed1d35698b39b60027454e0e29976a6415d867da51f93e423ca50959d7df3d"
    )


def test_p1_private_artifacts_survive_strict_json_round_trip() -> None:
    config = load_m5_r3_teacher_source_strategy_config(CONFIG)
    context = generate_m5_r3_p1_contexts(config)[0]
    generation = _generations()[0]

    decoded_context = json.loads(json.dumps(context.to_dict(), sort_keys=True))
    decoded_generation = json.loads(json.dumps(generation.to_dict(), sort_keys=True))

    assert (
        M5R3P1TaskContext.model_validate_json(json.dumps(decoded_context, sort_keys=True))
        == context
    )
    assert (
        M5R3P1StageGeneration.model_validate_json(json.dumps(decoded_generation, sort_keys=True))
        == generation
    )


def test_p1_synthetic_two_stage_build_passes_source_and_control_gates() -> None:
    build = _build(_generations())

    assert len(build.samples) == 40
    assert len(build.audits) == 40
    assert all(item.status == "accepted" for item in build.audits)
    assert all(item.gate_passed for item in build.family_results)
    assert build.control.status == "pass"
    assert build.control.structural_passes == 40
    assert build.control.training_source_authorized is False
    assert build.rejection_counts == {}


def test_p1_rejects_compressor_without_exact_evidence_anchor() -> None:
    build = _build(_generations(reasoning_override="short rationale with no direct anchor"))

    assert not build.samples
    assert build.rejection_counts == {"missing_evidence_anchor": 40}
    assert all(not item.gate_passed for item in build.family_results)


def test_p1_rejects_solver_lineage_drift() -> None:
    generations = list(_generations())
    generations[0] = generations[0].model_copy(update={"seed": 1})

    with pytest.raises(M5R3P1Error, match="solver lineage differs"):
        _build(tuple(generations))


def test_p1_rejects_parent_task_collision() -> None:
    config = load_m5_r3_teacher_source_strategy_config(CONFIG)
    contexts = generate_m5_r3_p1_contexts(config)
    p0, r1 = _parent_tasks()

    collision = contexts[0].task.model_copy(
        update={"template_family": contexts[0].task.template_family}
    )
    report = check_m5_r3_p1_contamination(
        contexts,
        dev_tasks=_frozen_tasks(200, split="dev"),
        historical_tasks=_frozen_tasks(100, split="historical"),
        p0_tasks=(collision, *p0[1:]),
        p0_r1_tasks=r1,
    )

    assert report.status == "fail"
    assert report.p0_exact_prompt_matches == 1
    assert report.p0_normalized_prompt_matches == 1
    assert report.p0_template_family_overlaps == 1


def test_p1_runner_cli_freezes_private_inputs_and_outputs() -> None:
    args = build_parser().parse_args(
        [
            "--historical-pilot-artifact",
            "/private/pilot.json",
            "--model-dir",
            "/private/model/revision",
            "--tokenizer-dir",
            "/private/tokenizer",
            "--gpu-index",
            "7",
            "--raw-output",
            "/private/raw.json",
            "--public-output",
            "reports/m5/raw/m5_r3_p1.json",
        ]
    )

    assert args.config == CONFIG
    assert args.gpu_index == 7
    assert args.policy_python == Path(".venv/bin/python")
    assert args.timeout_seconds == 7200
    assert args.public_output == Path("reports/m5/raw/m5_r3_p1.json")


def test_p1_runner_rejects_non_pinned_model_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "b968826d9c46dd6066d109eabc6255188de91218"
    snapshot.mkdir()

    with pytest.raises(M5R3P1EnvironmentError, match="snapshot is incomplete"):
        _verify_model_directory(
            snapshot,
            "b968826d9c46dd6066d109eabc6255188de91218",
        )
