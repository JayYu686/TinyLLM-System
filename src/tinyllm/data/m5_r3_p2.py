"""Parent-bound fallback selection and isolated compression for M5.2-R3 P2."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from tinyllm.data.m5_r3_p1 import (
    M5R3P1StagePromptKey,
    M5R3P1StageSeedKey,
    m5_r3_p1_stage_seed,
)
from tinyllm.data.m5_r3_p1_schema import (
    M5R3P1StageGeneration,
    M5R3P1TaskContext,
)
from tinyllm.data.m5_r3_p2_schema import (
    M5R3P2Config,
    M5R3P2FallbackReason,
)
from tinyllm.data.m5_r3_source_strategy_schema import M5R3TeacherSourceStrategyConfig
from tinyllm.data.reasoning import parse_teacher_output
from tinyllm.data.reasoning_schema import canonical_json, content_sha256


class M5R3P2Error(ValueError):
    """Raised when a P2 parent, delta, prompt, or lineage invariant differs."""


@dataclass(frozen=True, slots=True)
class M5R3P2Selection:
    """Selected parent/fallback solvers plus isolated compressor generations."""

    generations: tuple[M5R3P1StageGeneration, ...]
    expected_stage_seeds: dict[M5R3P1StageSeedKey, int]
    expected_stage_prompt_sha256: dict[M5R3P1StagePromptKey, str]
    fallback_trigger_counts: dict[M5R3P2FallbackReason, int]
    fallback_task_ids: tuple[str, ...]


def load_m5_r3_p2_config(path: Path) -> M5R3P2Config:
    """Load one strict P2 YAML contract."""

    if path.suffix not in {".yaml", ".yml"}:
        raise M5R3P2Error("M5 R3 P2 config must use YAML")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return M5R3P2Config.model_validate(payload)
    except OSError as exc:
        raise M5R3P2Error("M5 R3 P2 config cannot be read") from exc
    except yaml.YAMLError as exc:
        raise M5R3P2Error("M5 R3 P2 config is invalid YAML") from exc
    except ValidationError as exc:
        raise M5R3P2Error("M5 R3 P2 config violates its schema") from exc


def m5_r3_p2_config_sha256(config: M5R3P2Config) -> str:
    """Hash the resolved P2 configuration without path identity."""

    return content_sha256(config.to_dict())


def build_m5_r3_p2_fallback_solver_prompt(context: M5R3P1TaskContext) -> str:
    """Request a concise second Thinking candidate while retaining closed labels."""

    labels = ", ".join(context.allowed_labels)
    if context.task.language == "en":
        return (
            "Analyze only the direct evidence below. Think briefly, stop after identifying one "
            "decisive clue, and do not compare or enumerate alternatives.\n\n"
            f"Evidence:\n{context.evidence}\n\n"
            f"Choose {context.label_key} from exactly one of {labels}. After the brief reasoning, "
            f'return exactly {{"{context.label_key}":"selected_value"}}.'
        )
    return (
        "只分析下面的直接证据。推理应简短，找到一个决定性线索后立即停止，不要比较或枚举"
        "其他选项。\n\n"
        f"证据：\n{context.evidence}\n\n"
        f"{context.label_key} 必须且只能从 {labels} 中选择一个。简短推理后严格返回"
        f'{{"{context.label_key}":"所选值"}}。'
    )


def build_m5_r3_p2_isolated_compressor_prompt(
    context: M5R3P1TaskContext,
    solver_reasoning: str,
    verified_final_answer: str,
) -> str:
    """Build a compressor prompt that excludes raw reasoning and alternative labels."""

    del solver_reasoning
    if verified_final_answer != context.task.expected_answer_json:
        raise M5R3P2Error("M5 R3 P2 compressor received an unverified answer")
    anchor_json = json.dumps(context.evidence_anchor, ensure_ascii=False)
    answer_json = json.dumps(
        json.loads(verified_final_answer),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "A separate solver has already verified the answer. Produce exactly one JSON object with "
        "keys reasoning and final_answer. The reasoning string must quote the required evidence "
        "anchor exactly, remain concise, and mention only the selected label "
        f"{context.expected_label}. "
        "Do not emit Markdown, thinking tags, comparisons, or additional keys.\n\n"
        f"Required evidence anchor: {anchor_json}\n"
        f"Verified final_answer: {answer_json}"
    )


def classify_m5_r3_p2_solver(
    context: M5R3P1TaskContext,
    generation: M5R3P1StageGeneration,
) -> M5R3P2FallbackReason | None:
    """Return the frozen solver rejection reason or accept the exact answer."""

    if generation.status == "failed":
        return "solver_runtime_error"
    if generation.finish_reason == "length":
        return "solver_length_limit"
    assert generation.raw_output is not None
    parsed, parse_reason = parse_teacher_output(generation.raw_output)
    if parsed is None or parse_reason is not None:
        return "solver_invalid_output"
    try:
        answer = canonical_json(json.loads(parsed.final_answer))
    except json.JSONDecodeError:
        return "solver_invalid_output"
    if answer != context.task.expected_answer_json:
        return "solver_answer_mismatch"
    return None


def _unique_by_task(
    records: Iterable[M5R3P1StageGeneration],
    *,
    stage: str,
) -> dict[str, M5R3P1StageGeneration]:
    selected = tuple(records)
    if any(item.stage != stage for item in selected):
        raise M5R3P2Error(f"M5 R3 P2 {stage} records contain another stage")
    by_task = {item.task_id: item for item in selected}
    if len(by_task) != len(selected):
        raise M5R3P2Error(f"M5 R3 P2 {stage} task IDs are duplicated")
    return by_task


def select_m5_r3_p2_generations(
    contexts: Iterable[M5R3P1TaskContext],
    parent_generations: Iterable[M5R3P1StageGeneration],
    fallback_solvers: Iterable[M5R3P1StageGeneration],
    isolated_compressors: Iterable[M5R3P1StageGeneration],
    *,
    p1_config: M5R3TeacherSourceStrategyConfig,
    p2_config: M5R3P2Config,
) -> M5R3P2Selection:
    """Select P1 or fallback solvers and bind every P2 stage seed and prompt."""

    ordered_contexts = tuple(sorted(contexts, key=lambda item: item.task.id))
    parent_solver_map = _unique_by_task(
        (item for item in parent_generations if item.stage == "solver"),
        stage="solver",
    )
    fallback_map = _unique_by_task(fallback_solvers, stage="solver")
    compressor_map = _unique_by_task(isolated_compressors, stage="compressor")
    task_ids = {item.task.id for item in ordered_contexts}
    if set(parent_solver_map) != task_ids or not set(fallback_map).issubset(task_ids):
        raise M5R3P2Error("M5 R3 P2 parent or fallback solver task set differs")

    selected: list[M5R3P1StageGeneration] = []
    expected_seeds: dict[M5R3P1StageSeedKey, int] = {}
    expected_prompts: dict[M5R3P1StagePromptKey, str] = {}
    fallback_reasons: Counter[M5R3P2FallbackReason] = Counter()
    fallback_task_ids: list[str] = []
    compressor_required: set[str] = set()

    for index, context in enumerate(ordered_contexts):
        task_id = context.task.id
        parent = parent_solver_map[task_id]
        if (
            parent.seed != m5_r3_p1_stage_seed(p1_config.pilot.solver.base_seed, index)
            or parent.prompt_sha256 != context.task.prompt_sha256
        ):
            raise M5R3P2Error("M5 R3 P2 parent solver lineage differs")
        parent_reason = classify_m5_r3_p2_solver(context, parent)
        if parent_reason is None:
            solver = parent
            if task_id in fallback_map:
                raise M5R3P2Error("M5 R3 P2 fallback exists for an accepted parent solver")
        else:
            fallback_reasons[parent_reason] += 1
            fallback_task_ids.append(task_id)
            try:
                solver = fallback_map[task_id]
            except KeyError as exc:
                raise M5R3P2Error("M5 R3 P2 required fallback solver is missing") from exc
            fallback_prompt = build_m5_r3_p2_fallback_solver_prompt(context)
            if (
                solver.seed != m5_r3_p1_stage_seed(p2_config.fallback_solver.base_seed, index)
                or solver.prompt_sha256 != hashlib.sha256(fallback_prompt.encode()).hexdigest()
            ):
                raise M5R3P2Error("M5 R3 P2 fallback solver lineage differs")

        selected.append(solver)
        expected_seeds[(task_id, "solver")] = solver.seed
        expected_prompts[(task_id, "solver")] = solver.prompt_sha256
        compressor_seed = m5_r3_p1_stage_seed(
            p2_config.isolated_compressor.base_seed,
            index,
        )
        compressor_prompt = build_m5_r3_p2_isolated_compressor_prompt(
            context,
            "",
            context.task.expected_answer_json,
        )
        expected_seeds[(task_id, "compressor")] = compressor_seed
        expected_prompts[(task_id, "compressor")] = hashlib.sha256(
            compressor_prompt.encode()
        ).hexdigest()
        if classify_m5_r3_p2_solver(context, solver) is None:
            compressor_required.add(task_id)
            try:
                compressor = compressor_map[task_id]
            except KeyError as exc:
                raise M5R3P2Error("M5 R3 P2 isolated compressor is missing") from exc
            if (
                compressor.seed != compressor_seed
                or compressor.prompt_sha256 != expected_prompts[(task_id, "compressor")]
            ):
                raise M5R3P2Error("M5 R3 P2 isolated compressor lineage differs")
            selected.append(compressor)

    if set(fallback_map) != set(fallback_task_ids) or set(compressor_map) != compressor_required:
        raise M5R3P2Error("M5 R3 P2 delta contains an unexpected generation")
    return M5R3P2Selection(
        generations=tuple(sorted(selected, key=lambda item: item.generation_id)),
        expected_stage_seeds=expected_seeds,
        expected_stage_prompt_sha256=expected_prompts,
        fallback_trigger_counts=dict(sorted(fallback_reasons.items())),
        fallback_task_ids=tuple(sorted(fallback_task_ids)),
    )
