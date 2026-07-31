"""Deterministic tasks and two-stage selection for M5.2-R3 P1."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import ValidationError

from tinyllm.data.m5_r3_p0 import m5_r3_target_evidence_library
from tinyllm.data.m5_r3_p1_schema import (
    M5R3P1CandidateAudit,
    M5R3P1CompressedEnvelope,
    M5R3P1ContaminationReport,
    M5R3P1ControlResult,
    M5R3P1FamilyResult,
    M5R3P1RejectionReason,
    M5R3P1Stage,
    M5R3P1StageGeneration,
    M5R3P1TaskContext,
)
from tinyllm.data.m5_r3_schema import M5R3TargetFamily
from tinyllm.data.m5_r3_source_strategy_schema import (
    M5R3P1TracePolicy,
    M5R3TeacherSourceStrategyConfig,
)
from tinyllm.data.reasoning import parse_teacher_output
from tinyllm.data.reasoning_schema import (
    ReasoningLanguage,
    ReasoningSample,
    ReasoningTask,
    canonical_json,
    content_sha256,
)
from tinyllm.data.schema import ImportedMessage
from tinyllm.data.tokenization import OffsetTokenizer, render_qwen3_thinking


class M5R3P1Error(ValueError):
    """Raised when a P1 task, generation, or Gate violates its contract."""


@dataclass(frozen=True, slots=True)
class M5R3P1Build:
    """Validated P1 samples and path-free aggregate evidence."""

    contexts: tuple[M5R3P1TaskContext, ...]
    generations: tuple[M5R3P1StageGeneration, ...]
    samples: tuple[ReasoningSample, ...]
    audits: tuple[M5R3P1CandidateAudit, ...]
    contamination: M5R3P1ContaminationReport
    family_results: tuple[M5R3P1FamilyResult, M5R3P1FamilyResult]
    control: M5R3P1ControlResult
    rejection_counts: dict[M5R3P1RejectionReason, int]
    task_set_sha256: str
    samples_sha256: str


_CASE_REFERENCE = re.compile(
    r"\b(?:CFG|LOG)-R3P(?:0(?:R1)?|1)-\d{6}\b",
    flags=re.IGNORECASE,
)
_FAMILY_ORDER: tuple[M5R3TargetFamily, M5R3TargetFamily] = (
    "config",
    "log_diagnosis",
)
_STAGE_ORDER: tuple[M5R3P1Stage, M5R3P1Stage] = ("solver", "compressor")


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _make_context(
    *,
    family: M5R3TargetFamily,
    language: ReasoningLanguage,
    index: int,
    reference: str,
    evidence: str,
    label: str,
    allowed_labels: tuple[str, str, str, str],
) -> M5R3P1TaskContext:
    label_key: LiteralLabelKey = "issue" if family == "config" else "root_cause"
    short_family = "config" if family == "config" else "log"
    labels = ", ".join(allowed_labels)
    noun = (
        ("configuration fragment" if family == "config" else "training log")
        if language == "en"
        else ("配置片段" if family == "config" else "训练日志")
    )
    if language == "en":
        prompt = (
            f"Case {reference}. Analyze this synthetic {noun}:\n{evidence}\n"
            f"Choose {label_key} from exactly one of {labels}. Solve the task carefully and "
            f'return {{"{label_key}":"selected_value"}} after reasoning.'
        )
    else:
        prompt = (
            f"案例 {reference}。分析这段合成{noun}：\n{evidence}\n"
            f"{label_key} 必须且只能从 {labels} 中选择一个。请先完成分析，再返回"
            f'{{"{label_key}":"所选值"}}。'
        )
    answer = canonical_json({label_key: label})
    task = ReasoningTask(
        id=f"m5-reasoning:pilot:r3p1-{short_family}-{language}-{index:03d}",
        split="pilot_train",
        task_family=family,
        language=language,
        template_family=f"pilot.{family}.r3-two-stage-p1.v1",
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        expected_answer_json=answer,
        expected_answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
    )
    return M5R3P1TaskContext(
        task=task,
        evidence=evidence,
        evidence_anchor=_normalized(evidence),
        label_key=label_key,
        allowed_labels=allowed_labels,
        expected_label=label,
    )


LiteralLabelKey = Literal["issue", "root_cause"]
M5R3P1StageSeedKey = tuple[str, M5R3P1Stage]
M5R3P1StagePromptKey = tuple[str, M5R3P1Stage]
M5R3P1CompressorPromptBuilder = Callable[[M5R3P1TaskContext, str, str], str]


def generate_m5_r3_p1_contexts(
    config: M5R3TeacherSourceStrategyConfig,
) -> tuple[M5R3P1TaskContext, ...]:
    """Generate the fixed balanced 40-task P1 task/context set."""

    rng = random.Random(config.pilot.task_seed)
    library = m5_r3_target_evidence_library()
    contexts: list[M5R3P1TaskContext] = []
    for family in config.pilot.target_families:
        evidence_map = library[family]
        labels = cast(tuple[str, str, str, str], tuple(sorted(evidence_map)))
        if any(
            len(values) < config.pilot.evidence_variants_per_label
            for values in evidence_map.values()
        ):
            raise M5R3P1Error("M5 R3 P1 evidence library is below the diversity gate")
        prefix = "CFG-R3P1" if family == "config" else "LOG-R3P1"
        for index in range(config.pilot.tasks_per_family):
            language: ReasoningLanguage = (
                "en" if index < config.pilot.language_counts_per_family["en"] else "zh"
            )
            label = labels[index % len(labels)]
            variant = (index // len(labels)) % config.pilot.evidence_variants_per_label
            contexts.append(
                _make_context(
                    family=family,
                    language=language,
                    index=index,
                    reference=f"{prefix}-{rng.randrange(1_000_000):06d}",
                    evidence=evidence_map[label][variant],
                    label=label,
                    allowed_labels=labels,
                )
            )
    ordered = tuple(sorted(contexts, key=lambda item: item.task.id))
    if (
        len(ordered) != 40
        or len({item.task.id for item in ordered}) != 40
        or len({item.task.prompt_sha256 for item in ordered}) != 40
    ):
        raise M5R3P1Error("M5 R3 P1 task set is incomplete or duplicated")
    return ordered


def m5_r3_p1_stage_seed(base_seed: int, task_index: int) -> int:
    """Return one stable per-stage task seed."""

    if task_index < 0:
        raise M5R3P1Error("M5 R3 P1 task index must be non-negative")
    return (base_seed + task_index) % (2**32)


def build_m5_r3_p1_compressor_prompt(
    context: M5R3P1TaskContext,
    *,
    solver_reasoning: str,
    verified_final_answer: str,
) -> str:
    """Build the strict private compression request without executing model content."""

    return (
        "Compress the verified solution below into one short evidence-grounded rationale. "
        "Return exactly one JSON object with keys reasoning and final_answer; final_answer "
        f"must be a JSON object. Quote this evidence anchor exactly: {context.evidence_anchor}. "
        f"Use the selected label {context.expected_label} and do not mention other labels. "
        "Do not emit Markdown or thinking tags.\n\n"
        f"Original task:\n{context.task.prompt}\n\n"
        f"Verified solver reasoning:\n{solver_reasoning}\n\n"
        f"Verified final answer:\n{verified_final_answer}"
    )


def _default_compressor_prompt_builder(
    context: M5R3P1TaskContext,
    solver_reasoning: str,
    verified_final_answer: str,
) -> str:
    """Adapt the keyword-only P1 prompt builder to the shared selector interface."""

    return build_m5_r3_p1_compressor_prompt(
        context,
        solver_reasoning=solver_reasoning,
        verified_final_answer=verified_final_answer,
    )


def check_m5_r3_p1_contamination(
    contexts: Iterable[M5R3P1TaskContext],
    *,
    dev_tasks: Iterable[ReasoningTask],
    historical_tasks: Iterable[ReasoningTask],
    p0_tasks: Iterable[ReasoningTask],
    p0_r1_tasks: Iterable[ReasoningTask],
) -> M5R3P1ContaminationReport:
    """Check P1 against every frozen M5 task source."""

    p1 = tuple(sorted((item.task for item in contexts), key=lambda item: item.id))
    sources = {
        "dev": tuple(dev_tasks),
        "historical": tuple(historical_tasks),
        "p0": tuple(p0_tasks),
        "p0_r1": tuple(p0_r1_tasks),
    }
    if (
        tuple(len(sources[key]) for key in ("dev", "historical", "p0", "p0_r1"))
        != (
            200,
            100,
            40,
            40,
        )
        or len(p1) != 40
    ):
        raise M5R3P1Error("M5 R3 P1 contamination input sizes differ")
    p1_exact = {item.prompt_sha256 for item in p1}
    p1_normalized = {_normalized(_CASE_REFERENCE.sub("<case>", item.prompt)) for item in p1}
    p1_templates = {item.template_family for item in p1}

    def counts(tasks: tuple[ReasoningTask, ...]) -> tuple[int, int, int]:
        return (
            len(p1_exact & {item.prompt_sha256 for item in tasks}),
            len(
                p1_normalized
                & {_normalized(_CASE_REFERENCE.sub("<case>", item.prompt)) for item in tasks}
            ),
            len(p1_templates & {item.template_family for item in tasks}),
        )

    dev_exact, _dev_normalized, dev_templates = counts(sources["dev"])
    historical_exact, historical_normalized, historical_templates = counts(sources["historical"])
    p0_exact, p0_normalized, p0_templates = counts(sources["p0"])
    p0_r1_exact, p0_r1_normalized, p0_r1_templates = counts(sources["p0_r1"])
    all_counts = (
        dev_exact,
        dev_templates,
        historical_exact,
        historical_normalized,
        historical_templates,
        p0_exact,
        p0_normalized,
        p0_templates,
        p0_r1_exact,
        p0_r1_normalized,
        p0_r1_templates,
    )
    return M5R3P1ContaminationReport(
        algorithm="m5-r3-p1-exact-normalized-template-v1",
        task_set_sha256=content_sha256([item.to_dict() for item in p1]),
        p1_task_count=40,
        dev_task_count=200,
        historical_pilot_task_count=100,
        parent_p0_task_count=40,
        parent_p0_r1_task_count=40,
        dev_exact_prompt_matches=dev_exact,
        dev_template_family_overlaps=dev_templates,
        historical_exact_prompt_matches=historical_exact,
        historical_normalized_prompt_matches=historical_normalized,
        historical_template_family_overlaps=historical_templates,
        p0_exact_prompt_matches=p0_exact,
        p0_normalized_prompt_matches=p0_normalized,
        p0_template_family_overlaps=p0_templates,
        p0_r1_exact_prompt_matches=p0_r1_exact,
        p0_r1_normalized_prompt_matches=p0_r1_normalized,
        p0_r1_template_family_overlaps=p0_r1_templates,
        status="pass" if sum(all_counts) == 0 else "fail",
    )


def _trace_metrics(token_ids: tuple[int, ...], text: str) -> tuple[int, int]:
    windows = tuple(tuple(token_ids[index : index + 8]) for index in range(len(token_ids) - 7))
    repeated = round((len(windows) - len(set(windows))) * 10_000 / len(windows)) if windows else 0
    lines = tuple(line.strip().casefold() for line in text.splitlines() if line.strip())
    return repeated, max(Counter(lines).values()) if lines else 1


def _parse_compressor_output(raw_output: str) -> M5R3P1CompressedEnvelope | None:
    try:
        decoded: Any = json.loads(raw_output)
        return M5R3P1CompressedEnvelope.model_validate(decoded)
    except (json.JSONDecodeError, ValidationError):
        return None


def _early_audit(
    context: M5R3P1TaskContext,
    reason: M5R3P1RejectionReason,
    *,
    solver_sha256: str | None = None,
    compressor_sha256: str | None = None,
) -> M5R3P1CandidateAudit:
    return M5R3P1CandidateAudit(
        task_id=context.task.id,
        task_family=cast(M5R3TargetFamily, context.task.task_family),
        language=context.task.language,
        status="rejected",
        rejection_reason=reason,
        solver_output_sha256=solver_sha256,
        compressor_output_sha256=compressor_sha256,
    )


def select_m5_r3_two_stage_task(
    context: M5R3P1TaskContext,
    records: dict[M5R3P1Stage, M5R3P1StageGeneration],
    *,
    trace_policy: M5R3P1TracePolicy,
    tokenizer: OffsetTokenizer,
    existing_trace_hashes: frozenset[str],
    expected_stage_seeds: Mapping[M5R3P1StageSeedKey, int],
    expected_stage_prompt_sha256: Mapping[M5R3P1StagePromptKey, str] | None,
    compressor_prompt_builder: M5R3P1CompressorPromptBuilder,
) -> tuple[ReasoningSample | None, M5R3P1CandidateAudit, str | None]:
    solver = records.get("solver")
    if solver is None:
        raise M5R3P1Error("M5 R3 P1 solver record is missing")
    expected_solver_prompt_sha256 = (
        context.task.prompt_sha256
        if expected_stage_prompt_sha256 is None
        else expected_stage_prompt_sha256[(context.task.id, "solver")]
    )
    if (
        solver.seed != expected_stage_seeds[(context.task.id, "solver")]
        or solver.prompt_sha256 != expected_solver_prompt_sha256
    ):
        raise M5R3P1Error("M5 R3 P1 solver lineage differs")
    if solver.status == "failed":
        return None, _early_audit(context, "solver_runtime_error"), None
    solver_sha = solver.raw_output_sha256
    if solver.finish_reason == "length":
        return None, _early_audit(context, "solver_length_limit", solver_sha256=solver_sha), None
    assert solver.raw_output is not None
    parsed, parse_reason = parse_teacher_output(solver.raw_output)
    if parsed is None or parse_reason is not None:
        return None, _early_audit(context, "solver_invalid_output", solver_sha256=solver_sha), None
    try:
        solver_answer = canonical_json(json.loads(parsed.final_answer))
    except json.JSONDecodeError:
        return None, _early_audit(context, "solver_invalid_output", solver_sha256=solver_sha), None
    if solver_answer != context.task.expected_answer_json:
        return None, _early_audit(context, "solver_answer_mismatch", solver_sha256=solver_sha), None

    compressor = records.get("compressor")
    if compressor is None:
        raise M5R3P1Error("M5 R3 P1 compressor record is missing after solver acceptance")
    compressor_prompt = compressor_prompt_builder(
        context,
        parsed.reasoning_content,
        solver_answer,
    )
    compressor_prompt_sha256 = hashlib.sha256(compressor_prompt.encode()).hexdigest()
    expected_compressor_prompt_sha256 = (
        compressor_prompt_sha256
        if expected_stage_prompt_sha256 is None
        else expected_stage_prompt_sha256[(context.task.id, "compressor")]
    )
    if (
        compressor.seed != expected_stage_seeds[(context.task.id, "compressor")]
        or compressor.prompt_sha256 != expected_compressor_prompt_sha256
        or compressor.prompt_sha256 != compressor_prompt_sha256
    ):
        raise M5R3P1Error("M5 R3 P1 compressor lineage differs")
    if compressor.status == "failed":
        return (
            None,
            _early_audit(
                context,
                "compressor_runtime_error",
                solver_sha256=solver_sha,
            ),
            None,
        )
    compressor_sha = compressor.raw_output_sha256
    if compressor.finish_reason == "length":
        return (
            None,
            _early_audit(
                context,
                "compressor_length_limit",
                solver_sha256=solver_sha,
                compressor_sha256=compressor_sha,
            ),
            None,
        )
    assert compressor.raw_output is not None
    envelope = _parse_compressor_output(compressor.raw_output)
    if envelope is None:
        return (
            None,
            _early_audit(
                context,
                "compressor_invalid_json",
                solver_sha256=solver_sha,
                compressor_sha256=compressor_sha,
            ),
            None,
        )
    final_answer = canonical_json(envelope.final_answer)
    if final_answer != context.task.expected_answer_json:
        return (
            None,
            _early_audit(
                context,
                "compressor_answer_mismatch",
                solver_sha256=solver_sha,
                compressor_sha256=compressor_sha,
            ),
            None,
        )
    reasoning_ids = tokenizer.encode(envelope.reasoning).ids
    if not reasoning_ids:
        return (
            None,
            _early_audit(
                context,
                "compressor_empty_reasoning",
                solver_sha256=solver_sha,
                compressor_sha256=compressor_sha,
            ),
            None,
        )
    normalized_reasoning = _normalized(envelope.reasoning)
    anchor_match = context.evidence_anchor in normalized_reasoning
    other_mentions = sum(
        label in normalized_reasoning
        for label in context.allowed_labels
        if label != context.expected_label
    )
    repeated, line_repeat = _trace_metrics(reasoning_ids, envelope.reasoning)
    trace_hash = hashlib.sha256(normalized_reasoning.encode()).hexdigest()
    rendered = render_qwen3_thinking(
        (
            ImportedMessage(role="user", content=context.task.prompt),
            ImportedMessage(role="assistant", content=final_answer),
        ),
        assistant_reasoning=(envelope.reasoning,),
    )
    sequence_tokens = len(tokenizer.encode(rendered.text).ids)
    rejection: M5R3P1RejectionReason | None = None
    if len(reasoning_ids) > trace_policy.max_reasoning_tokens:
        rejection = "reasoning_over_192_tokens"
    elif not anchor_match:
        rejection = "missing_evidence_anchor"
    elif other_mentions:
        rejection = "other_label_mentioned"
    elif repeated > trace_policy.max_repeated_8gram_basis_points:
        rejection = "repeated_8gram_over_500bp"
    elif line_repeat > trace_policy.max_identical_line_hash_repetitions:
        rejection = "identical_line_repetition"
    elif trace_hash in existing_trace_hashes:
        rejection = "duplicate_normalized_trace"
    elif sequence_tokens > trace_policy.max_training_sequence_tokens:
        rejection = "sequence_over_1024_tokens"

    def complete_audit(
        status: Literal["accepted", "rejected"],
        reason: M5R3P1RejectionReason | None,
    ) -> M5R3P1CandidateAudit:
        return M5R3P1CandidateAudit(
            task_id=context.task.id,
            task_family=cast(M5R3TargetFamily, context.task.task_family),
            language=context.task.language,
            status=status,
            rejection_reason=reason,
            solver_output_sha256=solver_sha,
            compressor_output_sha256=compressor_sha,
            reasoning_tokens=len(reasoning_ids),
            repeated_8gram_basis_points=repeated,
            max_identical_line_hash_repetitions=line_repeat,
            normalized_trace_sha256=trace_hash,
            evidence_anchor_matched=anchor_match,
            other_label_mentions=other_mentions,
            training_sequence_tokens=sequence_tokens,
        )

    if rejection is not None:
        return None, complete_audit("rejected", rejection), None
    sample_payload = {
        "final_answer": final_answer,
        "prompt": context.task.prompt,
        "reasoning_content": envelope.reasoning,
    }
    sample = ReasoningSample(
        id=f"m5-reasoning-sample:{context.task.id.removeprefix('m5-reasoning:pilot:')}",
        task_id=context.task.id,
        task_family=context.task.task_family,
        language=context.task.language,
        split="pilot_train",
        template_family=context.task.template_family,
        prompt=context.task.prompt,
        reasoning_content=envelope.reasoning,
        final_answer=final_answer,
        generation_id=f"{context.task.id}:candidate-0",
        verification_id=f"{context.task.id}:verify-0",
        prompt_sha256=context.task.prompt_sha256,
        raw_output_sha256=cast(str, compressor_sha),
        content_sha256=content_sha256(sample_payload),
        observed_token_count=sequence_tokens,
    )
    return (
        sample,
        complete_audit("accepted", None),
        trace_hash,
    )


def _family_result(
    family: M5R3TargetFamily,
    *,
    samples: tuple[ReasoningSample, ...],
    audits: tuple[M5R3P1CandidateAudit, ...],
) -> M5R3P1FamilyResult:
    family_samples = tuple(item for item in samples if item.task_family == family)
    accepted_ids = {item.task_id for item in family_samples}
    lengths = sorted(
        cast(int, item.reasoning_tokens) for item in audits if item.task_id in accepted_ids
    )
    languages: Counter[ReasoningLanguage] = Counter(item.language for item in family_samples)
    if lengths:
        minimum: int | None = lengths[0]
        median: float | None = float(statistics.median(lengths))
        p90: int | None = lengths[math.ceil(0.9 * len(lengths)) - 1]
        maximum: int | None = lengths[-1]
    else:
        minimum = median = p90 = maximum = None
    return M5R3P1FamilyResult(
        task_family=family,
        input_tasks=20,
        input_language_counts={"en": 14, "zh": 6},
        accepted_items=len(family_samples),
        accepted_language_counts={"en": languages["en"], "zh": languages["zh"]},
        reasoning_tokens_min=minimum,
        reasoning_tokens_p50=median,
        reasoning_tokens_p90=p90,
        reasoning_tokens_max=maximum,
        gate_passed=(len(family_samples) >= 14 and languages["en"] >= 10 and languages["zh"] >= 4),
    )


def build_m5_r3_p1_rule_control(
    contexts: Iterable[M5R3P1TaskContext],
    *,
    tokenizer: OffsetTokenizer,
) -> M5R3P1ControlResult:
    """Build content-free structural evidence for the non-training rule control."""

    lengths: list[int] = []
    hashes: set[str] = set()
    passes = 0
    for context in sorted(contexts, key=lambda item: item.task.id):
        if context.task.language == "en":
            trace = (
                f'The evidence "{context.evidence_anchor}" directly supports '
                f"{context.expected_label}."
            )
        else:
            trace = f"证据“{context.evidence_anchor}”直接表明 {context.expected_label}。"
        token_count = len(tokenizer.encode(trace).ids)
        trace_hash = hashlib.sha256(_normalized(trace).encode()).hexdigest()
        other_mentions = sum(
            label in _normalized(trace)
            for label in context.allowed_labels
            if label != context.expected_label
        )
        if (
            token_count <= 192
            and context.evidence_anchor in _normalized(trace)
            and not other_mentions
            and trace_hash not in hashes
        ):
            passes += 1
            hashes.add(trace_hash)
            lengths.append(token_count)
    return M5R3P1ControlResult(
        source_kind="deterministic_rule_trace",
        input_tasks=40,
        structural_passes=passes,
        reasoning_tokens_max=max(lengths, default=1),
        unique_trace_count=len(hashes),
        status="pass" if passes == len(hashes) == 40 else "fail",
        training_source_authorized=False,
    )


def build_m5_r3_p1_dataset(
    contexts: Iterable[M5R3P1TaskContext],
    generations: Iterable[M5R3P1StageGeneration],
    *,
    config: M5R3TeacherSourceStrategyConfig,
    dev_tasks: Iterable[ReasoningTask],
    historical_tasks: Iterable[ReasoningTask],
    p0_tasks: Iterable[ReasoningTask],
    p0_r1_tasks: Iterable[ReasoningTask],
    tokenizer: OffsetTokenizer,
    expected_stage_seeds: Mapping[M5R3P1StageSeedKey, int] | None = None,
    expected_stage_prompt_sha256: Mapping[M5R3P1StagePromptKey, str] | None = None,
    compressor_prompt_builder: M5R3P1CompressorPromptBuilder = (_default_compressor_prompt_builder),
) -> M5R3P1Build:
    """Build deterministic P1 evidence from private two-stage generations."""

    ordered_contexts = tuple(sorted(contexts, key=lambda item: item.task.id))
    if ordered_contexts != generate_m5_r3_p1_contexts(config):
        raise M5R3P1Error("M5 R3 P1 tasks differ from the frozen generator")
    contamination = check_m5_r3_p1_contamination(
        ordered_contexts,
        dev_tasks=dev_tasks,
        historical_tasks=historical_tasks,
        p0_tasks=p0_tasks,
        p0_r1_tasks=p0_r1_tasks,
    )
    if contamination.status != "pass":
        raise M5R3P1Error("M5 R3 P1 contamination Gate failed")
    ordered_generations = tuple(sorted(generations, key=lambda item: item.generation_id))
    if len({item.generation_id for item in ordered_generations}) != len(ordered_generations):
        raise M5R3P1Error("M5 R3 P1 generation IDs are duplicated")
    grouped: dict[str, dict[M5R3P1Stage, M5R3P1StageGeneration]] = defaultdict(dict)
    for item in ordered_generations:
        grouped[item.task_id][item.stage] = item
    task_ids = {item.task.id for item in ordered_contexts}
    if not set(grouped).issubset(task_ids):
        raise M5R3P1Error("M5 R3 P1 generations contain an unknown task")
    if expected_stage_seeds is None:
        resolved_stage_seeds: Mapping[M5R3P1StageSeedKey, int] = {
            (context.task.id, stage): m5_r3_p1_stage_seed(
                (
                    config.pilot.solver.base_seed
                    if stage == "solver"
                    else config.pilot.compressor.base_seed
                ),
                task_index,
            )
            for task_index, context in enumerate(ordered_contexts)
            for stage in _STAGE_ORDER
        }
    else:
        expected_keys = {
            (context.task.id, stage)
            for context in ordered_contexts
            for stage in ("solver", "compressor")
        }
        if set(expected_stage_seeds) != expected_keys:
            raise M5R3P1Error("M5 R3 P1 expected stage seed mapping differs")
        resolved_stage_seeds = expected_stage_seeds
    if expected_stage_prompt_sha256 is not None:
        expected_prompt_keys = {
            (context.task.id, stage)
            for context in ordered_contexts
            for stage in ("solver", "compressor")
        }
        if set(expected_stage_prompt_sha256) != expected_prompt_keys:
            raise M5R3P1Error("M5 R3 P1 expected stage prompt mapping differs")
    samples: list[ReasoningSample] = []
    audits: list[M5R3P1CandidateAudit] = []
    hashes: set[str] = set()
    for context in ordered_contexts:
        sample, audit, trace_hash = select_m5_r3_two_stage_task(
            context,
            grouped.get(context.task.id, {}),
            trace_policy=config.pilot.trace_policy,
            tokenizer=tokenizer,
            existing_trace_hashes=frozenset(hashes),
            expected_stage_seeds=resolved_stage_seeds,
            expected_stage_prompt_sha256=expected_stage_prompt_sha256,
            compressor_prompt_builder=compressor_prompt_builder,
        )
        audits.append(audit)
        if sample is not None:
            samples.append(sample)
            hashes.add(cast(str, trace_hash))
    frozen_samples = tuple(sorted(samples, key=lambda item: item.id))
    frozen_audits = tuple(sorted(audits, key=lambda item: item.task_id))
    family_results = cast(
        tuple[M5R3P1FamilyResult, M5R3P1FamilyResult],
        tuple(
            _family_result(family, samples=frozen_samples, audits=frozen_audits)
            for family in _FAMILY_ORDER
        ),
    )
    rejection_counts: Counter[M5R3P1RejectionReason] = Counter(
        cast(M5R3P1RejectionReason, item.rejection_reason)
        for item in frozen_audits
        if item.status == "rejected"
    )
    return M5R3P1Build(
        contexts=ordered_contexts,
        generations=ordered_generations,
        samples=frozen_samples,
        audits=frozen_audits,
        contamination=contamination,
        family_results=family_results,
        control=build_m5_r3_p1_rule_control(ordered_contexts, tokenizer=tokenizer),
        rejection_counts=dict(sorted(rejection_counts.items())),
        task_set_sha256=content_sha256([item.to_dict() for item in ordered_contexts]),
        samples_sha256=content_sha256([item.to_dict() for item in frozen_samples]),
    )
