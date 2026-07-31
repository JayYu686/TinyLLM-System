"""Deterministic tasks, verification, and selection for M5.2-R3 formal sources."""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import ValidationError

from tinyllm.data.m5_r3_formal_schema import (
    M5R3FormalContaminationReport,
    M5R3FormalSourceConfig,
    M5R3FormalStratumResult,
)
from tinyllm.data.m5_r3_p1 import (
    M5R3P1StagePromptKey,
    M5R3P1StageSeedKey,
    m5_r3_p1_stage_seed,
    select_m5_r3_two_stage_task,
)
from tinyllm.data.m5_r3_p1_schema import (
    M5R3P1CandidateAudit,
    M5R3P1RejectionReason,
    M5R3P1StageGeneration,
    M5R3P1TaskContext,
)
from tinyllm.data.m5_r3_p2 import (
    build_m5_r3_p2_fallback_solver_prompt,
    build_m5_r3_p2_isolated_compressor_prompt,
)
from tinyllm.data.m5_r3_schema import M5R3TargetFamily
from tinyllm.data.reasoning_schema import (
    ReasoningLanguage,
    ReasoningSample,
    ReasoningTask,
    canonical_json,
    content_sha256,
)
from tinyllm.data.tokenization import OffsetTokenizer


class M5R3FormalSourceError(ValueError):
    """Raised when formal source generation or selection violates its contract."""


@dataclass(frozen=True, slots=True)
class M5R3FormalBuild:
    """Verified accepted samples and deterministic 160-item selection."""

    contexts: tuple[M5R3P1TaskContext, ...]
    generations: tuple[M5R3P1StageGeneration, ...]
    samples: tuple[ReasoningSample, ...]
    selected_samples: tuple[ReasoningSample, ...]
    audits: tuple[M5R3P1CandidateAudit, ...]
    stratum_results: tuple[
        M5R3FormalStratumResult,
        M5R3FormalStratumResult,
        M5R3FormalStratumResult,
        M5R3FormalStratumResult,
    ]
    rejection_counts: dict[M5R3P1RejectionReason, int]
    contamination: M5R3FormalContaminationReport
    task_set_sha256: str
    accepted_samples_sha256: str
    selected_samples_sha256: str


_FORMAL_REFERENCE = re.compile(r"\b(?:CFG|LOG)-R3F-\d{6}\b", flags=re.IGNORECASE)
_STRATA: tuple[
    tuple[M5R3TargetFamily, ReasoningLanguage],
    tuple[M5R3TargetFamily, ReasoningLanguage],
    tuple[M5R3TargetFamily, ReasoningLanguage],
    tuple[M5R3TargetFamily, ReasoningLanguage],
] = (
    ("config", "en"),
    ("config", "zh"),
    ("log_diagnosis", "en"),
    ("log_diagnosis", "zh"),
)


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def load_m5_r3_formal_source_config(path: Path) -> M5R3FormalSourceConfig:
    """Load one strict YAML formal-source contract."""

    if path.suffix not in {".yaml", ".yml"}:
        raise M5R3FormalSourceError("M5 R3 formal config must use YAML")
    try:
        return M5R3FormalSourceConfig.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    except OSError as exc:
        raise M5R3FormalSourceError("M5 R3 formal config cannot be read") from exc
    except yaml.YAMLError as exc:
        raise M5R3FormalSourceError("M5 R3 formal config is invalid YAML") from exc
    except ValidationError as exc:
        raise M5R3FormalSourceError("M5 R3 formal config violates its schema") from exc


def m5_r3_formal_source_config_sha256(config: M5R3FormalSourceConfig) -> str:
    """Hash one resolved path-free formal-source config."""

    return content_sha256(config.to_dict())


def _formal_evidence(
    family: M5R3TargetFamily,
    label: str,
    variant: int,
) -> str:
    """Return one unique, direct, non-evaluation formal evidence fragment."""

    slot = f"formal_{variant:02d}"
    if family == "config":
        if label == "forbidden_truncation":
            return (
                f"sequence_limit: {512 + 128 * variant}\n"
                f"overflow_policy: truncate\nsource_slot: {slot}"
            )
        if label == "missing_checkpoint":
            return f"resume_mode: exact\ncheckpoint_path: null\nrun_slot: {slot}"
        if label == "unsupported_precision":
            return f"precision: bf16\ngpu_family: v100\nnode_slot: {slot}"
        if label == "world_size_mismatch":
            world_size = 2 + variant % 7
            return (
                f"world_size: {world_size}\nvisible_device_count: {world_size - 1}\n"
                f"topology_slot: {slot}"
            )
    else:
        if label == "collective_timeout":
            return (
                f"rank={variant % 8} collective=all_reduce status=timeout sequence={1000 + variant}"
            )
        if label == "cuda_oom":
            return f"CUDA out of memory requested_mib={256 + 32 * variant} allocation_id={slot}"
        if label == "disk_full":
            return f"checkpoint write failed errno=28 free_bytes=0 shard={slot}"
        if label == "non_finite_gradient":
            return f"step={200 + variant} grad_norm=nan finite_check=false slot={slot}"
    raise M5R3FormalSourceError("M5 R3 formal evidence label differs")


def _formal_context(
    *,
    family: M5R3TargetFamily,
    language: ReasoningLanguage,
    index: int,
    reference: str,
    evidence: str,
    label: str,
    allowed_labels: tuple[str, str, str, str],
) -> M5R3P1TaskContext:
    label_key = "issue" if family == "config" else "root_cause"
    short_family = "config" if family == "config" else "log"
    labels = ", ".join(allowed_labels)
    if language == "en":
        noun = "configuration fragment" if family == "config" else "training log"
        prompt = (
            f"Case {reference}. Analyze this synthetic {noun}:\n{evidence}\n"
            f"Choose {label_key} from exactly one of {labels}. Give a concise evidence-grounded "
            f'reason, then return {{"{label_key}":"selected_value"}}.'
        )
    else:
        noun = "配置片段" if family == "config" else "训练日志"
        prompt = (
            f"案例 {reference}。分析这段合成{noun}：\n{evidence}\n"
            f"{label_key} 必须且只能从 {labels} 中选择一个。请给出简洁且基于证据的理由，"
            f'再返回{{"{label_key}":"所选值"}}。'
        )
    answer = canonical_json({label_key: label})
    task = ReasoningTask(
        id=f"m5-reasoning:pilot:r3formal-{short_family}-{language}-{index:03d}",
        split="pilot_train",
        task_family=family,
        language=language,
        template_family=f"pilot.{family}.r3-two-stage-formal.v1",
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        expected_answer_json=answer,
        expected_answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
    )
    return M5R3P1TaskContext(
        task=task,
        evidence=evidence,
        evidence_anchor=_normalized(evidence),
        label_key=cast(Literal["issue", "root_cause"], label_key),
        allowed_labels=allowed_labels,
        expected_label=label,
    )


def generate_m5_r3_formal_contexts(
    config: M5R3FormalSourceConfig,
) -> tuple[M5R3P1TaskContext, ...]:
    """Generate 240 unique balanced formal contexts."""

    rng = random.Random(config.task_policy.task_seed)
    labels_by_family: dict[M5R3TargetFamily, tuple[str, str, str, str]] = {
        "config": (
            "forbidden_truncation",
            "missing_checkpoint",
            "unsupported_precision",
            "world_size_mismatch",
        ),
        "log_diagnosis": (
            "collective_timeout",
            "cuda_oom",
            "disk_full",
            "non_finite_gradient",
        ),
    }
    contexts: list[M5R3P1TaskContext] = []
    for family in config.task_policy.target_families:
        labels = labels_by_family[family]
        prefix = "CFG-R3F" if family == "config" else "LOG-R3F"
        for index in range(config.task_policy.tasks_per_family):
            language: ReasoningLanguage = (
                "en" if index < config.task_policy.language_counts_per_family["en"] else "zh"
            )
            label = labels[index % len(labels)]
            variant = index // len(labels)
            contexts.append(
                _formal_context(
                    family=family,
                    language=language,
                    index=index,
                    reference=f"{prefix}-{rng.randrange(1_000_000):06d}",
                    evidence=_formal_evidence(family, label, variant),
                    label=label,
                    allowed_labels=labels,
                )
            )
    ordered = tuple(sorted(contexts, key=lambda item: item.task.id))
    if (
        len(ordered) != 240
        or len({item.task.id for item in ordered}) != 240
        or len({item.task.prompt_sha256 for item in ordered}) != 240
        or len({item.evidence_anchor for item in ordered}) != 240
    ):
        raise M5R3FormalSourceError("M5 R3 formal task set is incomplete or duplicated")
    return ordered


def check_m5_r3_formal_contamination(
    contexts: Iterable[M5R3P1TaskContext],
    *,
    dev_tasks: Iterable[ReasoningTask],
    historical_tasks: Iterable[ReasoningTask],
    p0_tasks: Iterable[ReasoningTask],
    p0_r1_tasks: Iterable[ReasoningTask],
    p1_tasks: Iterable[ReasoningTask],
) -> M5R3FormalContaminationReport:
    """Check exact, normalized, and template overlap against frozen prior sources."""

    formal_contexts = tuple(sorted(contexts, key=lambda item: item.task.id))
    formal = tuple(item.task for item in formal_contexts)
    sources = {
        "dev": tuple(dev_tasks),
        "historical": tuple(historical_tasks),
        "p0": tuple(p0_tasks),
        "p0_r1": tuple(p0_r1_tasks),
        "p1": tuple(p1_tasks),
    }
    if len(formal) != 240 or tuple(len(items) for items in sources.values()) != (
        200,
        100,
        40,
        40,
        40,
    ):
        raise M5R3FormalSourceError("M5 R3 formal contamination input sizes differ")
    exact = {item.prompt_sha256 for item in formal}
    normalized = {_normalized(_FORMAL_REFERENCE.sub("<case>", item.prompt)) for item in formal}
    templates = {item.template_family for item in formal}

    def counts(tasks: tuple[ReasoningTask, ...]) -> tuple[int, int, int]:
        return (
            len(exact & {item.prompt_sha256 for item in tasks}),
            len(
                normalized
                & {_normalized(_FORMAL_REFERENCE.sub("<case>", item.prompt)) for item in tasks}
            ),
            len(templates & {item.template_family for item in tasks}),
        )

    values = {name: counts(tasks) for name, tasks in sources.items()}
    flattened = tuple(value for counts_ in values.values() for value in counts_)
    return M5R3FormalContaminationReport(
        algorithm="m5-r3-formal-exact-normalized-template-v1",
        task_set_sha256=content_sha256([item.to_dict() for item in formal_contexts]),
        formal_task_count=240,
        dev_task_count=200,
        historical_pilot_task_count=100,
        parent_p0_task_count=40,
        parent_p0_r1_task_count=40,
        parent_p1_task_count=40,
        dev_exact_prompt_matches=values["dev"][0],
        dev_template_family_overlaps=values["dev"][2],
        historical_exact_prompt_matches=values["historical"][0],
        historical_normalized_prompt_matches=values["historical"][1],
        historical_template_family_overlaps=values["historical"][2],
        p0_exact_prompt_matches=values["p0"][0],
        p0_normalized_prompt_matches=values["p0"][1],
        p0_template_family_overlaps=values["p0"][2],
        p0_r1_exact_prompt_matches=values["p0_r1"][0],
        p0_r1_normalized_prompt_matches=values["p0_r1"][1],
        p0_r1_template_family_overlaps=values["p0_r1"][2],
        p1_exact_prompt_matches=values["p1"][0],
        p1_normalized_prompt_matches=values["p1"][1],
        p1_template_family_overlaps=values["p1"][2],
        status="pass" if sum(flattened) == 0 else "fail",
    )


def _selection_key(
    sample: ReasoningSample,
    audit_by_task: dict[str, M5R3P1CandidateAudit],
) -> tuple[int, int, str]:
    audit = audit_by_task[sample.task_id]
    assert audit.reasoning_tokens is not None
    assert audit.repeated_8gram_basis_points is not None
    return audit.reasoning_tokens, audit.repeated_8gram_basis_points, sample.id


def build_m5_r3_formal_source(
    contexts: Iterable[M5R3P1TaskContext],
    generations: Iterable[M5R3P1StageGeneration],
    *,
    config: M5R3FormalSourceConfig,
    dev_tasks: Iterable[ReasoningTask],
    historical_tasks: Iterable[ReasoningTask],
    p0_tasks: Iterable[ReasoningTask],
    p0_r1_tasks: Iterable[ReasoningTask],
    p1_tasks: Iterable[ReasoningTask],
    tokenizer: OffsetTokenizer,
) -> M5R3FormalBuild:
    """Verify generated sources and select the frozen 56/24 strata."""

    ordered_contexts = tuple(sorted(contexts, key=lambda item: item.task.id))
    if ordered_contexts != generate_m5_r3_formal_contexts(config):
        raise M5R3FormalSourceError("M5 R3 formal contexts differ from the frozen generator")
    contamination = check_m5_r3_formal_contamination(
        ordered_contexts,
        dev_tasks=dev_tasks,
        historical_tasks=historical_tasks,
        p0_tasks=p0_tasks,
        p0_r1_tasks=p0_r1_tasks,
        p1_tasks=p1_tasks,
    )
    if contamination.status != "pass":
        raise M5R3FormalSourceError("M5 R3 formal contamination Gate failed")

    ordered_generations = tuple(sorted(generations, key=lambda item: item.generation_id))
    if len({item.generation_id for item in ordered_generations}) != len(ordered_generations):
        raise M5R3FormalSourceError("M5 R3 formal generation IDs are duplicated")
    grouped: dict[str, dict[str, M5R3P1StageGeneration]] = defaultdict(dict)
    for item in ordered_generations:
        grouped[item.task_id][item.stage] = item
    task_ids = {item.task.id for item in ordered_contexts}
    if set(grouped) - task_ids:
        raise M5R3FormalSourceError("M5 R3 formal generations contain an unknown task")

    seeds: dict[M5R3P1StageSeedKey, int] = {}
    prompts: dict[M5R3P1StagePromptKey, str] = {}
    for index, context in enumerate(ordered_contexts):
        solver_prompt = build_m5_r3_p2_fallback_solver_prompt(context)
        compressor_prompt = build_m5_r3_p2_isolated_compressor_prompt(
            context,
            "",
            context.task.expected_answer_json,
        )
        seeds[(context.task.id, "solver")] = m5_r3_p1_stage_seed(config.solver.base_seed, index)
        seeds[(context.task.id, "compressor")] = m5_r3_p1_stage_seed(
            config.compressor.base_seed, index
        )
        prompts[(context.task.id, "solver")] = hashlib.sha256(solver_prompt.encode()).hexdigest()
        prompts[(context.task.id, "compressor")] = hashlib.sha256(
            compressor_prompt.encode()
        ).hexdigest()

    samples: list[ReasoningSample] = []
    audits: list[M5R3P1CandidateAudit] = []
    trace_hashes: set[str] = set()
    for context in ordered_contexts:
        records = cast(
            dict[Literal["solver", "compressor"], M5R3P1StageGeneration],
            grouped.get(context.task.id, {}),
        )
        sample, audit, trace_hash = select_m5_r3_two_stage_task(
            context,
            records,
            trace_policy=config.trace_policy,
            tokenizer=tokenizer,
            existing_trace_hashes=frozenset(trace_hashes),
            expected_stage_seeds=seeds,
            expected_stage_prompt_sha256=prompts,
            compressor_prompt_builder=build_m5_r3_p2_isolated_compressor_prompt,
        )
        audits.append(audit)
        if sample is not None:
            samples.append(sample)
            assert trace_hash is not None
            trace_hashes.add(trace_hash)

    frozen_samples = tuple(sorted(samples, key=lambda item: item.id))
    frozen_audits = tuple(sorted(audits, key=lambda item: item.task_id))
    audit_by_task = {item.task_id: item for item in frozen_audits}
    selected: list[ReasoningSample] = []
    strata: list[M5R3FormalStratumResult] = []
    for family, language in _STRATA:
        candidates = sorted(
            (
                item
                for item in frozen_samples
                if item.task_family == family and item.language == language
            ),
            key=lambda item: _selection_key(item, audit_by_task),
        )
        required = config.selection.selected_languages_per_family[family][language]
        chosen = candidates[:required]
        selected.extend(chosen)
        strata.append(
            M5R3FormalStratumResult(
                task_family=family,
                language=language,
                input_tasks=84 if language == "en" else 36,
                accepted_items=len(candidates),
                required_items=required,
                selected_items=len(chosen),
                gate_passed=len(chosen) == required,
            )
        )
    selected_samples = tuple(sorted(selected, key=lambda item: item.id))
    rejection_counts: Counter[M5R3P1RejectionReason] = Counter(
        cast(M5R3P1RejectionReason, item.rejection_reason)
        for item in frozen_audits
        if item.status == "rejected"
    )
    return M5R3FormalBuild(
        contexts=ordered_contexts,
        generations=ordered_generations,
        samples=frozen_samples,
        selected_samples=selected_samples,
        audits=frozen_audits,
        stratum_results=cast(
            tuple[
                M5R3FormalStratumResult,
                M5R3FormalStratumResult,
                M5R3FormalStratumResult,
                M5R3FormalStratumResult,
            ],
            tuple(strata),
        ),
        rejection_counts=dict(sorted(rejection_counts.items())),
        contamination=contamination,
        task_set_sha256=content_sha256([item.to_dict() for item in ordered_contexts]),
        accepted_samples_sha256=content_sha256([item.to_dict() for item in frozen_samples]),
        selected_samples_sha256=content_sha256([item.to_dict() for item in selected_samples]),
    )
