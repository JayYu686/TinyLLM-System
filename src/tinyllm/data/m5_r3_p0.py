"""Deterministic tasks, contamination checks, and trace selection for M5.2-R3-P0."""

from __future__ import annotations

import hashlib
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from tinyllm.data.m5_r3_p0_schema import (
    M5R3P0CandidateAudit,
    M5R3P0Config,
    M5R3P0ContaminationReport,
    M5R3P0FamilyResult,
    M5R3P0RejectionReason,
)
from tinyllm.data.m5_r3_schema import M5R3TargetFamily
from tinyllm.data.reasoning import (
    parse_teacher_output,
    verify_reasoning_answer,
)
from tinyllm.data.reasoning_schema import (
    M5ReasoningDataConfig,
    ReasoningLanguage,
    ReasoningSample,
    ReasoningTask,
    ReasoningVerifierResult,
    TeacherGenerationRecord,
    canonical_json,
    content_sha256,
)
from tinyllm.data.schema import ImportedMessage
from tinyllm.data.tokenization import (
    OffsetTokenizer,
    render_qwen3_thinking,
)


class M5R3P0Error(ValueError):
    """Raised when an R3-P0 contract or private input fails closed."""


@dataclass(frozen=True, slots=True)
class M5R3P0TaskSelection:
    """Selection result for one task and its generated candidates."""

    sample: ReasoningSample | None
    audits: tuple[M5R3P0CandidateAudit, ...]
    verifications: tuple[ReasoningVerifierResult, ...]
    normalized_trace_sha256: str | None


@dataclass(frozen=True, slots=True)
class M5R3P0Build:
    """Fully validated private P0 dataset and content-free aggregate evidence."""

    tasks: tuple[ReasoningTask, ...]
    generations: tuple[TeacherGenerationRecord, ...]
    samples: tuple[ReasoningSample, ...]
    candidate_audits: tuple[M5R3P0CandidateAudit, ...]
    verifications: tuple[ReasoningVerifierResult, ...]
    contamination: M5R3P0ContaminationReport
    family_results: tuple[M5R3P0FamilyResult, M5R3P0FamilyResult]
    rejection_counts: dict[M5R3P0RejectionReason, int]
    task_set_sha256: str
    samples_sha256: str


_CASE_REFERENCE = re.compile(r"\b(?:CFG|LOG)(?:-R3P0)?-\d{6}\b", flags=re.IGNORECASE)
_TARGET_FAMILY_ORDER: tuple[M5R3TargetFamily, M5R3TargetFamily] = (
    "config",
    "log_diagnosis",
)

_CONFIG_EVIDENCE: dict[str, tuple[str, ...]] = {
    "unsupported_precision": (
        "precision: bf16\ndevice: v100",
        "compute_dtype: bfloat16\naccelerator: Tesla V100",
        "mixed_precision: bf16\ngpu_architecture: volta",
        "use_bf16: true\nhardware_family: v100",
        "precision_policy: bf16\ncuda_device: V100-SXM2",
        "trainer_dtype: bfloat16\ngpu_capability: sm_70",
    ),
    "world_size_mismatch": (
        "world_size: 4\ngpu_ids: [4, 5]",
        "num_processes: 8\nvisible_devices: [0, 1, 2, 3]",
        "distributed_world_size: 2\nlocal_ranks: [0]",
        "expected_ranks: 6\nallocated_gpu_ids: [2, 3, 4, 5]",
        "process_count: 3\ndevices: [6, 7]",
        "world_size: 5\ncuda_visible_devices: 0,1,2,3",
    ),
    "forbidden_truncation": (
        "max_sequence_length: 1024\ntruncate_overlength: true",
        "sequence_length: 2048\noverlength_policy: truncate",
        "max_tokens: 4096\nclip_long_samples: true",
        "context_window: 1024\ndataset_overflow: truncate_tail",
        "packing_length: 2048\nallow_sample_truncation: true",
        "max_position: 1024\non_overlength: crop",
    ),
    "missing_checkpoint": (
        "resume_mode: exact\ncheckpoint: null",
        "resume: exact\ncheckpoint_path: ''",
        "restore_policy: exact\ncheckpoint_uri: null",
        "resume_kind: exact\nsource_checkpoint: none",
        "recovery_mode: exact\ncheckpoint_id: ''",
        "exact_resume: true\nresume_from: null",
    ),
}

_LOG_EVIDENCE: dict[str, tuple[str, ...]] = {
    "non_finite_gradient": (
        "step=91 loss=2.1\nstep=92 loss=nan grad_norm=inf",
        "optimizer_step=17 loss=inf gradient_norm=nan",
        "train_step=203 finite_loss=false grad_norm=inf",
        "step=44 loss=nan first_bad_tensor=decoder.layers.3",
        "iteration=12 loss_scale=1 grad_norm=nan",
        "step=8 loss=inf skipped_update=true",
    ),
    "disk_full": (
        "checkpoint write failed: errno=28",
        "save_state failed: No space left on device",
        "cannot write optimizer.pt: disk quota exhausted",
        "checkpoint commit aborted: filesystem free_bytes=0",
        "safetensors export failed with ENOSPC",
        "write manifest.json failed: device has no remaining space",
    ),
    "collective_timeout": (
        "rank=2 watchdog collective timeout",
        "NCCL operation AllReduce exceeded timeout on rank 1",
        "ProcessGroupNCCL watchdog detected stalled collective",
        "rank=3 collective sequence 71 timed out",
        "broadcast operation exceeded distributed timeout",
        "NCCL work item remained pending beyond watchdog limit",
    ),
    "cuda_oom": (
        "CUDA out of memory. Tried to allocate 512 MiB",
        "torch.cuda.OutOfMemoryError while allocating activation",
        "GPU allocator failed: requested_bytes=1073741824",
        "CUDA OOM at attention forward; free memory 64 MiB",
        "cannot allocate optimizer tensor on CUDA device",
        "device memory exhausted during backward pass",
    ),
}


def load_m5_r3_p0_config(path: Path) -> M5R3P0Config:
    """Load the strict P0 YAML config with no silent coercion beyond YAML sequences."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise M5R3P0Error("M5 R3 P0 config must use YAML")
    try:
        decoded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        return M5R3P0Config.model_validate(decoded)
    except OSError as exc:
        raise M5R3P0Error("M5 R3 P0 config cannot be read") from exc
    except yaml.YAMLError as exc:
        raise M5R3P0Error("M5 R3 P0 config is invalid YAML") from exc
    except ValidationError as exc:
        raise M5R3P0Error("M5 R3 P0 config violates its schema") from exc


def m5_r3_p0_config_sha256(config: M5R3P0Config) -> str:
    """Hash the canonical parsed P0 configuration."""

    return content_sha256(config.to_dict())


def _make_task(
    *,
    family: M5R3TargetFamily,
    language: ReasoningLanguage,
    index: int,
    reference: str,
    evidence: str,
    label: str,
) -> ReasoningTask:
    key = "issue" if family == "config" else "root_cause"
    labels = (
        "unsupported_precision, world_size_mismatch, forbidden_truncation, missing_checkpoint"
        if family == "config"
        else "non_finite_gradient, disk_full, collective_timeout, cuda_oom"
    )
    if language == "en":
        noun = "configuration fragment" if family == "config" else "training log"
        prompt = (
            f"Case {reference}. Analyze this synthetic {noun}:\n{evidence}\n"
            f"Choose {key} from exactly one of {labels}. Reason briefly without repeating "
            f"observations; keep visible reasoning under 192 tokens. Return only "
            f'{{"{key}":"selected_value"}} after reasoning, replacing selected_value.'
        )
    else:
        noun = "配置片段" if family == "config" else "训练日志"
        prompt = (
            f"案例 {reference}。分析这段合成{noun}：\n{evidence}\n"
            f"{key} 必须且只能从 {labels} 中选择一个。请简洁推理，不要重复观察，可见推理"
            f'不超过 192 Token；推理后只返回 {{"{key}":"所选值"}}，并替换占位文字。'
        )
    answer = canonical_json({key: label})
    short_family = "config" if family == "config" else "log"
    task_id = f"m5-reasoning:pilot:r3p0-{short_family}-{language}-{index:03d}"
    return ReasoningTask(
        id=task_id,
        split="pilot_train",
        task_family=family,
        language=language,
        template_family=f"pilot.{family}.r3-targeted.v2",
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        expected_answer_json=answer,
        expected_answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
    )


def generate_m5_r3_p0_tasks(config: M5R3P0Config) -> tuple[ReasoningTask, ...]:
    """Generate the fixed 40-task Config/Log P0 set with real evidence variation."""

    rng = random.Random(config.task_seed)
    tasks: list[ReasoningTask] = []
    evidence_by_family = {
        "config": _CONFIG_EVIDENCE,
        "log_diagnosis": _LOG_EVIDENCE,
    }
    prefixes = {"config": "CFG-R3P0", "log_diagnosis": "LOG-R3P0"}
    for family in config.target_families:
        evidence_map = evidence_by_family[family]
        labels = tuple(evidence_map)
        if any(
            len(values) < config.evidence_variants_per_label for values in evidence_map.values()
        ):
            raise M5R3P0Error("M5 R3 P0 evidence library is below the frozen diversity gate")
        for index in range(config.tasks_per_family):
            language: ReasoningLanguage = (
                "en" if index < config.language_counts_per_family["en"] else "zh"
            )
            label = labels[index % len(labels)]
            variant = (index // len(labels)) % config.evidence_variants_per_label
            reference = f"{prefixes[family]}-{rng.randrange(1_000_000):06d}"
            tasks.append(
                _make_task(
                    family=family,
                    language=language,
                    index=index,
                    reference=reference,
                    evidence=evidence_map[label][variant],
                    label=label,
                )
            )
    ordered = tuple(sorted(tasks, key=lambda item: item.id))
    if (
        len(ordered) != 40
        or len({item.id for item in ordered}) != 40
        or len({item.prompt_sha256 for item in ordered}) != 40
    ):
        raise M5R3P0Error("M5 R3 P0 task set is incomplete or duplicated")
    return ordered


def _normalize_prompt(prompt: str) -> str:
    return " ".join(_CASE_REFERENCE.sub("<case>", prompt).casefold().split())


def check_m5_r3_p0_contamination(
    tasks: Iterable[ReasoningTask],
    *,
    dev_tasks: Iterable[ReasoningTask],
    historical_tasks: Iterable[ReasoningTask],
) -> M5R3P0ContaminationReport:
    """Check exact, normalized, and template collisions against both frozen sets."""

    p0 = tuple(sorted(tasks, key=lambda item: item.id))
    dev = tuple(sorted(dev_tasks, key=lambda item: item.id))
    historical = tuple(sorted(historical_tasks, key=lambda item: item.id))
    if len(p0) != 40 or len(dev) != 200 or len(historical) != 100:
        raise M5R3P0Error("M5 R3 P0 contamination inputs have unexpected sizes")
    p0_prompts = {item.prompt_sha256 for item in p0}
    p0_templates = {item.template_family for item in p0}
    p0_normalized = {_normalize_prompt(item.prompt) for item in p0}
    dev_exact = len(p0_prompts & {item.prompt_sha256 for item in dev})
    dev_templates = len(p0_templates & {item.template_family for item in dev})
    historical_exact = len(p0_prompts & {item.prompt_sha256 for item in historical})
    historical_normalized = len(
        p0_normalized & {_normalize_prompt(item.prompt) for item in historical}
    )
    historical_templates = len(p0_templates & {item.template_family for item in historical})
    total = (
        dev_exact + dev_templates + historical_exact + historical_normalized + historical_templates
    )
    return M5R3P0ContaminationReport(
        algorithm="m5-r3-exact-normalized-template-v1",
        p0_tasks_sha256=content_sha256([item.to_dict() for item in p0]),
        p0_task_count=40,
        dev_task_count=200,
        historical_pilot_task_count=100,
        dev_exact_prompt_matches=dev_exact,
        dev_template_family_overlaps=dev_templates,
        historical_exact_prompt_matches=historical_exact,
        historical_normalized_prompt_matches=historical_normalized,
        historical_template_family_overlaps=historical_templates,
        status="pass" if total == 0 else "fail",
    )


def m5_r3_p0_generation_seed(
    base_seed: int,
    task_index: int,
    candidate_index: int,
) -> int:
    """Return the stable non-overlapping seed for one P0 Teacher candidate."""

    if task_index < 0 or candidate_index not in {0, 1}:
        raise M5R3P0Error("invalid M5 R3 P0 task or candidate index")
    return (base_seed + task_index * 2 + candidate_index) % (2**32)


def _trace_metrics(token_ids: tuple[int, ...], text: str) -> tuple[int, int]:
    windows = tuple(tuple(token_ids[index : index + 8]) for index in range(len(token_ids) - 7))
    repeated = round((len(windows) - len(set(windows))) * 10_000 / len(windows)) if windows else 0
    lines = tuple(line.strip().casefold() for line in text.splitlines() if line.strip())
    maximum_line_repeat = max(Counter(lines).values()) if lines else 1
    return repeated, maximum_line_repeat


def _rejected_audit(
    generation: TeacherGenerationRecord,
    reason: M5R3P0RejectionReason,
    *,
    reasoning_tokens: int | None = None,
    repeated_8gram_basis_points: int | None = None,
    maximum_line_repeat: int | None = None,
    normalized_trace_sha256: str | None = None,
    training_sequence_tokens: int | None = None,
    verification_id: str | None = None,
) -> M5R3P0CandidateAudit:
    return M5R3P0CandidateAudit(
        task_id=generation.task_id,
        generation_id=generation.generation_id,
        status="rejected",
        rejection_reason=reason,
        reasoning_tokens=reasoning_tokens,
        repeated_8gram_basis_points=repeated_8gram_basis_points,
        max_identical_line_hash_repetitions=maximum_line_repeat,
        normalized_trace_sha256=normalized_trace_sha256,
        training_sequence_tokens=training_sequence_tokens,
        verification_id=verification_id,
    )


def select_m5_r3_p0_candidate(
    task: ReasoningTask,
    generations: Iterable[TeacherGenerationRecord],
    *,
    config: M5R3P0Config,
    reasoning_config: M5ReasoningDataConfig,
    tokenizer: OffsetTokenizer,
    existing_trace_hashes: frozenset[str],
) -> M5R3P0TaskSelection:
    """Select the first verified concise unique trace without repairing output."""

    ordered = tuple(sorted(generations, key=lambda item: item.candidate_index))
    audits: list[M5R3P0CandidateAudit] = []
    verifications: list[ReasoningVerifierResult] = []
    for generation in ordered:
        if generation.task_id != task.id or generation.prompt_sha256 != task.prompt_sha256:
            raise M5R3P0Error("M5 R3 P0 generation identity differs from its task")
        if generation.status == "failed":
            audits.append(_rejected_audit(generation, "generation_runtime_error"))
            continue
        if generation.finish_reason == "length":
            audits.append(_rejected_audit(generation, "teacher_length_limit"))
            continue
        if generation.raw_output is None:
            raise M5R3P0Error("successful M5 R3 P0 generation lost raw output")
        parsed, parse_reason = parse_teacher_output(generation.raw_output)
        if parse_reason is not None:
            audits.append(
                _rejected_audit(
                    generation,
                    cast(M5R3P0RejectionReason, parse_reason),
                )
            )
            continue
        if parsed is None:
            raise M5R3P0Error("M5 R3 P0 parser returned no result")
        verification = verify_reasoning_answer(
            task=task,
            generation=generation,
            final_answer=parsed.final_answer,
            config=reasoning_config,
        )
        verifications.append(verification)
        if not verification.passed:
            reason: M5R3P0RejectionReason = (
                "invalid_final_json"
                if verification.reason == "invalid_final_json"
                else "answer_mismatch"
            )
            audits.append(
                _rejected_audit(
                    generation,
                    reason,
                    verification_id=verification.verification_id,
                )
            )
            continue
        reasoning_ids = tokenizer.encode(parsed.reasoning_content).ids
        if not reasoning_ids:
            audits.append(_rejected_audit(generation, "empty_reasoning"))
            continue
        repeated, maximum_line_repeat = _trace_metrics(
            reasoning_ids,
            parsed.reasoning_content,
        )
        normalized = " ".join(parsed.reasoning_content.split()).casefold()
        normalized_hash = hashlib.sha256(normalized.encode()).hexdigest()
        rendered = render_qwen3_thinking(
            (
                ImportedMessage(role="user", content=task.prompt),
                ImportedMessage(role="assistant", content=parsed.final_answer),
            ),
            assistant_reasoning=(parsed.reasoning_content,),
        )
        sequence_tokens = len(tokenizer.encode(rendered.text).ids)
        metrics = {
            "reasoning_tokens": len(reasoning_ids),
            "repeated_8gram_basis_points": repeated,
            "maximum_line_repeat": maximum_line_repeat,
            "normalized_trace_sha256": normalized_hash,
            "training_sequence_tokens": sequence_tokens,
            "verification_id": verification.verification_id,
        }
        rejection: M5R3P0RejectionReason | None = None
        if len(reasoning_ids) > config.trace_policy.max_reasoning_tokens:
            rejection = "reasoning_over_192_tokens"
        elif repeated > config.trace_policy.max_repeated_8gram_basis_points:
            rejection = "repeated_8gram_over_500bp"
        elif maximum_line_repeat > config.trace_policy.max_identical_line_hash_repetitions:
            rejection = "identical_line_repetition"
        elif normalized_hash in existing_trace_hashes:
            rejection = "duplicate_normalized_trace"
        elif sequence_tokens > config.max_sequence_length:
            rejection = "sequence_over_1024_tokens"
        if rejection is not None:
            audits.append(
                _rejected_audit(
                    generation,
                    rejection,
                    reasoning_tokens=cast(int, metrics["reasoning_tokens"]),
                    repeated_8gram_basis_points=cast(
                        int,
                        metrics["repeated_8gram_basis_points"],
                    ),
                    maximum_line_repeat=cast(int, metrics["maximum_line_repeat"]),
                    normalized_trace_sha256=cast(
                        str,
                        metrics["normalized_trace_sha256"],
                    ),
                    training_sequence_tokens=cast(
                        int,
                        metrics["training_sequence_tokens"],
                    ),
                    verification_id=cast(str, metrics["verification_id"]),
                )
            )
            continue
        sample_payload = {
            "final_answer": parsed.final_answer,
            "prompt": task.prompt,
            "reasoning_content": parsed.reasoning_content,
        }
        sample = ReasoningSample(
            id=f"m5-reasoning-sample:{task.id.removeprefix('m5-reasoning:pilot:')}",
            task_id=task.id,
            task_family=task.task_family,
            language=task.language,
            split="pilot_train",
            template_family=task.template_family,
            prompt=task.prompt,
            reasoning_content=parsed.reasoning_content,
            final_answer=parsed.final_answer,
            generation_id=generation.generation_id,
            verification_id=verification.verification_id,
            prompt_sha256=task.prompt_sha256,
            raw_output_sha256=cast(str, generation.raw_output_sha256),
            content_sha256=content_sha256(sample_payload),
            observed_token_count=generation.observed_token_count,
        )
        audits.append(
            M5R3P0CandidateAudit(
                task_id=task.id,
                generation_id=generation.generation_id,
                status="accepted",
                rejection_reason=None,
                reasoning_tokens=len(reasoning_ids),
                repeated_8gram_basis_points=repeated,
                max_identical_line_hash_repetitions=maximum_line_repeat,
                normalized_trace_sha256=normalized_hash,
                training_sequence_tokens=sequence_tokens,
                verification_id=verification.verification_id,
            )
        )
        return M5R3P0TaskSelection(
            sample=sample,
            audits=tuple(audits),
            verifications=tuple(verifications),
            normalized_trace_sha256=normalized_hash,
        )
    return M5R3P0TaskSelection(
        sample=None,
        audits=tuple(audits),
        verifications=tuple(verifications),
        normalized_trace_sha256=None,
    )


def _family_result(
    family: M5R3TargetFamily,
    *,
    tasks: tuple[ReasoningTask, ...],
    samples: tuple[ReasoningSample, ...],
    audits: tuple[M5R3P0CandidateAudit, ...],
) -> M5R3P0FamilyResult:
    if sum(item.task_family == family for item in tasks) != 20:
        raise M5R3P0Error("M5 R3 P0 family input count differs")
    family_samples = tuple(item for item in samples if item.task_family == family)
    accepted_ids = {item.generation_id for item in family_samples}
    accepted_audits = tuple(item for item in audits if item.generation_id in accepted_ids)
    lengths = sorted(cast(int, item.reasoning_tokens) for item in accepted_audits)
    repeated = [cast(int, item.repeated_8gram_basis_points) for item in accepted_audits]
    languages: Counter[ReasoningLanguage] = Counter(item.language for item in family_samples)
    if lengths:
        p90_index = math.ceil(0.9 * len(lengths)) - 1
        minimum: int | None = lengths[0]
        median: float | None = float(statistics.median(lengths))
        p90: int | None = lengths[p90_index]
        maximum: int | None = lengths[-1]
        repeated_mean: int | None = round(sum(repeated) / len(repeated))
    else:
        minimum = median = p90 = maximum = repeated_mean = None
    return M5R3P0FamilyResult(
        task_family=family,
        input_tasks=20,
        input_language_counts={"en": 14, "zh": 6},
        accepted_items=len(family_samples),
        accepted_language_counts={"en": languages["en"], "zh": languages["zh"]},
        reasoning_tokens_min=minimum,
        reasoning_tokens_p50=median,
        reasoning_tokens_p90=p90,
        reasoning_tokens_max=maximum,
        repeated_8gram_mean_basis_points=repeated_mean,
        gate_passed=(len(family_samples) >= 14 and languages["en"] >= 10 and languages["zh"] >= 4),
    )


def build_m5_r3_p0_dataset(
    tasks: Iterable[ReasoningTask],
    generations: Iterable[TeacherGenerationRecord],
    *,
    config: M5R3P0Config,
    reasoning_config: M5ReasoningDataConfig,
    dev_tasks: Iterable[ReasoningTask],
    historical_tasks: Iterable[ReasoningTask],
    tokenizer: OffsetTokenizer,
) -> M5R3P0Build:
    """Build deterministic P0 evidence from generated candidates."""

    ordered_tasks = tuple(sorted(tasks, key=lambda item: item.id))
    if ordered_tasks != generate_m5_r3_p0_tasks(config):
        raise M5R3P0Error("M5 R3 P0 tasks differ from the frozen generator")
    contamination = check_m5_r3_p0_contamination(
        ordered_tasks,
        dev_tasks=dev_tasks,
        historical_tasks=historical_tasks,
    )
    if contamination.status != "pass":
        raise M5R3P0Error("M5 R3 P0 contamination Gate failed")
    ordered_generations = tuple(sorted(generations, key=lambda item: item.generation_id))
    if len({item.generation_id for item in ordered_generations}) != len(ordered_generations):
        raise M5R3P0Error("M5 R3 P0 generation IDs are duplicated")
    grouped: dict[str, list[TeacherGenerationRecord]] = defaultdict(list)
    for generation in ordered_generations:
        grouped[generation.task_id].append(generation)
    samples: list[ReasoningSample] = []
    audits: list[M5R3P0CandidateAudit] = []
    verifications: list[ReasoningVerifierResult] = []
    trace_hashes: set[str] = set()
    rejected_tasks = 0
    for task in ordered_tasks:
        records = tuple(sorted(grouped.get(task.id, ()), key=lambda item: item.candidate_index))
        if len(records) > config.sampling.candidate_count or tuple(
            item.candidate_index for item in records
        ) != tuple(range(len(records))):
            raise M5R3P0Error("M5 R3 P0 candidates must use contiguous indices 0..1")
        selection = select_m5_r3_p0_candidate(
            task,
            records,
            config=config,
            reasoning_config=reasoning_config,
            tokenizer=tokenizer,
            existing_trace_hashes=frozenset(trace_hashes),
        )
        audits.extend(selection.audits)
        verifications.extend(selection.verifications)
        if selection.sample is None:
            rejected_tasks += 1
        else:
            samples.append(selection.sample)
            trace_hashes.add(cast(str, selection.normalized_trace_sha256))
    rejection_counts: Counter[M5R3P0RejectionReason] = Counter(
        cast(M5R3P0RejectionReason, item.rejection_reason)
        for item in audits
        if item.status == "rejected"
    )
    if rejected_tasks:
        rejection_counts["no_candidate_passed"] += rejected_tasks
    frozen_samples = tuple(sorted(samples, key=lambda item: item.id))
    frozen_audits = tuple(sorted(audits, key=lambda item: (item.task_id, item.generation_id)))
    family_results = cast(
        tuple[M5R3P0FamilyResult, M5R3P0FamilyResult],
        tuple(
            _family_result(
                family,
                tasks=ordered_tasks,
                samples=frozen_samples,
                audits=frozen_audits,
            )
            for family in _TARGET_FAMILY_ORDER
        ),
    )
    return M5R3P0Build(
        tasks=ordered_tasks,
        generations=ordered_generations,
        samples=frozen_samples,
        candidate_audits=frozen_audits,
        verifications=tuple(sorted(verifications, key=lambda item: item.verification_id)),
        contamination=contamination,
        family_results=family_results,
        rejection_counts=dict(sorted(rejection_counts.items())),
        task_set_sha256=content_sha256([item.to_dict() for item in ordered_tasks]),
        samples_sha256=content_sha256([item.to_dict() for item in frozen_samples]),
    )
