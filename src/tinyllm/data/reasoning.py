"""Deterministic M5.1 reasoning task generation, parsing, verification, and selection."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from tinyllm.data.reasoning_schema import (
    REASONING_TASK_FAMILIES,
    M5ReasoningDataConfig,
    M5ReasoningDatasetManifest,
    ReasoningLanguage,
    ReasoningRejectedRecord,
    ReasoningRejectionReason,
    ReasoningRejectionStage,
    ReasoningSample,
    ReasoningSplit,
    ReasoningTask,
    ReasoningTaskFamily,
    ReasoningTaskSetManifest,
    ReasoningVerifierResult,
    TeacherGenerationRecord,
    VerifierReason,
    canonical_json,
    content_sha256,
)


class ReasoningDataError(ValueError):
    """Raised when M5.1 inputs violate an integrity or configuration boundary."""


@dataclass(frozen=True, slots=True)
class ParsedTeacherOutput:
    """One structurally valid visible reasoning trace and final answer."""

    reasoning_content: str
    final_answer: str


@dataclass(frozen=True, slots=True)
class ReasoningDatasetBuild:
    """Selected M5 pilot samples and all content-free audit evidence."""

    task_manifest: ReasoningTaskSetManifest
    manifest: M5ReasoningDatasetManifest
    samples: tuple[ReasoningSample, ...]
    verifications: tuple[ReasoningVerifierResult, ...]
    rejected: tuple[ReasoningRejectedRecord, ...]


def _sequence_hash(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = canonical_json(value).encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def load_m5_reasoning_data_config(path: Path) -> M5ReasoningDataConfig:
    """Load a strict M5.1 YAML configuration with concise validation errors."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ReasoningDataError("M5 reasoning data config must use a .yaml or .yml extension")
    try:
        decoded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReasoningDataError(f"cannot read M5 reasoning data config: {path}") from exc
    except yaml.YAMLError as exc:
        raise ReasoningDataError(f"invalid YAML in M5 reasoning data config: {path}") from exc
    try:
        return M5ReasoningDataConfig.model_validate(decoded)
    except ValidationError as exc:
        messages: list[str] = []
        for error in exc.errors(include_url=False, include_context=False):
            location = ".".join(str(part) for part in error["loc"])
            messages.append(f"{location}: {error['msg']}" if location else str(error["msg"]))
        raise ReasoningDataError("; ".join(messages)) from exc


def _make_task(
    *,
    namespace: str,
    task_family: ReasoningTaskFamily,
    language: ReasoningLanguage,
    index: int,
    prompt: str,
    answer: dict[str, object],
    template_name: str,
) -> ReasoningTask:
    prompt = (
        f"{prompt} Keep the visible reasoning under 192 tokens."
        if language == "en"
        else f"{prompt} 可见推理过程应控制在 192 个 Token 以内。"
    )
    answer_json = canonical_json(answer)
    split: ReasoningSplit = "pilot_train" if namespace == "pilot" else "reasoning_dev"
    return ReasoningTask(
        id=f"m5-reasoning:{namespace}:{task_family}-{language}-{index:03d}",
        split=split,
        task_family=task_family,
        language=language,
        template_family=f"{namespace}.{task_family}.{template_name}.v1",
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        expected_answer_json=answer_json,
        expected_answer_sha256=hashlib.sha256(answer_json.encode("utf-8")).hexdigest(),
    )


def _python_task(
    namespace: str,
    language: ReasoningLanguage,
    index: int,
    rng: random.Random,
) -> ReasoningTask:
    values = [rng.randint(1, 12) for _ in range(4)]
    offset = rng.randint(1, 5)
    result = sum(value * 2 if value % 2 == 0 else value + offset for value in values)
    if language == "en":
        prompt = (
            "Trace this deterministic rule without executing code: for each even value add twice "
            f"the value; for each odd value add the value plus {offset}. Values: {values}. "
            'Return only JSON in the form {"result": integer}.'
        )
    else:
        prompt = (
            f"不要执行代码，按规则推导：偶数计为其两倍，奇数计为该数加 {offset}。输入为 "
            f'{values}。只返回形如 {{"result": 整数}} 的 JSON。'
        )
    return _make_task(
        namespace=namespace,
        task_family="python",
        language=language,
        index=index,
        prompt=prompt,
        answer={"result": result},
        template_name="trace-accumulator",
    )


def _json_task(
    namespace: str,
    language: ReasoningLanguage,
    index: int,
    rng: random.Random,
) -> ReasoningTask:
    target = rng.randint(10, 999)
    position = rng.randint(0, 2)
    records = [{"id": item, "value": rng.randint(1, 50)} for item in range(3)]
    records[position]["value"] = target
    payload = canonical_json({"records": records, "target_id": position})
    if language == "en":
        prompt = (
            f"Read this JSON object: {payload}. Find the record whose id equals target_id. "
            'Return only canonical JSON in the form {"value": integer}.'
        )
    else:
        prompt = (
            f"读取 JSON 对象：{payload}。找到 id 等于 target_id 的记录，只返回形如 "
            '{"value": 整数} 的规范 JSON。'
        )
    return _make_task(
        namespace=namespace,
        task_family="json",
        language=language,
        index=index,
        prompt=prompt,
        answer={"value": target},
        template_name="nested-lookup",
    )


def _linux_task(namespace: str, language: ReasoningLanguage, index: int) -> ReasoningTask:
    cases = (
        ("cat: /srv/app/config.yaml: Permission denied", "permission_denied"),
        ("bash: tinyllm: command not found", "command_not_found"),
        ("No space left on device while writing checkpoint", "disk_full"),
        ("Connection refused while contacting 127.0.0.1:5000", "connection_refused"),
    )
    evidence, diagnosis = cases[index % len(cases)]
    if language == "en":
        prompt = (
            f"Diagnose the primary Linux failure from this synthetic message: {evidence}. "
            'Do not propose or execute a command. Return only {"diagnosis":"code"} as JSON.'
        )
    else:
        prompt = (
            f"根据这条合成 Linux 信息判断首要故障：{evidence}。不要提出或执行命令，只返回 "
            '{"diagnosis":"代码"} 形式的 JSON。'
        )
    return _make_task(
        namespace=namespace,
        task_family="linux",
        language=language,
        index=index,
        prompt=prompt,
        answer={"diagnosis": diagnosis},
        template_name="error-classification",
    )


def _config_task(namespace: str, language: ReasoningLanguage, index: int) -> ReasoningTask:
    cases = (
        ("precision: bf16\ndevice: v100", "unsupported_precision"),
        ("world_size: 4\ngpu_ids: [4, 5]", "world_size_mismatch"),
        ("max_sequence_length: 1024\ntruncate_overlength: true", "forbidden_truncation"),
        ("resume_mode: exact\ncheckpoint: null", "missing_checkpoint"),
    )
    snippet, issue = cases[index % len(cases)]
    if language == "en":
        prompt = (
            f"Inspect this synthetic YAML fragment:\n{snippet}\n"
            "Identify its single contract violation. "
            'Return only JSON in the form {"issue":"code"}.'
        )
    else:
        prompt = (
            f"检查这段合成 YAML：\n{snippet}\n识别唯一的契约错误，只返回 "
            '{"issue":"代码"} 形式的 JSON。'
        )
    return _make_task(
        namespace=namespace,
        task_family="config",
        language=language,
        index=index,
        prompt=prompt,
        answer={"issue": issue},
        template_name="contract-violation",
    )


def _log_task(namespace: str, language: ReasoningLanguage, index: int) -> ReasoningTask:
    cases = (
        ("step=91 loss=2.1\nstep=92 loss=nan grad_norm=inf", "non_finite_gradient"),
        ("checkpoint write failed: errno=28", "disk_full"),
        ("rank=2 watchdog collective timeout", "collective_timeout"),
        ("CUDA out of memory. Tried to allocate 512 MiB", "cuda_oom"),
    )
    log_text, root_cause = cases[index % len(cases)]
    if language == "en":
        prompt = (
            f"Find the primary root cause in this synthetic training log:\n{log_text}\n"
            'Return only JSON in the form {"root_cause":"code"}.'
        )
    else:
        prompt = (
            f"判断这段合成训练日志的首要根因：\n{log_text}\n只返回 "
            '{"root_cause":"代码"} 形式的 JSON。'
        )
    return _make_task(
        namespace=namespace,
        task_family="log_diagnosis",
        language=language,
        index=index,
        prompt=prompt,
        answer={"root_cause": root_cause},
        template_name="root-cause",
    )


def _task_for_family(
    namespace: str,
    task_family: ReasoningTaskFamily,
    language: ReasoningLanguage,
    index: int,
    rng: random.Random,
) -> ReasoningTask:
    if task_family == "python":
        return _python_task(namespace, language, index, rng)
    if task_family == "json":
        return _json_task(namespace, language, index, rng)
    if task_family == "linux":
        return _linux_task(namespace, language, index)
    if task_family == "config":
        return _config_task(namespace, language, index)
    return _log_task(namespace, language, index)


def generate_reasoning_dev_tasks(config: M5ReasoningDataConfig) -> tuple[ReasoningTask, ...]:
    """Build the frozen 200-item M5 Dev set with exact 5x40 and 70/30 item counts."""

    rng = random.Random(config.dev.seed)
    tasks: list[ReasoningTask] = []
    for task_family in REASONING_TASK_FAMILIES:
        for index in range(config.dev.task_family_counts[task_family]):
            language: ReasoningLanguage = (
                "en" if index < config.dev.language_counts_per_family["en"] else "zh"
            )
            tasks.append(_task_for_family("dev", task_family, language, index, rng))
    return tuple(sorted(tasks, key=lambda task: task.id))


def generate_reasoning_pilot_tasks(
    *,
    seed: int,
    tasks_per_family: int,
) -> tuple[ReasoningTask, ...]:
    """Build deterministic private-pilot task inputs using namespaces disjoint from Dev."""

    if tasks_per_family < 10 or tasks_per_family % 10 != 0:
        raise ReasoningDataError("pilot tasks per family must be a positive multiple of 10")
    rng = random.Random(seed)
    english_count = tasks_per_family * 7 // 10
    tasks: list[ReasoningTask] = []
    for task_family in REASONING_TASK_FAMILIES:
        for index in range(tasks_per_family):
            language: ReasoningLanguage = "en" if index < english_count else "zh"
            tasks.append(_task_for_family("pilot", task_family, language, index, rng))
    return tuple(sorted(tasks, key=lambda task: task.id))


def build_reasoning_task_manifest(
    tasks: Iterable[ReasoningTask], *, config: M5ReasoningDataConfig
) -> ReasoningTaskSetManifest:
    """Validate one split, unique identities, distributions, and content address."""

    ordered = tuple(sorted(tasks, key=lambda task: task.id))
    if not ordered:
        raise ReasoningDataError("reasoning task set cannot be empty")
    if len({task.id for task in ordered}) != len(ordered):
        raise ReasoningDataError("reasoning task IDs must be unique")
    splits = {task.split for task in ordered}
    if len(splits) != 1:
        raise ReasoningDataError("reasoning task set must contain exactly one split")
    split = ordered[0].split
    if split == "reasoning_dev":
        if len(ordered) != config.dev.total_tasks:
            raise ReasoningDataError("reasoning Dev must contain exactly 200 tasks")
        family_counts = Counter(task.task_family for task in ordered)
        language_by_family = Counter((task.task_family, task.language) for task in ordered)
        if dict(sorted(family_counts.items())) != config.dev.task_family_counts:
            raise ReasoningDataError("reasoning Dev family distribution does not match config")
        for task_family in REASONING_TASK_FAMILIES:
            for language in ("en", "zh"):
                expected = config.dev.language_counts_per_family[language]
                if language_by_family[(task_family, language)] != expected:
                    raise ReasoningDataError(
                        "reasoning Dev language distribution does not match config"
                    )
    tasks_sha256 = _sequence_hash(task.to_dict() for task in ordered)
    namespace = "pilot-tasks" if split == "pilot_train" else "dev"
    return ReasoningTaskSetManifest(
        task_set_version=f"m5-reasoning-{namespace}-v1-{tasks_sha256[:8]}",
        split=split,
        config_sha256=content_sha256(config.to_dict()),
        tasks_sha256=tasks_sha256,
        task_count=len(ordered),
        task_family_counts=dict(sorted(Counter(task.task_family for task in ordered).items())),
        language_counts=dict(sorted(Counter(task.language for task in ordered).items())),
        template_family_counts=dict(
            sorted(Counter(task.template_family for task in ordered).items())
        ),
    )


def parse_teacher_output(
    raw_output: str,
) -> tuple[ParsedTeacherOutput | None, ReasoningRejectionReason | None]:
    """Parse exactly one native visible Think block without repairing malformed output."""

    open_tag = "<think>"
    close_tag = "</think>"
    open_count = raw_output.count(open_tag)
    close_count = raw_output.count(close_tag)
    if open_count == 0 or close_count == 0:
        return None, "missing_think_block"
    if open_count != 1 or close_count != 1:
        return None, "multiple_think_blocks"
    open_index = raw_output.find(open_tag)
    close_index = raw_output.find(close_tag)
    if open_index > close_index or raw_output[:open_index].strip():
        return None, "nested_think_tag"
    reasoning = raw_output[open_index + len(open_tag) : close_index].strip()
    final_answer = raw_output[close_index + len(close_tag) :].strip()
    if final_answer.endswith("<|im_end|>"):
        final_answer = final_answer.removesuffix("<|im_end|>").strip()
    if open_tag in reasoning or close_tag in reasoning or "<think" in final_answer:
        return None, "nested_think_tag"
    if not reasoning:
        return None, "empty_reasoning"
    if not final_answer:
        return None, "empty_final_answer"
    return ParsedTeacherOutput(reasoning_content=reasoning, final_answer=final_answer), None


def verify_reasoning_answer(
    *,
    task: ReasoningTask,
    generation: TeacherGenerationRecord,
    final_answer: str,
    config: M5ReasoningDataConfig,
) -> ReasoningVerifierResult:
    """Compare canonical JSON objects without executing code from either field."""

    answer_sha256 = hashlib.sha256(final_answer.encode("utf-8")).hexdigest()
    try:
        decoded = json.loads(final_answer)
    except json.JSONDecodeError:
        reason: VerifierReason = "invalid_final_json"
        passed = False
    else:
        passed = isinstance(decoded, dict) and canonical_json(decoded) == task.expected_answer_json
        reason = "accepted" if passed else "answer_mismatch"
    return ReasoningVerifierResult(
        verification_id=f"{task.id}:verify-{generation.candidate_index}",
        task_id=task.id,
        generation_id=generation.generation_id,
        verifier_id=config.verifier.verifier_id,
        expected_answer_sha256=task.expected_answer_sha256,
        final_answer_sha256=answer_sha256,
        passed=passed,
        reason=reason,
    )


def _candidate_rejection(
    task: ReasoningTask,
    generation: TeacherGenerationRecord,
    *,
    stage: ReasoningRejectionStage,
    reason: ReasoningRejectionReason,
    config: M5ReasoningDataConfig,
    verification: ReasoningVerifierResult | None = None,
) -> ReasoningRejectedRecord:
    return ReasoningRejectedRecord(
        task_id=task.id,
        generation_id=generation.generation_id,
        stage=stage,
        reason=reason,
        prompt_sha256=task.prompt_sha256,
        raw_output_sha256=generation.raw_output_sha256,
        observed_token_count=generation.observed_token_count,
        max_sequence_length=(config.max_sequence_length if reason == "sequence_too_long" else None),
        verification_id=verification.verification_id if verification is not None else None,
    )


def _validate_generation_inputs(
    tasks: tuple[ReasoningTask, ...], generations: tuple[TeacherGenerationRecord, ...]
) -> dict[str, tuple[TeacherGenerationRecord, ...]]:
    task_by_id = {task.id: task for task in tasks}
    generation_ids = [generation.generation_id for generation in generations]
    if len(generation_ids) != len(set(generation_ids)):
        raise ReasoningDataError("teacher generation IDs must be unique")
    unknown = sorted({generation.task_id for generation in generations} - set(task_by_id))
    if unknown:
        raise ReasoningDataError(f"teacher generation references unknown task: {unknown[0]}")
    grouped: dict[str, list[TeacherGenerationRecord]] = defaultdict(list)
    for generation in generations:
        task = task_by_id[generation.task_id]
        if generation.prompt_sha256 != task.prompt_sha256:
            raise ReasoningDataError(
                f"teacher generation prompt hash differs from task: {generation.generation_id}"
            )
        grouped[generation.task_id].append(generation)
    frozen: dict[str, tuple[TeacherGenerationRecord, ...]] = {}
    for task_id, records in grouped.items():
        records.sort(key=lambda record: record.candidate_index)
        indices = [record.candidate_index for record in records]
        if indices != list(range(len(indices))) or len(indices) > 2:
            raise ReasoningDataError(
                f"teacher candidates must use contiguous indices 0..1: {task_id}"
            )
        frozen[task_id] = tuple(records)
    return frozen


def build_reasoning_dataset(
    tasks: Iterable[ReasoningTask],
    generations: Iterable[TeacherGenerationRecord],
    *,
    config: M5ReasoningDataConfig,
) -> ReasoningDatasetBuild:
    """Select the first verified candidate per task and preserve every failure path."""

    ordered_tasks = tuple(sorted(tasks, key=lambda task: task.id))
    if any(task.split != "pilot_train" for task in ordered_tasks):
        raise ReasoningDataError("only pilot_train tasks may produce an M5 reasoning dataset")
    task_manifest = build_reasoning_task_manifest(ordered_tasks, config=config)
    ordered_generations = tuple(sorted(generations, key=lambda item: item.generation_id))
    generations_by_task = _validate_generation_inputs(ordered_tasks, ordered_generations)

    samples: list[ReasoningSample] = []
    verifications: list[ReasoningVerifierResult] = []
    rejected: list[ReasoningRejectedRecord] = []
    unused_candidates = 0
    for task in ordered_tasks:
        task_generations = generations_by_task.get(task.id, ())
        accepted = False
        for position, generation in enumerate(task_generations):
            if generation.status == "failed":
                rejected.append(
                    _candidate_rejection(
                        task,
                        generation,
                        stage="generation",
                        reason="teacher_generation_failed",
                        config=config,
                    )
                )
                continue
            if generation.finish_reason == "length":
                rejected.append(
                    _candidate_rejection(
                        task,
                        generation,
                        stage="generation",
                        reason="teacher_length_limit",
                        config=config,
                    )
                )
                continue
            if generation.observed_token_count > config.max_sequence_length:
                rejected.append(
                    _candidate_rejection(
                        task,
                        generation,
                        stage="tokenization",
                        reason="sequence_too_long",
                        config=config,
                    )
                )
                continue
            if generation.raw_output is None or generation.raw_output_sha256 is None:
                raise ReasoningDataError("successful teacher generation lost required output")
            parsed, parse_reason = parse_teacher_output(generation.raw_output)
            if parse_reason is not None:
                rejected.append(
                    _candidate_rejection(
                        task,
                        generation,
                        stage="parsing",
                        reason=parse_reason,
                        config=config,
                    )
                )
                continue
            if parsed is None:
                raise ReasoningDataError("teacher parser returned no output or rejection reason")
            verification = verify_reasoning_answer(
                task=task,
                generation=generation,
                final_answer=parsed.final_answer,
                config=config,
            )
            verifications.append(verification)
            if not verification.passed:
                rejection_reason: ReasoningRejectionReason = (
                    "invalid_final_json"
                    if verification.reason == "invalid_final_json"
                    else "verifier_failed"
                )
                rejected.append(
                    _candidate_rejection(
                        task,
                        generation,
                        stage="verification",
                        reason=rejection_reason,
                        config=config,
                        verification=verification,
                    )
                )
                continue
            sample_payload = {
                "final_answer": parsed.final_answer,
                "prompt": task.prompt,
                "reasoning_content": parsed.reasoning_content,
            }
            suffix = task.id.removeprefix("m5-reasoning:pilot:")
            samples.append(
                ReasoningSample(
                    id=f"m5-reasoning-sample:{suffix}",
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
                    raw_output_sha256=generation.raw_output_sha256,
                    content_sha256=content_sha256(sample_payload),
                    observed_token_count=generation.observed_token_count,
                )
            )
            unused_candidates += len(task_generations) - position - 1
            accepted = True
            break
        if not accepted:
            rejected.append(
                ReasoningRejectedRecord(
                    task_id=task.id,
                    stage="selection",
                    reason="no_candidate_passed",
                    prompt_sha256=task.prompt_sha256,
                )
            )

    samples.sort(key=lambda sample: sample.id)
    verifications.sort(key=lambda result: result.verification_id)
    rejected.sort(key=lambda record: (record.task_id, record.generation_id or "~", record.reason))
    samples_sha256 = _sequence_hash(sample.to_dict() for sample in samples)
    rejections_sha256 = _sequence_hash(record.to_dict() for record in rejected)
    verifications_sha256 = _sequence_hash(result.to_dict() for result in verifications)
    generations_sha256 = _sequence_hash(generation.to_dict() for generation in ordered_generations)
    content_identity = content_sha256(
        {
            "generations_sha256": generations_sha256,
            "rejections_sha256": rejections_sha256,
            "samples_sha256": samples_sha256,
            "tasks_sha256": task_manifest.tasks_sha256,
            "verifications_sha256": verifications_sha256,
        }
    )
    failed_generations = sum(generation.status == "failed" for generation in ordered_generations)
    rejected_candidates = sum(record.generation_id is not None for record in rejected)
    rejected_tasks = sum(record.reason == "no_candidate_passed" for record in rejected)
    manifest = M5ReasoningDatasetManifest(
        dataset_name="m5-reasoning-pilot",
        dataset_version=f"m5-reasoning-pilot-v1-{content_identity[:8]}",
        parent_dataset_version=config.parent_dataset_version,
        task_set_version=task_manifest.task_set_version,
        thinking_template_id=config.thinking_template_id,
        config_sha256=content_sha256(config.to_dict()),
        tasks_sha256=task_manifest.tasks_sha256,
        generations_sha256=generations_sha256,
        verifications_sha256=verifications_sha256,
        samples_sha256=samples_sha256,
        rejections_sha256=rejections_sha256,
        content_sha256=content_identity,
        input_tasks=len(ordered_tasks),
        generation_attempts=len(ordered_generations),
        successful_generations=len(ordered_generations) - failed_generations,
        failed_generations=failed_generations,
        verified_candidates=len(verifications),
        unused_candidates=unused_candidates,
        accepted_samples=len(samples),
        rejected_candidates=rejected_candidates,
        rejected_tasks=rejected_tasks,
        rejected_records=len(rejected),
        task_family_counts=task_manifest.task_family_counts,
        language_counts=task_manifest.language_counts,
        template_family_counts=task_manifest.template_family_counts,
        rejection_counts=dict(sorted(Counter(record.reason for record in rejected).items())),
        teacher=config.teacher,
        sampling=config.sampling,
        verifier=config.verifier,
    )
    return ReasoningDatasetBuild(
        task_manifest=task_manifest,
        manifest=manifest,
        samples=tuple(samples),
        verifications=tuple(verifications),
        rejected=tuple(rejected),
    )


def build_synthetic_teacher_generations(
    tasks: Iterable[ReasoningTask], *, config: M5ReasoningDataConfig
) -> tuple[TeacherGenerationRecord, ...]:
    """Create CPU-only public fixture output; never present it as model-generated evidence."""

    ordered = tuple(sorted(tasks, key=lambda task: task.id))
    records: list[TeacherGenerationRecord] = []
    for task_index, task in enumerate(ordered):
        for candidate_index in range(config.sampling.candidate_count):
            if candidate_index == 0 and task_index % 5 == 0:
                output = "<think>synthetic rejected path</think>\n\nnot-json"
            else:
                output = (
                    "<think>synthetic deterministic fixture reasoning; this is not model output"
                    f"</think>\n\n{task.expected_answer_json}"
                )
            records.append(
                TeacherGenerationRecord(
                    generation_id=f"{task.id}:candidate-{candidate_index}",
                    task_id=task.id,
                    candidate_index=candidate_index,
                    seed=(config.sampling.base_seed + task_index * 2 + candidate_index) % (2**32),
                    prompt_sha256=task.prompt_sha256,
                    status="succeeded",
                    finish_reason="stop",
                    raw_output=output,
                    raw_output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
                    observed_token_count=64,
                )
            )
    return tuple(records)


def summarize_reasoning_build(build: ReasoningDatasetBuild) -> dict[str, object]:
    """Return path-free JSON suitable for public CPU-smoke evidence."""

    return cast(
        dict[str, object],
        {
            "manifest": build.manifest.to_dict(),
            "rejected": [record.to_dict() for record in build.rejected],
            "sample_ids": [sample.id for sample in build.samples],
            "task_manifest": build.task_manifest.to_dict(),
            "verification_results": [result.to_dict() for result in build.verifications],
        },
    )
