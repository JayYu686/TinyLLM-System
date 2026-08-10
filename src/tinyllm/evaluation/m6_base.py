"""M6 Base evidence import and deterministic domain-evidence helpers."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from tinyllm.evaluation.baseline import (
    load_baseline_config,
    load_human_rubric_judgments,
)
from tinyllm.evaluation.baseline_schema import (
    BaselineEvaluationResult,
    DomainItemResult,
    HumanRubricJudgment,
)
from tinyllm.evaluation.contamination import load_evaluation_items
from tinyllm.evaluation.m6 import load_m6_release_config
from tinyllm.evaluation.m6_schema import (
    M6BaseImportResult,
    M6DomainItemScore,
    M6DomainModeResult,
    M6GeneralResult,
    M6GeneralTaskResult,
    M6ModelIdentity,
    M6ReleaseConfig,
)
from tinyllm.evaluation.schema import EvaluationItem
from tinyllm.schemas import RunManifest, RunStatus, canonical_config_hash


class M6BaseImportError(RuntimeError):
    """Raised when historical evidence cannot be reused without ambiguity."""


def sha256_file(path: Path) -> str:
    """Hash one regular non-symlink file."""

    if not path.is_file() or path.is_symlink():
        raise M6BaseImportError("M6 evidence file is missing or unsafe")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash a directory as sorted relative path, size, and content identity."""

    if not path.is_dir() or path.is_symlink():
        raise M6BaseImportError("M6 evidence tree is missing or unsafe")
    entries: list[dict[str, object]] = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise M6BaseImportError("M6 evidence tree cannot contain symlinks")
        if child.is_file():
            entries.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "size_bytes": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
    if not entries:
        raise M6BaseImportError("M6 evidence tree contains no files")
    payload = json.dumps(
        {"algorithm": "tinyllm-tree-sha256-v1", "files": entries},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def domain_cluster_id(item: EvaluationItem) -> str:
    """Map the frozen bilingual tag contract to one stable bootstrap cluster."""

    pair_tags = tuple(tag for tag in item.tags if tag.startswith("bilingual-pair-"))
    if len(pair_tags) == 1:
        pair_number = pair_tags[0].removeprefix("bilingual-pair-")
        return f"pair:{item.category}:{pair_number}"
    if not pair_tags and "english-only" in item.tags and item.language == "en":
        return f"singleton:{item.id}"
    raise M6BaseImportError("M6 domain item has an invalid bilingual cluster identity")


def model_artifact_sha256(model_dir: Path, expected_files: tuple[object, ...]) -> str:
    """Verify every pinned Base file and return a directory-independent identity."""

    if not model_dir.is_absolute() or not model_dir.is_dir() or model_dir.is_symlink():
        raise M6BaseImportError("M6 Base model directory is missing or unsafe")
    entries: list[dict[str, object]] = []
    for untyped in expected_files:
        item = cast(Any, untyped)
        file_path = model_dir / str(item.filename)
        if (
            not file_path.is_file()
            or file_path.is_symlink()
            or file_path.stat().st_size != int(item.size_bytes)
            or sha256_file(file_path) != str(item.sha256)
        ):
            raise M6BaseImportError("M6 Base model file identity differs from pinned input")
        entries.append(
            {
                "filename": str(item.filename),
                "size_bytes": int(item.size_bytes),
                "sha256": str(item.sha256),
            }
        )
    payload = json.dumps(
        {"algorithm": "tinyllm-model-artifact-sha256-v1", "files": entries},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_jsonl(path: Path, model: type[Any]) -> tuple[Any, ...]:
    if not path.is_file() or path.is_symlink():
        raise M6BaseImportError("M6 source JSONL is missing or unsafe")
    values: list[Any] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise M6BaseImportError(
                        f"M6 source JSONL contains a blank line at {line_number}"
                    )
                values.append(model.model_validate_json(line))
    except (OSError, ValidationError) as exc:
        raise M6BaseImportError("M6 source JSONL is invalid") from exc
    return tuple(values)


def _validate_protocol_equivalence(
    release: M6ReleaseConfig,
    source: Any,
) -> None:
    if release.protocol_version == "m6-release-v1":
        domain = source.domain
        nonthinking = release.domain_execution.nonthinking
        if (
            domain.suite_version != release.suite_version
            or domain.content_sha256 != release.suite_content_sha256
            or domain.expected_items != release.expected_domain_items
            or domain.max_sequence_length != release.domain_execution.max_sequence_length
            or domain.max_new_tokens != nonthinking.max_new_tokens
            or domain.do_sample != nonthinking.do_sample
            or source.generation_template.template_sha256 != nonthinking.template_sha256
        ):
            raise M6BaseImportError("M2 domain protocol is incompatible with M6 Non-thinking")
    general = source.general
    expected_tasks = release.general_execution.tasks
    actual_tasks = tuple(
        (task.task, task.adapter_sha256, task.dataset_revision, task.expected_samples)
        for task in general.tasks
    )
    required_tasks = tuple(
        (task.task, task.adapter_sha256, task.dataset_revision, task.expected_samples)
        for task in expected_tasks
    )
    if (
        general.harness_version != release.general_execution.harness_version
        or general.tokenizer_chat_template_sha256
        != release.general_execution.tokenizer_chat_template_sha256
        or general.apply_chat_template != release.general_execution.apply_chat_template
        or general.enable_thinking != release.general_execution.enable_thinking
        or general.num_fewshot != release.general_execution.num_fewshot
        or general.batch_size != release.general_execution.batch_size
        or general.max_length != release.general_execution.max_length
        or general.log_samples != release.general_execution.log_samples
        or actual_tasks != required_tasks
    ):
        raise M6BaseImportError("M2 general protocol is incompatible with M6 regression")


def _nonthinking_mode(
    items: tuple[EvaluationItem, ...],
    results: tuple[DomainItemResult, ...],
    judgments: tuple[HumanRubricJudgment, ...],
) -> M6DomainModeResult:
    if tuple(item.id for item in items) != tuple(result.item_id for result in results):
        raise M6BaseImportError("M2 domain result identities differ from the frozen suite")
    judgment_map = {judgment.item_id: judgment for judgment in judgments}
    expected_human = tuple(item.id for item in items if item.scorer.kind == "human_rubric")
    if tuple(judgment_map) != expected_human:
        raise M6BaseImportError("M2 human-review identities differ from the frozen suite")
    scores: list[M6DomainItemScore] = []
    for item, result in zip(items, results, strict=True):
        correct = (
            judgment_map[item.id].passed
            if result.human_review_required
            else bool(result.automatic_correct)
        )
        format_valid = result.json_valid is not False
        leakage = "<think>" in result.response or "</think>" in result.response
        scores.append(
            M6DomainItemScore(
                item_id=item.id,
                cluster_id=domain_cluster_id(item),
                language=item.language,
                category=item.category,
                scorer_kind=item.scorer.kind,
                correct=correct,
                json_valid=result.json_valid,
                format_valid=format_valid,
                visible_reasoning_leakage=leakage,
            )
        )
    ordered = tuple(scores)
    correct_items = sum(item.correct for item in ordered)
    format_items = sum(item.format_valid for item in ordered)
    json_valid = sum(item.json_valid is True for item in ordered)
    leakage_items = sum(item.visible_reasoning_leakage for item in ordered)
    return M6DomainModeResult(
        mode="nonthinking",
        items=ordered,
        evaluated_items=300,
        correct_items=correct_items,
        score_basis_points=round(correct_items * 10000 / 300),
        format_valid_items=format_items,
        format_valid_basis_points=round(format_items * 10000 / 300),
        json_items=80,
        json_valid_items=json_valid,
        json_valid_basis_points=round(json_valid * 10000 / 80),
        visible_reasoning_leakage_items=leakage_items,
        visible_reasoning_leakage_basis_points=round(leakage_items * 10000 / 300),
        natural_thinking_closed_items=0,
        budget_forced_close_items=0,
        forced_close_basis_points=0,
        generated_tokens=sum(result.generated_tokens for result in results),
        injected_tokens=0,
    )


def _general_result(summary: BaselineEvaluationResult) -> M6GeneralResult:
    tasks = tuple(
        M6GeneralTaskResult(
            task=item.task,
            samples=item.samples,
            acc=item.acc,
            acc_stderr=item.acc_stderr,
            acc_norm=item.acc_norm,
            acc_norm_stderr=item.acc_norm_stderr,
        )
        for item in summary.general.tasks
    )
    typed = cast(
        tuple[M6GeneralTaskResult, M6GeneralTaskResult, M6GeneralTaskResult],
        tasks,
    )
    return M6GeneralResult(
        harness_version=summary.general.harness_version,
        metric="acc_norm",
        aggregation="equal-task-mean",
        tasks=typed,
        aggregate_basis_points=round(sum(item.acc_norm for item in tasks) * 10000 / 3),
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def import_m2_base_evidence(
    *,
    release_config_path: Path,
    source_run: Path,
    model_dir: Path,
    project_root: Path,
    output_path: Path | None = None,
) -> M6BaseImportResult:
    """Import Base identity and only evidence compatible with the release protocol."""

    if not source_run.is_absolute() or not source_run.is_dir() or source_run.is_symlink():
        raise M6BaseImportError("M6 source Run must be an existing absolute directory")
    project_root = project_root.resolve()
    release = load_m6_release_config(release_config_path)
    source_config = load_baseline_config(source_run / "config.original.yaml")
    _validate_protocol_equivalence(release, source_config)
    try:
        manifest = RunManifest.model_validate_json(
            (source_run / "run.json").read_text(encoding="utf-8")
        )
        summary = BaselineEvaluationResult.model_validate_json(
            (source_run / "evaluations/summary.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise M6BaseImportError("M2 Base Run metadata is invalid") from exc
    source_config_sha = canonical_config_hash(source_config)
    if (
        manifest.status != RunStatus.SUCCEEDED
        or manifest.run_id != source_run.name
        or manifest.config_hash != source_config_sha
        or summary.status != "succeeded"
        or summary.mode != "formal"
        or summary.run_id != manifest.run_id
        or summary.config_sha256 != source_config_sha
        or summary.git_dirty
        or summary.git_commit != manifest.git_commit
        or summary.model_repository != source_config.model.repository
        or summary.model_revision != source_config.model.revision
    ):
        raise M6BaseImportError("M2 Base Run lineage is incomplete or inconsistent")
    domain_path = source_run / "evaluations/domain/results.jsonl"
    review_path = source_run / "evaluations/domain/human_review/judgments.jsonl"
    domain_sha: str | None = None
    review_sha: str | None = None
    nonthinking: M6DomainModeResult | None = None
    if release.protocol_version == "m6-release-v1":
        items = load_evaluation_items(project_root / "evals/domain/v1/items.jsonl")
        if len(items) != 300:
            raise M6BaseImportError("M6 frozen domain suite must contain exactly 300 items")
        results = cast(tuple[DomainItemResult, ...], _load_jsonl(domain_path, DomainItemResult))
        judgments = load_human_rubric_judgments(review_path)
        domain_sha = sha256_file(domain_path)
        review_sha = sha256_file(review_path)
        nonthinking = _nonthinking_mode(items, results, judgments)
    model_sha = model_artifact_sha256(model_dir, source_config.model.files)
    imported = M6BaseImportResult(
        status="succeeded",
        protocol_version=release.protocol_version,
        config_sha256=canonical_config_hash(release),
        source_run_id=manifest.run_id,
        source_config_sha256=source_config_sha,
        source_git_commit=summary.git_commit,
        source_evaluation_sha256=sha256_file(source_run / "evaluations/summary.json"),
        source_domain_results_sha256=domain_sha,
        source_human_review_sha256=review_sha,
        source_general_tree_sha256=sha256_tree(source_run / "evaluations/general/raw"),
        source_environment_sha256=sha256_file(source_run / "environment.json"),
        source_hardware_sha256=sha256_file(source_run / "hardware.json"),
        model=M6ModelIdentity(
            role="base",
            repository=source_config.model.repository,
            base_revision=source_config.model.revision,
            attention_architecture="gqa",
            adaptation="base",
            model_artifact_sha256=model_sha,
            model_parameters=summary.general.model_parameters,
        ),
        nonthinking=nonthinking,
        general=_general_result(summary),
    )
    if output_path is not None:
        if not output_path.is_absolute():
            raise M6BaseImportError("M6 Base import output path must be absolute")
        _atomic_json(output_path, imported.to_dict())
    return imported


def load_m6_base_import(path: Path) -> M6BaseImportResult:
    """Load one persisted Base import and bind its exact file identity externally."""

    try:
        return M6BaseImportResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise M6BaseImportError("M6 Base import result is invalid") from exc
