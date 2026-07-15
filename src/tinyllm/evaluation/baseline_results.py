"""Strict aggregation and lm-eval output parsing for the M2.4c Baseline."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from tinyllm.evaluation.baseline_runtime import BaselineRuntimeError
from tinyllm.evaluation.baseline_schema import (
    BaselineRunConfig,
    DomainBaselineSummary,
    DomainItemResult,
    GeneralBaselineSummary,
    GeneralTaskResult,
)


def build_domain_summary(
    config: BaselineRunConfig,
    results: Sequence[DomainItemResult],
) -> DomainBaselineSummary:
    """Build a path-free aggregate while preserving human review as an explicit state."""

    ordered = tuple(results)
    expected_items = config.domain.limit or config.domain.expected_items
    if len(ordered) != expected_items or len({result.item_id for result in ordered}) != len(
        ordered
    ):
        raise BaselineRuntimeError("Domain result count or item identity is invalid")
    objective = tuple(result for result in ordered if result.automatic_correct is not None)
    human_pending = sum(result.human_review_required for result in ordered)
    json_results = tuple(result for result in ordered if result.scorer_kind == "json_object")
    return DomainBaselineSummary(
        status="awaiting_human_review" if human_pending else "complete",
        suite_version=config.domain.suite_version,
        evaluated_items=len(ordered),
        objective_items=len(objective),
        objective_correct=sum(result.automatic_correct is True for result in objective),
        human_review_pending=human_pending,
        human_reviewed=0,
        human_passed=0,
        json_items=len(json_results),
        json_valid=sum(result.json_valid is True for result in json_results),
    )


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BaselineRuntimeError(f"lm-eval {field} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineRuntimeError(f"lm-eval {field} must be numeric")
    decoded = float(value)
    if not 0.0 <= decoded <= 1.0:
        raise BaselineRuntimeError(f"lm-eval {field} is outside [0, 1]")
    return decoded


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BaselineRuntimeError(f"lm-eval {field} must be an integer")
    return value


def _validate_run_metadata(config: BaselineRunConfig, payload: Mapping[str, Any]) -> int:
    if payload.get("lm_eval_version") != config.software.lm_eval:
        raise BaselineRuntimeError("lm-eval result version does not match Baseline config")
    if payload.get("transformers_version") != config.software.transformers:
        raise BaselineRuntimeError("Transformers result version does not match Baseline config")
    if payload.get("max_length") != config.general.max_length:
        raise BaselineRuntimeError("lm-eval result maximum length does not match config")
    if payload.get("chat_template_sha") != config.general.tokenizer_chat_template_sha256:
        raise BaselineRuntimeError("lm-eval Chat Template hash does not match config")
    if payload.get("model_source") != "hf" or payload.get("fewshot_as_multiturn") is not True:
        raise BaselineRuntimeError("lm-eval model source or chat formatting does not match config")
    run_config = _mapping(payload.get("config"), field="config")
    if run_config.get("model_dtype") != "torch.bfloat16":
        raise BaselineRuntimeError("lm-eval result dtype does not match BF16 contract")
    expected_limit = float(config.general.limit) if config.general.limit is not None else None
    if run_config.get("limit") != expected_limit:
        raise BaselineRuntimeError("lm-eval result limit does not match execution mode")
    expected_seeds = {
        "random_seed": config.seeds.python,
        "numpy_seed": config.seeds.numpy,
        "torch_seed": config.seeds.torch,
        "fewshot_seed": config.seeds.fewshot,
    }
    if any(run_config.get(name) != value for name, value in expected_seeds.items()):
        raise BaselineRuntimeError("lm-eval result seeds do not match Baseline config")
    parameters = _integer(run_config.get("model_num_parameters"), field="model parameters")
    if parameters <= 0:
        raise BaselineRuntimeError("lm-eval model parameter count must be positive")
    return parameters


def load_general_summary(
    config: BaselineRunConfig,
    *,
    output_path: Path,
) -> GeneralBaselineSummary:
    """Parse exactly one private lm-eval aggregate and reject partial or drifted results."""

    result_files = tuple(sorted(output_path.rglob("results_*.json")))
    if len(result_files) != 1:
        raise BaselineRuntimeError("lm-eval output must contain exactly one aggregate result")
    try:
        decoded: object = json.loads(result_files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineRuntimeError("cannot read lm-eval aggregate result") from exc
    payload = _mapping(decoded, field="result")
    parameters = _validate_run_metadata(config, payload)
    raw_results = _mapping(payload.get("results"), field="task results")
    raw_configs = _mapping(payload.get("configs"), field="task configs")
    expected_names = tuple(task.task for task in config.general.tasks)
    if tuple(raw_results) != expected_names:
        raise BaselineRuntimeError("lm-eval task result identities or order do not match config")
    expected_samples = config.general.limit
    tasks: list[GeneralTaskResult] = []
    for identity in config.general.tasks:
        task_config = _mapping(raw_configs.get(identity.task), field=f"config {identity.task}")
        dataset_kwargs = _mapping(
            task_config.get("dataset_kwargs"), field=f"dataset kwargs {identity.task}"
        )
        if (
            task_config.get("dataset_path") != identity.dataset
            or dataset_kwargs.get("revision") != identity.dataset_revision
            or task_config.get("unsafe_code") is not False
            or task_config.get("num_fewshot") != config.general.num_fewshot
            or task_config.get("output_type") != "multiple_choice"
        ):
            raise BaselineRuntimeError(f"lm-eval task config mismatch: {identity.task}")
        result = _mapping(raw_results.get(identity.task), field=f"result {identity.task}")
        sample_count = _integer(result.get("sample_len"), field=f"{identity.task} samples")
        if sample_count != (expected_samples or identity.expected_samples):
            raise BaselineRuntimeError(f"lm-eval sample count mismatch: {identity.task}")
        tasks.append(
            GeneralTaskResult(
                task=identity.task,
                samples=sample_count,
                acc=_float(result.get("acc,none"), field=f"{identity.task} acc"),
                acc_stderr=_float(
                    result.get("acc_stderr,none"), field=f"{identity.task} acc stderr"
                ),
                acc_norm=_float(result.get("acc_norm,none"), field=f"{identity.task} acc_norm"),
                acc_norm_stderr=_float(
                    result.get("acc_norm_stderr,none"),
                    field=f"{identity.task} acc_norm stderr",
                ),
            )
        )
    evaluation_seconds = payload.get("total_evaluation_time_seconds")
    if isinstance(evaluation_seconds, bool) or not isinstance(
        evaluation_seconds, (int, float, str)
    ):
        raise BaselineRuntimeError("lm-eval evaluation time must be numeric")
    try:
        parsed_seconds = float(evaluation_seconds)
    except ValueError as exc:
        raise BaselineRuntimeError("lm-eval evaluation time must be numeric") from exc
    if not math.isfinite(parsed_seconds) or parsed_seconds <= 0:
        raise BaselineRuntimeError("lm-eval evaluation time must be positive and finite")
    return GeneralBaselineSummary(
        harness_version=config.general.harness_version,
        model_parameters=parameters,
        tasks=cast(tuple[GeneralTaskResult, GeneralTaskResult, GeneralTaskResult], tuple(tasks)),
        evaluation_seconds=parsed_seconds,
    )
