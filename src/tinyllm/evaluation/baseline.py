"""Pure M2.4c Baseline contracts, prompt rendering, scoring, and command planning."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import ValidationError

from tinyllm.evaluation.baseline_schema import (
    BaselineRunConfig,
    DomainItemResult,
    HumanRubricJudgment,
)
from tinyllm.evaluation.contamination import load_evaluation_items
from tinyllm.evaluation.schema import (
    EvaluationItem,
    EvaluationSetManifest,
    ExactMatchScorer,
    HumanRubricScorer,
    JsonObjectScorer,
    MultipleChoiceScorer,
    RequiredTermsScorer,
)

_CHATML_START = "<|im_start|>"
_CHATML_END = "<|im_end|>"
_MESSAGE_FORMAT = f"{_CHATML_START}{{role}}\n{{content}}{_CHATML_END}\n"
_GENERATION_PROMPT = f"{_CHATML_START}assistant\n<think>\n\n</think>\n\n"
QWEN3_GENERATION_TEMPLATE_SPEC: dict[str, object] = {
    "add_generation_prompt": True,
    "generation_prompt": _GENERATION_PROMPT,
    "id": "qwen3-chatml-nonthinking-generation-v1",
    "message_format": _MESSAGE_FORMAT,
    "mode": "non-thinking",
}
QWEN3_GENERATION_TEMPLATE_SHA256 = (
    "b9a510e2f016a112860e47056f770b04e5c93131cc4a8ecd47fcc950cfdb6273"
)


class BaselineContractError(ValueError):
    """Raised when Baseline inputs or outputs violate the frozen M2.4c contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_template_hash() -> str:
    encoded = json.dumps(
        QWEN3_GENERATION_TEMPLATE_SPEC,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_baseline_config(path: Path) -> BaselineRunConfig:
    """Load strict formal/Smoke YAML and verify the built-in generation template."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise BaselineContractError("Baseline config must use a .yaml or .yml extension")
    try:
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = BaselineRunConfig.model_validate(payload)
    except OSError as exc:
        raise BaselineContractError("cannot read Baseline config") from exc
    except yaml.YAMLError as exc:
        raise BaselineContractError("Baseline config is invalid YAML") from exc
    except ValidationError as exc:
        messages = []
        for error in exc.errors(include_url=False, include_context=False):
            location = ".".join(str(part) for part in error["loc"])
            messages.append(f"{location}: {error['msg']}")
        raise BaselineContractError("invalid Baseline config: " + "; ".join(messages)) from exc
    if _generation_template_hash() != config.generation_template.template_sha256:
        raise BaselineContractError("built-in generation Template hash does not match config")
    return config


def load_human_rubric_judgments(path: Path) -> tuple[HumanRubricJudgment, ...]:
    """Load a strict, ordered private JSONL file of maintainer rubric decisions."""

    if not path.is_file() or path.is_symlink():
        raise BaselineContractError("human-rubric Judgment JSONL is missing")
    judgments: list[HumanRubricJudgment] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise BaselineContractError(
                        f"human-rubric Judgment JSONL has a blank line at {line_number}"
                    )
                try:
                    judgments.append(HumanRubricJudgment.model_validate_json(line))
                except ValidationError as exc:
                    raise BaselineContractError(
                        f"invalid human-rubric Judgment at line {line_number}"
                    ) from exc
    except OSError as exc:
        raise BaselineContractError("cannot read human-rubric Judgment JSONL") from exc
    if not judgments:
        raise BaselineContractError("human-rubric Judgment JSONL cannot be empty")
    if len({judgment.item_id for judgment in judgments}) != len(judgments):
        raise BaselineContractError("human-rubric Judgment item IDs must be unique")
    return tuple(judgments)


def validate_baseline_inputs(
    config: BaselineRunConfig,
    *,
    project_root: Path,
) -> tuple[EvaluationItem, ...]:
    """Verify frozen evaluation identity and local task Adapter hashes before model loading."""

    if not project_root.is_absolute() or not project_root.is_dir():
        raise BaselineContractError("project root must be an existing absolute directory")
    items_path = project_root / config.domain.items_path
    items = load_evaluation_items(items_path)
    manifest_path = items_path.parent / "manifest.json"
    try:
        manifest = EvaluationSetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise BaselineContractError("cannot load frozen evaluation Manifest") from exc
    if (
        manifest.suite_version != config.domain.suite_version
        or manifest.content_sha256 != config.domain.content_sha256
        or manifest.item_count != config.domain.expected_items
        or len(items) != config.domain.expected_items
    ):
        raise BaselineContractError("Baseline Domain identity does not match frozen evaluation set")
    for task in config.general.tasks:
        adapter = project_root / task.adapter_path
        if not adapter.is_file() or adapter.is_symlink():
            raise BaselineContractError(f"lm-eval Adapter is missing: {task.task}")
        if _sha256_file(adapter) != task.adapter_sha256:
            raise BaselineContractError(f"lm-eval Adapter hash mismatch: {task.task}")
    limit = config.domain.limit
    return items if limit is None else items[:limit]


def render_generation_prompt(item: EvaluationItem) -> str:
    """Render frozen Qwen3 non-thinking ChatML, including its empty reasoning block."""

    messages = "".join(
        _MESSAGE_FORMAT.format(role=message.role, content=message.content)
        for message in item.prompt_messages
    )
    return messages + _GENERATION_PROMPT


def _normalized_exact(response: str, scorer: ExactMatchScorer) -> bool:
    candidate = response.strip() if scorer.strip_outer_whitespace else response
    accepted = scorer.accepted_answers
    if scorer.case_sensitive:
        return candidate in accepted
    return candidate.casefold() in {answer.casefold() for answer in accepted}


def _required_terms_match(response: str, scorer: RequiredTermsScorer) -> bool:
    candidate = response if scorer.case_sensitive else response.casefold()
    required = scorer.required_terms
    forbidden = scorer.forbidden_terms
    if not scorer.case_sensitive:
        required = tuple(term.casefold() for term in required)
        forbidden = tuple(term.casefold() for term in forbidden)
    return all(term in candidate for term in required) and not any(
        term in candidate for term in forbidden
    )


def score_domain_response(
    item: EvaluationItem,
    response: str,
    *,
    prompt_tokens: int,
    generated_tokens: int,
    finish_reason: str,
) -> DomainItemResult:
    """Apply only the scorer frozen on the item and retain raw private response text."""

    automatic_correct: bool | None
    json_valid: bool | None = None
    scorer = item.scorer
    if isinstance(scorer, ExactMatchScorer):
        automatic_correct = _normalized_exact(response, scorer)
    elif isinstance(scorer, MultipleChoiceScorer):
        automatic_correct = response.strip() == scorer.choices[scorer.answer_index]
    elif isinstance(scorer, JsonObjectScorer):
        try:
            decoded = json.loads(response.strip())
        except json.JSONDecodeError:
            decoded = None
        json_valid = isinstance(decoded, dict)
        expected = cast(dict[str, object], json.loads(scorer.expected_json))
        automatic_correct = json_valid and decoded == expected
    elif isinstance(scorer, RequiredTermsScorer):
        automatic_correct = _required_terms_match(response, scorer)
    elif isinstance(scorer, HumanRubricScorer):
        automatic_correct = None
    else:  # pragma: no cover - discriminated Schema makes this unreachable
        raise BaselineContractError("unsupported Domain scorer")
    if finish_reason not in {"eos", "length"}:
        raise BaselineContractError("finish reason must be eos or length")
    return DomainItemResult(
        item_id=item.id,
        scorer_kind=scorer.kind,
        response=response,
        response_sha256=hashlib.sha256(response.encode()).hexdigest(),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        finish_reason=cast(Literal["eos", "length"], finish_reason),
        automatic_correct=automatic_correct,
        json_valid=json_valid,
        human_review_required=isinstance(scorer, HumanRubricScorer),
    )


def build_lm_eval_command(
    config: BaselineRunConfig,
    *,
    project_root: Path,
    model_path: Path,
    output_path: Path,
    device: str,
) -> tuple[str, ...]:
    """Build the reviewable lm-eval v0.4.12 invocation without executing it."""

    if not model_path.is_absolute() or not output_path.is_absolute():
        raise BaselineContractError("model and lm-eval output paths must be absolute")
    if device not in {"cpu", "cuda", "cuda:0"}:
        raise BaselineContractError("Baseline device must be cpu, cuda, or cuda:0")
    model_args = ",".join(
        (
            f"pretrained={model_path}",
            f"dtype={config.model.dtype}",
            f"attn_implementation={config.model.attention_implementation}",
            f"enable_thinking={str(config.general.enable_thinking)}",
            f"max_length={config.general.max_length}",
        )
    )
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(task.task for task in config.general.tasks),
        "--include_path",
        str(project_root / config.general.include_path),
        "--device",
        device,
        "--batch_size",
        str(config.general.batch_size),
        "--num_fewshot",
        str(config.general.num_fewshot),
        "--seed",
        ",".join(
            str(seed)
            for seed in (
                config.seeds.python,
                config.seeds.numpy,
                config.seeds.torch,
                config.seeds.fewshot,
            )
        ),
        "--apply_chat_template",
        "--log_samples",
        "--output_path",
        str(output_path),
    ]
    if config.general.limit is not None:
        command.extend(("--limit", str(config.general.limit)))
    return tuple(command)


def build_lm_eval_validation_command(
    config: BaselineRunConfig,
    *,
    project_root: Path,
) -> tuple[str, ...]:
    """Build the required task-validation command used before model initialization."""

    return (
        sys.executable,
        "-m",
        "lm_eval",
        "validate",
        "--tasks",
        ",".join(task.task for task in config.general.tasks),
        "--include_path",
        str(project_root / config.general.include_path),
    )
