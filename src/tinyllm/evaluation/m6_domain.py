"""Independent M6 dual-mode domain generation and human-review finalization."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from tinyllm.evaluation.baseline import (
    load_baseline_config,
    load_human_rubric_judgments,
    score_domain_response,
)
from tinyllm.evaluation.baseline_schema import HumanRubricJudgment
from tinyllm.evaluation.contamination import load_evaluation_items
from tinyllm.evaluation.m5_thinking_budget_schema import EARLY_STOPPING_TEXT
from tinyllm.evaluation.m6 import load_m6_release_config
from tinyllm.evaluation.m6_base import (
    domain_cluster_id,
    model_artifact_sha256,
    sha256_file,
)
from tinyllm.evaluation.m6_candidate import model_export_sha256
from tinyllm.evaluation.m6_schema import (
    M6DomainItemScore,
    M6DomainModeResult,
    M6DomainPassSummary,
    M6DomainTranscript,
    M6ModelIdentity,
    M6ProtocolVersion,
    M6SuiteVersion,
)
from tinyllm.evaluation.schema import EvaluationItem
from tinyllm.lineage import read_git_identity
from tinyllm.schemas import canonical_config_hash


class M6DomainError(RuntimeError):
    """Raised when M6 domain generation or review fails closed."""


EVIDENCE_GROUNDING_SYSTEM_PROMPT = (
    "Evidence-grounding policy / 证据约束：When the request asks for a root cause but "
    "explicitly says evidence is missing, do not name or repeat any suspected component as "
    "the cause. State that the supplied evidence is insufficient, then request every missing "
    "evidence item named by the request. 当请求在证据缺失时要求判断根因，不得把被怀疑组件复述为"
    "根因；必须明确说明现有证据不足，并请求题目列出的全部缺失证据。"
)
THINKING_FINAL_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class _PendingContinuation:
    """One Thinking result awaiting the batched Final-Answer generation stage."""

    item_index: int
    item: EvaluationItem
    prompt: str
    prompt_tokens: int
    first_ids: list[int]
    first_response: str
    controller_text: str
    injected_tokens: int
    forced: bool
    input_text: str


def _suite_items_path(project_root: Path, suite_version: M6SuiteVersion) -> Path:
    relative = {
        "tinyllm-domain-v1-83bdd8ef": Path("evals/domain/v1/items.jsonl"),
        "tinyllm-domain-holdout-v1-c0c948cc": Path("evals/domain/v2/items.jsonl"),
        "tinyllm-domain-holdout-v1-2b167ce6": Path("evals/domain/v3/items.jsonl"),
        "tinyllm-domain-final-audit-v1-bac25144": Path("evals/domain/v4/items.jsonl"),
    }[suite_version]
    return project_root / relative


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, values: Sequence[M6DomainTranscript]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_transcripts(path: Path) -> tuple[M6DomainTranscript, ...]:
    if not path.is_file() or path.is_symlink():
        raise M6DomainError("M6 transcript JSONL is missing or unsafe")
    values: list[M6DomainTranscript] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise M6DomainError(
                        f"M6 transcript JSONL contains a blank line at {line_number}"
                    )
                values.append(M6DomainTranscript.model_validate_json(line))
    except (OSError, ValueError) as exc:
        raise M6DomainError("M6 transcript JSONL is invalid") from exc
    return tuple(values)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _eos_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    result = {value} if isinstance(value, int) else set(cast(list[int], value))
    if not result:
        raise M6DomainError("Qwen tokenizer has no EOS Token")
    return result


def _trim_at_eos(
    raw_ids: Sequence[int],
    eos_ids: set[int],
    *,
    include_eos: bool,
) -> tuple[list[int], bool]:
    token_ids = list(raw_ids)
    for index, token_id in enumerate(token_ids):
        if token_id in eos_ids:
            end = index + 1 if include_eos else index
            return token_ids[:end], True
    return token_ids, False


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


def parse_m6_final_answer(
    response: str,
    *,
    mode: Literal["thinking", "nonthinking"],
) -> tuple[str, bool, bool]:
    """Return final answer, controlled-format state, and visible-reasoning leakage."""

    if mode == "nonthinking":
        answer = response.strip()
        leakage = "<think>" in answer or "</think>" in answer
        return answer, bool(answer) and not leakage, leakage
    closing_count = response.count("</think>")
    if closing_count != 1:
        return "", False, False
    answer = response.split("</think>", 1)[1].strip()
    format_valid = bool(answer) and "<think>" not in answer and "</think>" not in answer
    return answer, format_valid, False


def repair_m6_json_answer(
    item: EvaluationItem,
    answer: str,
    *,
    policy: Literal[
        "json-syntax-only-v1",
        "json-syntax-only-v2",
        "json-syntax-only-v3",
    ]
    | None,
) -> tuple[
    str,
    Literal[
        "none",
        "wrap_single_key",
        "brace_member_fragment",
        "close_object",
        "unwrap_json_fence",
        "quote_bare_keys",
        "arrow_single_key",
        "wrap_bareword_single_key",
        "promote_required_keys",
        "close_object_promote_required_keys",
    ],
]:
    """Repair only a JSON object's syntax shell without changing any decoded leaf value."""

    if policy is None or item.scorer.kind != "json_object":
        return answer, "none"
    required_keys = tuple(item.scorer.required_keys)
    stripped = answer.strip()
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        if set(required_keys).issubset(decoded):
            return answer, "none"
        if policy in {"json-syntax-only-v2", "json-syntax-only-v3"}:
            missing = tuple(key for key in required_keys if key not in decoded)
            containers = tuple(
                (key, value)
                for key, value in decoded.items()
                if isinstance(value, dict) and set(missing).issubset(value)
            )
            if missing and len(containers) == 1:
                container_key, container = containers[0]
                promoted = dict(decoded)
                retained = dict(container)
                for key in missing:
                    promoted[key] = retained.pop(key)
                promoted[container_key] = retained
                return (
                    json.dumps(
                        promoted,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "promote_required_keys",
                )
        return answer, "none"
    if decoded is not None and len(required_keys) == 1:
        repaired = {required_keys[0]: decoded}
        return (
            json.dumps(repaired, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            "wrap_single_key",
        )
    if policy in {"json-syntax-only-v2", "json-syntax-only-v3"}:
        fenced = re.fullmatch(
            r"```(?:json)?\s*\n?(.*?)\n?```\s*",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced is not None:
            fenced_value: Any
            try:
                fenced_value = json.loads(fenced.group(1).strip())
            except json.JSONDecodeError:
                fenced_value = None
            if isinstance(fenced_value, dict) and set(required_keys).issubset(fenced_value):
                return (
                    json.dumps(
                        fenced_value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "unwrap_json_fence",
                )
        arrow = re.fullmatch(r'\[\s*"([^"\\]+)"\s*\]\s*=>\s*(.+)', stripped, re.DOTALL)
        if arrow is not None and len(required_keys) == 1 and arrow.group(1) == required_keys[0]:
            try:
                value = json.loads(arrow.group(2))
            except json.JSONDecodeError:
                value = None
            if value is not None:
                return (
                    json.dumps(
                        {required_keys[0]: value},
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "arrow_single_key",
                )
        if len(required_keys) == 1 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", stripped):
            return (
                json.dumps(
                    {required_keys[0]: stripped},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "wrap_bareword_single_key",
            )
        quoted_keys = re.sub(
            r"(?P<prefix>[{,]\s*)(?P<key>[A-Za-z_][A-Za-z0-9_-]*)(?P<colon>\s*:)",
            r'\g<prefix>"\g<key>"\g<colon>',
            stripped,
        )
        if quoted_keys != stripped:
            quoted_value: Any
            try:
                quoted_value = json.loads(quoted_keys)
            except json.JSONDecodeError:
                quoted_value = None
            if isinstance(quoted_value, dict) and set(required_keys).issubset(quoted_value):
                return (
                    json.dumps(
                        quoted_value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "quote_bare_keys",
                )
    candidates: tuple[tuple[str, Literal["brace_member_fragment", "close_object"]], ...] = (
        (f"{{{stripped}}}", "brace_member_fragment"),
        (f"{stripped}}}", "close_object"),
    )
    for candidate, action in candidates:
        try:
            repaired = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(repaired, dict) and set(required_keys).issubset(repaired):
            return (
                json.dumps(repaired, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                action,
            )
        if (
            policy == "json-syntax-only-v3"
            and action == "close_object"
            and isinstance(repaired, dict)
        ):
            missing = tuple(key for key in required_keys if key not in repaired)
            containers = tuple(
                (key, value)
                for key, value in repaired.items()
                if isinstance(value, dict) and set(missing).issubset(value)
            )
            if missing and len(containers) == 1:
                container_key, container = containers[0]
                promoted = dict(repaired)
                retained = dict(container)
                for key in missing:
                    promoted[key] = retained.pop(key)
                promoted[container_key] = retained
                return (
                    json.dumps(
                        promoted,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "close_object_promote_required_keys",
                )
    return answer, "none"


def build_m6_domain_transcript(
    item: EvaluationItem,
    *,
    mode: Literal["thinking", "nonthinking"],
    prompt: str,
    response: str,
    first_pass_response: str,
    continuation_response: str,
    controller_action: Literal[
        "not_applicable",
        "natural_complete",
        "natural_close_continue",
        "forced_close_continue",
    ],
    prompt_tokens: int,
    first_pass_tokens: int,
    continuation_tokens: int,
    injected_tokens: int,
    finish_reason: Literal["eos", "length"],
    json_repair_policy: Literal[
        "json-syntax-only-v1",
        "json-syntax-only-v2",
        "json-syntax-only-v3",
    ]
    | None = None,
) -> M6DomainTranscript:
    """Parse and score one transcript without fabricating human-rubric judgments."""

    raw_final_answer, format_valid, leakage = parse_m6_final_answer(response, mode=mode)
    final_answer, repair_action = repair_m6_json_answer(
        item,
        raw_final_answer,
        policy=json_repair_policy if format_valid else None,
    )
    scored = score_domain_response(
        item,
        final_answer,
        prompt_tokens=prompt_tokens,
        generated_tokens=first_pass_tokens + continuation_tokens,
        finish_reason=finish_reason,
    )
    automatic_correct = scored.automatic_correct
    if automatic_correct is not None:
        automatic_correct = automatic_correct and format_valid
    forced = controller_action == "forced_close_continue"
    natural = mode == "thinking" and not forced
    controller_injected_text = ""
    if forced:
        controller_injected_text = EARLY_STOPPING_TEXT
    elif controller_action == "natural_close_continue" and injected_tokens:
        controller_injected_text = THINKING_FINAL_SEPARATOR
    return M6DomainTranscript(
        item_id=item.id,
        mode=mode,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        response=response,
        response_sha256=hashlib.sha256(response.encode()).hexdigest(),
        first_pass_response=first_pass_response,
        continuation_response=continuation_response,
        controller_injected_text=controller_injected_text,
        controller_action=controller_action,
        final_answer=final_answer,
        final_answer_sha256=hashlib.sha256(final_answer.encode()).hexdigest(),
        raw_final_answer=raw_final_answer if repair_action != "none" else "",
        raw_final_answer_sha256=(
            hashlib.sha256(raw_final_answer.encode()).hexdigest()
            if repair_action != "none"
            else None
        ),
        output_repair_action=repair_action,
        prompt_tokens=prompt_tokens,
        first_pass_tokens=first_pass_tokens,
        continuation_tokens=continuation_tokens,
        injected_tokens=injected_tokens,
        generated_tokens=first_pass_tokens + continuation_tokens,
        finish_reason=finish_reason,
        scorer_kind=item.scorer.kind,
        automatic_correct=automatic_correct,
        json_valid=scored.json_valid,
        human_review_required=scored.human_review_required,
        format_valid=format_valid,
        visible_reasoning_leakage=leakage,
        natural_thinking_closed=natural,
        budget_forced_close=forced,
    )


def _generate(
    model: Any,
    *,
    model_inputs: dict[str, Any],
    input_width: int,
    tokenizer: Any,
    eos_ids: set[int],
    seed: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    stop_strings: tuple[str, ...] | None = None,
) -> list[list[int]]:
    _set_seed(seed)
    stopping: dict[str, object] = {}
    if stop_strings is not None:
        stopping = {"stop_strings": stop_strings, "tokenizer": tokenizer}
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=sorted(eos_ids),
            use_cache=True,
            **stopping,
        )
    return cast(list[list[int]], generated[:, input_width:].detach().cpu().tolist())


def _validate_runtime(model_dir: Path, tokenizer: Any, model: M6ModelIdentity) -> None:
    probe = [{"role": "user", "content": "TinyLLM template probe."}]
    thinking = tokenizer.apply_chat_template(
        probe,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    nonthinking = tokenizer.apply_chat_template(
        probe,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if thinking != (
        "<|im_start|>user\nTinyLLM template probe.<|im_end|>\n<|im_start|>assistant\n"
    ) or nonthinking != (
        "<|im_start|>user\nTinyLLM template probe.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ):
        raise M6DomainError("Qwen3 M6 generation Template drifted")
    try:
        config = cast(
            dict[str, object],
            json.loads((model_dir / "config.json").read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise M6DomainError("M6 model config cannot be parsed") from exc
    expected_heads = 16 if model.repository.endswith("0.6B") else 32
    if {
        "model_type": config.get("model_type"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
    } != {
        "model_type": "qwen3",
        "num_attention_heads": expected_heads,
        "num_key_value_heads": 8,
    }:
        raise M6DomainError("M6 evaluated model is not the frozen Qwen3 GQA route")


def _environment_payload() -> dict[str, object]:
    import transformers  # type: ignore[import-not-found]

    return {
        "schema_version": "1.0",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _hardware_payload(device: torch.device, physical_gpu_index: int) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": "1.0",
        "physical_gpu_index": physical_gpu_index,
        "logical_gpu_index": 0,
        "gpu_name": torch.cuda.get_device_name(device),
        "memory_total_bytes": int(properties.total_memory),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def _summary(
    *,
    model: M6ModelIdentity,
    mode: Literal["thinking", "nonthinking"],
    config_sha256: str,
    git_commit: str,
    transcripts: tuple[M6DomainTranscript, ...],
    evaluation_id: str,
    duration_seconds: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    physical_gpu_index: int,
    gpu_name: str,
    environment_sha256: str,
    hardware_sha256: str,
    raw_results_sha256: str,
    protocol_version: M6ProtocolVersion,
    suite_version: M6SuiteVersion,
) -> M6DomainPassSummary:
    return M6DomainPassSummary(
        status="awaiting_human_review",
        evaluation_id=evaluation_id,
        protocol_version=protocol_version,
        suite_version=suite_version,
        config_sha256=config_sha256,
        git_commit=git_commit,
        git_dirty=False,
        model=model,
        mode=mode,
        evaluated_items=300,
        objective_items=260,
        objective_correct_items=sum(item.automatic_correct is True for item in transcripts),
        human_review_pending=40,
        human_reviewed=0,
        human_passed=0,
        json_items=80,
        json_valid_items=sum(item.json_valid is True for item in transcripts),
        json_repaired_items=sum(item.output_repair_action != "none" for item in transcripts),
        format_valid_items=sum(item.format_valid for item in transcripts),
        visible_reasoning_leakage_items=sum(item.visible_reasoning_leakage for item in transcripts),
        natural_thinking_closed_items=sum(item.natural_thinking_closed for item in transcripts),
        budget_forced_close_items=sum(item.budget_forced_close for item in transcripts),
        generated_tokens=sum(item.generated_tokens for item in transcripts),
        injected_tokens=sum(item.injected_tokens for item in transcripts),
        duration_seconds=duration_seconds,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        physical_gpu_index=physical_gpu_index,
        gpu_name=gpu_name,
        environment_sha256=environment_sha256,
        hardware_sha256=hardware_sha256,
        raw_results_sha256=raw_results_sha256,
    )


def run_m6_domain_pass(
    *,
    release_config_path: Path,
    model_dir: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    project_root: Path,
    physical_gpu_index: int,
    model_identity: M6ModelIdentity,
    mode: Literal["thinking", "nonthinking"],
    expected_config_sha256: str,
) -> M6DomainPassSummary:
    """Run one clean-Git, single-GPU M6 domain mode into private artifacts."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise M6DomainError("M6 domain evaluation requires exactly one visible CUDA GPU")
    project_root = project_root.resolve()
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M6DomainError("formal M6 evaluation requires a clean Git worktree")
    if output_dir.exists() or not output_dir.is_absolute():
        raise M6DomainError("M6 output directory must be absolute and absent")
    release = load_m6_release_config(release_config_path)
    if canonical_config_hash(release) != expected_config_sha256:
        raise M6DomainError("M6 Base import and Release config identities differ")
    source_config = load_baseline_config(project_root / "configs/eval/m2_baseline.yaml")
    base_artifact_sha256 = model_artifact_sha256(tokenizer_dir, source_config.model.files)
    if model_identity.role == "base":
        if (
            model_identity.adaptation != "base"
            or model_dir.resolve() != tokenizer_dir.resolve()
            or base_artifact_sha256 != model_identity.model_artifact_sha256
        ):
            raise M6DomainError(
                "M6 Base model or Tokenizer identity differs from imported evidence"
            )
    elif (
        model_identity.adaptation != "full_sft"
        or model_identity.repository != "Qwen/Qwen3-0.6B"
        or model_identity.base_revision != source_config.model.revision
        or model_export_sha256(model_dir) != model_identity.model_artifact_sha256
    ):
        raise M6DomainError("M6 Candidate model or Tokenizer identity differs from its import")
    items = load_evaluation_items(_suite_items_path(project_root, release.suite_version))
    if len(items) != 300:
        raise M6DomainError("M6 domain suite must contain exactly 300 items")
    output_dir.mkdir(parents=True)
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        raise M6DomainError("M6 tokenizer has no padding Token")
    _validate_runtime(model_dir, tokenizer, model_identity)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    eos_ids = _eos_ids(tokenizer)
    injection_ids = list(tokenizer.encode(EARLY_STOPPING_TEXT, add_special_tokens=False))
    if not injection_ids or _decode(tokenizer, injection_ids) != EARLY_STOPPING_TEXT:
        raise M6DomainError("M6 controller injection does not round-trip")
    separator_ids: list[int] = []
    if release.domain_execution.output_control is not None:
        separator = release.domain_execution.output_control
        if (
            hashlib.sha256(THINKING_FINAL_SEPARATOR.encode()).hexdigest()
            != separator.thinking_final_separator_sha256
        ):
            raise M6DomainError("M6 Thinking final separator identity drifted")
        separator_ids = list(tokenizer.encode(THINKING_FINAL_SEPARATOR, add_special_tokens=False))
        if not separator_ids or _decode(tokenizer, separator_ids) != THINKING_FINAL_SEPARATOR:
            raise M6DomainError("M6 Thinking final separator does not round-trip")
    config_sha256 = canonical_config_hash(release)
    environment_path = output_dir / "environment.json"
    hardware_path = output_dir / "hardware.json"
    _atomic_json(environment_path, _environment_payload())
    _atomic_json(hardware_path, _hardware_payload(device, physical_gpu_index))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    transcripts: list[M6DomainTranscript] = []
    generation = release.domain_execution
    output_control = generation.output_control
    if output_control is not None and (
        hashlib.sha256(EVIDENCE_GROUNDING_SYSTEM_PROMPT.encode()).hexdigest()
        != output_control.evidence_system_prompt_sha256
    ):
        raise M6DomainError("M6 evidence-grounding System Prompt identity drifted")
    for offset in range(0, len(items), generation.batch_size):
        batch_items = items[offset : offset + generation.batch_size]
        prompts: list[str] = []
        for item in batch_items:
            messages = [message.to_dict() for message in item.prompt_messages]
            if output_control is not None and item.scorer.kind == "human_rubric":
                messages.insert(
                    0,
                    {"role": "system", "content": EVIDENCE_GROUNDING_SYSTEM_PROMPT},
                )
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=mode == "thinking",
                )
            )
        encoded: dict[str, Any] = tokenizer(prompts, padding=True, return_tensors="pt")
        prompt_lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1)]
        if any(length > generation.max_sequence_length for length in prompt_lengths):
            raise M6DomainError("M6 domain prompt exceeds maximum sequence length")
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        input_width = int(model_inputs["input_ids"].shape[1])
        first_rows = _generate(
            model,
            model_inputs=model_inputs,
            input_width=input_width,
            tokenizer=tokenizer,
            eos_ids=eos_ids,
            seed=generation.thinking.seed + offset,
            max_new_tokens=(
                generation.thinking.thinking_budget_tokens
                if mode == "thinking"
                else generation.nonthinking.max_new_tokens
            ),
            do_sample=mode == "thinking",
            temperature=generation.thinking.temperature if mode == "thinking" else None,
            top_p=generation.thinking.top_p if mode == "thinking" else None,
            top_k=generation.thinking.top_k if mode == "thinking" else None,
            stop_strings=("</think>",) if mode == "thinking" else None,
        )
        batch_transcripts: list[M6DomainTranscript | None] = [None] * len(batch_items)
        pending: list[_PendingContinuation] = []
        json_repair_policy = (
            output_control.json_repair_policy if output_control is not None else None
        )
        for item_index, (item, prompt, prompt_tokens, raw_ids) in enumerate(
            zip(batch_items, prompts, prompt_lengths, first_rows, strict=True)
        ):
            first_ids, first_eos = _trim_at_eos(raw_ids, eos_ids, include_eos=True)
            first_response = _decode(tokenizer, first_ids)
            if mode == "nonthinking":
                batch_transcripts[item_index] = build_m6_domain_transcript(
                    item,
                    mode=mode,
                    prompt=prompt,
                    response=first_response,
                    first_pass_response=first_response,
                    continuation_response="",
                    controller_action="not_applicable",
                    prompt_tokens=prompt_tokens,
                    first_pass_tokens=len(first_ids),
                    continuation_tokens=0,
                    injected_tokens=0,
                    finish_reason="eos" if first_eos else "length",
                    json_repair_policy=json_repair_policy,
                )
                continue
            natural_close = "</think>" in first_response
            if natural_close and first_eos:
                batch_transcripts[item_index] = build_m6_domain_transcript(
                    item,
                    mode=mode,
                    prompt=prompt,
                    response=first_response,
                    first_pass_response=first_response,
                    continuation_response="",
                    controller_action="natural_complete",
                    prompt_tokens=prompt_tokens,
                    first_pass_tokens=len(first_ids),
                    continuation_tokens=0,
                    injected_tokens=0,
                    finish_reason="eos",
                    json_repair_policy=json_repair_policy,
                )
                continue
            forced = not natural_close
            controller_ids = injection_ids if forced else separator_ids
            controller_text = EARLY_STOPPING_TEXT if forced else THINKING_FINAL_SEPARATOR
            pending.append(
                _PendingContinuation(
                    item_index=item_index,
                    item=item,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    first_ids=first_ids,
                    first_response=first_response,
                    controller_text=controller_text,
                    injected_tokens=len(controller_ids),
                    forced=forced,
                    input_text=prompt + first_response + controller_text,
                )
            )

        final_batch_size = generation.thinking.final_answer_batch_size
        for pending_offset in range(0, len(pending), final_batch_size):
            pending_batch = pending[pending_offset : pending_offset + final_batch_size]
            continuation_encoded: dict[str, Any] = tokenizer(
                [row.input_text for row in pending_batch],
                padding=True,
                return_tensors="pt",
            )
            continuation_inputs = {
                key: value.to(device) for key, value in continuation_encoded.items()
            }
            continuation_width = int(continuation_inputs["input_ids"].shape[1])
            continuation_rows = _generate(
                model,
                model_inputs=continuation_inputs,
                input_width=continuation_width,
                tokenizer=tokenizer,
                eos_ids=eos_ids,
                seed=(generation.thinking.seed + 2_000_000 + offset + pending_batch[0].item_index),
                max_new_tokens=generation.thinking.final_answer_max_new_tokens,
                do_sample=generation.thinking.final_answer_do_sample,
                temperature=(
                    generation.thinking.temperature
                    if generation.thinking.final_answer_do_sample
                    else None
                ),
                top_p=(
                    generation.thinking.top_p
                    if generation.thinking.final_answer_do_sample
                    else None
                ),
                top_k=(
                    generation.thinking.top_k
                    if generation.thinking.final_answer_do_sample
                    else None
                ),
            )
            for row, raw_continuation in zip(pending_batch, continuation_rows, strict=True):
                continuation_ids, continuation_eos = _trim_at_eos(
                    raw_continuation,
                    eos_ids,
                    include_eos=True,
                )
                continuation_response = _decode(tokenizer, continuation_ids)
                response = row.first_response + row.controller_text + continuation_response
                batch_transcripts[row.item_index] = build_m6_domain_transcript(
                    row.item,
                    mode=mode,
                    prompt=row.prompt,
                    response=response,
                    first_pass_response=row.first_response,
                    continuation_response=continuation_response,
                    controller_action=(
                        "forced_close_continue" if row.forced else "natural_close_continue"
                    ),
                    prompt_tokens=row.prompt_tokens,
                    first_pass_tokens=len(row.first_ids),
                    continuation_tokens=len(continuation_ids),
                    injected_tokens=row.injected_tokens,
                    finish_reason="eos" if continuation_eos else "length",
                    json_repair_policy=json_repair_policy,
                )
        if any(item is None for item in batch_transcripts):
            raise M6DomainError("M6 domain batch did not produce every transcript")
        transcripts.extend(cast(M6DomainTranscript, item) for item in batch_transcripts)
    raw_path = output_dir / "results.jsonl"
    ordered = tuple(transcripts)
    if tuple(item.item_id for item in ordered) != tuple(item.id for item in items):
        raise M6DomainError("M6 domain transcript identities differ from frozen suite")
    _atomic_jsonl(raw_path, ordered)
    evaluation_id = (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-m6-domain-{mode}-"
        f"{model_identity.role}-{model_identity.model_artifact_sha256[:8]}"
    )
    summary = _summary(
        model=model_identity,
        mode=mode,
        config_sha256=config_sha256,
        git_commit=git_commit,
        transcripts=ordered,
        evaluation_id=evaluation_id,
        duration_seconds=time.monotonic() - started,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        physical_gpu_index=physical_gpu_index,
        gpu_name=torch.cuda.get_device_name(device),
        environment_sha256=sha256_file(environment_path),
        hardware_sha256=sha256_file(hardware_path),
        raw_results_sha256=sha256_file(raw_path),
        protocol_version=release.protocol_version,
        suite_version=release.suite_version,
    )
    _atomic_json(output_dir / "summary.json", summary.to_dict())
    return summary


def finalize_m6_domain_pass(
    *,
    project_root: Path,
    pass_directory: Path,
    judgments_path: Path,
) -> M6DomainModeResult:
    """Apply all 40 maintainer judgments and emit content-free item scores."""

    try:
        summary = M6DomainPassSummary.model_validate_json(
            (pass_directory / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise M6DomainError("M6 domain pass summary is invalid") from exc
    raw_path = pass_directory / "results.jsonl"
    if summary.raw_results_sha256 != sha256_file(raw_path):
        raise M6DomainError("M6 transcript identity differs from pass summary")
    transcripts = _load_transcripts(raw_path)
    items = load_evaluation_items(_suite_items_path(project_root.resolve(), summary.suite_version))
    judgments = load_human_rubric_judgments(judgments_path)
    judgment_map: dict[str, HumanRubricJudgment] = {
        judgment.item_id: judgment for judgment in judgments
    }
    expected_human = tuple(item.id for item in items if item.scorer.kind == "human_rubric")
    if tuple(judgment_map) != expected_human:
        raise M6DomainError("M6 review must cover the exact 40 human-rubric items in order")
    if tuple(item.id for item in items) != tuple(item.item_id for item in transcripts):
        raise M6DomainError("M6 transcript identities differ from frozen suite")
    scores: list[M6DomainItemScore] = []
    for item, transcript in zip(items, transcripts, strict=True):
        correct = (
            judgment_map[item.id].passed
            if transcript.human_review_required
            else bool(transcript.automatic_correct)
        )
        correct = correct and transcript.format_valid
        scores.append(
            M6DomainItemScore(
                item_id=item.id,
                cluster_id=domain_cluster_id(item),
                language=item.language,
                category=item.category,
                scorer_kind=item.scorer.kind,
                correct=correct,
                json_valid=transcript.json_valid,
                json_repaired=transcript.output_repair_action != "none",
                format_valid=transcript.format_valid,
                visible_reasoning_leakage=transcript.visible_reasoning_leakage,
            )
        )
    result_items = tuple(scores)
    correct_items = sum(item.correct for item in result_items)
    format_items = sum(item.format_valid for item in result_items)
    json_valid = sum(item.json_valid is True for item in result_items)
    json_repaired = sum(item.json_repaired for item in result_items)
    leakage = sum(item.visible_reasoning_leakage for item in result_items)
    forced = sum(item.budget_forced_close for item in transcripts)
    result = M6DomainModeResult(
        mode=summary.mode,
        items=result_items,
        evaluated_items=300,
        correct_items=correct_items,
        score_basis_points=round(correct_items * 10000 / 300),
        format_valid_items=format_items,
        format_valid_basis_points=round(format_items * 10000 / 300),
        json_items=80,
        json_valid_items=json_valid,
        json_valid_basis_points=round(json_valid * 10000 / 80),
        json_repaired_items=json_repaired,
        visible_reasoning_leakage_items=leakage,
        visible_reasoning_leakage_basis_points=round(leakage * 10000 / 300),
        natural_thinking_closed_items=sum(item.natural_thinking_closed for item in transcripts),
        budget_forced_close_items=forced,
        forced_close_basis_points=round(forced * 10000 / 300),
        generated_tokens=sum(item.generated_tokens for item in transcripts),
        injected_tokens=sum(item.injected_tokens for item in transcripts),
    )
    _atomic_json(pass_directory / "mode_result.json", result.to_dict())
    judgment_hash = sha256_file(judgments_path)
    completed = summary.model_copy(
        update={
            "status": "succeeded",
            "human_review_pending": 0,
            "human_reviewed": 40,
            "human_passed": sum(judgment.passed for judgment in judgments),
            "human_review_sha256": judgment_hash,
        }
    )
    _atomic_json(pass_directory / "summary.json", completed.to_dict())
    return result
