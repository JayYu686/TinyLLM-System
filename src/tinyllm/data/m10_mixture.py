"""Build the immutable exact-token M10 Agent SFT mixture."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import unicodedata
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from pydantic import ValidationError
from torch import Tensor
from torch.utils.data import Dataset

from tinyllm.data.m5_dual_mode_correction import align_legacy_nonthinking_sequence_v2
from tinyllm.data.m5_mixture import (
    M5MixtureSequence,
    open_m5_ablation_mixture,
)
from tinyllm.data.m5_mixture_schema import M6DomainGeneralizationMixtureManifest
from tinyllm.data.m10_agent import load_m10_agent_data_config
from tinyllm.data.m10_agent_schema import M10SourceId
from tinyllm.data.m10_canonical_schema import (
    M10CanonicalTrainingSample,
    M10ExternalImportManifest,
)
from tinyllm.data.m10_devops import (
    _candidate_pairs,
    _shingles,
    _signature,
    _similarity,
    load_bfcl_target,
    load_dataset,
    load_m6_domain_target,
    load_m9_target,
)
from tinyllm.data.m10_devops_schema import (
    M10DevOpsContentReviewResult,
    M10DevOpsDatasetManifest,
    M10DevOpsTrainingSample,
    canonical_json_sha256,
)
from tinyllm.data.m10_mixture_schema import (
    M10FrozenMixtureConfig,
    M10FrozenMixtureManifest,
    M10FrozenMixtureReport,
    M10MixtureArtifact,
    M10MixtureInputEvidence,
    M10MixtureLanguage,
    M10MixtureMode,
)
from tinyllm.data.registry import open_registered_dataset
from tinyllm.data.tokenization import (
    TokenizersBackend,
    load_m2_tokenization_config,
)

_SEQUENCE_FILE = "sequences.npz"
_MANIFEST_FILE = "manifest.json"
_COMMIT_FILE = "COMMITTED.json"
_PAD_TOKEN_ID = 151643
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SOURCE_CODES: dict[M10SourceId, int] = {
    "toolace": 0,
    "hermes_function_calling": 1,
    "tinyllm_devops": 2,
    "m6_domain_replay": 3,
    "m2_no_tool_replay": 4,
}
_LANGUAGE_CODES: dict[M10MixtureLanguage, int] = {"en": 0, "zh": 1}
_MODE_CODES: dict[M10MixtureMode, int] = {"nonthinking": 0, "thinking": 1}
_TEXT_SOURCE_PRIORITY: dict[M10SourceId, int] = {
    "tinyllm_devops": 0,
    "hermes_function_calling": 1,
    "toolace": 2,
    "m6_domain_replay": 3,
    "m2_no_tool_replay": 4,
}
_AGENT_TEMPLATE_SPEC: dict[str, object] = {
    "assistant_context": "<think>\n\n</think>\n\n",
    "assistant_supervision": "content_tool_calls_and_im_end",
    "id": "qwen3-agent-chatml-nonthinking-v1",
    "tool_catalog": "qwen3-tools-xml-v1",
    "tool_results": "qwen3-tool-response-xml-v1",
}
_AGENT_TEMPLATE_SHA256 = canonical_json_sha256(_AGENT_TEMPLATE_SPEC)


class M10MixtureError(ValueError):
    """Raised when any M10 source, gate, or output identity fails closed."""


@dataclass(frozen=True, slots=True)
class M10TextCandidate:
    """One verified textual source row before deduplication and tokenization."""

    source_id: M10SourceId
    version: str
    record_id: str
    record_sha256: str
    group_id: str
    language: M10MixtureLanguage
    tools: tuple[dict[str, object], ...]
    messages: tuple[dict[str, object], ...]
    prompt: str
    prompt_sha256: str
    tool_schema_text: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class M10TrainingSequence:
    """One fixed-length sequence with explicit mixture stratum identity."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    source_id: M10SourceId
    language: M10MixtureLanguage
    mode: M10MixtureMode
    record_sha256: str

    @property
    def supervised_tokens(self) -> int:
        return sum(value != -100 for value in self.labels[1:])


@dataclass(frozen=True, slots=True)
class M10SourceCandidates:
    """Candidate sequences and source-level accounting after verification."""

    source_id: M10SourceId
    version: str
    content_sha256: str
    manifest_sha256: str
    input_candidates: int
    sequences: tuple[M10TrainingSequence, ...]
    overlength_rejections: int


@dataclass(frozen=True, slots=True)
class M10MixtureBuild:
    """In-memory build ready for atomic persistence."""

    manifest: M10FrozenMixtureManifest
    arrays: dict[str, np.ndarray]
    duplicate_report: dict[str, object]
    contamination_report: dict[str, object]


def _json_bytes(value: object, *, indent: int | None = None) -> bytes:
    separators = (",", ":") if indent is None else None
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=indent, separators=separators
    )
    return (text + ("\n" if indent is not None else "")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise M10MixtureError(f"required immutable JSON is missing or unsafe: {path}")
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M10MixtureError(f"cannot decode immutable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise M10MixtureError(f"immutable JSON must be an object: {path}")
    return cast(dict[str, Any], value)


def load_frozen_mixture_config(path: Path) -> M10FrozenMixtureConfig:
    """Load a strict YAML recipe without accepting unknown fields."""

    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return M10FrozenMixtureConfig.model_validate(value)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise M10MixtureError("M10 frozen mixture config is invalid") from exc


def _verify_committed_files(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise M10MixtureError(f"committed source directory is missing or unsafe: {root}")
    marker = _safe_json(root / _COMMIT_FILE)
    files = marker.get("files")
    if not isinstance(files, dict) or not files:
        raise M10MixtureError("committed source marker has no file identities")
    verified: dict[str, str] = {}
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str) or Path(name).name != name:
            raise M10MixtureError("committed source marker contains an unsafe filename")
        path = root / name
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
            raise M10MixtureError(f"committed source file hash differs: {name}")
        verified[name] = expected
    return verified


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file() or path.is_symlink():
        raise M10MixtureError(f"required JSONL is missing or unsafe: {path}")
    values: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value: object = json.loads(line)
                    if not isinstance(value, dict):
                        raise M10MixtureError("M10 JSONL row is not an object")
                    values.append(cast(dict[str, Any], value))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M10MixtureError(f"cannot decode JSONL: {path}") from exc
    return tuple(values)


def _external_tool(value: object) -> dict[str, object]:
    tool = cast(Any, value)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


def _external_message(value: object) -> dict[str, object]:
    message = cast(Any, value)
    calls = tuple(
        {"id": call.id, "name": call.name, "arguments": call.arguments}
        for call in message.tool_calls
    )
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": calls,
        "tool_call_ids": tuple(message.tool_call_ids),
    }


def _external_candidate(sample: M10CanonicalTrainingSample, *, version: str) -> M10TextCandidate:
    tools = tuple(_external_tool(item) for item in sample.tools)
    messages = tuple(_external_message(item) for item in sample.messages)
    prompt = "\n".join(str(item["content"] or "") for item in messages if item["role"] == "user")
    return M10TextCandidate(
        source_id=sample.source_id,
        version=version,
        record_id=sample.source_record_id,
        record_sha256=sample.source_record_sha256,
        group_id=sample.group_id,
        language=sample.language,
        tools=tools,
        messages=messages,
        prompt=prompt,
        prompt_sha256=sample.prompt_sha256,
        tool_schema_text=json.dumps(tools, ensure_ascii=False, sort_keys=True),
        content_sha256=sample.content_sha256,
    )


def _devops_tool(value: object) -> dict[str, object]:
    tool = cast(Any, value)
    return {
        "type": "function",
        "function": {
            "name": tool.tool_name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _devops_message(value: object) -> dict[str, object]:
    message = cast(Any, value)
    calls = tuple(
        {
            "id": call.id,
            "name": call.function.name,
            "arguments": call.function.arguments,
        }
        for call in message.tool_calls
    )
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": calls,
        "tool_call_ids": (message.tool_call_id,) if message.tool_call_id else (),
    }


def _devops_candidate(sample: M10DevOpsTrainingSample, *, version: str) -> M10TextCandidate:
    tools = tuple(_devops_tool(item) for item in sample.available_tools)
    messages = tuple(_devops_message(item) for item in sample.messages)
    prompt = "\n".join(str(item["content"] or "") for item in messages if item["role"] == "user")
    return M10TextCandidate(
        source_id="tinyllm_devops",
        version=version,
        record_id=sample.sample_id,
        record_sha256=sample.source_record_sha256,
        group_id=sample.group_id,
        language=sample.language,
        tools=tools,
        messages=messages,
        prompt=prompt,
        prompt_sha256=sample.prompt_sha256,
        tool_schema_text=json.dumps(tools, ensure_ascii=False, sort_keys=True),
        content_sha256=sample.content_sha256,
    )


def load_external_candidates(
    root: Path, *, expected_version: str, expected_content: str, expected_manifest: str
) -> tuple[M10ExternalImportManifest, tuple[M10TextCandidate, ...]]:
    """Open one canonical import only after rebuilding all immutable identities."""

    _verify_committed_files(root)
    manifest_path = root / _MANIFEST_FILE
    if _sha256_file(manifest_path) != expected_manifest:
        raise M10MixtureError("M10 external manifest hash differs from frozen config")
    try:
        manifest = M10ExternalImportManifest.model_validate_json(manifest_path.read_bytes())
        samples = tuple(
            M10CanonicalTrainingSample.model_validate(value)
            for value in _load_jsonl(root / "items.jsonl")
        )
    except ValidationError as exc:
        raise M10MixtureError("M10 canonical import failed Schema validation") from exc
    rebuilt_content = canonical_json_sha256([item.content_sha256 for item in samples])
    if (
        root.name != expected_version
        or manifest.import_version != expected_version
        or manifest.content_sha256 != expected_content
        or rebuilt_content != expected_content
        or len(samples) != manifest.accepted_rows
    ):
        raise M10MixtureError("M10 canonical import identity or row count differs")
    return manifest, tuple(_external_candidate(item, version=expected_version) for item in samples)


def load_approved_devops_candidates(
    dataset_root: Path,
    approval_root: Path,
    *,
    expected_version: str,
    expected_content: str,
    expected_manifest: str,
    expected_approval: str,
) -> tuple[M10DevOpsDatasetManifest, tuple[M10TextCandidate, ...]]:
    """Bind pending source bytes to the immutable maintainer approval packet."""

    pending, samples = load_dataset(dataset_root)
    approved_manifest_path = approval_root / "approved-manifest.json"
    approval_path = approval_root / "approval.json"
    if (
        _sha256_file(approved_manifest_path) != expected_manifest
        or _sha256_file(approval_path) != expected_approval
    ):
        raise M10MixtureError("M10 authored approval file hash differs")
    try:
        approved = M10DevOpsDatasetManifest.model_validate_json(approved_manifest_path.read_bytes())
        approval = M10DevOpsContentReviewResult.model_validate_json(approval_path.read_bytes())
    except ValidationError as exc:
        raise M10MixtureError("M10 authored approval failed Schema validation") from exc
    if (
        dataset_root.name != expected_version
        or approved.dataset_version != expected_version
        or approved.content_sha256 != expected_content
        or pending.content_sha256 != expected_content
        or not approved.training_permitted
        or approval.source_dataset_version != expected_version
        or approval.source_content_sha256 != expected_content
        or approval.approved_manifest_sha256 != expected_manifest
    ):
        raise M10MixtureError("M10 authored approval is not bound to the source")
    return approved, tuple(_devops_candidate(item, version=expected_version) for item in samples)


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _exact_candidate_identity(item: M10TextCandidate) -> str:
    """Hash normalized semantic conversation content without source-local call IDs."""

    messages = []
    for message in item.messages:
        calls = tuple(
            {
                "name": call["name"],
                "arguments": call["arguments"],
            }
            for call in cast(Sequence[dict[str, object]], message["tool_calls"])
        )
        messages.append(
            {
                "role": message["role"],
                "content": _normalized(str(message["content"] or "")),
                "tool_calls": calls,
            }
        )
    return canonical_json_sha256(
        {
            "messages": messages,
            "tools": json.loads(item.tool_schema_text),
        }
    )


def deduplicate_text_candidates(
    candidates: Sequence[M10TextCandidate],
) -> tuple[tuple[M10TextCandidate, ...], dict[str, object]]:
    """Deduplicate normalized prompt+schema identities, then cross-source near matches."""

    ordered = sorted(
        candidates,
        key=lambda item: (
            _TEXT_SOURCE_PRIORITY[item.source_id],
            item.source_id,
            item.record_id,
            item.content_sha256,
        ),
    )
    exact_seen: set[str] = set()
    exact_kept: list[M10TextCandidate] = []
    exact_drops = 0
    exact_source_drops: Counter[M10SourceId] = Counter()
    for item in ordered:
        identity = _exact_candidate_identity(item)
        if identity in exact_seen:
            exact_drops += 1
            exact_source_drops[item.source_id] += 1
            continue
        exact_seen.add(identity)
        exact_kept.append(item)

    prompt_sets = tuple(_shingles(item.prompt) for item in exact_kept)
    prompt_signatures = tuple(_signature(item) for item in prompt_sets)
    schema_sets = tuple(_shingles(item.tool_schema_text) for item in exact_kept)
    dropped: set[int] = set()
    maximum_prompt = 0.0
    maximum_schema = 0.0
    near_pairs = 0
    for left, right in sorted(_candidate_pairs(prompt_signatures)):
        if exact_kept[left].source_id == exact_kept[right].source_id:
            continue
        prompt_similarity = _similarity(prompt_sets[left], prompt_sets[right])
        maximum_prompt = max(maximum_prompt, prompt_similarity)
        if prompt_similarity < 0.85:
            continue
        schema_similarity = _similarity(schema_sets[left], schema_sets[right])
        maximum_schema = max(maximum_schema, schema_similarity)
        if schema_similarity < 0.85:
            continue
        near_pairs += 1
        loser = max(
            (left, right),
            key=lambda index: (
                _TEXT_SOURCE_PRIORITY[exact_kept[index].source_id],
                exact_kept[index].record_id,
            ),
        )
        dropped.add(loser)
    near_source_drops: Counter[M10SourceId] = Counter(
        exact_kept[index].source_id for index in dropped
    )
    kept = tuple(item for index, item in enumerate(exact_kept) if index not in dropped)
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "algorithm": "normalized-exact-plus-minhash-5gram-v1",
        "input_items": len(candidates),
        "output_items": len(kept),
        "exact_duplicate_drops": exact_drops,
        "cross_source_near_pairs": near_pairs,
        "near_duplicate_drops": len(dropped),
        "source_duplicate_drops": {
            source: exact_source_drops[source] + near_source_drops[source]
            for source in cast(
                tuple[M10SourceId, ...],
                ("toolace", "hermes_function_calling", "tinyllm_devops"),
            )
        },
        "maximum_cross_source_prompt_similarity_basis_points": round(maximum_prompt * 10_000),
        "maximum_qualifying_tool_schema_similarity_basis_points": round(maximum_schema * 10_000),
        "prompt_threshold_basis_points": 8500,
        "tool_schema_threshold_basis_points": 8500,
        "contains_source_content": False,
        "status": "pass",
    }
    payload["report_sha256"] = canonical_json_sha256(payload)
    return kept, payload


def scan_text_contamination(
    candidates: Sequence[M10TextCandidate], *, targets: Sequence[object]
) -> dict[str, object]:
    """Scan textual M10 sources against the four frozen evaluation boundaries."""

    expected = ("m9_dev", "m9_release", "bfcl_core", "m6_domain")
    if tuple(cast(Any, item).target_id for item in targets) != expected:
        raise M10MixtureError("M10 contamination targets differ from frozen order")
    source_sets = tuple(_shingles(item.prompt) for item in candidates)
    source_signatures = tuple(_signature(item) for item in source_sets)
    normalized_source = Counter(_normalized(item.prompt) for item in candidates)
    results: list[dict[str, object]] = []
    for raw_target in targets:
        target = cast(Any, raw_target)
        target_sets = tuple(_shingles(item) for item in target.prompts)
        target_signatures = tuple(_signature(item) for item in target_sets)
        normalized_target = Counter(_normalized(item) for item in target.prompts)
        exact = sum(
            count * normalized_target.get(value, 0) for value, count in normalized_source.items()
        )
        near = 0
        maximum = 0.0
        for left, right in _candidate_pairs(source_signatures, target_signatures):
            similarity = _similarity(source_sets[left], target_sets[right])
            maximum = max(maximum, similarity)
            if similarity >= 0.85 and _normalized(candidates[left].prompt) != _normalized(
                target.prompts[right]
            ):
                near += 1
        results.append(
            {
                "target_id": target.target_id,
                "target_version": target.version,
                "target_content_sha256": target.content_sha256,
                "target_items": len(target.prompts),
                "exact_matches": exact,
                "near_matches": near,
                "maximum_candidate_prompt_similarity_basis_points": round(maximum * 10_000),
                "contains_target_content": False,
            }
        )
    status = (
        "pass"
        if all(not row["exact_matches"] and not row["near_matches"] for row in results)
        else "fail"
    )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "scan_version": "m10-frozen-mixture-contamination-v1",
        "algorithm": "normalized-exact-plus-minhash-5gram-v1",
        "source_items": len(candidates),
        "scanned_source_ids": ["toolace", "hermes_function_calling", "tinyllm_devops"],
        "registered_replay_evidence": {
            "m6_domain_replay": "source-manifest-evaluation-overlap-zero",
            "m2_no_tool_replay": "registered-m2-train-split-only",
        },
        "targets": results,
        "status": status,
        "contains_evaluation_content": False,
    }
    payload["report_sha256"] = canonical_json_sha256(payload)
    if status != "pass":
        raise M10MixtureError("M10 textual sources overlap a frozen evaluation boundary")
    return payload


def _render_tool_catalog(tools: Sequence[dict[str, object]]) -> str:
    rendered = "\n".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in tools
    )
    return (
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{rendered}\n</tools>\n\n"
        "For each function call, return a JSON object with function name and arguments within "
        "<tool_call></tool_call> XML tags."
    )


def render_agent_conversation(
    candidate: M10TextCandidate,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Render Qwen3 tool conversations with masked hard Non-thinking prefixes."""

    messages = list(candidate.messages)
    system = "Use the provided tools when required and answer from evidence."
    if messages and messages[0]["role"] == "system":
        system = str(messages.pop(0)["content"] or system)
    if candidate.tools:
        system = f"{system}\n\n{_render_tool_catalog(candidate.tools)}"
    parts: list[str] = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    spans: list[tuple[int, int]] = []
    length = len(parts[0])
    index = 0
    while index < len(messages):
        message = messages[index]
        role = str(message["role"])
        if role in {"user", "system"}:
            value = f"<|im_start|>{role}\n{message['content'] or ''}<|im_end|>\n"
            parts.append(value)
            length += len(value)
            index += 1
            continue
        if role == "assistant":
            prefix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            parts.append(prefix)
            length += len(prefix)
            supervised_start = length
            content = str(message["content"] or "")
            if content:
                parts.append(content)
                length += len(content)
            calls = cast(Sequence[dict[str, object]], message["tool_calls"])
            for call_index, call_value in enumerate(calls):
                if content or call_index:
                    parts.append("\n")
                    length += 1
                call = call_value
                call_text = (
                    "<tool_call>\n"
                    + json.dumps(
                        {"name": call["name"], "arguments": call["arguments"]},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n</tool_call>"
                )
                parts.append(call_text)
                length += len(call_text)
            parts.append("<|im_end|>")
            length += len("<|im_end|>")
            spans.append((supervised_start, length))
            parts.append("\n")
            length += 1
            index += 1
            continue
        if role == "tool":
            responses: list[str] = []
            while index < len(messages) and messages[index]["role"] == "tool":
                responses.append(
                    f"<tool_response>\n{messages[index]['content'] or ''}\n</tool_response>"
                )
                index += 1
            value = "<|im_start|>user\n" + "\n".join(responses) + "<|im_end|>\n"
            parts.append(value)
            length += len(value)
            continue
        raise M10MixtureError(f"unsupported M10 conversation role: {role}")
    if not spans:
        raise M10MixtureError("M10 textual candidate has no supervised assistant message")
    return "".join(parts), tuple(spans)


def _labels_from_offsets(
    ids: tuple[int, ...], offsets: tuple[tuple[int, int], ...], spans: Sequence[tuple[int, int]]
) -> tuple[int, ...]:
    labels: list[int] = []
    for token_id, (start, end) in zip(ids, offsets, strict=True):
        overlaps = tuple((left, right) for left, right in spans if start < right and end > left)
        if not overlaps:
            labels.append(-100)
        elif any(start >= left and end <= right for left, right in overlaps):
            labels.append(token_id)
        else:
            raise M10MixtureError("token crosses an M10 assistant supervision boundary")
    return tuple(labels)


def _pad_sequence(
    input_ids: Sequence[int],
    labels: Sequence[int],
    *,
    source_id: M10SourceId,
    language: M10MixtureLanguage,
    mode: M10MixtureMode,
    record_sha256: str,
) -> M10TrainingSequence:
    if len(input_ids) != len(labels) or not 1 < len(input_ids) <= 2048:
        raise M10MixtureError("M10 sequence length is outside the frozen contract")
    padding = 2048 - len(input_ids)
    sequence = M10TrainingSequence(
        input_ids=tuple(input_ids) + (_PAD_TOKEN_ID,) * padding,
        labels=tuple(labels) + (-100,) * padding,
        attention_mask=(1,) * len(input_ids) + (0,) * padding,
        source_id=source_id,
        language=language,
        mode=mode,
        record_sha256=record_sha256,
    )
    if sequence.supervised_tokens <= 0:
        raise M10MixtureError("M10 sequence has no shifted supervised tokens")
    return sequence


def tokenize_text_candidates(
    candidates: Sequence[M10TextCandidate], *, backend: TokenizersBackend
) -> tuple[tuple[M10TrainingSequence, ...], Counter[M10SourceId]]:
    """Tokenize all textual sources with exact assistant/tool-call labels."""

    sequences: list[M10TrainingSequence] = []
    rejected: Counter[M10SourceId] = Counter()
    for candidate in candidates:
        text, spans = render_agent_conversation(candidate)
        encoded = backend.encode(text)
        if len(encoded.ids) > 2048:
            rejected[candidate.source_id] += 1
            continue
        labels = _labels_from_offsets(encoded.ids, encoded.offsets, spans)
        sequences.append(
            _pad_sequence(
                encoded.ids,
                labels,
                source_id=candidate.source_id,
                language=candidate.language,
                mode="nonthinking",
                record_sha256=candidate.record_sha256,
            )
        )
    return tuple(sequences), rejected


def _active_ids(sequence: M10TrainingSequence) -> tuple[int, ...]:
    return tuple(
        token
        for token, active in zip(sequence.input_ids, sequence.attention_mask, strict=True)
        if active
    )


def _decoded_language(backend: TokenizersBackend, token_ids: Sequence[int]) -> M10MixtureLanguage:
    return "zh" if _CJK_RE.search(backend.decode(token_ids)) else "en"


def load_m6_replay_candidates(
    root: Path, *, backend: TokenizersBackend, expected_version: str
) -> tuple[M6DomainGeneralizationMixtureManifest, tuple[M10TrainingSequence, ...]]:
    """Open hash-verified M6 arrays and retain their native mode labels."""

    opened = open_m5_ablation_mixture(root)
    if not isinstance(opened.manifest, M6DomainGeneralizationMixtureManifest):
        raise M10MixtureError("M10 M6 replay source has the wrong manifest kind")
    manifest = opened.manifest
    if root.name != expected_version or manifest.mixture_version != expected_version:
        raise M10MixtureError("M10 M6 replay version differs")
    sequences: list[M10TrainingSequence] = []
    with np.load(root / manifest.artifact.path, allow_pickle=False) as arrays:
        for index in range(manifest.sequence_count):
            active = int(np.asarray(arrays["attention_masks"][index]).sum())
            ids = tuple(int(value) for value in arrays["input_ids"][index][:active])
            labels = tuple(int(value) for value in arrays["labels"][index][:active])
            raw = np.asarray(ids, dtype="<i4").tobytes() + np.asarray(labels, dtype="<i4").tobytes()
            sequences.append(
                _pad_sequence(
                    ids,
                    labels,
                    source_id="m6_domain_replay",
                    language=_decoded_language(backend, ids),
                    mode="thinking" if int(arrays["modes"][index]) else "nonthinking",
                    record_sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
    return manifest, tuple(sequences)


def load_m2_replay_candidates(
    *, artifact_root: Path, backend: TokenizersBackend, expected_version: str
) -> tuple[object, tuple[M10TrainingSequence, ...], int]:
    """Split registered Train Packs, align the Qwen3 hard switch, and classify language."""

    registered = open_registered_dataset(
        artifact_root=artifact_root, dataset_version=expected_version
    )
    sequences: list[M10TrainingSequence] = []
    overlength = 0
    for pack in registered.iter_packs():
        if str(pack.split) != "train":
            continue
        cursor = 0
        for sample_id, count in zip(pack.sample_ids, pack.sample_token_counts, strict=True):
            end = cursor + count
            ids = pack.input_ids[cursor:end]
            labels = pack.labels[cursor:end]
            cursor = end
            raw = M5MixtureSequence(
                input_ids=ids + (_PAD_TOKEN_ID,) * (1024 - len(ids)),
                labels=labels + (-100,) * (1024 - len(labels)),
                attention_mask=(1,) * len(ids) + (0,) * (1024 - len(ids)),
                mode=0,
            )
            try:
                aligned = align_legacy_nonthinking_sequence_v2(raw)
            except ValueError:
                overlength += 1
                continue
            active = sum(aligned.attention_mask)
            aligned_ids = aligned.input_ids[:active]
            language: M10MixtureLanguage = (
                "en"
                if sample_id.startswith("commitpackft:")
                else _decoded_language(backend, aligned_ids)
            )
            sequences.append(
                _pad_sequence(
                    aligned_ids,
                    aligned.labels[:active],
                    source_id="m2_no_tool_replay",
                    language=language,
                    mode="nonthinking",
                    record_sha256=hashlib.sha256(sample_id.encode()).hexdigest(),
                )
            )
    return registered.manifest, tuple(sequences), overlength


def _trim_supervision(sequence: M10TrainingSequence, keep: int) -> M10TrainingSequence:
    if not 0 < keep < sequence.supervised_tokens:
        raise M10MixtureError("partial M10 supervision count is invalid")
    labels = list(sequence.labels)
    remaining = keep
    for index in range(1, len(labels)):
        if labels[index] == -100:
            continue
        if remaining:
            remaining -= 1
        else:
            labels[index] = -100
    if remaining:
        raise M10MixtureError("could not trim M10 supervision exactly")
    return replace(sequence, labels=tuple(labels))


def select_exact_sequences(
    candidates: Sequence[M10TrainingSequence], *, target: int, seed: int
) -> tuple[tuple[M10TrainingSequence, ...], int, int]:
    """Cycle deterministic shuffled epochs until the exact stratum budget is reached."""

    if not candidates or target <= 0:
        raise M10MixtureError("M10 stratum has no candidates or invalid target")
    rng = random.Random(seed)
    selected: list[M10TrainingSequence] = []
    consumed = 0
    reuse = 0
    partial = 0
    epoch = 0
    while consumed < target:
        order = list(range(len(candidates)))
        rng.shuffle(order)
        if epoch:
            reuse += len(order)
        for index in order:
            sequence = candidates[index]
            remaining = target - consumed
            if sequence.supervised_tokens > remaining:
                sequence = _trim_supervision(sequence, remaining)
                partial += 1
            selected.append(sequence)
            consumed += sequence.supervised_tokens
            if consumed == target:
                return tuple(selected), reuse, partial
        epoch += 1
    raise AssertionError("unreachable exact M10 token selector state")


def _array_content_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = arrays[name]
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_frozen_mixture(
    *,
    config_path: Path,
    source_config_path: Path,
    tokenizer_config_path: Path,
    model_dir: Path,
    artifact_root: Path,
    m9_dev_dir: Path,
    m9_release_dir: Path,
    bfcl_data_root: Path,
    m6_domain_dir: Path,
) -> M10MixtureBuild:
    """Verify five inputs, run leakage gates, and create exact-token arrays."""

    config = load_frozen_mixture_config(config_path)
    if _sha256_file(source_config_path) != config.source_config_sha256:
        raise M10MixtureError("M10 preregistered source config hash differs")
    source_config = load_m10_agent_data_config(source_config_path)
    if source_config.training_permitted or source_config.status != "preregistered":
        raise M10MixtureError("M10 source preregistration must remain immutable and fail closed")
    frozen_config_sha = _sha256_file(config_path)
    tokenization = load_m2_tokenization_config(tokenizer_config_path)
    backend = TokenizersBackend.from_files(
        model_dir / tokenization.tokenizer.tokenizer_file,
        model_dir / tokenization.tokenizer.tokenizer_config_file,
        tokenization.tokenizer,
    )

    toolace_spec = config.inputs.toolace
    toolace_manifest, toolace_text = load_external_candidates(
        artifact_root / "datasets/m10-agent/external/toolace" / toolace_spec.version,
        expected_version=toolace_spec.version,
        expected_content=toolace_spec.content_sha256,
        expected_manifest=toolace_spec.manifest_sha256,
    )
    hermes_spec = config.inputs.hermes_function_calling
    hermes_manifest, hermes_text = load_external_candidates(
        artifact_root / "datasets/m10-agent/external/hermes" / hermes_spec.version,
        expected_version=hermes_spec.version,
        expected_content=hermes_spec.content_sha256,
        expected_manifest=hermes_spec.manifest_sha256,
    )
    devops_spec = config.inputs.tinyllm_devops
    devops_manifest, devops_text = load_approved_devops_candidates(
        artifact_root / "datasets/m10-agent/devops" / devops_spec.version,
        artifact_root / "reviews" / devops_spec.version / "approval-v1",
        expected_version=devops_spec.version,
        expected_content=devops_spec.content_sha256,
        expected_manifest=devops_spec.manifest_sha256,
        expected_approval=cast(str, devops_spec.approval_sha256),
    )
    textual, duplicate_report = deduplicate_text_candidates(
        toolace_text + hermes_text + devops_text
    )
    targets = (
        load_m9_target(m9_dev_dir, target_id="m9_dev"),
        load_m9_target(m9_release_dir, target_id="m9_release"),
        load_bfcl_target(bfcl_data_root),
        load_m6_domain_target(m6_domain_dir),
    )
    observed_versions = tuple(cast(Any, item).version for item in targets)
    expected_versions = (
        config.contamination.m9_dev_version,
        config.contamination.m9_release_version,
        config.contamination.bfcl_version,
        config.contamination.m6_domain_version,
    )
    if observed_versions != expected_versions:
        raise M10MixtureError("M10 evaluation boundary version differs from frozen config")
    contamination = scan_text_contamination(textual, targets=targets)
    text_sequences, text_rejections = tokenize_text_candidates(textual, backend=backend)

    m6_spec = config.inputs.m6_domain_replay
    m6_manifest, m6_sequences = load_m6_replay_candidates(
        artifact_root / "datasets/m6-domain-generalization" / m6_spec.version,
        backend=backend,
        expected_version=m6_spec.version,
    )
    if (
        m6_manifest.content_sha256 != m6_spec.content_sha256
        or _sha256_file(
            artifact_root / "datasets/m6-domain-generalization" / m6_spec.version / _MANIFEST_FILE
        )
        != m6_spec.manifest_sha256
    ):
        raise M10MixtureError("M10 M6 replay lineage differs")
    m2_spec = config.inputs.m2_no_tool_replay
    m2_manifest, m2_sequences, m2_rejections = load_m2_replay_candidates(
        artifact_root=artifact_root, backend=backend, expected_version=m2_spec.version
    )
    if (
        cast(Any, m2_manifest).content_sha256 != m2_spec.content_sha256
        or _sha256_file(artifact_root / "datasets/m2-sft" / m2_spec.version / _MANIFEST_FILE)
        != m2_spec.manifest_sha256
    ):
        raise M10MixtureError("M10 M2 replay lineage differs")

    all_sequences = text_sequences + m6_sequences + m2_sequences
    selected: list[M10TrainingSequence] = []
    reuse_counts: dict[str, int] = {}
    partial_counts: dict[str, int] = {}
    stratum_counts: dict[str, int] = {}
    for stratum in config.strata:
        key = f"{stratum.source_id}:{stratum.language}:{stratum.mode}"
        pool = tuple(
            item
            for item in all_sequences
            if item.source_id == stratum.source_id
            and item.language == stratum.language
            and item.mode == stratum.mode
        )
        seed = int(hashlib.sha256(f"{config.build_seed}:{key}".encode()).hexdigest()[:8], 16)
        chosen, reuse, partial = select_exact_sequences(
            pool, target=stratum.supervised_tokens, seed=seed
        )
        selected.extend(chosen)
        reuse_counts[key] = reuse
        partial_counts[key] = partial
        stratum_counts[key] = sum(item.supervised_tokens for item in chosen)
    random.Random(config.build_seed).shuffle(selected)
    arrays = {
        "attention_masks": np.asarray([item.attention_mask for item in selected], dtype="u1"),
        "input_ids": np.asarray([item.input_ids for item in selected], dtype="<i4"),
        "labels": np.asarray([item.labels for item in selected], dtype="<i4"),
        "languages": np.asarray([_LANGUAGE_CODES[item.language] for item in selected], dtype="u1"),
        "modes": np.asarray([_MODE_CODES[item.mode] for item in selected], dtype="u1"),
        "source_ids": np.asarray([_SOURCE_CODES[item.source_id] for item in selected], dtype="u1"),
    }
    arrays_sha = _array_content_hash(arrays)
    input_evidence = (
        M10MixtureInputEvidence(
            source_id="toolace",
            version=toolace_spec.version,
            content_sha256=toolace_spec.content_sha256,
            manifest_sha256=toolace_spec.manifest_sha256,
            input_candidates=toolace_manifest.accepted_rows,
            accepted_candidates=sum(item.source_id == "toolace" for item in text_sequences),
            duplicate_rejections=cast(dict[str, int], duplicate_report["source_duplicate_drops"])[
                "toolace"
            ],
            overlength_rejections=text_rejections["toolace"],
        ),
        M10MixtureInputEvidence(
            source_id="hermes_function_calling",
            version=hermes_spec.version,
            content_sha256=hermes_spec.content_sha256,
            manifest_sha256=hermes_spec.manifest_sha256,
            input_candidates=hermes_manifest.accepted_rows,
            accepted_candidates=sum(
                item.source_id == "hermes_function_calling" for item in text_sequences
            ),
            duplicate_rejections=cast(dict[str, int], duplicate_report["source_duplicate_drops"])[
                "hermes_function_calling"
            ],
            overlength_rejections=text_rejections["hermes_function_calling"],
        ),
        M10MixtureInputEvidence(
            source_id="tinyllm_devops",
            version=devops_spec.version,
            content_sha256=devops_spec.content_sha256,
            manifest_sha256=devops_spec.manifest_sha256,
            input_candidates=devops_manifest.item_count,
            accepted_candidates=sum(item.source_id == "tinyllm_devops" for item in text_sequences),
            duplicate_rejections=cast(dict[str, int], duplicate_report["source_duplicate_drops"])[
                "tinyllm_devops"
            ],
            overlength_rejections=text_rejections["tinyllm_devops"],
        ),
        M10MixtureInputEvidence(
            source_id="m6_domain_replay",
            version=m6_spec.version,
            content_sha256=m6_spec.content_sha256,
            manifest_sha256=m6_spec.manifest_sha256,
            input_candidates=len(m6_sequences),
            accepted_candidates=len(m6_sequences),
            duplicate_rejections=0,
            overlength_rejections=0,
        ),
        M10MixtureInputEvidence(
            source_id="m2_no_tool_replay",
            version=m2_spec.version,
            content_sha256=m2_spec.content_sha256,
            manifest_sha256=m2_spec.manifest_sha256,
            input_candidates=len(m2_sequences) + m2_rejections,
            accepted_candidates=len(m2_sequences),
            duplicate_rejections=0,
            overlength_rejections=m2_rejections,
        ),
    )
    identity = {
        "arrays_sha256": arrays_sha,
        "build_seed": config.build_seed,
        "contamination_report_sha256": contamination["report_sha256"],
        "duplicate_report_sha256": duplicate_report["report_sha256"],
        "frozen_config_sha256": frozen_config_sha,
        "input_content_sha256s": [item.content_sha256 for item in input_evidence],
        "stratum_supervised_tokens": stratum_counts,
        "template_sha256": _AGENT_TEMPLATE_SHA256,
        "tokenizer_sha256": tokenization.tokenizer.tokenizer_sha256,
    }
    content_sha = canonical_json_sha256(identity)
    source_counts = {
        source: sum(item.supervised_tokens for item in selected if item.source_id == source)
        for source in _SOURCE_CODES
    }
    language_counts = {
        language: sum(item.supervised_tokens for item in selected if item.language == language)
        for language in _LANGUAGE_CODES
    }
    mode_counts = {
        mode: sum(item.supervised_tokens for item in selected if item.mode == mode)
        for mode in _MODE_CODES
    }
    manifest = M10FrozenMixtureManifest(
        dataset_version=f"m10-agent-sft-v1-{content_sha[:8]}",
        content_sha256=content_sha,
        frozen_config_sha256=frozen_config_sha,
        source_config_sha256=config.source_config_sha256,
        build_seed=config.build_seed,
        tokenizer_revision=config.tokenizer_revision,
        tokenizer_sha256=tokenization.tokenizer.tokenizer_sha256,
        template_id="qwen3-agent-chatml-nonthinking-v1",
        template_sha256=_AGENT_TEMPLATE_SHA256,
        sequence_length=config.sequence_length,
        sequence_count=len(selected),
        target_supervised_tokens=config.target_supervised_tokens,
        source_supervised_tokens=source_counts,
        language_supervised_tokens=language_counts,
        mode_supervised_tokens=mode_counts,
        stratum_supervised_tokens=stratum_counts,
        reuse_counts=reuse_counts,
        partial_sequence_counts=partial_counts,
        input_evidence=input_evidence,
        exact_duplicate_drops=cast(int, duplicate_report["exact_duplicate_drops"]),
        near_duplicate_drops=cast(int, duplicate_report["near_duplicate_drops"]),
        duplicate_report_sha256=cast(str, duplicate_report["report_sha256"]),
        contamination_report_sha256=cast(str, contamination["report_sha256"]),
        contamination_status="pass",
        artifact=M10MixtureArtifact(path="sequences.npz", size_bytes=1, sha256="0" * 64),
    )
    return M10MixtureBuild(
        manifest=manifest,
        arrays=arrays,
        duplicate_report=duplicate_report,
        contamination_report=contamination,
    )


def write_frozen_mixture(output_root: Path, build: M10MixtureBuild) -> M10FrozenMixtureManifest:
    """Atomically persist arrays, reports, manifest, and a complete commit marker."""

    if output_root.is_symlink():
        raise M10MixtureError("M10 mixture output root cannot be a symbolic link")
    version = build.manifest.dataset_version
    target = output_root / version
    staging = output_root / f".{version}.staging-{uuid.uuid4().hex}"
    output_root.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        sequence_path = staging / _SEQUENCE_FILE
        with sequence_path.open("xb") as handle:
            np.savez(
                handle,
                attention_masks=build.arrays["attention_masks"],
                input_ids=build.arrays["input_ids"],
                labels=build.arrays["labels"],
                languages=build.arrays["languages"],
                modes=build.arrays["modes"],
                source_ids=build.arrays["source_ids"],
            )
            handle.flush()
            os.fsync(handle.fileno())
        artifact = M10MixtureArtifact(
            path="sequences.npz",
            size_bytes=sequence_path.stat().st_size,
            sha256=_sha256_file(sequence_path),
        )
        manifest = build.manifest.model_copy(update={"artifact": artifact})
        files = {
            _MANIFEST_FILE: _json_bytes(manifest.to_dict(), indent=2),
            "duplicate-report.json": _json_bytes(build.duplicate_report, indent=2),
            "contamination-report.json": _json_bytes(build.contamination_report, indent=2),
        }
        for name, payload in files.items():
            path = staging / name
            path.write_bytes(payload)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        commit_files = {
            _SEQUENCE_FILE: _sha256_file(sequence_path),
            **{name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
        }
        commit = _json_bytes(
            {"schema_version": "1.0", "dataset_version": version, "files": commit_files},
            indent=2,
        )
        (staging / _COMMIT_FILE).write_bytes(commit)
        if target.exists():
            if target.is_symlink():
                raise M10MixtureError("existing M10 mixture target is unsafe")
            existing = open_frozen_mixture(target)
            if existing != manifest:
                raise M10MixtureError("existing M10 mixture version has different content")
            shutil.rmtree(staging)
            return existing
        os.rename(staging, target)
        return open_frozen_mixture(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def open_frozen_mixture(root: Path) -> M10FrozenMixtureManifest:
    """Reopen a committed mixture and revalidate arrays and all token accounting."""

    files = _verify_committed_files(root)
    try:
        manifest = M10FrozenMixtureManifest.model_validate_json(
            (root / _MANIFEST_FILE).read_bytes()
        )
    except (OSError, ValidationError) as exc:
        raise M10MixtureError("M10 mixture manifest is invalid") from exc
    if (
        root.name != manifest.dataset_version
        or files.get(_SEQUENCE_FILE) != manifest.artifact.sha256
    ):
        raise M10MixtureError("M10 mixture directory or artifact identity differs")
    if (root / _SEQUENCE_FILE).stat().st_size != manifest.artifact.size_bytes:
        raise M10MixtureError("M10 mixture artifact size differs")
    duplicate = _safe_json(root / "duplicate-report.json")
    contamination = _safe_json(root / "contamination-report.json")
    for report, declared, name in (
        (duplicate, manifest.duplicate_report_sha256, "duplicate"),
        (contamination, manifest.contamination_report_sha256, "contamination"),
    ):
        report_sha = report.pop("report_sha256", None)
        if report_sha != declared or canonical_json_sha256(report) != declared:
            raise M10MixtureError(f"M10 {name} report content identity differs")
        report["report_sha256"] = report_sha
    if contamination.get("status") != "pass":
        raise M10MixtureError("M10 contamination report does not pass")
    try:
        with np.load(root / _SEQUENCE_FILE, allow_pickle=False) as arrays:
            expected = {
                "attention_masks",
                "input_ids",
                "labels",
                "languages",
                "modes",
                "source_ids",
            }
            if set(arrays.files) != expected:
                raise M10MixtureError("M10 mixture arrays differ from the frozen set")
            shape = (manifest.sequence_count, manifest.sequence_length)
            if arrays["input_ids"].shape != shape or arrays["labels"].shape != shape:
                raise M10MixtureError("M10 token arrays have invalid shapes")
            if arrays["attention_masks"].shape != shape:
                raise M10MixtureError("M10 attention masks have invalid shape")
            for name in ("languages", "modes", "source_ids"):
                if arrays[name].shape != (manifest.sequence_count,):
                    raise M10MixtureError(f"M10 {name} array has invalid shape")
            expected_dtypes = {
                "attention_masks": np.dtype("u1"),
                "input_ids": np.dtype("<i4"),
                "labels": np.dtype("<i4"),
                "languages": np.dtype("u1"),
                "modes": np.dtype("u1"),
                "source_ids": np.dtype("u1"),
            }
            if any(arrays[name].dtype != dtype for name, dtype in expected_dtypes.items()):
                raise M10MixtureError("M10 array dtype differs from the frozen contract")
            if not np.logical_or(
                arrays["attention_masks"] == 0, arrays["attention_masks"] == 1
            ).all():
                raise M10MixtureError("M10 attention mask is not binary")
            if np.logical_and(arrays["attention_masks"] == 0, arrays["labels"] != -100).any():
                raise M10MixtureError("M10 padding carries supervised labels")
            if not np.logical_or(
                arrays["labels"] == -100, arrays["labels"] == arrays["input_ids"]
            ).all():
                raise M10MixtureError("M10 labels are not masked or equal to input IDs")
            if (
                not set(arrays["source_ids"].tolist()) <= set(_SOURCE_CODES.values())
                or not set(arrays["languages"].tolist()) <= set(_LANGUAGE_CODES.values())
                or not set(arrays["modes"].tolist()) <= set(_MODE_CODES.values())
            ):
                raise M10MixtureError("M10 stratum identity arrays contain unknown codes")
            valid = arrays["labels"][:, 1:] != -100
            source_counts = {
                source: int(valid[arrays["source_ids"] == code].sum())
                for source, code in _SOURCE_CODES.items()
            }
            language_counts = {
                language: int(valid[arrays["languages"] == code].sum())
                for language, code in _LANGUAGE_CODES.items()
            }
            mode_counts = {
                mode: int(valid[arrays["modes"] == code].sum())
                for mode, code in _MODE_CODES.items()
            }
            if (
                source_counts != manifest.source_supervised_tokens
                or language_counts != manifest.language_supervised_tokens
                or mode_counts != manifest.mode_supervised_tokens
            ):
                raise M10MixtureError("M10 array token accounting differs from manifest")
            stratum_counts: dict[str, int] = {}
            for key in manifest.stratum_supervised_tokens:
                source, language, mode = key.split(":")
                mask = np.logical_and.reduce(
                    (
                        arrays["source_ids"] == _SOURCE_CODES[cast(M10SourceId, source)],
                        arrays["languages"] == _LANGUAGE_CODES[cast(M10MixtureLanguage, language)],
                        arrays["modes"] == _MODE_CODES[cast(M10MixtureMode, mode)],
                    )
                )
                stratum_counts[key] = int(valid[mask].sum())
            if stratum_counts != manifest.stratum_supervised_tokens:
                raise M10MixtureError("M10 array strata differ from manifest")
            arrays_sha = _array_content_hash({name: np.asarray(arrays[name]) for name in expected})
            identity = {
                "arrays_sha256": arrays_sha,
                "build_seed": manifest.build_seed,
                "contamination_report_sha256": manifest.contamination_report_sha256,
                "duplicate_report_sha256": manifest.duplicate_report_sha256,
                "frozen_config_sha256": manifest.frozen_config_sha256,
                "input_content_sha256s": [item.content_sha256 for item in manifest.input_evidence],
                "stratum_supervised_tokens": manifest.stratum_supervised_tokens,
                "template_sha256": manifest.template_sha256,
                "tokenizer_sha256": manifest.tokenizer_sha256,
            }
            if canonical_json_sha256(identity) != manifest.content_sha256:
                raise M10MixtureError("M10 array content hash differs from manifest")
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, M10MixtureError):
            raise
        raise M10MixtureError("M10 mixture arrays cannot be decoded") from exc
    return manifest


def build_public_report(manifest: M10FrozenMixtureManifest) -> M10FrozenMixtureReport:
    """Create a path-free public report containing only aggregate build facts."""

    overlength = {item.source_id: item.overlength_rejections for item in manifest.input_evidence}
    manifest_sha = hashlib.sha256(_json_bytes(manifest.to_dict(), indent=2)).hexdigest()
    return M10FrozenMixtureReport(
        status="pass",
        dataset_version=manifest.dataset_version,
        manifest_sha256=manifest_sha,
        content_sha256=manifest.content_sha256,
        sequence_count=manifest.sequence_count,
        target_supervised_tokens=manifest.target_supervised_tokens,
        source_supervised_tokens=manifest.source_supervised_tokens,
        language_supervised_tokens=manifest.language_supervised_tokens,
        mode_supervised_tokens=manifest.mode_supervised_tokens,
        overlength_rejections=overlength,
        exact_duplicate_drops=manifest.exact_duplicate_drops,
        near_duplicate_drops=manifest.near_duplicate_drops,
        duplicate_report_sha256=manifest.duplicate_report_sha256,
        contamination_report_sha256=manifest.contamination_report_sha256,
        training_permitted=True,
    )


class M10FrozenDataset(Dataset[dict[str, Tensor]]):
    """Torch Dataset over one fully revalidated private M10.1 mixture."""

    def __init__(self, root: Path) -> None:
        self.manifest = open_frozen_mixture(root)
        with np.load(root / _SEQUENCE_FILE, allow_pickle=False) as arrays:
            self._input_ids = arrays["input_ids"].copy()
            self._labels = arrays["labels"].copy()
            self._attention_masks = arrays["attention_masks"].copy()

    def __len__(self) -> int:
        return self.manifest.sequence_count

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        import torch

        return {
            "input_ids": torch.from_numpy(self._input_ids[index].astype(np.int64, copy=False)),
            "labels": torch.from_numpy(self._labels[index].astype(np.int64, copy=False)),
            "attention_mask": torch.from_numpy(
                self._attention_masks[index].astype(np.int64, copy=False)
            ),
        }
