"""Canonical importers for the pinned ToolACE and Hermes Agent sources."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from tinyllm.data.m10_agent import (
    _HERMES_TOOL_CALL_RE,
    _TOOLACE_SCHEMA_PREFIX,
    _TOOLACE_SCHEMA_SUFFIX,
    M10AgentDataError,
    _extract_toolace_tools,
    _load_json_array,
    _safe_tool_name,
    _source,
    _tool_name_collision,
    _toolace_role_path_valid,
    _valid_hermes_tools,
    _valid_messages,
    _valid_toolace_tools,
    _verify_artifact,
    load_m10_agent_data_config,
    parse_toolace_calls,
)
from tinyllm.data.m10_canonical_schema import (
    M10CanonicalMessage,
    M10CanonicalRejectReason,
    M10CanonicalSourceId,
    M10CanonicalToolCall,
    M10CanonicalToolDefinition,
    M10CanonicalTrainingSample,
    M10ExternalImportManifest,
    M10ExternalImportReport,
    M10ExternalImportSummary,
    M10ExternalRejectedRecord,
)
from tinyllm.data.m10_devops_schema import canonical_json_sha256

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


class M10CanonicalImportError(ValueError):
    """Raised when canonical import or committed evidence is invalid."""


class _RowError(ValueError):
    def __init__(self, reason: M10CanonicalRejectReason) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class M10ExternalImportBuild:
    """In-memory private import before atomic persistence."""

    manifest: M10ExternalImportManifest
    samples: tuple[M10CanonicalTrainingSample, ...]
    rejected: tuple[M10ExternalRejectedRecord, ...]


def _json_bytes(value: object, *, indent: int | None = None) -> bytes:
    separators = (",", ":") if indent is None else None
    rendered = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=indent, separators=separators
    )
    return (rendered + ("\n" if indent is not None else "")).encode()


def _jsonl_bytes(values: tuple[object, ...]) -> bytes:
    return b"".join(_json_bytes(value) + b"\n" for value in values)


def _raw_hash(value: object) -> str:
    return canonical_json_sha256(value)


def _tool(raw: object, *, wrapped: bool) -> tuple[str, M10CanonicalToolDefinition]:
    if not isinstance(raw, dict):
        raise _RowError("invalid_tool_schema")
    function = raw.get("function") if wrapped else raw
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise _RowError("invalid_tool_schema")
    original_name = function["name"]
    name = _safe_tool_name(original_name)
    parameters = deepcopy(function.get("parameters"))
    if not name or not isinstance(parameters, dict):
        raise _RowError("invalid_tool_schema")
    if parameters.get("type") == "dict":
        parameters["type"] = "object"
    payload: dict[str, object] = {
        "name": name,
        "description": function.get("description") or None,
        "input_schema": parameters,
    }
    payload["tool_sha256"] = canonical_json_sha256(payload)
    try:
        return original_name, M10CanonicalToolDefinition.model_validate(payload)
    except ValueError as exc:
        raise _RowError("invalid_tool_schema") from exc


def _tools(
    raw_tools: list[object], *, wrapped: bool
) -> tuple[tuple[M10CanonicalToolDefinition, ...], dict[str, str]]:
    pairs = tuple(_tool(item, wrapped=wrapped) for item in raw_tools)
    mapping = {original: normalized.name for original, normalized in pairs}
    normalized = tuple(item for _, item in pairs)
    if len(mapping) != len(pairs) or len({item.name for item in normalized}) != len(normalized):
        raise _RowError("invalid_tool_schema")
    return normalized, mapping


def _call(*, call_id: str, name: str, arguments: dict[str, object]) -> M10CanonicalToolCall:
    payload: dict[str, object] = {"id": call_id, "name": name, "arguments": arguments}
    payload["call_sha256"] = canonical_json_sha256(payload)
    return M10CanonicalToolCall.model_validate(payload)


def _message(
    role: Literal["system", "user", "assistant", "tool"],
    *,
    content: str | None = None,
    calls: tuple[M10CanonicalToolCall, ...] = (),
    call_ids: tuple[str, ...] = (),
) -> M10CanonicalMessage:
    payload: dict[str, object] = {
        "role": role,
        "content": content,
        "tool_calls": [item.to_dict() for item in calls],
        "tool_call_ids": list(call_ids),
        "supervised": role == "assistant",
    }
    payload["message_sha256"] = canonical_json_sha256(payload)
    try:
        return M10CanonicalMessage.model_validate(payload)
    except ValueError as exc:
        if role == "assistant" and content and "<think>" in content.casefold():
            raise _RowError("visible_reasoning") from exc
        raise _RowError("invalid_row_shape") from exc


def _language(messages: tuple[M10CanonicalMessage, ...]) -> Literal["en", "zh"]:
    user_text = "\n".join(item.content or "" for item in messages if item.role == "user")
    return "zh" if _CJK_RE.search(user_text) else "en"


def _sample(
    *,
    source_id: M10CanonicalSourceId,
    revision: str,
    row_index: int,
    raw_row: object,
    tools: tuple[M10CanonicalToolDefinition, ...],
    messages: tuple[M10CanonicalMessage, ...],
) -> M10CanonicalTrainingSample:
    short_source = "toolace" if source_id == "toolace" else "hermes"
    raw_sha = _raw_hash(raw_row)
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "source_id": source_id,
        "source_revision": revision,
        "source_record_id": f"{short_source}-{row_index:05d}",
        "source_record_sha256": raw_sha,
        "license": "Apache-2.0",
        "language": _language(messages),
        "mode": "nonthinking",
        "group_id": f"group-{short_source}-{raw_sha[:16]}",
        "tools": [item.to_dict() for item in tools],
        "messages": [item.to_dict() for item in messages],
        "prompt_sha256": canonical_json_sha256(
            [
                {"role": item.role, "content": item.content}
                for item in messages
                if item.role == "user"
            ]
        ),
        "tool_schema_sha256": canonical_json_sha256([item.to_dict() for item in tools]),
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return M10CanonicalTrainingSample.model_validate(payload)


def _toolace_system_without_catalog(value: str) -> str:
    start = value.find(_TOOLACE_SCHEMA_PREFIX)
    end = value.rfind(_TOOLACE_SCHEMA_SUFFIX)
    if start < 0 or end < 0:
        raise _RowError("invalid_tool_schema")
    content = (value[:start] + value[end + len(_TOOLACE_SCHEMA_SUFFIX) :]).strip()
    return content or "Use the provided tools when they are required and answer from evidence."


def canonicalize_toolace_row(
    raw_row: object, *, revision: str, row_index: int
) -> M10CanonicalTrainingSample:
    """Convert one ToolACE row or raise a stable private rejection reason."""

    if (
        not isinstance(raw_row, dict)
        or set(raw_row) != {"system", "conversations"}
        or not isinstance(raw_row.get("system"), str)
        or not _valid_messages(raw_row.get("conversations"), role_key="from", content_key="value")
    ):
        raise _RowError("invalid_row_shape")
    conversations = cast(list[dict[str, object]], raw_row["conversations"])
    try:
        raw_tools = _extract_toolace_tools(raw_row["system"])
    except M10AgentDataError as exc:
        raise _RowError("invalid_tool_schema") from exc
    valid, _, _ = _valid_toolace_tools(raw_tools)
    if not valid or _tool_name_collision(raw_tools, wrapped=False):
        raise _RowError("invalid_tool_schema")
    tools, name_map = _tools(raw_tools, wrapped=False)

    parsed_by_index: dict[int, tuple[M10CanonicalToolCall, ...]] = {}
    call_indexes: set[int] = set()
    for message_index, message in enumerate(conversations):
        if message.get("from") != "assistant" or not isinstance(message.get("value"), str):
            continue
        value = str(message["value"]).strip()
        looks_like_call = value.startswith("[") and value.endswith("]")
        followed_by_tool = (
            message_index + 1 < len(conversations)
            and conversations[message_index + 1].get("from") == "tool"
        )
        if not (looks_like_call or followed_by_tool):
            continue
        call_indexes.add(message_index)
        try:
            parsed = parse_toolace_calls(value)
        except M10AgentDataError as exc:
            raise _RowError("malformed_tool_call") from exc
        row_calls: list[M10CanonicalToolCall] = []
        for call_index, (original_name, arguments) in enumerate(parsed):
            normalized_name = name_map.get(original_name)
            if normalized_name is None:
                raise _RowError("malformed_tool_call")
            row_calls.append(
                _call(
                    call_id=f"call_toolace_{row_index}_{message_index}_{call_index}",
                    name=normalized_name,
                    arguments=arguments,
                )
            )
        parsed_by_index[message_index] = tuple(row_calls)
    if not _toolace_role_path_valid(conversations, call_indexes):
        raise _RowError("invalid_role_path")

    canonical: list[M10CanonicalMessage] = [
        _message("system", content=_toolace_system_without_catalog(raw_row["system"]))
    ]
    pending: tuple[str, ...] = ()
    for message_index, message in enumerate(conversations):
        role = str(message["from"])
        value = str(message["value"]).strip()
        if role == "user":
            canonical.append(_message("user", content=value))
            pending = ()
        elif role == "assistant":
            calls = parsed_by_index.get(message_index, ())
            canonical.append(_message("assistant", content=None if calls else value, calls=calls))
            pending = tuple(item.id for item in calls)
        elif role == "tool":
            if not pending:
                raise _RowError("unpaired_tool_result")
            canonical.append(_message("tool", content=value, call_ids=pending))
            pending = ()
        else:
            raise _RowError("invalid_role_path")
    return _sample(
        source_id="toolace",
        revision=revision,
        row_index=row_index,
        raw_row=raw_row,
        tools=tools,
        messages=tuple(canonical),
    )


def _decode_hermes_call(raw: str, *, names: dict[str, str]) -> tuple[str, dict[str, object]]:
    normalized = raw.strip()
    if normalized.startswith(r"\n") and normalized.endswith(r"\n"):
        normalized = normalized[2:-2].strip()
    try:
        decoded: object = json.loads(normalized)
    except json.JSONDecodeError:
        try:
            decoded = ast.literal_eval(normalized)
        except (ValueError, SyntaxError) as exc:
            raise _RowError("malformed_tool_call") from exc
    if not isinstance(decoded, dict) or not isinstance(decoded.get("arguments"), dict):
        raise _RowError("malformed_tool_call")
    arguments = cast(dict[str, object], deepcopy(decoded["arguments"]))
    original_name = decoded.get("name")
    if original_name is None and len(names) == 1 and isinstance(arguments.get("name"), str):
        original_name = arguments.pop("name")
    if not isinstance(original_name, str) or original_name not in names:
        raise _RowError("malformed_tool_call")
    return names[original_name], arguments


def _hermes_assistant(
    value: str,
    *,
    names: dict[str, str],
    row_index: int,
    message_index: int,
) -> tuple[str | None, tuple[M10CanonicalToolCall, ...]]:
    blocks = _HERMES_TOOL_CALL_RE.findall(value)
    if value.count("<tool_call>") != len(blocks) or value.count("</tool_call>") != len(blocks):
        raise _RowError("malformed_tool_call")
    calls = tuple(
        _call(
            call_id=f"call_hermes_{row_index}_{message_index}_{call_index}",
            name=name,
            arguments=arguments,
        )
        for call_index, block in enumerate(blocks)
        for name, arguments in (_decode_hermes_call(block, names=names),)
    )
    remainder = _HERMES_TOOL_CALL_RE.sub("", value).strip()
    return remainder or None, calls


def _hermes_tool_result(value: str) -> str:
    stripped = value.strip()
    prefix = "<tool_response>"
    suffix = "</tool_response>"
    if stripped.startswith(prefix) and stripped.endswith(suffix):
        return stripped[len(prefix) : -len(suffix)].strip()
    return stripped


def canonicalize_hermes_row(
    raw_row: object, *, revision: str, row_index: int
) -> M10CanonicalTrainingSample:
    """Convert one Hermes row or raise a stable private rejection reason."""

    expected = {"category", "conversations", "id", "subcategory", "task", "tools"}
    if (
        not isinstance(raw_row, dict)
        or set(raw_row) != expected
        or not _valid_messages(raw_row.get("conversations"), role_key="from", content_key="value")
    ):
        raise _RowError("invalid_row_shape")
    conversations = cast(list[dict[str, object]], raw_row["conversations"])
    role_path = ">".join(str(item["from"]) for item in conversations)
    if role_path not in {
        "system>human>gpt",
        "system>human>gpt>tool",
        "system>human>gpt>tool>gpt",
    }:
        raise _RowError("invalid_role_path")
    try:
        raw_tools: object = json.loads(cast(str, raw_row["tools"]))
    except (json.JSONDecodeError, TypeError) as exc:
        raise _RowError("invalid_tool_schema") from exc
    if not _valid_hermes_tools(raw_tools) or _tool_name_collision(
        cast(list[object], raw_tools), wrapped=True
    ):
        raise _RowError("invalid_tool_schema")
    tools, names = _tools(cast(list[object], raw_tools), wrapped=True)
    if not tools and any(
        "<tool_call>" in str(message["value"])
        for message in conversations
        if message.get("from") == "gpt"
    ):
        raise _RowError("invalid_tool_schema")

    canonical: list[M10CanonicalMessage] = []
    pending: tuple[str, ...] = ()
    for message_index, message in enumerate(conversations):
        role = str(message["from"])
        value = str(message["value"]).strip()
        if role == "system":
            canonical.append(_message("system", content=value))
        elif role == "human":
            canonical.append(_message("user", content=value))
            pending = ()
        elif role == "gpt":
            content, calls = _hermes_assistant(
                value, names=names, row_index=row_index, message_index=message_index
            )
            canonical.append(_message("assistant", content=content, calls=calls))
            pending = tuple(item.id for item in calls)
        elif role == "tool":
            if not pending:
                raise _RowError("unpaired_tool_result")
            canonical.append(_message("tool", content=_hermes_tool_result(value), call_ids=pending))
            pending = ()
        else:
            raise _RowError("invalid_role_path")
    return _sample(
        source_id="hermes_function_calling",
        revision=revision,
        row_index=row_index,
        raw_row=raw_row,
        tools=tools,
        messages=tuple(canonical),
    )


def _render_samples(samples: tuple[M10CanonicalTrainingSample, ...]) -> bytes:
    return _jsonl_bytes(tuple(item.to_dict() for item in samples))


def _render_rejected(rejected: tuple[M10ExternalRejectedRecord, ...]) -> bytes:
    return _jsonl_bytes(tuple(item.to_dict() for item in rejected))


def import_external_source(
    *, config_path: Path, source_id: M10CanonicalSourceId, artifact_path: Path
) -> M10ExternalImportBuild:
    """Verify one pinned artifact and convert every row deterministically."""

    config = load_m10_agent_data_config(config_path)
    source = _source(config, source_id)
    artifact = _verify_artifact(source, artifact_path)
    rows = _load_json_array(artifact_path)
    samples: list[M10CanonicalTrainingSample] = []
    rejected: list[M10ExternalRejectedRecord] = []
    converter = canonicalize_toolace_row if source_id == "toolace" else canonicalize_hermes_row
    for row_index, raw_row in enumerate(rows):
        try:
            samples.append(converter(raw_row, revision=source.revision, row_index=row_index))
        except _RowError as exc:
            rejected.append(
                M10ExternalRejectedRecord(
                    source_id=source_id,
                    row_index=row_index,
                    source_record_sha256=_raw_hash(raw_row),
                    reason=exc.reason,
                )
            )
    frozen_samples = tuple(samples)
    frozen_rejected = tuple(rejected)
    items = _render_samples(frozen_samples)
    rejected_bytes = _render_rejected(frozen_rejected)
    content_sha = canonical_json_sha256([item.content_sha256 for item in frozen_samples])
    short_source = "toolace" if source_id == "toolace" else "hermes"
    manifest = M10ExternalImportManifest(
        import_version=f"m10-{short_source}-canonical-v1-{content_sha[:8]}",
        source_id=source_id,
        dataset_id=source.dataset_id,
        source_revision=source.revision,
        source_artifact_sha256=artifact.sha256,
        source_rows=len(rows),
        accepted_rows=len(frozen_samples),
        rejected_rows=len(frozen_rejected),
        rejection_counts=dict(Counter(item.reason for item in frozen_rejected)),
        language_counts=dict(Counter(item.language for item in frozen_samples)),
        supervised_messages=sum(
            item.supervised for sample in frozen_samples for item in sample.messages
        ),
        masked_messages=sum(
            not item.supervised for sample in frozen_samples for item in sample.messages
        ),
        tool_calls=sum(
            len(item.tool_calls) for sample in frozen_samples for item in sample.messages
        ),
        items_sha256=hashlib.sha256(items).hexdigest(),
        rejected_sha256=hashlib.sha256(rejected_bytes).hexdigest(),
        content_sha256=content_sha,
    )
    return M10ExternalImportBuild(
        manifest=manifest, samples=frozen_samples, rejected=frozen_rejected
    )


def write_external_import(output_root: Path, build: M10ExternalImportBuild) -> Path:
    """Atomically commit one private canonical source with per-file hashes."""

    if output_root.is_symlink():
        raise M10CanonicalImportError("M10 external import root cannot be a symbolic link")
    target = output_root / build.manifest.import_version
    staging = output_root / f".{build.manifest.import_version}.staging"
    output_root.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    files = {
        "items.jsonl": _render_samples(build.samples),
        "rejected.jsonl": _render_rejected(build.rejected),
        "manifest.json": _json_bytes(build.manifest.to_dict(), indent=2),
    }
    for name, payload in files.items():
        (staging / name).write_bytes(payload)
    committed = _json_bytes(
        {
            "schema_version": "1.0",
            "import_version": build.manifest.import_version,
            "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
        },
        indent=2,
    )
    (staging / "COMMITTED.json").write_bytes(committed)
    if target.exists():
        expected = {**files, "COMMITTED.json": committed}
        if all(
            (target / name).is_file() and (target / name).read_bytes() == payload
            for name, payload in expected.items()
        ):
            shutil.rmtree(staging)
            return target
        raise M10CanonicalImportError("M10 external import version already differs")
    staging.rename(target)
    return target


def build_external_import_report(
    builds: tuple[M10ExternalImportBuild, M10ExternalImportBuild],
) -> M10ExternalImportReport:
    """Create one path-free public report from both private imports."""

    summaries = tuple(
        M10ExternalImportSummary(
            source_id=build.manifest.source_id,
            import_version=build.manifest.import_version,
            manifest_sha256=hashlib.sha256(
                _json_bytes(build.manifest.to_dict(), indent=2)
            ).hexdigest(),
            content_sha256=build.manifest.content_sha256,
            source_rows=build.manifest.source_rows,
            accepted_rows=build.manifest.accepted_rows,
            rejected_rows=build.manifest.rejected_rows,
            rejection_counts=build.manifest.rejection_counts,
            language_counts=build.manifest.language_counts,
            supervised_messages=build.manifest.supervised_messages,
            masked_messages=build.manifest.masked_messages,
            tool_calls=build.manifest.tool_calls,
        )
        for build in builds
    )
    return M10ExternalImportReport(
        status="pass" if all(item.accepted_rows for item in summaries) else "fail",
        sources=cast(tuple[M10ExternalImportSummary, M10ExternalImportSummary], summaries),
        total_source_rows=sum(item.source_rows for item in summaries),
        total_accepted_rows=sum(item.accepted_rows for item in summaries),
        total_rejected_rows=sum(item.rejected_rows for item in summaries),
    )
