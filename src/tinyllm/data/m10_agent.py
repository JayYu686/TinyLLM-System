"""Fail-closed loading and content-free profiling for M10 Agent sources."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tinyllm.data.m10_agent_schema import (
    M10AgentArtifactSpec,
    M10AgentDataConfig,
    M10AgentSourceSpec,
    M10ExternalSourceProfile,
    M10ExternalSourceProfileReport,
    M10SourceRejectionCount,
    M10SourceRolePathCount,
)
from tinyllm.schemas import canonical_config_hash

_TOOLACE_SCHEMA_PREFIX = "Here is a list of functions in JSON format that you can invoke:\n"
_TOOLACE_SCHEMA_SUFFIX = "]. \nShould you decide to return the function call(s)."
_HERMES_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_SAFE_TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")
_REJECTION_ORDER = (
    "invalid_row_shape",
    "invalid_role_path",
    "invalid_tool_schema",
    "malformed_tool_call",
)


class M10AgentDataError(ValueError):
    """Raised when M10 data contracts or frozen sources fail validation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise M10AgentDataError("M10 source artifact cannot be read") from exc
    return digest.hexdigest()


def load_m10_agent_data_config(path: Path) -> M10AgentDataConfig:
    """Load the strict preregistered M10 data contract from YAML."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise M10AgentDataError("M10 Agent data config must use YAML")
    try:
        decoded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        return M10AgentDataConfig.model_validate(decoded)
    except OSError as exc:
        raise M10AgentDataError("M10 Agent data config cannot be read") from exc
    except yaml.YAMLError as exc:
        raise M10AgentDataError("M10 Agent data config is invalid YAML") from exc
    except ValidationError as exc:
        raise M10AgentDataError("M10 Agent data config violates its schema") from exc


def m10_agent_data_config_sha256(config: M10AgentDataConfig) -> str:
    """Hash the resolved strict contract using canonical JSON."""

    return canonical_config_hash(config.to_dict())


def _source(config: M10AgentDataConfig, source_id: str) -> M10AgentSourceSpec:
    matches = tuple(item for item in config.sources if item.source_id == source_id)
    if len(matches) != 1:
        raise M10AgentDataError("M10 data config must contain each source exactly once")
    return matches[0]


def _verify_artifact(source: M10AgentSourceSpec, path: Path) -> M10AgentArtifactSpec:
    if len(source.artifacts) != 1:
        raise M10AgentDataError("M10 external profiler requires exactly one selected artifact")
    expected = source.artifacts[0]
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise M10AgentDataError("M10 source artifact cannot be inspected") from exc
    if path.name != expected.filename or size != expected.size_bytes:
        raise M10AgentDataError("M10 source artifact name or size differs from its contract")
    if _sha256_file(path) != expected.sha256:
        raise M10AgentDataError("M10 source artifact SHA256 differs from its contract")
    return expected


def _load_json_array(path: Path) -> list[object]:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise M10AgentDataError("M10 source artifact cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise M10AgentDataError("M10 source artifact is invalid JSON") from exc
    if not isinstance(decoded, list) or not decoded:
        raise M10AgentDataError("M10 source artifact must be a non-empty JSON array")
    return decoded


def _role_path(messages: object, *, role_key: str) -> str:
    if not isinstance(messages, list) or not messages:
        return "invalid"
    roles: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get(role_key), str):
            return "invalid"
        role = str(message[role_key]).strip().lower()
        if not role or not role.isalpha():
            return "invalid"
        roles.append(role)
    return ">".join(roles)


def _valid_messages(messages: object, *, role_key: str, content_key: str) -> bool:
    return bool(
        isinstance(messages, list)
        and messages
        and all(
            isinstance(message, dict)
            and set(message) == {role_key, content_key}
            and isinstance(message[role_key], str)
            and isinstance(message[content_key], str)
            and bool(message[content_key].strip())
            for message in messages
        )
    )


def _safe_tool_name(name: str) -> str:
    normalized = _SAFE_TOOL_NAME_RE.sub("_", name.strip()).strip("_")
    if normalized and normalized[0].isdigit():
        normalized = f"tool_{normalized}"
    return normalized.casefold()


def _tool_name_collision(tools: Sequence[object], *, wrapped: bool) -> bool:
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            return False
        candidate = tool.get("function") if wrapped else tool
        if not isinstance(candidate, dict) or not isinstance(candidate.get("name"), str):
            return False
        normalized = _safe_tool_name(candidate["name"])
        if not normalized:
            return False
        names.append(normalized)
    return len(names) != len(set(names))


def _valid_hermes_tools(tools: object) -> bool:
    if not isinstance(tools, list):
        return False
    for item in tools:
        if not isinstance(item, dict) or set(item) != {"type", "function"}:
            return False
        function = item.get("function")
        if (
            item.get("type") != "function"
            or not isinstance(function, dict)
            or not isinstance(function.get("name"), str)
            or not function["name"].strip()
            or not isinstance(function.get("parameters"), dict)
        ):
            return False
    return True


def _parse_hermes_calls(
    messages: Sequence[Mapping[str, object]],
    tools: Sequence[object],
) -> tuple[int, int]:
    parsed = 0
    malformed = 0
    available_names = {
        str(function["name"])
        for tool in tools
        if isinstance(tool, dict)
        and isinstance((function := tool.get("function")), dict)
        and isinstance(function.get("name"), str)
    }
    for message in messages:
        if message.get("from") != "gpt" or not isinstance(message.get("value"), str):
            continue
        value = str(message["value"])
        blocks = _HERMES_TOOL_CALL_RE.findall(value)
        if value.count("<tool_call>") != len(blocks) or value.count("</tool_call>") != len(blocks):
            malformed += max(value.count("<tool_call>"), value.count("</tool_call>"), 1)
            continue
        for block in blocks:
            normalized_block = block.strip()
            if normalized_block.startswith(r"\n") and normalized_block.endswith(r"\n"):
                normalized_block = normalized_block[2:-2].strip()
            try:
                call = json.loads(normalized_block)
            except json.JSONDecodeError:
                try:
                    call = ast.literal_eval(normalized_block)
                except (ValueError, SyntaxError):
                    malformed += 1
                    continue
            name = call.get("name") if isinstance(call, dict) else None
            arguments = call.get("arguments") if isinstance(call, dict) else None
            if (
                name is None
                and isinstance(arguments, dict)
                and len(available_names) == 1
                and arguments.get("name") in available_names
            ):
                name = arguments["name"]
            if (
                not isinstance(call, dict)
                or not isinstance(name, str)
                or name not in available_names
                or not isinstance(arguments, dict)
            ):
                malformed += 1
                continue
            parsed += 1
    return parsed, malformed


def _split_top_level(value: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    depth = 0
    for index, character in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
            if depth < 0:
                raise M10AgentDataError("ToolACE call has unbalanced delimiters")
        elif character == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    if quote is not None or depth != 0:
        raise M10AgentDataError("ToolACE call has unbalanced quotes or delimiters")
    parts.append(value[start:].strip())
    if any(not item for item in parts):
        raise M10AgentDataError("ToolACE call contains an empty component")
    return tuple(parts)


def _matching_parenthesis(value: str, open_index: int) -> int:
    quote: str | None = None
    escaped = False
    depth = 0
    for index in range(open_index, len(value)):
        character = value[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
    raise M10AgentDataError("ToolACE call has unbalanced arguments")


def _toolace_expressions(value: str) -> tuple[tuple[str, str], ...]:
    expressions: list[tuple[str, str]] = []
    cursor = 0
    while cursor < len(value):
        selected: tuple[int, int] | None = None
        for open_index in range(cursor, len(value)):
            if value[open_index] != "(":
                continue
            close_index = _matching_parenthesis(value, open_index)
            trailing = value[close_index + 1 :].lstrip()
            if not trailing or trailing.startswith(","):
                selected = (open_index, close_index)
                break
        if selected is None:
            raise M10AgentDataError("ToolACE call expression is malformed")
        open_index, close_index = selected
        name = value[cursor:open_index].strip()
        arguments = value[open_index + 1 : close_index].strip()
        if not name:
            raise M10AgentDataError("ToolACE call name is invalid")
        expressions.append((name, arguments))
        cursor = close_index + 1
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if cursor == len(value):
            break
        if value[cursor] != ",":
            raise M10AgentDataError("ToolACE calls must be comma separated")
        cursor += 1
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if cursor == len(value):
            raise M10AgentDataError("ToolACE call list has a trailing comma")
    return tuple(expressions)


def _literal_argument(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise M10AgentDataError("ToolACE argument is not a safe literal") from exc


def parse_toolace_calls(value: str) -> tuple[tuple[str, dict[str, object]], ...]:
    """Parse ToolACE bracket calls without evaluating executable expressions."""

    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise M10AgentDataError("ToolACE call must use outer brackets")
    body = stripped[1:-1].strip()
    if not body:
        raise M10AgentDataError("ToolACE call list cannot be empty")
    calls: list[tuple[str, dict[str, object]]] = []
    for name, raw_arguments in _toolace_expressions(body):
        if not name or not _safe_tool_name(name):
            raise M10AgentDataError("ToolACE call name is invalid")
        arguments: dict[str, object] = {}
        if raw_arguments:
            for argument in _split_top_level(raw_arguments):
                key, separator, raw_value = argument.partition("=")
                key = key.strip()
                if not separator or not re.fullmatch(r"[$A-Za-z_][$A-Za-z0-9_.-]*", key):
                    raise M10AgentDataError("ToolACE argument name is invalid")
                if key in arguments:
                    raise M10AgentDataError("ToolACE call repeats an argument")
                arguments[key] = _literal_argument(raw_value.strip())
        calls.append((name, arguments))
    return tuple(calls)


def _extract_toolace_tools(system: str) -> list[object]:
    prefix_index = system.find(_TOOLACE_SCHEMA_PREFIX)
    suffix_index = system.rfind(_TOOLACE_SCHEMA_SUFFIX)
    if prefix_index < 0 or suffix_index < 0:
        raise M10AgentDataError("ToolACE system message lacks the frozen schema envelope")
    start = prefix_index + len(_TOOLACE_SCHEMA_PREFIX)
    raw = system[start : suffix_index + 1]
    try:
        tools: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise M10AgentDataError("ToolACE embedded tool schema is invalid JSON") from exc
    if not isinstance(tools, list):
        raise M10AgentDataError("ToolACE embedded tool schema must be an array")
    return tools


def _valid_toolace_tools(tools: object) -> tuple[bool, int, int]:
    if not isinstance(tools, list):
        return False, 0, 0
    dict_types = 0
    null_required = 0
    for item in tools:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"].strip()
            or not isinstance(item.get("parameters"), dict)
        ):
            return False, dict_types, null_required
        parameters = item["parameters"]
        if parameters.get("type") == "dict":
            dict_types += 1
        elif parameters.get("type") != "object":
            return False, dict_types, null_required
        if "properties" not in parameters or not isinstance(parameters["properties"], dict):
            return False, dict_types, null_required
        if item.get("required") is None and "required" in item:
            null_required += 1
        elif "required" in item and not isinstance(item["required"], list):
            return False, dict_types, null_required
    return True, dict_types, null_required


def _toolace_role_path_valid(
    messages: Sequence[Mapping[str, object]],
    call_indexes: set[int],
) -> bool:
    roles = tuple(message.get("from") for message in messages)
    if not roles or roles[0] != "user" or roles[-1] != "assistant":
        return False
    for index, role in enumerate(roles):
        previous = roles[index - 1] if index else None
        if role == "user" and previous not in {None, "assistant"}:
            return False
        if role == "assistant" and previous not in {"user", "tool"}:
            return False
        if role == "tool" and (previous != "assistant" or index - 1 not in call_indexes):
            return False
        if role not in {"user", "assistant", "tool"}:
            return False
    return True


def _counts_as_models(
    values: Counter[str],
) -> tuple[M10SourceRejectionCount, ...]:
    return tuple(
        M10SourceRejectionCount(reason=reason, rows=values[reason])  # type: ignore[arg-type]
        for reason in _REJECTION_ORDER
        if values[reason]
    )


def _role_counts_as_models(values: Counter[str]) -> tuple[M10SourceRolePathCount, ...]:
    return tuple(
        M10SourceRolePathCount(role_path=role_path, rows=count)
        for role_path, count in sorted(values.items())
    )


def _profile_hermes(source: M10AgentSourceSpec, path: Path) -> M10ExternalSourceProfile:
    artifact = _verify_artifact(source, path)
    rows = _load_json_array(path)
    role_counts: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    tool_counts: list[int] = []
    tool_call_rows = 0
    parsed_calls = 0
    malformed_calls = 0
    collision_rows = 0
    for raw_row in rows:
        primary_rejection: str | None = None
        role_path = "invalid"
        tools: list[object] = []
        messages: list[dict[str, object]] = []
        if (
            not isinstance(raw_row, dict)
            or set(raw_row)
            != {
                "category",
                "conversations",
                "id",
                "subcategory",
                "task",
                "tools",
            }
            or not _valid_messages(
                raw_row.get("conversations"),
                role_key="from",
                content_key="value",
            )
        ):
            primary_rejection = "invalid_row_shape"
        else:
            messages = raw_row["conversations"]
            role_path = _role_path(messages, role_key="from")
            try:
                decoded_tools = json.loads(raw_row["tools"])
            except (json.JSONDecodeError, TypeError):
                decoded_tools = None
            if not _valid_hermes_tools(decoded_tools):
                primary_rejection = "invalid_tool_schema"
            else:
                tools = decoded_tools
                if _tool_name_collision(tools, wrapped=True):
                    collision_rows += 1
                    primary_rejection = "invalid_tool_schema"
            if (
                role_path
                not in {
                    "system>human>gpt",
                    "system>human>gpt>tool",
                    "system>human>gpt>tool>gpt",
                }
                and primary_rejection is None
            ):
                primary_rejection = "invalid_role_path"
        role_counts[role_path] += 1
        tool_counts.append(len(tools))
        calls, malformed = _parse_hermes_calls(messages, tools)
        parsed_calls += calls
        malformed_calls += malformed
        if calls or malformed:
            tool_call_rows += 1
        if malformed and not tools and primary_rejection is None:
            primary_rejection = "invalid_tool_schema"
        elif malformed and primary_rejection is None:
            primary_rejection = "malformed_tool_call"
        if primary_rejection:
            rejections[primary_rejection] += 1
    rejected = sum(rejections.values())
    return M10ExternalSourceProfile(
        source_id="hermes_function_calling",
        dataset_id=source.dataset_id,
        revision=source.revision,
        artifacts=(artifact,),
        source_rows=len(rows),
        accepted_shape_rows=len(rows) - rejected,
        rejected_shape_rows=rejected,
        role_paths=_role_counts_as_models(role_counts),
        rejection_counts=_counts_as_models(rejections),
        rows_with_tool_definitions=sum(bool(count) for count in tool_counts),
        tool_definitions=sum(tool_counts),
        tools_per_row_min=min(tool_counts),
        tools_per_row_max=max(tool_counts),
        tools_per_row_mean_milli=round(sum(tool_counts) * 1000 / len(tool_counts)),
        tool_call_candidate_rows=tool_call_rows,
        no_tool_candidate_rows=len(rows) - tool_call_rows,
        parsed_tool_calls=parsed_calls,
        malformed_tool_calls=malformed_calls,
        dict_to_object_normalizations=0,
        null_required_normalizations=0,
        tool_name_collision_rows=collision_rows,
    )


def _profile_toolace(source: M10AgentSourceSpec, path: Path) -> M10ExternalSourceProfile:
    artifact = _verify_artifact(source, path)
    rows = _load_json_array(path)
    role_counts: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    tool_counts: list[int] = []
    tool_call_rows = 0
    parsed_calls = 0
    malformed_calls = 0
    dict_types = 0
    null_required = 0
    collision_rows = 0
    for raw_row in rows:
        primary_rejection: str | None = None
        role_path = "invalid"
        tools: list[object] = []
        messages: list[dict[str, object]] = []
        if (
            not isinstance(raw_row, dict)
            or set(raw_row) != {"system", "conversations"}
            or not isinstance(raw_row.get("system"), str)
            or not _valid_messages(
                raw_row.get("conversations"),
                role_key="from",
                content_key="value",
            )
        ):
            primary_rejection = "invalid_row_shape"
        else:
            messages = raw_row["conversations"]
            role_path = _role_path(messages, role_key="from")
            try:
                tools = _extract_toolace_tools(raw_row["system"])
            except M10AgentDataError:
                tools = []
                primary_rejection = "invalid_tool_schema"
            valid_tools, row_dict_types, row_null_required = _valid_toolace_tools(tools)
            dict_types += row_dict_types
            null_required += row_null_required
            if not valid_tools:
                primary_rejection = primary_rejection or "invalid_tool_schema"
            elif _tool_name_collision(tools, wrapped=False):
                collision_rows += 1
                primary_rejection = primary_rejection or "invalid_tool_schema"
        call_indexes: set[int] = set()
        row_malformed = 0
        for index, message in enumerate(messages):
            if message.get("from") != "assistant" or not isinstance(message.get("value"), str):
                continue
            value = str(message["value"]).strip()
            looks_like_call = value.startswith("[") and value.endswith("]")
            followed_by_tool = (
                index + 1 < len(messages) and messages[index + 1].get("from") == "tool"
            )
            if not (looks_like_call or followed_by_tool):
                continue
            call_indexes.add(index)
            try:
                calls = parse_toolace_calls(value)
            except M10AgentDataError:
                row_malformed += 1
            else:
                parsed_calls += len(calls)
        if call_indexes:
            tool_call_rows += 1
        if messages and not _toolace_role_path_valid(messages, call_indexes):
            primary_rejection = primary_rejection or "invalid_role_path"
        malformed_calls += row_malformed
        if row_malformed and primary_rejection is None:
            primary_rejection = "malformed_tool_call"
        role_counts[role_path] += 1
        tool_counts.append(len(tools))
        if primary_rejection:
            rejections[primary_rejection] += 1
    rejected = sum(rejections.values())
    return M10ExternalSourceProfile(
        source_id="toolace",
        dataset_id=source.dataset_id,
        revision=source.revision,
        artifacts=(artifact,),
        source_rows=len(rows),
        accepted_shape_rows=len(rows) - rejected,
        rejected_shape_rows=rejected,
        role_paths=_role_counts_as_models(role_counts),
        rejection_counts=_counts_as_models(rejections),
        rows_with_tool_definitions=sum(bool(count) for count in tool_counts),
        tool_definitions=sum(tool_counts),
        tools_per_row_min=min(tool_counts),
        tools_per_row_max=max(tool_counts),
        tools_per_row_mean_milli=round(sum(tool_counts) * 1000 / len(tool_counts)),
        tool_call_candidate_rows=tool_call_rows,
        no_tool_candidate_rows=len(rows) - tool_call_rows,
        parsed_tool_calls=parsed_calls,
        malformed_tool_calls=malformed_calls,
        dict_to_object_normalizations=dict_types,
        null_required_normalizations=null_required,
        tool_name_collision_rows=collision_rows,
    )


def profile_m10_external_sources(
    *,
    config_path: Path,
    toolace_artifact: Path,
    hermes_artifact: Path,
) -> M10ExternalSourceProfileReport:
    """Verify and profile the two pinned external sources without publishing content."""

    config = load_m10_agent_data_config(config_path)
    toolace = _profile_toolace(_source(config, "toolace"), toolace_artifact)
    hermes = _profile_hermes(_source(config, "hermes_function_calling"), hermes_artifact)
    return M10ExternalSourceProfileReport(
        profile_version="m10-external-source-profile-v1",
        data_config_sha256=m10_agent_data_config_sha256(config),
        profiles=(toolace, hermes),
        total_source_rows=toolace.source_rows + hermes.source_rows,
        total_accepted_shape_rows=toolace.accepted_shape_rows + hermes.accepted_shape_rows,
        total_rejected_shape_rows=toolace.rejected_shape_rows + hermes.rejected_shape_rows,
    )
