"""Defense-in-depth ASGI middleware for the fixed legacy cu118 vLLM backend."""

from __future__ import annotations

import hmac
import json
import os
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_BACKEND_BODY_BYTES = 1_048_576
MAX_BACKEND_HEADER_BYTES = 16_384
MAX_TOOL_SCHEMA_DEPTH = 16
MAX_TOOL_SCHEMA_NODES = 4096
MAX_JSON_WIRE_DEPTH = 32
FORBIDDEN_SCHEMA_KEYWORDS = frozenset(
    {
        "$defs",
        "$ref",
        "allOf",
        "anyOf",
        "definitions",
        "dependentSchemas",
        "else",
        "if",
        "not",
        "oneOf",
        "pattern",
        "patternProperties",
        "then",
        "unevaluatedProperties",
    }
)
ALLOWED_PATHS = frozenset({"/health", "/v1/chat/completions", "/v1/models"})


class VLLMBackendGuard:
    """Authenticate and bound every request before old vLLM/Starlette parses it."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        token = os.environ.get("TINYLLM_VLLM_INTERNAL_TOKEN", "")
        if len(token) < 32:
            raise RuntimeError("TINYLLM_VLLM_INTERNAL_TOKEN must contain at least 32 characters")
        self._token = token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        raw_path = scope.get("raw_path", b"")
        path = scope.get("path", "")
        headers = scope.get("headers", [])
        if (
            not isinstance(raw_path, bytes)
            or not raw_path.startswith(b"/")
            or not isinstance(path, str)
            or path not in ALLOWED_PATHS
        ):
            await _reject(send, 404, "backend endpoint is unavailable")
            return
        if sum(len(name) + len(value) for name, value in headers) > MAX_BACKEND_HEADER_BYTES:
            await _reject(send, 431, "backend headers are too large")
            return
        header_map = {name.lower(): value for name, value in headers}
        supplied = header_map.get(b"authorization", b"")
        if not hmac.compare_digest(supplied, b"Bearer " + self._token):
            await _reject(send, 401, "backend authentication failed")
            return
        try:
            content_length = int(header_map.get(b"content-length", b"0"))
        except ValueError:
            await _reject(send, 400, "backend content length is invalid")
            return
        if content_length < 0 or content_length > MAX_BACKEND_BODY_BYTES:
            await _reject(send, 413, "backend request is too large")
            return
        if path != "/v1/chat/completions":
            await self.app(scope, receive, send)
            return
        try:
            body = await _read_body(receive)
            _validate_chat_body(body)
        except ValueError as exc:
            await _reject(send, 400, str(exc))
            return

        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)


async def _read_body(receive: Receive) -> bytes:
    body = bytearray()
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise ValueError("backend client disconnected")
        chunk = message.get("body", b"")
        if not isinstance(chunk, bytes):
            raise ValueError("backend request body is invalid")
        body.extend(chunk)
        if len(body) > MAX_BACKEND_BODY_BYTES:
            raise ValueError("backend request is too large")
        if not message.get("more_body", False):
            return bytes(body)


def _validate_chat_body(body: bytes) -> None:
    if not _json_wire_depth_is_safe(body):
        raise ValueError("backend request exceeds the JSON nesting limit")
    try:
        payload: Any = json.loads(body)
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError) as exc:
        raise ValueError("backend request is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("backend request must be a JSON object")
    if any(key in payload for key in ("chat_template", "guided_regex", "guided_json")):
        raise ValueError("backend request contains a forbidden dynamic field")
    template_arguments = payload.get("chat_template_kwargs")
    if template_arguments is not None and (
        not isinstance(template_arguments, dict)
        or set(template_arguments) != {"enable_thinking"}
        or not isinstance(template_arguments["enable_thinking"], bool)
    ):
        raise ValueError("backend chat template arguments are outside the fixed contract")
    if _structure_depth(payload) > MAX_TOOL_SCHEMA_DEPTH:
        raise ValueError("backend request exceeds the schema nesting limit")
    tools = payload.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                raise ValueError("backend tool definition is invalid")
            function = tool.get("function")
            parameters = function.get("parameters") if isinstance(function, dict) else None
            _validate_json_schema(parameters)


def _validate_json_schema(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("backend tool parameters must be a JSON Schema object")
    stack: list[tuple[object, int, bool]] = [(value, 1, True)]
    visited = 0
    while stack:
        current, depth, is_schema_node = stack.pop()
        if depth > MAX_TOOL_SCHEMA_DEPTH:
            raise ValueError("backend tool schema exceeds the nesting limit")
        visited += 1
        if visited > MAX_TOOL_SCHEMA_NODES:
            raise ValueError("backend tool schema is too complex")
        if isinstance(current, dict):
            if is_schema_node and FORBIDDEN_SCHEMA_KEYWORDS.intersection(current):
                raise ValueError("backend tool schema uses an unsafe keyword")
            for key, item in current.items():
                # Keys below `properties` are user-visible argument names, not
                # JSON Schema keywords. Their values are schema nodes again.
                child_is_schema_node = not (is_schema_node and key == "properties")
                if not is_schema_node:
                    child_is_schema_node = True
                stack.append((item, depth + 1, child_is_schema_node))
        elif isinstance(current, list):
            stack.extend((item, depth + 1, True) for item in current)


def _structure_depth(value: object) -> int:
    stack: list[tuple[object, int]] = [(value, 1)]
    maximum = 1
    visited = 0
    while stack:
        current, depth = stack.pop()
        maximum = max(maximum, depth)
        if maximum > MAX_TOOL_SCHEMA_DEPTH:
            return maximum
        visited += 1
        if visited > MAX_TOOL_SCHEMA_NODES:
            return MAX_TOOL_SCHEMA_DEPTH + 1
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
    return maximum


def _json_wire_depth_is_safe(body: bytes, limit: int = MAX_JSON_WIRE_DEPTH) -> bool:
    """Reject excessive structural nesting before calling the recursive JSON decoder."""

    depth = 0
    quoted = False
    escaped = False
    for byte in body:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                quoted = False
            continue
        if byte == 0x22:
            quoted = True
        elif byte in (0x5B, 0x7B):
            depth += 1
            if depth > limit:
                return False
        elif byte in (0x5D, 0x7D):
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not quoted


async def _reject(send: Send, status_code: int, message: str) -> None:
    response = JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "tinyllm_backend_guard"}},
        headers={"cache-control": "no-store", "x-content-type-options": "nosniff"},
    )
    await response({"type": "http"}, _empty_receive, send)


async def _empty_receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}
