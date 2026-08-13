from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from starlette.types import Message, Receive, Scope, Send

from tinyllm.serving.vllm_guard import VLLMBackendGuard

TOKEN = "internal-unit-token-that-is-longer-than-32-characters"


async def _call(
    monkeypatch: pytest.MonkeyPatch,
    *,
    path: str = "/v1/chat/completions",
    token: str | None = TOKEN,
    body: dict[str, Any] | None = None,
    raw_path: bytes | None = None,
) -> tuple[int, bool]:
    monkeypatch.setenv("TINYLLM_VLLM_INTERNAL_TOKEN", TOKEN)
    reached = False

    async def inner(_scope: Scope, _receive: Receive, send: Send) -> None:
        nonlocal reached
        reached = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    encoded = json.dumps(
        body
        or {
            "model": "unit",
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).encode()
    headers = [(b"content-length", str(len(encoded)).encode())]
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "raw_path": raw_path if raw_path is not None else path.encode(),
        "headers": headers,
    }
    request_sent = False

    async def receive() -> Message:
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": encoded, "more_body": False}

    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    await VLLMBackendGuard(inner)(scope, receive, send)
    status = next(
        message["status"] for message in messages if message["type"] == "http.response.start"
    )
    return status, reached


def test_guard_accepts_only_authenticated_allowed_request(monkeypatch: pytest.MonkeyPatch) -> None:
    assert asyncio.run(_call(monkeypatch)) == (200, True)
    assert asyncio.run(_call(monkeypatch, token=None)) == (401, False)
    assert asyncio.run(_call(monkeypatch, path="/metrics")) == (404, False)
    assert asyncio.run(_call(monkeypatch, raw_path=b"@evil")) == (404, False)


def test_guard_rejects_dynamic_or_pathological_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    status, reached = asyncio.run(
        _call(monkeypatch, body={"messages": [], "guided_regex": "(a+)+"})
    )
    assert (status, reached) == (400, False)
    schema: dict[str, Any] = {"type": "string"}
    for _ in range(20):
        schema = {"type": "array", "items": schema}
    body = {
        "model": "unit",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "deep", "parameters": schema},
            }
        ],
    }
    assert asyncio.run(_call(monkeypatch, body=body)) == (400, False)
    tools = body["tools"]
    assert isinstance(tools, list) and isinstance(tools[0], dict)
    function = tools[0]["function"]
    assert isinstance(function, dict)
    function["parameters"] = {
        "type": "string",
        "pattern": "(a+)+$",
    }
    assert asyncio.run(_call(monkeypatch, body=body)) == (400, False)


def test_guard_allows_only_fixed_thinking_template_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = {
        "model": "unit",
        "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert asyncio.run(_call(monkeypatch, body=valid)) == (200, True)
    invalid = valid | {"chat_template_kwargs": {"enable_thinking": False, "evil": "x"}}
    assert asyncio.run(_call(monkeypatch, body=invalid)) == (400, False)


def test_guard_requires_internal_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TINYLLM_VLLM_INTERNAL_TOKEN", raising=False)

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        return None

    with pytest.raises(RuntimeError, match="must contain"):
        VLLMBackendGuard(app)


def test_guard_rejects_wire_nesting_before_json_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = ('{"x":' * 40 + "0" + "}" * 40).encode()
    status, reached = asyncio.run(
        _call(monkeypatch, body={"messages": [], "value": nested.decode()})
    )
    assert (status, reached) == (200, True)
    monkeypatch.setenv("TINYLLM_VLLM_INTERNAL_TOKEN", TOKEN)

    async def inner(_scope: Scope, _receive: Receive, _send: Send) -> None:
        raise AssertionError("unsafe body reached backend")

    sent = False

    async def receive() -> Message:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": nested, "more_body": False}

    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "headers": [
            (b"content-length", str(len(nested)).encode()),
            (b"authorization", f"Bearer {TOKEN}".encode()),
        ],
    }
    asyncio.run(VLLMBackendGuard(inner)(scope, receive, send))
    assert messages[0]["status"] == 400
