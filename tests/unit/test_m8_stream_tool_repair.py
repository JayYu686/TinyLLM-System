from __future__ import annotations

import json
from typing import Any, cast

import pytest

from tinyllm.serving.gateway import (
    _normalize_tool_completion,
    _repair_auto_tool_completion,
    _SSEAutoToolRepair,
    _SSEToolNormalizer,
)


def _frames(content_parts: list[str]) -> bytes:
    frames = []
    for index, content in enumerate(content_parts):
        value = {
            "id": "chatcmpl-golden",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant" if index == 0 else None,
                        "content": content,
                    },
                    "finish_reason": None,
                }
            ],
        }
        frames.append(
            b"data: "
            + json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
            + b"\n\n"
        )
    return b"".join(frames) + b"data: [DONE]\n\n"


def _feed(raw: bytes, boundaries: list[int]) -> bytes:
    repair = _SSEAutoToolRepair(frozenset({"search_evidence"}))
    output: list[bytes] = []
    start = 0
    for end in boundaries:
        output.extend(repair.feed(raw[start:end]))
        start = end
    output.extend(repair.feed(raw[start:]))
    output.extend(repair.flush())
    return b"".join(output)


def _tool_calls(wire: bytes) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for line in wire.splitlines():
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        value = json.loads(line[6:])
        for choice in value.get("choices", []):
            calls.extend(choice.get("delta", {}).get("tool_calls", []))
    return calls


@pytest.mark.parametrize("width", [1, 2, 3, 5, 7, 13, 31, 127])
def test_auto_tool_repair_is_stable_across_arbitrary_byte_boundaries(width: int) -> None:
    raw = _frames(
        [
            '{"',
            "name",
            '": "search_evidence", "arguments": {"query": "M7 Production"}}\n',
        ]
    )
    output = _feed(raw, list(range(width, len(raw), width)))
    calls = _tool_calls(output)

    assert len(calls) == 1
    function = calls[0]["function"]
    assert isinstance(function, dict)
    assert function["name"] == "search_evidence"
    assert json.loads(str(function["arguments"])) == {"query": "M7 Production"}
    assert b'"finish_reason":"tool_calls"' in output
    assert b"data: [DONE]" in output
    assert b'"name": "search_evidence"' not in output


def test_auto_tool_repair_passes_plain_text_and_native_tool_calls() -> None:
    plain = _frames(["The", " answer"])
    assert _feed(plain, [1, 17, 53]) == plain

    native = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        b'"function":{"name":"search_evidence","arguments":"{}"}}]}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    assert _feed(native, [1, 3, 6, 11, 19, 32]) == native


def test_auto_tool_repair_fails_closed_for_unknown_or_malformed_json() -> None:
    unknown = _frames(['{"name":"shell","arguments":{}}'])
    output = _feed(unknown, [1, 9, 27])
    assert b"tool_parser_error" in output
    assert not _tool_calls(output)


def test_nonstream_auto_tool_repair_converts_only_fixed_allowed_json() -> None:
    completion: dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"name":"search_evidence","arguments":{"query":"M7"}}',
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }
        ]
    }

    repaired = _repair_auto_tool_completion(completion, frozenset({"search_evidence"}))

    choice = cast(dict[str, Any], cast(list[object], repaired["choices"])[0])
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "search_evidence"
    assert json.loads(call["function"]["arguments"]) == {"query": "M7"}


@pytest.mark.parametrize(
    "content",
    [
        "plain text",
        '{"answer":"JSON is a legitimate answer"}',
        '{"name":"unknown","arguments":{}}',
        '{"name":"search_evidence","arguments":"invalid"}',
    ],
)
def test_nonstream_auto_tool_repair_preserves_non_tool_content(content: str) -> None:
    completion: dict[str, Any] = {
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "tool_calls": []},
                "finish_reason": "stop",
            }
        ]
    }

    repaired = _repair_auto_tool_completion(completion, frozenset({"search_evidence"}))

    choice = cast(dict[str, Any], cast(list[object], repaired["choices"])[0])
    message = cast(dict[str, Any], choice["message"])
    assert message["content"] == content
    assert message["tool_calls"] == []


def test_native_stream_tool_normalizer_emits_name_once_and_tool_finish() -> None:
    wire = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
        b'{"name":"search_evidence","arguments":"{\\""}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
        b'{"name":"search_evidence","arguments":"query\\":\\"M7\\"}"}}]},'
        b'"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    normalizer = _SSEToolNormalizer()
    output = b"".join(normalizer.feed(wire[:37]))
    output += b"".join(normalizer.feed(wire[37:]))
    output += b"".join(normalizer.flush())

    assert output.count(b'"name":"search_evidence"') == 1
    assert b'"finish_reason":"tool_calls"' in output
    assert b'"id":"call_' in output


def test_native_stream_tool_normalizer_synthesizes_missing_finish_before_done() -> None:
    wire = (
        b'data: {"id":"chat","choices":[{"index":0,"delta":{"tool_calls":'
        b'[{"index":0,"function":{"name":"search_evidence","arguments":"{}"}}]},'
        b'"finish_reason":null}]}\n\ndata: [DONE]\n\n'
    )
    normalizer = _SSEToolNormalizer()
    output = b"".join(normalizer.feed(wire)) + b"".join(normalizer.flush())

    assert output.count(b'"finish_reason":"tool_calls"') == 1
    assert output.endswith(b"data: [DONE]\n\n")


def test_nonstream_native_tool_finish_is_normalized() -> None:
    completion: dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "search_evidence"}}],
                },
                "finish_reason": "stop",
            }
        ]
    }

    normalized = _normalize_tool_completion(completion)

    choice = cast(dict[str, Any], cast(list[object], normalized["choices"])[0])
    assert choice["finish_reason"] == "tool_calls"
