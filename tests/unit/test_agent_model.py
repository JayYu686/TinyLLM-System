from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx
import pytest

from tinyllm.agent.model import (
    AgentModelError,
    GatewayAgentModel,
    _compact_observation,
    _function_name,
    _parse_text_tool_calls,
)
from tinyllm.agent.schema import AgentMessage, AgentToolDefinition

TOKEN = "gateway-agent-token-with-at-least-32-characters"


class _MCPClient:
    tool_names = ("search_evidence",)

    async def discover_tools(self) -> tuple[AgentToolDefinition, ...]:
        return (
            AgentToolDefinition(
                server_id="tinyllm-devops",
                tool_name="search_evidence",
                description="Search evidence",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
        )


def _model(handler: Any, *, seed: int | None = None) -> GatewayAgentModel:
    return GatewayAgentModel(
        base_url="http://127.0.0.1:8000",
        bearer_token=TOKEN,
        model="production",
        clients={"tinyllm-devops": _MCPClient()},  # type: ignore[dict-item]
        seed=seed,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_gateway_agent_model_forwards_optional_evaluation_seed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert payload["seed"] == 20260820
        policy = payload["messages"][0]["content"]
        assert "Follow this decision order before every tool call" in policy
        assert "JSON Schema's required array" in policy
        assert "sufficient authorization for a read" in policy
        assert "ask one concise clarification and make no tool call" in policy
        assert "incident/reference ID is metadata" in policy
        assert "search_evidence: policy, documentation, or evidence search" in policy
        assert "read_log_excerpt: .log/.txt log inspection" in policy
        assert "literal Run ID or relative path" in policy
        assert "independent or parallel operations, emit all calls together" in policy
        assert "emit all calls in the requested order" in policy
        assert "retries are performed by the runtime" in policy
        assert "explicitly repeat the user-requested subject" in policy
        assert "or entity and cite every supporting call" in policy
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    decision = asyncio.run(
        _model(handler, seed=20260820).decide(
            messages=(AgentMessage(role="user", content="diagnose"),),
            observations=(),
            mode="nonthinking",
            allowed_tools=("tinyllm-devops.search_evidence",),
        )
    )
    assert decision.message == "done"


def test_gateway_agent_model_maps_openai_tool_call_to_local_authority() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert payload["tool_choice"] == "auto"
        assert payload["tools"][0]["function"]["name"] == "search_evidence"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_search_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_evidence",
                                        "arguments": '{"query":"failure"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    decision = asyncio.run(
        _model(handler).decide(
            messages=(AgentMessage(role="user", content="diagnose"),),
            observations=(),
            mode="nonthinking",
            allowed_tools=("tinyllm-devops.search_evidence",),
        )
    )

    assert decision.tool_calls[0].call_id.startswith("call_")
    assert decision.tool_calls[0].server_id == "tinyllm-devops"
    assert decision.tool_calls[0].tool_name == "search_evidence"
    assert decision.tool_calls[0].arguments == {"query": "failure"}


def test_gateway_agent_model_omits_tool_fields_for_final_answer_retry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert "tools" not in payload
        assert "tool_choice" not in payload
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    decision = asyncio.run(
        _model(handler).decide(
            messages=(AgentMessage(role="user", content="summarize the verified evidence"),),
            observations=(),
            mode="nonthinking",
            allowed_tools=(),
        )
    )

    assert decision.message == "done"


def test_gateway_agent_model_rejects_unknown_tool_and_reasoning_leakage() -> None:
    def unknown(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_unknown_1",
                                    "function": {"name": "shell", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            },
        )

    with pytest.raises(AgentModelError, match="unknown tool"):
        asyncio.run(
            _model(unknown).decide(
                messages=(AgentMessage(role="user", content="x"),),
                observations=(),
                mode="nonthinking",
                allowed_tools=("tinyllm-devops.search_evidence",),
            )
        )

    def leak(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer", "reasoning_content": "private"}}]},
        )

    with pytest.raises(AgentModelError, match="private reasoning"):
        asyncio.run(
            _model(leak).decide(
                messages=(AgentMessage(role="user", content="x"),),
                observations=(),
                mode="nonthinking",
                allowed_tools=("tinyllm-devops.search_evidence",),
            )
        )


def test_gateway_agent_model_serializes_observation_with_evidence_identity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        assert payload["messages"][-2]["tool_calls"][0]["id"] == "call_search_1"
        assert payload["messages"][-1]["tool_call_id"] == "call_search_1"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "done [evidence:call_search_1]"}}]},
        )

    decision = asyncio.run(
        _model(handler).decide(
            messages=(AgentMessage(role="user", content="diagnose"),),
            observations=(
                {
                    "call_id": "call_search_1",
                    "server_id": "tinyllm-devops",
                    "tool_name": "search_evidence",
                    "result": {"results": []},
                },
            ),
            mode="nonthinking",
            allowed_tools=("tinyllm-devops.search_evidence",),
        )
    )

    assert decision.message == "done [evidence:call_search_1]"


def test_gateway_agent_model_bounds_retrieval_observations() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        observation = __import__("json").loads(payload["messages"][-1]["content"])
        assert len(observation["result"]["results"]) == 3
        assert all(len(item["excerpt"]) == 600 for item in observation["result"]["results"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "done [evidence:call_search_1]"}}]},
        )

    results = [
        {
            "document_id": f"doc-{index:016x}",
            "relative_path": "reports/m7/m7_acceptance.md",
            "start_line": 1,
            "end_line": 40,
            "content_sha256": "a" * 64,
            "relevance_score": 1.0,
            "excerpt": "x" * 1200,
        }
        for index in range(8)
    ]
    decision = asyncio.run(
        _model(handler).decide(
            messages=(AgentMessage(role="user", content="diagnose"),),
            observations=(
                {
                    "call_id": "call_search_1",
                    "server_id": "tinyllm-devops",
                    "tool_name": "search_evidence",
                    "result": {"schema_version": "1.0", "results": results},
                },
            ),
            mode="nonthinking",
            allowed_tools=("tinyllm-devops.search_evidence",),
        )
    )
    assert decision.message == "done [evidence:call_search_1]"


def test_gateway_agent_model_repairs_leading_text_tool_markup() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '<tool_call>\n{"name":"search_evidence",'
                                '"arguments":{"query":"M7"}}\n'
                                "<证据>fabricated content must be ignored"
                            )
                        }
                    }
                ]
            },
        )

    decision = asyncio.run(
        _model(handler).decide(
            messages=(AgentMessage(role="user", content="diagnose"),),
            observations=(),
            mode="nonthinking",
            allowed_tools=("tinyllm-devops.search_evidence",),
        )
    )

    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].tool_name == "search_evidence"
    assert decision.tool_calls[0].arguments == {"query": "M7"}


def test_gateway_agent_model_rejects_unparsed_tool_or_evidence_markup() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "<证据>fake"}}]})

    with pytest.raises(AgentModelError, match="unparsed"):
        asyncio.run(
            _model(handler).decide(
                messages=(AgentMessage(role="user", content="diagnose"),),
                observations=(),
                mode="nonthinking",
                allowed_tools=("tinyllm-devops.search_evidence",),
            )
        )


@pytest.mark.parametrize(
    "content",
    (
        "<call_id>tinyllm_devops__get_run</call_id>",
        "<search_evidence>tinyllm_devops__search_evidence</search_evidence>",
        '<call_search_evidence>{"query":"M7"}</call_search_evidence>',
    ),
)
def test_gateway_agent_model_rejects_legacy_pseudo_tool_tags(content: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    with pytest.raises(AgentModelError, match="unparsed"):
        asyncio.run(
            _model(handler).decide(
                messages=(AgentMessage(role="user", content="diagnose"),),
                observations=(),
                mode="nonthinking",
                allowed_tools=("tinyllm-devops.search_evidence",),
            )
        )


def test_gateway_agent_model_rejects_alternate_xml_tool_markup() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "<call apply_sandbox_config_patchArguments>"
                                "<updates>unsafe</updates></call>"
                            ),
                            "tool_calls": [],
                        }
                    }
                ]
            },
        )

    with pytest.raises(AgentModelError, match="unparsed"):
        asyncio.run(
            _model(handler).decide(
                messages=(AgentMessage(role="user", content="patch"),),
                observations=(),
                mode="nonthinking",
                allowed_tools=("tinyllm-devops.search_evidence",),
            )
        )


def test_gateway_agent_model_requires_loopback_and_strong_token() -> None:
    with pytest.raises(AgentModelError, match="loopback"):
        GatewayAgentModel(
            base_url="https://gateway.example.com",
            bearer_token=TOKEN,
            model="production",
            clients={},
        )
    with pytest.raises(AgentModelError, match="too short"):
        GatewayAgentModel(
            base_url="http://localhost:8000/",
            bearer_token="short",
            model="production",
            clients={},
        )


def test_gateway_agent_model_closes_only_owned_http_client() -> None:
    owned = GatewayAgentModel(
        base_url="http://localhost:8000/",
        bearer_token=TOKEN,
        model="production",
        clients={},
    )
    asyncio.run(owned.close())
    assert owned._http.is_closed

    supplied = httpx.AsyncClient()
    external = GatewayAgentModel(
        base_url="http://localhost:8000",
        bearer_token=TOKEN,
        model="production",
        clients={},
        http_client=supplied,
    )
    asyncio.run(external.close())
    assert not supplied.is_closed
    asyncio.run(supplied.aclose())


def test_function_encoding_and_text_parser_are_bounded() -> None:
    definition = AgentToolDefinition(
        server_id="server",
        tool_name="t" * 64,
        input_schema={"type": "object"},
    )
    assert _function_name(definition) == "t" * 64

    encoded = {"search_evidence": ("tinyllm-devops", "search_evidence")}
    assert _parse_text_tool_calls("ordinary answer", encoded) == ()
    for content, pattern in (
        ("<tool_call>{", "malformed"),
        ('<tool_call>{"name":"x"}', "unsupported"),
        ('<tool_call>{"name":"unknown","arguments":{}}', "unknown"),
        (
            '<tool_call>{"name":"search_evidence","arguments":[]}',
            "JSON object",
        ),
    ):
        with pytest.raises(AgentModelError, match=pattern):
            _parse_text_tool_calls(content, encoded)

    one = '<tool_call>{"name":"search_evidence","arguments":{}}'
    with pytest.raises(AgentModelError, match="too many"):
        _parse_text_tool_calls(one * 9, encoded)


def test_compact_observation_handles_errors_invalid_items_and_large_values() -> None:
    assert _compact_observation(
        {"call_id": "call_1", "server_id": "s", "tool_name": "t", "error": "Timeout"}
    ) == {
        "call_id": "call_1",
        "server_id": "s",
        "tool_name": "t",
        "error": "Timeout",
    }
    compact = _compact_observation(
        {
            "call_id": "call_1",
            "result": {
                "schema_version": "1.0",
                "results": [None, {"document_id": "doc", "excerpt": 3}],
            },
        }
    )
    assert compact["result"] == {
        "schema_version": "1.0",
        "results": [{"document_id": "doc"}],
    }
    assert _compact_observation({"result": {"payload": "x" * 5000}})["result"] == {
        "truncated": True
    }


def test_gateway_agent_model_rejects_allowed_tool_drift() -> None:
    with pytest.raises(AgentModelError, match="differ"):
        asyncio.run(
            _model(lambda _request: httpx.Response(500)).decide(
                messages=(AgentMessage(role="user", content="x"),),
                observations=(),
                mode="nonthinking",
                allowed_tools=("tinyllm-devops.search_evidence", "tinyllm-devops.shell"),
            )
        )


def test_gateway_agent_model_rejects_duplicate_openai_names_across_servers() -> None:
    class DuplicateClient:
        def __init__(self, server_id: str) -> None:
            self.server_id = server_id

        async def discover_tools(self) -> tuple[AgentToolDefinition, ...]:
            return (
                AgentToolDefinition(
                    server_id=self.server_id,
                    tool_name="search_evidence",
                    input_schema={"type": "object"},
                ),
            )

    model = GatewayAgentModel(
        base_url="http://127.0.0.1:8000",
        bearer_token=TOKEN,
        model="production",
        clients=cast(
            Any,
            {
                "server-one": DuplicateClient("server-one"),
                "server-two": DuplicateClient("server-two"),
            },
        ),
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(500))
        ),
    )
    with pytest.raises(AgentModelError, match="collide"):
        asyncio.run(
            model.decide(
                messages=(AgentMessage(role="user", content="x"),),
                observations=(),
                mode="nonthinking",
                allowed_tools=("server-one.search_evidence",),
            )
        )
    asyncio.run(model.close())


@pytest.mark.parametrize(
    ("observation", "pattern"),
    [
        ({"call_id": 3, "server_id": "tinyllm-devops", "tool_name": "search_evidence"}, "identity"),
        (
            {"call_id": "call_1", "server_id": "tinyllm-devops", "tool_name": "unknown"},
            "unavailable",
        ),
    ],
)
def test_gateway_agent_model_rejects_untrusted_observation_identity(
    observation: dict[str, object], pattern: str
) -> None:
    with pytest.raises(AgentModelError, match=pattern):
        asyncio.run(
            _model(lambda _request: httpx.Response(500)).decide(
                messages=(AgentMessage(role="user", content="x"),),
                observations=(observation,),
                mode="nonthinking",
                allowed_tools=("tinyllm-devops.search_evidence",),
            )
        )


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503, json={"error": "backend unavailable"}),
        httpx.Response(200, content=b"not-json"),
    ],
)
def test_gateway_agent_model_maps_transport_and_json_errors(response: httpx.Response) -> None:
    with pytest.raises(AgentModelError, match="request failed"):
        asyncio.run(
            _model(lambda _request: response).decide(
                messages=(AgentMessage(role="user", content="x"),),
                observations=(),
                mode="nonthinking",
                allowed_tools=("tinyllm-devops.search_evidence",),
            )
        )


@pytest.mark.parametrize(
    "message",
    [
        {},
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "search_evidence",
                        "arguments": "[]",
                    },
                }
            ]
        },
        {"tool_calls": [{"id": "call_1", "function": {"arguments": "{}"}}]},
    ],
)
def test_gateway_agent_model_rejects_malformed_decisions(message: dict[str, object]) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": message}]})

    with pytest.raises(AgentModelError, match="neither|JSON object|malformed"):
        asyncio.run(
            _model(handler).decide(
                messages=(AgentMessage(role="user", content="x"),),
                observations=(),
                mode="nonthinking",
                allowed_tools=("tinyllm-devops.search_evidence",),
            )
        )
