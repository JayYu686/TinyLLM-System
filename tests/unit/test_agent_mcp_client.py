from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from tinyllm.agent.mcp_client import MCPClientError, MCPPolicyClient
from tinyllm.agent.schema import MCPServerConfig, MCPToolPolicy


def _client(
    *, access: Literal["read", "sandbox_write"] = "read", max_attempts: int = 3
) -> MCPPolicyClient:
    policy = MCPToolPolicy(
        name="search_evidence" if access == "read" else "apply_sandbox_config_patch",
        access=access,
        approval_required=access == "sandbox_write",
        max_attempts=max_attempts,
    )
    server = MCPServerConfig(
        server_id="tinyllm-devops",
        transport="stdio",
        command=Path("/usr/bin/env"),
        tools=(policy,),
    )
    return MCPPolicyClient(
        server,
        artifact_root=Path("/tmp/artifacts"),
        project_root=Path("/tmp/project"),
        evidence_index=Path("/tmp/index"),
        retry_delays_ms=(0, 0),
    )


def _tool(
    *,
    name: str = "search_evidence",
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Tool:
    return Tool(
        name=name,
        description="bounded tool",
        inputSchema=input_schema
        or {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        outputSchema=output_schema,
    )


class _Session:
    def __init__(self, tools: list[Tool]) -> None:
        self.tools = tools
        self.initialized = 0

    async def initialize(self) -> object:
        self.initialized += 1
        return object()

    async def list_tools(self) -> object:
        return SimpleNamespace(tools=self.tools)


def _install_session(
    monkeypatch: pytest.MonkeyPatch, client: MCPPolicyClient, session: _Session
) -> None:
    @asynccontextmanager
    async def fake_session() -> Any:
        yield session

    monkeypatch.setattr(client, "_session", fake_session)


def test_mcp_policy_and_catalog_reject_unregistered_or_invalid_tools() -> None:
    client = _client()
    assert client.tool_names == ("search_evidence",)
    with pytest.raises(MCPClientError, match="allowlist"):
        client.policy("shell")
    with pytest.raises(MCPClientError, match="invalid response"):
        client._catalog(object())
    tool = _tool()
    with pytest.raises(MCPClientError, match="invalid or duplicate"):
        client._catalog(SimpleNamespace(tools=[tool, tool]))
    with pytest.raises(MCPClientError, match="invalid or duplicate"):
        client._catalog(SimpleNamespace(tools=[object()]))


def test_mcp_nullable_and_write_schema_are_model_safe() -> None:
    client = _client(access="sandbox_write", max_attempts=1)
    policy = client.policy("apply_sandbox_config_patch")
    tool = _tool(
        name="apply_sandbox_config_patch",
        input_schema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "approval_id": {"type": "string"},
                "call_id": {"type": "string"},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["run_id", "approval_id", "call_id"],
        },
    )
    public = client._public_input_schema(tool, policy)
    assert set(public["properties"]) == {"note"}
    assert public["required"] == []
    assert public["properties"]["note"]["type"] == "string"
    assert client._normalize_nullable_schema([{"anyOf": [{"type": "integer"}]}]) == [
        {"anyOf": [{"type": "integer"}]}
    ]

    malformed = Tool.model_construct(name="broken", inputSchema=["not", "an", "object"])
    with pytest.raises(MCPClientError, match="not an object"):
        client._public_input_schema(malformed, policy)


def test_mcp_input_validation_and_discovery_enforce_server_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    with pytest.raises(MCPClientError, match="arguments"):
        client._validate_input(_tool().inputSchema, {"query": 3})

    session = _Session([_tool()])
    _install_session(monkeypatch, client, session)
    asyncio.run(client.validate_call("search_evidence", {"query": "M7"}))
    definitions = asyncio.run(client.discover_tools())
    assert definitions[0].tool_name == "search_evidence"
    assert session.initialized == 2

    missing = _Session([])
    _install_session(monkeypatch, client, missing)
    with pytest.raises(MCPClientError, match="registered MCP tool is missing"):
        asyncio.run(client.validate_call("search_evidence", {"query": "M7"}))
    with pytest.raises(MCPClientError, match="registered MCP tools are missing"):
        asyncio.run(client.discover_tools())


def test_mcp_discovery_rejects_invalid_advertised_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    invalid = _tool(input_schema={"type": "invalid"})
    _install_session(monkeypatch, client, _Session([invalid]))
    with pytest.raises(MCPClientError, match="invalid input Schema"):
        asyncio.run(client.discover_tools())


def test_mcp_result_accepts_structured_or_text_json() -> None:
    tool = _tool(output_schema={"type": "object", "required": ["ok"]})
    structured = CallToolResult(content=[], structuredContent={"ok": True})
    assert MCPPolicyClient._result(structured, tool) == {"ok": True}
    text = CallToolResult(content=[TextContent(type="text", text='{"ok":true}')])
    assert MCPPolicyClient._result(text, tool) == {"ok": True}


@pytest.mark.parametrize(
    ("result", "pattern"),
    [
        (CallToolResult(content=[], isError=True), "execution error"),
        (
            CallToolResult(content=[ImageContent(type="image", data="eA==", mimeType="image/png")]),
            "unsupported content",
        ),
        (CallToolResult(content=[TextContent(type="text", text="not-json")]), "not structured"),
        (
            CallToolResult(content=[TextContent(type="text", text="[]")]),
            "JSON object",
        ),
        (
            CallToolResult(content=[], structuredContent={"reasoning_content": "secret"}),
            "private reasoning",
        ),
        (
            CallToolResult(content=[], structuredContent={"payload": "x" * 70_000}),
            "output limit",
        ),
    ],
)
def test_mcp_result_rejects_unsafe_server_outputs(result: CallToolResult, pattern: str) -> None:
    with pytest.raises((MCPClientError, ValueError), match=pattern):
        MCPPolicyClient._result(result, _tool())


def test_mcp_result_validates_output_schema() -> None:
    tool = _tool(output_schema={"type": "object", "required": ["ok"]})
    with pytest.raises(MCPClientError, match="output Schema"):
        MCPPolicyClient._result(CallToolResult(content=[], structuredContent={"other": 1}), tool)


def test_mcp_call_retries_reads_but_never_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    reads = _client()
    read_attempts = 0

    async def flaky(_name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal read_attempts
        read_attempts += 1
        if read_attempts < 3:
            raise OSError("temporary")
        return {"ok": True}

    monkeypatch.setattr(reads, "_call_once", flaky)
    assert asyncio.run(reads.call("search_evidence", {"query": "M7"})) == {"ok": True}
    assert read_attempts == 3

    write = _client(access="sandbox_write", max_attempts=1)
    write_attempts = 0

    async def fail_write(_name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal write_attempts
        write_attempts += 1
        raise TimeoutError

    monkeypatch.setattr(write, "_call_once", fail_write)
    with pytest.raises(MCPClientError, match="1 attempt"):
        asyncio.run(write.call("apply_sandbox_config_patch", {}))
    assert write_attempts == 1
