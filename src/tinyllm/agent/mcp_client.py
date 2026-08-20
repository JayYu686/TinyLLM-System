"""Policy-enforcing MCP client for allowlisted TinyLLM Agent tools."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
from jsonschema import Draft202012Validator, SchemaError, ValidationError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, Tool

from tinyllm.agent.schema import (
    AgentToolDefinition,
    MCPServerConfig,
    MCPToolPolicy,
    _reject_private_reasoning,
)

MAX_MCP_RESULT_BYTES = 65_536
INTERNAL_WRITE_ARGUMENTS = frozenset({"run_id", "approval_id", "call_id"})


class MCPClientError(RuntimeError):
    """Raised for policy, protocol, timeout, or tool execution failures."""


class _Session(Protocol):
    async def initialize(self) -> object: ...

    async def list_tools(self) -> object: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
    ) -> CallToolResult: ...


class MCPPolicyClient:
    """Invoke only configured tools and treat server annotations as untrusted metadata."""

    def __init__(
        self,
        server: MCPServerConfig,
        *,
        artifact_root: Path,
        project_root: Path,
        evidence_index: Path,
        retry_delays_ms: tuple[int, ...] = (250, 500),
    ) -> None:
        self.server = server
        self.artifact_root = artifact_root
        self.project_root = project_root
        self.evidence_index = evidence_index
        self.retry_delays_ms = retry_delays_ms
        self._policy = {item.name: item for item in server.tools}

    def policy(self, tool_name: str) -> MCPToolPolicy:
        try:
            return self._policy[tool_name]
        except KeyError as exc:
            raise MCPClientError("tool is not present in the local allowlist") from exc

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Expose the administrator allowlist without server-provided annotations."""

        return tuple(self._policy)

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        if self.server.transport == "stdio":
            assert self.server.command is not None
            executable = str(Path(sys.executable).parent)
            environment = {
                "PATH": f"{executable}:/usr/bin:/bin",
                "PYTHONUNBUFFERED": "1",
                "TINYLLM_ARTIFACT_ROOT": str(self.artifact_root),
                "TINYLLM_PROJECT_ROOT": str(self.project_root),
                "TINYLLM_EVIDENCE_INDEX": str(self.evidence_index),
            }
            parameters = StdioServerParameters(
                command=str(self.server.command),
                args=list(self.server.args),
                env=environment,
                cwd=self.project_root,
            )
            async with (
                stdio_client(parameters) as (reader, writer),
                ClientSession(reader, writer) as session,
            ):
                yield session
            return
        assert self.server.url is not None and self.server.bearer_token_env is not None
        token = os.environ.get(self.server.bearer_token_env, "")
        if len(token) < 32:
            raise MCPClientError("MCP HTTP Bearer secret is missing or too short")
        async with (
            httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token}"},
                follow_redirects=False,
                timeout=15,
            ) as client,
            streamable_http_client(self.server.url, http_client=client) as streams,
            ClientSession(streams[0], streams[1]) as session,
        ):
            yield session

    @staticmethod
    def _catalog(value: object) -> dict[str, Tool]:
        tools = getattr(value, "tools", None)
        if not isinstance(tools, list):
            raise MCPClientError("MCP tools/list returned an invalid response")
        catalog: dict[str, Tool] = {}
        for raw in tools:
            if not isinstance(raw, Tool) or raw.name in catalog:
                raise MCPClientError("MCP tools/list contains an invalid or duplicate tool")
            catalog[raw.name] = raw
        return catalog

    @staticmethod
    def _validate_input(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
        try:
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(arguments)
        except (SchemaError, ValidationError) as exc:
            raise MCPClientError("tool arguments failed the discovered JSON Schema") from exc

    def _public_input_schema(self, tool: Tool, policy: MCPToolPolicy) -> dict[str, Any]:
        normalized = self._normalize_nullable_schema(tool.inputSchema)
        if not isinstance(normalized, dict):
            raise MCPClientError("MCP tool input Schema is not an object")
        input_schema = cast(dict[str, Any], normalized)
        if policy.access == "sandbox_write":
            properties = dict(input_schema.get("properties", {}))
            for internal in INTERNAL_WRITE_ARGUMENTS:
                properties.pop(internal, None)
            required = [
                item
                for item in input_schema.get("required", [])
                if item not in INTERNAL_WRITE_ARGUMENTS
            ]
            input_schema["properties"] = properties
            input_schema["required"] = required
        return input_schema

    @classmethod
    def _normalize_nullable_schema(cls, value: object) -> Any:
        """Remove Pydantic's optional-null branch from model-facing input Schemas."""

        if isinstance(value, list):
            return [cls._normalize_nullable_schema(item) for item in value]
        if not isinstance(value, dict):
            return value
        normalized = {
            str(key): cls._normalize_nullable_schema(item)
            for key, item in value.items()
            if key != "anyOf"
        }
        branches = value.get("anyOf")
        if isinstance(branches, list) and len(branches) == 2:
            concrete = [
                item for item in branches if isinstance(item, dict) and item.get("type") != "null"
            ]
            nulls = [
                item for item in branches if isinstance(item, dict) and item.get("type") == "null"
            ]
            if len(concrete) == len(nulls) == 1:
                normalized.update(cls._normalize_nullable_schema(concrete[0]))
                return normalized
        if branches is not None:
            normalized["anyOf"] = cls._normalize_nullable_schema(branches)
        return normalized

    async def validate_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Validate model arguments against the server Schema before policy approval."""

        policy = self.policy(tool_name)
        async with self._session() as session:
            await session.initialize()
            catalog = self._catalog(await session.list_tools())
        try:
            tool = catalog[tool_name]
        except KeyError as exc:
            raise MCPClientError("registered MCP tool is missing") from exc
        self._validate_input(self._public_input_schema(tool, policy), arguments)

    async def discover_tools(self) -> tuple[AgentToolDefinition, ...]:
        """Discover Schemas, retaining authority only for locally registered names."""

        async with self._session() as session:
            await session.initialize()
            catalog = self._catalog(await session.list_tools())
        missing = sorted(set(self._policy) - set(catalog))
        if missing:
            raise MCPClientError(f"registered MCP tools are missing: {missing}")
        definitions: list[AgentToolDefinition] = []
        for name in self._policy:
            tool = catalog[name]
            try:
                Draft202012Validator.check_schema(tool.inputSchema)
                input_schema = self._public_input_schema(tool, self._policy[name])
                definitions.append(
                    AgentToolDefinition(
                        server_id=self.server.server_id,
                        tool_name=name,
                        description=tool.description,
                        input_schema=input_schema,
                    )
                )
            except (SchemaError, ValueError) as exc:
                raise MCPClientError("MCP tool advertises an invalid input Schema") from exc
        return tuple(definitions)

    @staticmethod
    def _result(result: CallToolResult, tool: Tool) -> dict[str, Any]:
        if result.isError:
            raise MCPClientError("MCP tool returned an execution error")
        value: object | None = result.structuredContent
        if value is None:
            texts = [getattr(item, "text", None) for item in result.content]
            if len(texts) != 1 or not isinstance(texts[0], str):
                raise MCPClientError("MCP tool returned an unsupported content type")
            try:
                value = json.loads(texts[0])
            except json.JSONDecodeError as exc:
                raise MCPClientError("MCP tool result is not structured JSON") from exc
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        if len(payload) > MAX_MCP_RESULT_BYTES:
            raise MCPClientError("MCP tool result exceeds the output limit")
        if tool.outputSchema is not None:
            try:
                Draft202012Validator.check_schema(tool.outputSchema)
                Draft202012Validator(tool.outputSchema).validate(value)
            except (SchemaError, ValidationError) as exc:
                raise MCPClientError("MCP tool result failed its output Schema") from exc
        _reject_private_reasoning(value)
        if not isinstance(value, dict):
            raise MCPClientError("MCP tool result must be a JSON object")
        return cast(dict[str, Any], value)

    async def _call_once(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        policy = self.policy(tool_name)
        async with self._session() as session:
            await session.initialize()
            catalog = self._catalog(await session.list_tools())
            missing = sorted(set(self._policy) - set(catalog))
            if missing:
                raise MCPClientError(f"registered MCP tools are missing: {missing}")
            tool = catalog[tool_name]
            self._validate_input(tool.inputSchema, arguments)
            result = await session.call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=timedelta(seconds=policy.timeout_seconds),
            )
            return self._result(result, tool)

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one tool with bounded read retries and no write retry."""

        policy = self.policy(tool_name)
        attempts = policy.max_attempts
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                async with asyncio.timeout(policy.timeout_seconds):
                    return await self._call_once(tool_name, arguments)
            except (MCPClientError, TimeoutError, OSError) as exc:
                last_error = exc
                if policy.access != "read" or attempt + 1 >= attempts:
                    break
                delay_index = min(attempt, len(self.retry_delays_ms) - 1)
                await asyncio.sleep(self.retry_delays_ms[delay_index] / 1000)
        raise MCPClientError(f"MCP tool failed after {attempts} attempt(s)") from last_error
