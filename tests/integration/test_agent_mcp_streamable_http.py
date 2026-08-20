from __future__ import annotations

import asyncio

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings


def test_streamable_http_protocol_conformance() -> None:
    async def exercise() -> None:
        server = FastMCP(
            "tinyllm-streamable-http-conformance",
            stateless_http=True,
            json_response=True,
            transport_security=TransportSecuritySettings(allowed_hosts=["mcp.test"]),
        )

        @server.tool(structured_output=True)
        def inspect_config(relative_path: str) -> dict[str, str]:
            """Return one deterministic protocol-test result."""

            return {"schema_version": "1.0", "relative_path": relative_path}

        app = server.streamable_http_app()
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://mcp.test",
            ) as http,
            streamable_http_client("http://mcp.test/mcp", http_client=http) as streams,
            ClientSession(streams[0], streams[1]) as session,
        ):
            initialization = await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool(
                "inspect_config", {"relative_path": "configs/train.yaml"}
            )

        assert initialization.protocolVersion
        assert [tool.name for tool in tools.tools] == ["inspect_config"]
        assert result.isError is False
        assert result.structuredContent == {
            "schema_version": "1.0",
            "relative_path": "configs/train.yaml",
        }

    asyncio.run(exercise())
