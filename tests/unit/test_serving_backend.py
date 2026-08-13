from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tinyllm.serving.backend import BackendError, VLLMHTTPBackend


def test_backend_health_completion_retry_and_close() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path == "/health":
            return httpx.Response(200)
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"id": "ok"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = VLLMHTTPBackend(
                "http://127.0.0.1:8001",
                request_timeout_seconds=1,
                health_timeout_seconds=1,
                client=client,
            )
            assert await backend.health() is True
            assert await backend.complete({"model": "x"}, {}) == {"id": "ok"}
            await backend.close()

    asyncio.run(run())
    assert attempts == 2


def test_backend_maps_errors_and_streams_without_retry() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            raise httpx.ConnectError("offline")
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=b"data: one\n\ndata: [DONE]\n\n")
        return httpx.Response(400, json={"detail": "private detail"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = VLLMHTTPBackend(
                "http://127.0.0.1:8001",
                request_timeout_seconds=1,
                health_timeout_seconds=1,
                client=client,
            )
            assert await backend.health() is False
            with pytest.raises(BackendError) as failure:
                await backend.complete({"model": "x"}, {})
            assert failure.value.status_code == 400
            assert "private detail" not in str(failure.value)

            disconnected = False

            async def is_disconnected() -> bool:
                return disconnected

            chunks = [
                chunk
                async for chunk in backend.stream(
                    {"model": "x", "stream": True}, {}, is_disconnected
                )
            ]
            assert b"".join(chunks).endswith(b"[DONE]\n\n")

    asyncio.run(run())


def test_backend_rejects_invalid_json_and_exhausted_transport() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise httpx.ConnectError("offline")
        return httpx.Response(200, content=b"not-json")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = VLLMHTTPBackend(
                "http://127.0.0.1:8001",
                request_timeout_seconds=1,
                health_timeout_seconds=1,
                client=client,
            )
            with pytest.raises(BackendError, match="unavailable"):
                await backend.complete({}, {})
            with pytest.raises(BackendError, match="invalid JSON"):
                await backend.complete({}, {})

    asyncio.run(run())
