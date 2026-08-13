"""Bounded async backend contract and a vLLM HTTP implementation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

import httpx


class BackendError(RuntimeError):
    """A safe, classified backend failure."""

    def __init__(self, message: str, *, status_code: int = 502, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class ChatBackend(Protocol):
    """Minimal backend surface required by the Gateway."""

    async def health(self) -> bool: ...

    async def complete(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]: ...

    def stream(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        disconnected: Callable[[], Awaitable[bool]],
    ) -> AsyncIterator[bytes]: ...

    async def close(self) -> None: ...


class VLLMHTTPBackend:
    """Loopback-only vLLM proxy with bounded retry and cancellation semantics."""

    _RETRYABLE_STATUS = {502, 503, 504}

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_seconds: float,
        health_timeout_seconds: float,
        internal_token: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = request_timeout_seconds
        self._health_timeout = health_timeout_seconds
        self._internal_headers = (
            {"Authorization": f"Bearer {internal_token}"} if internal_token is not None else {}
        )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)

    async def health(self) -> bool:
        try:
            response = await self._client.get(
                f"{self._base_url}/health",
                headers=self._internal_headers,
                timeout=self._health_timeout,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def complete(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        for attempt in range(2):
            try:
                response = await self._client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    headers={**headers, **self._internal_headers},
                    timeout=self._timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == 0:
                    await asyncio.sleep(0)
                    continue
                raise BackendError("model backend is unavailable", retryable=True) from exc
            if response.status_code in self._RETRYABLE_STATUS and attempt == 0:
                continue
            if response.status_code >= 400:
                raise _response_error(response)
            try:
                decoded: Any = response.json()
            except json.JSONDecodeError as exc:
                raise BackendError("model backend returned invalid JSON") from exc
            if not isinstance(decoded, dict):
                raise BackendError("model backend returned an invalid response")
            return decoded
        raise BackendError("model backend retry was exhausted", retryable=True)

    async def stream(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        disconnected: Callable[[], Awaitable[bool]],
    ) -> AsyncIterator[bytes]:
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers={**headers, **self._internal_headers},
                timeout=self._timeout,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise _response_error(response)
                async for chunk in response.aiter_bytes():
                    if await disconnected():
                        return
                    if chunk:
                        yield chunk
        except BackendError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise BackendError("model backend stream failed", retryable=False) from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _response_error(response: httpx.Response) -> BackendError:
    status = response.status_code
    mapped = status if 400 <= status < 500 else 502
    return BackendError(
        "model backend rejected the request" if mapped < 500 else "model backend failed",
        status_code=mapped,
        retryable=status in VLLMHTTPBackend._RETRYABLE_STATUS,
    )
