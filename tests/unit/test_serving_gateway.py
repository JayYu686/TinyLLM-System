from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tinyllm.deployment import ResolvedModel
from tinyllm.evaluation import M6ModelIdentity
from tinyllm.serving.backend import BackendError, ChatBackend
from tinyllm.serving.config import GatewayConfig
from tinyllm.serving.gateway import create_gateway

TOKEN = "unit-test-token-that-is-longer-than-32-characters"


class FakeBackend(ChatBackend):
    def __init__(self) -> None:
        self.healthy = True
        self.payloads: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.failure: BackendError | None = None

    async def health(self) -> bool:
        return self.healthy

    async def complete(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        if self.failure is not None:
            raise self.failure
        content = (
            "<think>private chain</think>\n\nfinal"
            if payload["chat_template_kwargs"]["enable_thinking"]
            else "ok"
        )
        self.payloads.append(payload)
        self.headers.append(headers)
        return {
            "id": "chatcmpl_unit",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": "private",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def stream(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        disconnected: Callable[[], Awaitable[bool]],
    ) -> AsyncIterator[bytes]:
        self.payloads.append(payload)
        self.headers.append(headers)
        if self.failure is not None:
            raise self.failure
        if not await disconnected():
            yield b'data: {"choices":[{"delta":{"content":"<thi"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"nk>private"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"</think>\\nfinal"}}]}\n\n'
            yield b'data: {"choices":[],"usage":{"prompt_tokens":3,'
            yield b'"completion_tokens":2,"total_tokens":5}}\n\n'
            yield b"data: [DONE]\n\n"

    async def close(self) -> None:
        return None


def _resolved() -> ResolvedModel:
    return ResolvedModel(
        requested_ref="qwen3-0-6b-m6-aaaaaaaa",
        status="Candidate",
        model_version="qwen3-0-6b-m6-aaaaaaaa",
        candidate_model_version="qwen3-0-6b-m6-aaaaaaaa",
        candidate_record_sha256="a" * 64,
        model=M6ModelIdentity(
            role="candidate",
            repository="Qwen/Qwen3-0.6B",
            base_revision="c1899de289a04d12100db370d81485cdf75e47ca",
            attention_architecture="gqa",
            adaptation="full_sft",
            model_artifact_sha256="b" * 64,
            model_parameters=596049920,
            training_run_id="20260813T000000Z-m7-unit-test-aaaaaaaa-beef",
            training_checkpoint_id="checkpoint-tokens-0001000000",
            training_tokens=1_000_000,
            training_config_sha256="c" * 64,
            dataset_version="m7-unit-data-v1",
            dataset_manifest_sha256="d" * 64,
        ),
        model_dir=Path("/data/tinyllm/model"),
        model_artifact_sha256="b" * 64,
        tokenizer_dir=Path("/data/tinyllm/tokenizer"),
        tokenizer_artifact_sha256="e" * 64,
        verified_at=datetime(2026, 8, 13, tzinfo=UTC),
    )


def _client(backend: FakeBackend, **config: Any) -> TestClient:
    gateway_config = GatewayConfig(
        config_id="m7-gateway-unit",
        trusted_hosts=("testserver",),
        **config,
    )
    return TestClient(
        create_gateway(
            config=gateway_config,
            resolved_model=_resolved(),
            backend=backend,
            bearer_token=TOKEN,
        )
    )


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_health_version_auth_and_docs_are_secure_by_default() -> None:
    backend = FakeBackend()
    with _client(backend) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").json()["ready"] is True
        assert client.get("/version").json()["model"] == "qwen3-0-6b-m6-aaaaaaaa"
        assert client.get("/docs").status_code == 404
        unauthorized = client.get("/v1/models")
        assert unauthorized.status_code == 401
        assert unauthorized.json()["error"]["code"] == "authentication_error"
        assert client.get("/v1/models", headers=_headers()).json()["object"] == "list"
        metrics = client.get("/metrics", headers=_headers())
        assert metrics.status_code == 200
        assert "tinyllm_gateway_tokens_total" in metrics.text
        assert "tinyllm_gateway_backend_restarts" in metrics.text

        backend.healthy = False
        readiness = client.get("/health/ready")
        assert readiness.status_code == 503
        assert readiness.json()["ready"] is False


def test_nonstream_chat_forwards_trace_without_logging_content() -> None:
    backend = FakeBackend()
    traceparent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    with _client(backend) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={**_headers(), "traceparent": traceparent, "x-request-id": "req-test"},
            json={
                "model": "production",
                "messages": [{"role": "user", "content": "secret prompt"}],
                "max_completion_tokens": 64,
            },
        )
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-test"
    assert backend.payloads[0]["model"] == "qwen3-0-6b-m6-aaaaaaaa"
    assert backend.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert backend.headers[0]["traceparent"] == traceparent


def test_stream_tool_request_and_error_mapping() -> None:
    backend = FakeBackend()
    with _client(backend) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers=_headers(),
            json={
                "model": "production",
                "messages": [{"role": "user", "content": "inspect"}],
                "mode": "thinking",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "inspect_config",
                            "description": "Inspect a config",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "auto",
            },
        ) as response:
            assert response.status_code == 200
            stream = "".join(response.iter_text())
            assert "data: [DONE]" in stream
            assert "private" not in stream
            assert "<think>" not in stream
            assert "final" in stream

        completion = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json={
                "model": "production",
                "messages": [{"role": "user", "content": "inspect"}],
                "mode": "thinking",
            },
        )
        assert completion.json()["choices"][0]["message"]["content"] == "final"
        assert "reasoning_content" not in completion.json()["choices"][0]["message"]

        backend.failure = BackendError("backend exploded", status_code=502)
        failure = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json={
                "model": "production",
                "messages": [{"role": "user", "content": "inspect"}],
            },
        )
        assert failure.status_code == 502
        assert failure.json()["error"]["code"] == "backend_error"


def test_schema_size_rate_and_model_limits() -> None:
    backend = FakeBackend()
    with _client(backend, requests_per_minute=2, max_request_bytes=1024) as client:
        invalid = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json={"model": "production", "messages": [], "unknown": True},
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "invalid_request_error"

        wrong_model = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json={"model": "unknown", "messages": [{"role": "user", "content": "x"}]},
        )
        assert wrong_model.status_code == 404

        limited = client.get("/v1/models", headers=_headers())
        assert limited.status_code == 429

    with _client(FakeBackend(), max_request_bytes=1024) as client:
        oversized = client.post(
            "/v1/chat/completions",
            headers={**_headers(), "content-length": "2048"},
            content=b"{}",
        )
        assert oversized.status_code == 413
        chunked = client.post(
            "/v1/chat/completions",
            headers={**_headers(), "transfer-encoding": "chunked"},
            content=b"x" * 1025,
        )
        assert chunked.status_code == 413
        nested = b'{"model":"production","messages":' + b"[" * 40 + b"]" * 40 + b"}"
        deeply_nested = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            content=nested,
        )
        assert deeply_nested.status_code == 400
