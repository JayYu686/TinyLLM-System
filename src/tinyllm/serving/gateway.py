"""Secure FastAPI application factory for the M7 Model Gateway."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import cast

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from opentelemetry import propagate, trace
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import ClientDisconnect

from tinyllm import __version__
from tinyllm.deployment import ResolvedModel
from tinyllm.serving.backend import BackendError, ChatBackend
from tinyllm.serving.config import GatewayConfig
from tinyllm.serving.observability import StructuredEventLog
from tinyllm.serving.schema import (
    ChatCompletionRequest,
    HealthResponse,
    ModelCard,
    ModelList,
    VersionResponse,
)
from tinyllm.serving.supervisor import BackendSupervisor, BackendSupervisorError
from tinyllm.serving.vllm_guard import _json_wire_depth_is_safe

TRACEPARENT = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-0[01]$")
REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class SlidingWindowRateLimiter:
    """Process-local fixed-identity rate limiter using a monotonic clock."""

    def __init__(self, limit: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._limit = limit
        self._clock = clock
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        now = self._clock()
        async with self._lock:
            while self._timestamps and self._timestamps[0] <= now - 60:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._limit:
                return False
            self._timestamps.append(now)
            return True


class GatewayMetrics:
    """Per-app Prometheus collectors without sensitive labels."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "tinyllm_gateway_requests_total",
            "Gateway requests by endpoint and status class.",
            ("endpoint", "status_class"),
            registry=self.registry,
        )
        self.inflight = Gauge(
            "tinyllm_gateway_inflight_requests",
            "Current Chat Completions requests.",
            registry=self.registry,
        )
        self.latency = Histogram(
            "tinyllm_gateway_request_seconds",
            "End-to-end Gateway request latency.",
            ("stream",),
            registry=self.registry,
        )
        self.queue = Histogram(
            "tinyllm_gateway_queue_seconds",
            "Time spent waiting for a Gateway concurrency slot.",
            registry=self.registry,
        )
        self.ttft = Histogram(
            "tinyllm_gateway_ttft_seconds",
            "Time to the first streamed backend chunk.",
            registry=self.registry,
        )
        self.tpot = Histogram(
            "tinyllm_gateway_tpot_seconds",
            "Approximate time per output token after the first streamed chunk.",
            registry=self.registry,
        )
        self.throughput = Histogram(
            "tinyllm_gateway_output_tokens_per_second",
            "Completion-token throughput observed by the Gateway.",
            registry=self.registry,
        )
        self.tokens = Counter(
            "tinyllm_gateway_tokens_total",
            "Tokens reported by the backend, separated by kind.",
            ("kind",),
            registry=self.registry,
        )
        self.backend_errors = Counter(
            "tinyllm_gateway_backend_errors_total",
            "Backend failures by mapped HTTP status.",
            ("status",),
            registry=self.registry,
        )
        self.backend_restarts = Gauge(
            "tinyllm_gateway_backend_restarts",
            "Managed backend restarts since Gateway startup.",
            registry=self.registry,
        )
        self.backend_ready = Gauge(
            "tinyllm_gateway_backend_ready",
            "Whether the model backend is currently ready.",
            registry=self.registry,
        )
        self.gpu_memory = Gauge(
            "tinyllm_backend_gpu_memory_used_bytes",
            "Physical GPU memory currently used by the managed backend GPU.",
            registry=self.registry,
        )
        self.stream_disconnects = Counter(
            "tinyllm_gateway_stream_disconnects_total",
            "Streaming requests disconnected before backend completion.",
            registry=self.registry,
        )


def _openai_error(message: str, *, code: str, status_code: int, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"message": message, "type": "tinyllm_error", "param": None, "code": code}
        },
        headers={"x-request-id": request_id},
    )


def _visible_content(value: str) -> str:
    """Return final-answer text while withholding Qwen-style reasoning blocks."""

    opening = "<think>"
    closing = "</think>"
    start = value.find(opening)
    if start < 0:
        return "" if opening.startswith(value) else value
    end = value.find(closing, start + len(opening))
    if end < 0:
        return ""
    return value[end + len(closing) :].lstrip()


def _sanitize_completion(result: dict[str, object]) -> dict[str, object]:
    """Remove raw reasoning fields and tags from a non-stream OpenAI response."""

    choices = result.get("choices")
    if not isinstance(choices, list):
        return result
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            continue
        message = choice["message"]
        message.pop("reasoning_content", None)
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _visible_content(content)
    return result


def _repair_auto_tool_completion(
    result: dict[str, object], tool_names: frozenset[str]
) -> dict[str, object]:
    """Convert the fixed legacy Qwen JSON fallback into one OpenAI tool call."""

    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return result
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        return result
    message = choice["message"]
    if message.get("tool_calls"):
        return result
    content = message.get("content")
    if not isinstance(content, str) or not content.lstrip().startswith("{"):
        return result
    try:
        value: object = json.loads(content)
    except json.JSONDecodeError:
        return result
    if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
        return result
    name = value.get("name")
    arguments = value.get("arguments")
    if not isinstance(name, str) or name not in tool_names or not isinstance(arguments, dict):
        return result
    identity = hashlib.sha256(content.encode()).hexdigest()[:24]
    message["content"] = None
    message["tool_calls"] = [
        {
            "id": f"call_{identity}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            },
        }
    ]
    choice["finish_reason"] = "tool_calls"
    return result


def _normalize_tool_completion(result: dict[str, object]) -> dict[str, object]:
    """Normalize legacy vLLM's tool-call termination to the OpenAI contract."""

    choices = result.get("choices")
    if not isinstance(choices, list):
        return result
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            continue
        if choice["message"].get("tool_calls"):
            choice["finish_reason"] = "tool_calls"
    return result


class _SSECoTFilter:
    """Filter content deltas across arbitrary byte and token boundaries."""

    def __init__(self) -> None:
        self._wire = b""
        self._raw_content = ""
        self._emitted_content = ""

    def feed(self, chunk: bytes) -> tuple[bytes, ...]:
        self._wire += chunk
        frames = self._wire.split(b"\n\n")
        self._wire = frames.pop()
        return tuple(self._filter_frame(frame) + b"\n\n" for frame in frames if frame)

    def flush(self) -> tuple[bytes, ...]:
        if not self._wire:
            return ()
        frame, self._wire = self._wire, b""
        return (self._filter_frame(frame),)

    def _filter_frame(self, frame: bytes) -> bytes:
        lines = frame.splitlines()
        for index, line in enumerate(lines):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            try:
                decoded = json.loads(line[6:])
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(decoded, dict) or not isinstance(decoded.get("choices"), list):
                continue
            for choice in decoded["choices"]:
                if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                    continue
                delta = choice["delta"]
                delta.pop("reasoning_content", None)
                content = delta.get("content")
                if not isinstance(content, str):
                    continue
                self._raw_content += content
                visible = _visible_content(self._raw_content)
                if not visible.startswith(self._emitted_content):
                    delta["content"] = ""
                    continue
                delta["content"] = visible[len(self._emitted_content) :]
                self._emitted_content = visible
            lines[index] = (
                b"data: " + json.dumps(decoded, ensure_ascii=False, separators=(",", ":")).encode()
            )
        return b"\n".join(lines)


class _SSEAutoToolRepair:
    """Repair fixed Qwen/Hermes JSON tool markup across arbitrary SSE boundaries."""

    def __init__(self, tool_names: frozenset[str]) -> None:
        self._wire = b""
        self._tool_names = tool_names
        self._candidate = ""
        self._collecting = False
        self._repaired = False
        self._finish_emitted = False
        self._template: dict[str, object] | None = None

    def feed(self, chunk: bytes) -> tuple[bytes, ...]:
        self._wire += chunk
        frames = self._wire.split(b"\n\n")
        self._wire = frames.pop()
        output: list[bytes] = []
        for frame in frames:
            output.extend(self._repair_frame(frame))
        return tuple(output)

    def flush(self) -> tuple[bytes, ...]:
        output: list[bytes] = []
        if self._wire:
            output.extend(self._repair_frame(self._wire))
            self._wire = b""
        if self._collecting and not self._repaired:
            output.append(self._error_frame("streamed tool JSON was incomplete or malformed"))
        return tuple(output)

    def _repair_frame(self, frame: bytes) -> list[bytes]:
        if not frame:
            return []
        if frame == b"data: [DONE]":
            output: list[bytes] = []
            if self._collecting and not self._repaired:
                output.append(self._error_frame("streamed tool JSON was incomplete or malformed"))
            elif self._repaired and not self._finish_emitted:
                output.append(self._finish_frame())
                self._finish_emitted = True
            output.append(frame + b"\n\n")
            return output
        lines = frame.splitlines()
        for line in lines:
            if not line.startswith(b"data: "):
                continue
            try:
                decoded: object = json.loads(line[6:])
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(decoded, dict) or not isinstance(decoded.get("choices"), list):
                continue
            self._template = {key: value for key, value in decoded.items() if key != "choices"}
            for choice in decoded["choices"]:
                if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                    continue
                delta = choice["delta"]
                if delta.get("tool_calls"):
                    return [frame + b"\n\n"]
                content = delta.get("content")
                if self._repaired:
                    return []
                if not isinstance(content, str) or not content:
                    continue
                if not self._collecting:
                    if content.lstrip().startswith("{"):
                        self._collecting = True
                    else:
                        return [frame + b"\n\n"]
                self._candidate += content
                delta["content"] = ""
                parsed = self._parse_candidate()
                if parsed is None:
                    return []
                name, arguments = parsed
                call_hash = hashlib.sha256(self._candidate.encode()).hexdigest()[:24]
                delta.pop("content", None)
                delta["tool_calls"] = [
                    {
                        "index": 0,
                        "id": f"call_{call_hash}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                arguments, ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    }
                ]
                self._repaired = True
                return [self._encoded_frame(decoded)]
        return [frame + b"\n\n"]

    def _parse_candidate(self) -> tuple[str, dict[str, object]] | None:
        try:
            value: object = json.loads(self._candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
            return None
        name = value.get("name")
        arguments = value.get("arguments")
        if not isinstance(name, str) or name not in self._tool_names:
            return None
        if not isinstance(arguments, dict):
            return None
        return name, cast(dict[str, object], arguments)

    @staticmethod
    def _encoded_frame(value: dict[str, object]) -> bytes:
        return (
            b"data: "
            + json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
            + b"\n\n"
        )

    def _finish_frame(self) -> bytes:
        template = dict(self._template or {})
        template["choices"] = [
            {
                "index": 0,
                "delta": {"content": ""},
                "logprobs": None,
                "finish_reason": "tool_calls",
            }
        ]
        return self._encoded_frame(template)

    @staticmethod
    def _error_frame(message: str) -> bytes:
        return (
            b"data: "
            + json.dumps(
                {
                    "error": {
                        "message": message,
                        "type": "tinyllm_error",
                        "param": None,
                        "code": "tool_parser_error",
                    }
                },
                separators=(",", ":"),
            ).encode()
            + b"\n\n"
        )


class _SSEToolNormalizer:
    """Normalize native legacy tool deltas and their terminal finish reason."""

    def __init__(self) -> None:
        self._wire = b""
        self._seen: set[int] = set()
        self._finish_emitted = False
        self._template: dict[str, object] | None = None

    def feed(self, chunk: bytes) -> tuple[bytes, ...]:
        self._wire += chunk
        frames = self._wire.split(b"\n\n")
        self._wire = frames.pop()
        return tuple(self._normalize(frame) for frame in frames if frame)

    def flush(self) -> tuple[bytes, ...]:
        if not self._wire:
            return ()
        frame, self._wire = self._wire, b""
        return (self._normalize(frame),)

    def _normalize(self, frame: bytes) -> bytes:
        if frame == b"data: [DONE]":
            if self._seen and not self._finish_emitted:
                template = dict(self._template or {})
                template["choices"] = [
                    {
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "tool_calls",
                    }
                ]
                self._finish_emitted = True
                return (
                    b"data: "
                    + json.dumps(template, ensure_ascii=False, separators=(",", ":")).encode()
                    + b"\n\ndata: [DONE]\n\n"
                )
            return frame + b"\n\n"
        lines = frame.splitlines()
        changed = False
        for line_index, line in enumerate(lines):
            if not line.startswith(b"data: "):
                continue
            try:
                value: object = json.loads(line[6:])
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(value, dict) or not isinstance(value.get("choices"), list):
                continue
            self._template = {key: item for key, item in value.items() if key != "choices"}
            for choice in value["choices"]:
                if not isinstance(choice, dict) or not isinstance(choice.get("delta"), dict):
                    continue
                delta = choice["delta"]
                calls = delta.get("tool_calls")
                if isinstance(calls, list):
                    for raw in calls:
                        if not isinstance(raw, dict):
                            continue
                        index = raw.get("index", 0)
                        if not isinstance(index, int) or index < 0:
                            continue
                        function = raw.get("function")
                        if not isinstance(function, dict):
                            continue
                        if index not in self._seen:
                            self._seen.add(index)
                            raw.setdefault("id", f"call_{uuid.uuid4().hex[:24]}")
                            raw.setdefault("type", "function")
                        else:
                            function.pop("name", None)
                        changed = True
                if self._seen and choice.get("finish_reason") == "stop":
                    choice["finish_reason"] = "tool_calls"
                    changed = True
                if choice.get("finish_reason") == "tool_calls":
                    self._finish_emitted = True
            if changed:
                lines[line_index] = (
                    b"data: "
                    + json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
                )
        return b"\n".join(lines) + b"\n\n"


def create_gateway(
    *,
    config: GatewayConfig,
    resolved_model: ResolvedModel,
    backend: ChatBackend,
    bearer_token: str,
    clock: Callable[[], float] = time.monotonic,
    supervisor: BackendSupervisor | None = None,
    event_log: StructuredEventLog | None = None,
    agent_router: APIRouter | None = None,
    agent_startup: Callable[[], object] | None = None,
    agent_shutdown: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """Build an authenticated, bounded Gateway around one verified deployment."""

    if len(bearer_token) < 32:
        raise ValueError("Gateway Bearer Token must contain at least 32 characters")
    metrics = GatewayMetrics()
    tracer = trace.get_tracer("tinyllm.serving.gateway", __version__)
    limiter = SlidingWindowRateLimiter(config.requests_per_minute, clock=clock)
    concurrency = asyncio.Semaphore(config.max_concurrency)
    security = HTTPBearer(auto_error=False)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if supervisor is not None:
            await supervisor.start()
        if event_log is not None:
            await event_log.write(
                "gateway.started",
                model=resolved_model.model_version,
                backend_managed=supervisor is not None,
            )
        if agent_startup is not None:
            agent_startup()
        try:
            yield
        finally:
            if agent_shutdown is not None:
                await agent_shutdown()
            if event_log is not None:
                await event_log.write("gateway.stopping", model=resolved_model.model_version)
            await backend.close()
            if supervisor is not None:
                await supervisor.stop()

    app = FastAPI(
        title="TinyLLM Model Gateway",
        version=__version__,
        debug=False,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(config.trusted_hosts))
    if agent_router is not None:
        app.include_router(agent_router)

    auth_dependency = Security(security)

    async def authenticate(
        credentials: HTTPAuthorizationCredentials | None = auth_dependency,
    ) -> None:  # noqa: B008
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or not hmac.compare_digest(credentials.credentials, bearer_token)
        ):
            raise HTTPException(
                status_code=401,
                detail="invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not await limiter.allow():
            raise HTTPException(status_code=429, detail="rate limit exceeded")

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started_at = clock()
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if REQUEST_ID.fullmatch(supplied) else f"req_{uuid.uuid4().hex}"
        request.state.request_id = request_id
        raw_path = request.scope.get("raw_path", b"")
        if not isinstance(raw_path, bytes) or not raw_path.startswith(b"/"):
            return _openai_error(
                "request target is invalid",
                code="invalid_request_target",
                status_code=400,
                request_id=request_id,
            )
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                too_large = int(content_length) > config.max_request_bytes
            except ValueError:
                too_large = True
            if too_large:
                return _openai_error(
                    "request body is too large",
                    code="request_too_large",
                    status_code=413,
                    request_id=request_id,
                )
        if request.method in {"POST", "PUT", "PATCH"}:
            try:
                raw_body = await request.body()
            except ClientDisconnect:
                return _openai_error(
                    "request body was interrupted",
                    code="invalid_request_error",
                    status_code=400,
                    request_id=request_id,
                )
            if len(raw_body) > config.max_request_bytes:
                return _openai_error(
                    "request body is too large",
                    code="request_too_large",
                    status_code=413,
                    request_id=request_id,
                )
            if raw_body and not _json_wire_depth_is_safe(raw_body):
                return _openai_error(
                    "request JSON nesting is invalid or excessive",
                    code="invalid_request_error",
                    status_code=400,
                    request_id=request_id,
                )
        traceparent = request.headers.get("traceparent")
        parent = (
            propagate.extract({"traceparent": traceparent})
            if traceparent and TRACEPARENT.fullmatch(traceparent)
            else None
        )
        with tracer.start_as_current_span("tinyllm.gateway.request", context=parent) as span:
            span.set_attribute("http.request.method", request.method)
            span.set_attribute("url.path", str(request.scope.get("path", "")))
            span.set_attribute("tinyllm.request_id", request_id)
            response = await call_next(request)
            span.set_attribute("http.response.status_code", response.status_code)
        if event_log is not None:
            await event_log.write(
                "gateway.request",
                request_id=request_id,
                method=request.method,
                path=str(request.scope.get("path", "")),
                status_code=response.status_code,
                duration_milliseconds=round(max(0.0, clock() - started_at) * 1000, 3),
            )
        response.headers["x-request-id"] = request_id
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["cache-control"] = "no-store"
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        response = _openai_error(
            str(exc.detail),
            code="authentication_error" if exc.status_code == 401 else "request_rejected",
            status_code=exc.status_code,
            request_id=request.state.request_id,
        )
        response.headers.update(exc.headers or {})
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _exc: RequestValidationError) -> JSONResponse:
        return _openai_error(
            "request body failed schema validation",
            code="invalid_request_error",
            status_code=422,
            request_id=request.state.request_id,
        )

    @app.get("/health/live")
    async def live() -> dict[str, object]:
        return HealthResponse(status="ok", ready=True).to_dict()

    @app.get("/health/ready")
    async def ready() -> Response:
        healthy = (supervisor is None or supervisor.ready) and await backend.health()
        metrics.backend_ready.set(1 if healthy else 0)
        if supervisor is not None:
            metrics.backend_restarts.set(supervisor.restart_count)
        body = HealthResponse(
            status="ok" if healthy else "unavailable",
            ready=healthy,
            model=resolved_model.model_version if healthy else None,
        )
        return JSONResponse(status_code=200 if healthy else 503, content=body.to_dict())

    @app.get("/version")
    async def version() -> dict[str, object]:
        return VersionResponse(
            version=__version__,
            model=resolved_model.model_version,
            candidate_model_version=resolved_model.candidate_model_version,
            model_artifact_sha256=resolved_model.model_artifact_sha256,
        ).to_dict()

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    async def models() -> dict[str, object]:
        return ModelList(
            data=(
                ModelCard(
                    id=resolved_model.model_version,
                    created=int(resolved_model.verified_at.timestamp()),
                ),
            )
        ).to_dict()

    @app.get("/metrics", dependencies=[Depends(authenticate)])
    async def prometheus_metrics() -> Response:
        if supervisor is not None:
            metrics.backend_restarts.set(supervisor.restart_count)
            metrics.backend_ready.set(1 if supervisor.ready else 0)
            try:
                telemetry = await asyncio.to_thread(supervisor.inspect_selected_gpu)
                metrics.gpu_memory.set(telemetry["memory_used_mib"] * 1024 * 1024)
            except (BackendSupervisorError, RuntimeError):
                pass
        return Response(
            content=generate_latest(metrics.registry),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/v1/chat/completions", dependencies=[Depends(authenticate)])
    async def chat(request: Request, body: ChatCompletionRequest) -> Response:
        request_id = request.state.request_id
        if body.model not in {resolved_model.model_version, "production"}:
            return _openai_error(
                "requested model is not deployed",
                code="model_not_found",
                status_code=404,
                request_id=request_id,
            )
        payload = body.model_dump(mode="json", exclude_none=True)
        payload["model"] = resolved_model.model_version
        payload["chat_template_kwargs"] = {"enable_thinking": body.mode == "thinking"}
        payload.pop("mode", None)
        upstream_headers = {"x-request-id": request_id}
        traceparent = request.headers.get("traceparent")
        if traceparent and TRACEPARENT.fullmatch(traceparent):
            upstream_headers["traceparent"] = traceparent
        acquired = False
        queued_at = clock()
        try:
            await asyncio.wait_for(concurrency.acquire(), timeout=config.request_timeout_seconds)
            acquired = True
        except TimeoutError:
            return _openai_error(
                "request queue timeout",
                code="queue_timeout",
                status_code=503,
                request_id=request_id,
            )
        metrics.queue.observe(max(0.0, clock() - queued_at))
        metrics.inflight.inc()
        started = clock()
        if body.stream:

            async def iterator() -> AsyncIterator[bytes]:
                status = "2xx"
                first_chunk_at: float | None = None
                completion_tokens: int | None = None
                usage_buffer = b""
                cot_filter = _SSECoTFilter()
                tool_normalizer = _SSEToolNormalizer() if body.tools else None
                auto_tool_repair = (
                    _SSEAutoToolRepair(frozenset(tool.function.name for tool in body.tools or ()))
                    if body.tools and body.tool_choice == "auto"
                    else None
                )
                disconnected_observed = False

                async def disconnected() -> bool:
                    nonlocal disconnected_observed
                    current = await request.is_disconnected()
                    if current and not disconnected_observed:
                        disconnected_observed = True
                        metrics.stream_disconnects.inc()
                    return current

                try:
                    async for chunk in backend.stream(payload, upstream_headers, disconnected):
                        if first_chunk_at is None and chunk:
                            first_chunk_at = clock()
                            metrics.ttft.observe(max(0.0, first_chunk_at - started))
                        usage_buffer += chunk
                        complete_lines = usage_buffer.split(b"\n")
                        usage_buffer = complete_lines.pop()
                        for line in complete_lines:
                            usage = _stream_usage(line)
                            if usage is not None:
                                prompt_tokens, completion_tokens = usage
                                metrics.tokens.labels(kind="prompt").inc(prompt_tokens)
                                metrics.tokens.labels(kind="completion").inc(completion_tokens)
                        for safe_chunk in cot_filter.feed(chunk):
                            repaired = (
                                (safe_chunk,)
                                if auto_tool_repair is None
                                else auto_tool_repair.feed(safe_chunk)
                            )
                            for repaired_chunk in repaired:
                                if tool_normalizer is None:
                                    yield repaired_chunk
                                else:
                                    for normalized in tool_normalizer.feed(repaired_chunk):
                                        yield normalized
                except asyncio.CancelledError:
                    if not disconnected_observed:
                        disconnected_observed = True
                        metrics.stream_disconnects.inc()
                    raise
                except BackendError as exc:
                    status = f"{exc.status_code // 100}xx"
                    metrics.backend_errors.labels(status=str(exc.status_code)).inc()
                    error = {
                        "error": {
                            "message": str(exc),
                            "type": "tinyllm_error",
                            "param": None,
                            "code": "backend_stream_error",
                        }
                    }
                    yield f"data: {json.dumps(error)}\n\n".encode()
                finally:
                    for safe_chunk in cot_filter.flush():
                        repaired = (
                            (safe_chunk,)
                            if auto_tool_repair is None
                            else auto_tool_repair.feed(safe_chunk)
                        )
                        for repaired_chunk in repaired:
                            if tool_normalizer is None:
                                yield repaired_chunk
                            else:
                                for normalized in tool_normalizer.feed(repaired_chunk):
                                    yield normalized
                    if auto_tool_repair is not None:
                        for repaired_chunk in auto_tool_repair.flush():
                            if tool_normalizer is None:
                                yield repaired_chunk
                            else:
                                for normalized in tool_normalizer.feed(repaired_chunk):
                                    yield normalized
                    if tool_normalizer is not None:
                        for normalized in tool_normalizer.flush():
                            yield normalized
                    elapsed = max(0.0, clock() - started)
                    if completion_tokens is not None and completion_tokens > 0:
                        metrics.throughput.observe(completion_tokens / max(elapsed, 1e-9))
                        if first_chunk_at is not None and completion_tokens > 1:
                            metrics.tpot.observe(
                                max(0.0, clock() - first_chunk_at) / (completion_tokens - 1)
                            )
                    metrics.requests.labels(endpoint="chat", status_class=status).inc()
                    metrics.latency.labels(stream="true").observe(elapsed)
                    metrics.inflight.dec()
                    concurrency.release()

            return StreamingResponse(iterator(), media_type="text/event-stream")
        try:
            result = await backend.complete(payload, upstream_headers)
        except BackendError as exc:
            metrics.backend_errors.labels(status=str(exc.status_code)).inc()
            metrics.requests.labels(
                endpoint="chat", status_class=f"{exc.status_code // 100}xx"
            ).inc()
            return _openai_error(
                str(exc),
                code="backend_error",
                status_code=exc.status_code,
                request_id=request_id,
            )
        finally:
            if acquired:
                metrics.latency.labels(stream="false").observe(max(0.0, clock() - started))
                metrics.inflight.dec()
                concurrency.release()
        result = _sanitize_completion(result)
        if body.tools and body.tool_choice == "auto":
            result = _repair_auto_tool_completion(
                result,
                frozenset(tool.function.name for tool in body.tools),
            )
        if body.tools:
            result = _normalize_tool_completion(result)
        metrics.requests.labels(endpoint="chat", status_class="2xx").inc()
        usage = result.get("usage")
        if isinstance(usage, dict):
            for kind, field in (("prompt", "prompt_tokens"), ("completion", "completion_tokens")):
                value = usage.get(field)
                if isinstance(value, int) and value >= 0:
                    metrics.tokens.labels(kind=kind).inc(value)
        return JSONResponse(content=result, headers={"x-request-id": request_id})

    return app


def _stream_usage(chunk: bytes) -> tuple[int, int] | None:
    """Extract content-free usage from one complete SSE chunk when present."""

    for line in chunk.splitlines():
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        try:
            decoded = json.loads(line[6:])
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(decoded, dict) or not isinstance(decoded.get("usage"), dict):
            continue
        usage = decoded["usage"]
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if (
            isinstance(prompt, int)
            and prompt >= 0
            and isinstance(completion, int)
            and completion >= 0
        ):
            return prompt, completion
    return None
