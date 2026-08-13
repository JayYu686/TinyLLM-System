"""M7 request-level Direct/Gateway inference benchmark runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from tinyllm.benchmark.inference_schema import (
    InferenceBenchmarkConfig,
    InferenceBenchmarkSummary,
    InferenceRequestResult,
)
from tinyllm.deployment import ResolvedModel
from tinyllm.schemas import canonical_config_hash


class InferenceBenchmarkError(RuntimeError):
    """Raised when a formal benchmark cannot preserve its evidence contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o600)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _require_loopback_url(value: str) -> str:
    normalized = value.rstrip("/")
    if not normalized.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise InferenceBenchmarkError("benchmark endpoints must be loopback HTTP URLs")
    return normalized


def build_exact_chat_prompt(tokenizer_dir: Path, target_tokens: int) -> str:
    """Construct deterministic content whose rendered Qwen chat prompt has an exact length."""

    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise InferenceBenchmarkError(
            "Transformers is required to build exact benchmark prompts"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    content = ""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(rendered) > target_tokens:
        raise InferenceBenchmarkError("target input length is shorter than the chat template")
    filler_id = tokenizer.encode(" x", add_special_tokens=False)
    if len(filler_id) != 1:
        raise InferenceBenchmarkError("reviewed one-token benchmark filler is unavailable")
    content = " x" * (target_tokens - len(rendered))
    actual = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(actual) != target_tokens:
        raise InferenceBenchmarkError("cannot construct an exact-length Qwen chat prompt")
    return content


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("data: "):
        return None
    payload = line[6:]
    if payload == "[DONE]":
        return None
    decoded: Any = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("SSE data is not a JSON object")
    return decoded


async def _one_request(
    client: httpx.AsyncClient,
    *,
    request_number: int,
    backend: Literal["direct", "gateway"],
    base_url: str,
    bearer_token: str | None,
    model_version: str,
    content: str,
    concurrency: Literal[1, 4, 8, 16, 32],
    input_tokens: Literal[128, 512, 1024],
    output_tokens: Literal[128, 256],
    repeat: Literal[1, 2, 3],
    timeout: float,
) -> InferenceRequestResult:
    request_id = f"m7req-{request_number:06d}"
    headers = {"x-request-id": request_id}
    if bearer_token is not None:
        headers["Authorization"] = f"Bearer {bearer_token}"
    payload = {
        "model": model_version,
        "messages": [{"role": "user", "content": content}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": output_tokens,
        "temperature": 0,
    }
    if backend == "direct":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        payload["mode"] = "nonthinking"
    started = time.perf_counter()
    first_token_at: float | None = None
    prompt_count: int | None = None
    completion_count: int | None = None
    status_code = 0
    error_code: str | None = None
    saw_done = False
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        ) as response:
            status_code = response.status_code
            if status_code >= 400:
                await response.aread()
                error_code = f"http_{status_code}"
            else:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        saw_done = True
                        continue
                    decoded = _parse_sse_line(line)
                    if decoded is None:
                        continue
                    choices = decoded.get("choices")
                    if choices and first_token_at is None:
                        delta = choices[0].get("delta", {})
                        if delta.get("content") or delta.get("tool_calls"):
                            first_token_at = time.perf_counter()
                    usage = decoded.get("usage")
                    if isinstance(usage, dict):
                        prompt_count = usage.get("prompt_tokens")
                        completion_count = usage.get("completion_tokens")
    except (httpx.HTTPError, OSError, ValueError) as exc:
        error_code = type(exc).__name__
    ended = time.perf_counter()
    if (
        200 <= status_code < 300
        and error_code is None
        and (not saw_done or prompt_count is None or completion_count is None)
    ):
        error_code = "missing_stream_completion"
    success = 200 <= status_code < 300 and error_code is None
    ttft = (first_token_at - started) * 1000 if first_token_at is not None else None
    tpot = None
    if ttft is not None and completion_count is not None and completion_count > 1:
        tpot = max(0.0, ((ended - started) * 1000 - ttft) / (completion_count - 1))
    return InferenceRequestResult(
        request_id=request_id,
        backend=backend,
        concurrency=concurrency,
        input_tokens_target=input_tokens,
        output_tokens_target=output_tokens,
        repeat=repeat,
        success=success,
        status_code=status_code,
        prompt_tokens=prompt_count,
        completion_tokens=completion_count,
        ttft_milliseconds=ttft,
        tpot_milliseconds=tpot,
        latency_milliseconds=(ended - started) * 1000,
        error_code=None if success else error_code or "missing_stream_completion",
    )


async def _batch(
    client: httpx.AsyncClient,
    *,
    count: int,
    start_number: int,
    concurrency: Literal[1, 4, 8, 16, 32],
    kwargs: dict[str, Any],
) -> tuple[list[InferenceRequestResult], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(index: int) -> InferenceRequestResult:
        async with semaphore:
            return await _one_request(client, request_number=start_number + index, **kwargs)

    started = time.perf_counter()
    results = await asyncio.gather(*(bounded(index) for index in range(count)))
    return list(results), time.perf_counter() - started


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise InferenceBenchmarkError("cannot aggregate an empty latency cell")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile)))
    return ordered[index]


async def run_inference_benchmark(
    *,
    config: InferenceBenchmarkConfig,
    resolved_model: ResolvedModel,
    direct_url: str,
    direct_bearer_token: str,
    gateway_url: str,
    gateway_bearer_token: str,
    output_dir: Path,
    environment_sha256: str,
    hardware_sha256: str,
    gateway_config_sha256: str,
) -> InferenceBenchmarkSummary:
    """Execute the full frozen M7 matrix and atomically publish raw evidence."""

    if not output_dir.is_absolute() or output_dir.exists():
        raise InferenceBenchmarkError("benchmark output directory must be new and absolute")
    direct_url = _require_loopback_url(direct_url)
    gateway_url = _require_loopback_url(gateway_url)
    prompts = {
        target: build_exact_chat_prompt(resolved_model.tokenizer_dir, target)
        for target in config.input_tokens
    }
    all_results: list[InferenceRequestResult] = []
    throughputs: dict[tuple[str, int, int, int, int], float] = {}
    request_number = 1
    async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as client:
        for input_tokens in config.input_tokens:
            for output_tokens in config.output_tokens:
                for concurrency in config.concurrency:
                    for repeat in range(1, config.repeats + 1):
                        endpoints = (
                            (
                                ("direct", direct_url, direct_bearer_token),
                                ("gateway", gateway_url, gateway_bearer_token),
                            )
                            if repeat % 2
                            else (
                                ("gateway", gateway_url, gateway_bearer_token),
                                ("direct", direct_url, direct_bearer_token),
                            )
                        )
                        for backend, base_url, token in endpoints:
                            shared: dict[str, Any] = {
                                "backend": backend,
                                "base_url": base_url,
                                "bearer_token": token,
                                "model_version": resolved_model.model_version,
                                "content": prompts[input_tokens],
                                "concurrency": concurrency,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "repeat": repeat,
                                "timeout": config.request_timeout_seconds,
                            }
                            _, _ = await _batch(
                                client,
                                count=config.warmup_requests,
                                start_number=request_number,
                                concurrency=concurrency,
                                kwargs=shared,
                            )
                            request_number += config.warmup_requests
                            results, duration = await _batch(
                                client,
                                count=config.measured_requests,
                                start_number=request_number,
                                concurrency=concurrency,
                                kwargs=shared,
                            )
                            request_number += config.measured_requests
                            all_results.extend(results)
                            completed = sum(item.completion_tokens or 0 for item in results)
                            key = (backend, input_tokens, output_tokens, concurrency, repeat)
                            throughputs[key] = completed / duration if duration > 0 else 0.0
                            print(
                                json.dumps(
                                    {
                                        "event": "m7.inference.cell_completed",
                                        "backend": backend,
                                        "input_tokens": input_tokens,
                                        "output_tokens": output_tokens,
                                        "concurrency": concurrency,
                                        "repeat": repeat,
                                        "successful_requests": sum(
                                            item.success for item in results
                                        ),
                                        "measured_requests": len(results),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
    lines = b"".join(
        (json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode()
        for item in all_results
    )
    output_dir.mkdir(parents=True)
    request_path = output_dir / "requests.jsonl"
    _atomic_bytes(request_path, lines)
    ratios: list[int] = []
    overheads: list[int] = []
    for input_tokens in config.input_tokens:
        for output_tokens in config.output_tokens:
            for concurrency in config.concurrency:
                for repeat in range(1, config.repeats + 1):
                    direct_key = ("direct", input_tokens, output_tokens, concurrency, repeat)
                    gateway_key = ("gateway", input_tokens, output_tokens, concurrency, repeat)
                    direct_rate = throughputs[direct_key]
                    gateway_rate = throughputs[gateway_key]
                    ratios.append(round(gateway_rate * 10_000 / direct_rate) if direct_rate else 0)
                    direct_p95 = _percentile(
                        [
                            item.latency_milliseconds
                            for item in all_results
                            if (
                                item.backend,
                                item.input_tokens_target,
                                item.output_tokens_target,
                                item.concurrency,
                                item.repeat,
                            )
                            == direct_key
                        ],
                        0.95,
                    )
                    gateway_p95 = _percentile(
                        [
                            item.latency_milliseconds
                            for item in all_results
                            if (
                                item.backend,
                                item.input_tokens_target,
                                item.output_tokens_target,
                                item.concurrency,
                                item.repeat,
                            )
                            == gateway_key
                        ],
                        0.95,
                    )
                    overheads.append(round((gateway_p95 - direct_p95) * 10_000 / direct_p95))
    successful = sum(item.success for item in all_results)
    success_rate_basis_points = successful * 10_000 // len(all_results)
    oom_events = sum(item.error_code == "backend_oom" for item in all_results)
    hung_processes = sum(
        item.error_code in {"ReadTimeout", "TimeoutException"} for item in all_results
    )
    unexplained_5xx = sum(
        1
        for item in all_results
        if not item.success and item.status_code >= 500 and item.error_code != "backend_oom"
    )
    summary = InferenceBenchmarkSummary(
        benchmark_id=config.benchmark_id,
        completed_at=datetime.now(UTC),
        model_version=resolved_model.model_version,
        model_artifact_sha256=resolved_model.model_artifact_sha256,
        tokenizer_artifact_sha256=resolved_model.tokenizer_artifact_sha256,
        config_sha256=canonical_config_hash(config),
        gateway_config_sha256=gateway_config_sha256,
        environment_sha256=environment_sha256,
        hardware_sha256=hardware_sha256,
        request_results_sha256=_sha256_file(request_path),
        total_requests=len(all_results),
        successful_requests=successful,
        success_rate_basis_points=success_rate_basis_points,
        gateway_throughput_ratio_basis_points=round(statistics.median(ratios)),
        p95_latency_overhead_median_basis_points=round(statistics.median(overheads)),
        oom_events=oom_events,
        hung_processes=hung_processes,
        unexplained_5xx=unexplained_5xx,
        status=(
            "succeeded"
            if success_rate_basis_points >= 9950
            and unexplained_5xx == 0
            and oom_events == 0
            and hung_processes == 0
            else "failed"
        ),
    )
    _atomic_bytes(
        output_dir / "summary.json",
        (
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )
    return summary
