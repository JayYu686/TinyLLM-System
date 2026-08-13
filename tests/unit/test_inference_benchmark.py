from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from tinyllm.benchmark.inference import (
    InferenceBenchmarkError,
    _one_request,
    _parse_sse_line,
    _percentile,
    build_exact_chat_prompt,
    run_inference_benchmark,
)
from tinyllm.benchmark.inference_schema import (
    InferenceBenchmarkConfig,
    InferenceBenchmarkSummary,
    InferenceRequestResult,
    load_inference_benchmark_config,
)
from tinyllm.deployment import ResolvedModel
from tinyllm.evaluation import M6ModelIdentity


def test_formal_inference_matrix_is_frozen() -> None:
    config = load_inference_benchmark_config(Path("configs/benchmark/m7_inference.yaml"))

    assert config.concurrency == (1, 4, 8, 16, 32)
    assert config.input_tokens == (128, 512, 1024)
    assert config.output_tokens == (128, 256)
    assert config.warmup_requests == 20
    assert config.measured_requests == 100
    assert config.repeats == 3

    payload = config.to_dict()
    payload["concurrency"] = [1, 4]
    with pytest.raises(ValidationError, match="concurrency matrix"):
        InferenceBenchmarkConfig.model_validate(payload)


def test_summary_recomputes_success_rate() -> None:
    with pytest.raises(ValidationError, match="success rate"):
        InferenceBenchmarkSummary(
            benchmark_id="m7-inference-unit-v1",
            completed_at=datetime(2026, 8, 13, tzinfo=UTC),
            model_version="qwen3-0-6b-m6-aaaaaaaa",
            model_artifact_sha256="a" * 64,
            tokenizer_artifact_sha256="b" * 64,
            config_sha256="c" * 64,
            gateway_config_sha256="0" * 64,
            environment_sha256="d" * 64,
            hardware_sha256="e" * 64,
            request_results_sha256="f" * 64,
            total_requests=100,
            successful_requests=99,
            success_rate_basis_points=10_000,
            gateway_throughput_ratio_basis_points=9500,
            p95_latency_overhead_median_basis_points=500,
            oom_events=0,
            hung_processes=0,
            unexplained_5xx=0,
            status="failed",
        )


def test_request_runner_extracts_usage_and_ttft() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        payload = json.loads(request.content)
        assert payload["stream_options"] == {"include_usage": True}
        body = (
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":128,'
            '"completion_tokens":2,"total_tokens":130}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async def run() -> InferenceRequestResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _one_request(
                client,
                request_number=1,
                backend="gateway",
                base_url="http://127.0.0.1:8000",
                bearer_token="secret",
                model_version="qwen3-0-6b-m6-aaaaaaaa",
                content="x",
                concurrency=1,
                input_tokens=128,
                output_tokens=128,
                repeat=1,
                timeout=1,
            )

    result = asyncio.run(run())

    assert result.success is True
    assert result.prompt_tokens == 128
    assert result.completion_tokens == 2
    assert result.ttft_milliseconds is not None


def test_exact_prompt_builder_and_sse_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages: list[dict[str, str]], **_kwargs: object) -> list[int]:
            return [0] * (10 + messages[0]["content"].count(" x"))

        @staticmethod
        def encode(_value: str, **_kwargs: object) -> list[int]:
            return [1]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> Tokenizer:
            return Tokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))
    assert build_exact_chat_prompt(Path("/tokenizer"), 12) == " x x"
    assert _parse_sse_line(": heartbeat") is None
    assert _parse_sse_line("data: [DONE]") is None
    with pytest.raises(ValueError, match="JSON object"):
        _parse_sse_line("data: []")


def test_exact_prompt_builder_rejects_impossible_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        filler = [1]
        drift = False

        @classmethod
        def apply_chat_template(
            cls, messages: list[dict[str, str]], **_kwargs: object
        ) -> list[int]:
            content = messages[0]["content"]
            extra = 1 if cls.drift and content else 0
            return [0] * (10 + content.count(" x") + extra)

        @classmethod
        def encode(cls, _value: str, **_kwargs: object) -> list[int]:
            return cls.filler

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> Tokenizer:
            return Tokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer))
    with pytest.raises(InferenceBenchmarkError, match="shorter"):
        build_exact_chat_prompt(Path("/tokenizer"), 9)

    Tokenizer.filler = [1, 2]
    with pytest.raises(InferenceBenchmarkError, match="filler"):
        build_exact_chat_prompt(Path("/tokenizer"), 12)

    Tokenizer.filler = [1]
    Tokenizer.drift = True
    with pytest.raises(InferenceBenchmarkError, match="exact-length"):
        build_exact_chat_prompt(Path("/tokenizer"), 12)


def test_request_runner_records_http_and_stream_failures() -> None:
    async def run(handler: Any) -> InferenceRequestResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await _one_request(
                client,
                request_number=2,
                backend="direct",
                base_url="http://127.0.0.1:8001",
                bearer_token=None,
                model_version="qwen3-0-6b-m6-aaaaaaaa",
                content="x",
                concurrency=1,
                input_tokens=128,
                output_tokens=128,
                repeat=1,
                timeout=1,
            )

    http_failure = asyncio.run(run(lambda _request: httpx.Response(503)))
    assert not http_failure.success
    assert http_failure.error_code == "http_503"

    missing_usage = asyncio.run(
        run(
            lambda _request: httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"tool_calls":[{}]}}]}\n\ndata: [DONE]\n\n',
            )
        )
    )
    assert not missing_usage.success
    assert missing_usage.status_code == 200
    assert missing_usage.error_code == "missing_stream_completion"

    invalid_sse = asyncio.run(run(lambda _request: httpx.Response(200, text="data: []\n\n")))
    assert not invalid_sse.success
    assert invalid_sse.error_code == "ValueError"


def test_percentile_rejects_empty_cells() -> None:
    with pytest.raises(InferenceBenchmarkError, match="empty"):
        _percentile([], 0.95)


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


def test_small_mock_matrix_publishes_request_level_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host in {"127.0.0.1", "localhost"}
        body = (
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":128,'
            '"completion_tokens":2,"total_tokens":130}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    real_client = httpx.AsyncClient

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr("tinyllm.benchmark.inference.httpx.AsyncClient", client_factory)
    monkeypatch.setattr(
        "tinyllm.benchmark.inference.build_exact_chat_prompt", lambda _path, _target: "x"
    )
    config = InferenceBenchmarkConfig.model_construct(
        schema_version="1.0",
        benchmark_id="m7-inference-unit-v1",
        concurrency=(1,),
        input_tokens=(128,),
        output_tokens=(128,),
        warmup_requests=1,
        measured_requests=2,
        repeats=1,
        request_timeout_seconds=1.0,
    )
    output = tmp_path.resolve() / "benchmark"
    summary = asyncio.run(
        run_inference_benchmark(
            config=config,
            resolved_model=_resolved(),
            direct_url="http://127.0.0.1:8001",
            direct_bearer_token="internal-unit-token-that-is-longer-than-32-characters",
            gateway_url="http://127.0.0.1:8000",
            gateway_bearer_token="secret",
            output_dir=output,
            environment_sha256="1" * 64,
            hardware_sha256="2" * 64,
            gateway_config_sha256="3" * 64,
        )
    )

    assert summary.status == "succeeded"
    assert summary.total_requests == 4
    assert summary.successful_requests == 4
    lines = (output / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert (
        summary.request_results_sha256
        == __import__("hashlib").sha256((output / "requests.jsonl").read_bytes()).hexdigest()
    )


def test_formal_runner_rejects_remote_or_existing_output(tmp_path: Path) -> None:
    config = InferenceBenchmarkConfig.model_construct(
        schema_version="1.0",
        benchmark_id="m7-inference-unit-v1",
        concurrency=(1,),
        input_tokens=(128,),
        output_tokens=(128,),
        warmup_requests=1,
        measured_requests=1,
        repeats=1,
        request_timeout_seconds=1.0,
    )
    existing = tmp_path.resolve()
    with pytest.raises(InferenceBenchmarkError, match="new and absolute"):
        asyncio.run(
            run_inference_benchmark(
                config=config,
                resolved_model=_resolved(),
                direct_url="http://127.0.0.1:8001",
                direct_bearer_token="internal-unit-token-that-is-longer-than-32-characters",
                gateway_url="http://127.0.0.1:8000",
                gateway_bearer_token="secret",
                output_dir=existing,
                environment_sha256="1" * 64,
                hardware_sha256="2" * 64,
                gateway_config_sha256="3" * 64,
            )
        )

    with pytest.raises(InferenceBenchmarkError, match="loopback"):
        asyncio.run(
            run_inference_benchmark(
                config=config,
                resolved_model=_resolved(),
                direct_url="https://example.com",
                direct_bearer_token="internal-unit-token-that-is-longer-than-32-characters",
                gateway_url="http://127.0.0.1:8000",
                gateway_bearer_token="secret",
                output_dir=tmp_path.resolve() / "new",
                environment_sha256="1" * 64,
                hardware_sha256="2" * 64,
                gateway_config_sha256="3" * 64,
            )
        )
