from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tinyllm.deployment import ResolvedModel
from tinyllm.evaluation import M6ModelIdentity
from tinyllm.serving.config import GatewayConfig
from tinyllm.serving.supervisor import BackendSupervisor, BackendSupervisorError
from tinyllm.training.smoke_preflight import GpuPreflight


def _gpu(
    *, memory: int = 0, utilization: int = 0, temperature: int = 32
) -> tuple[GpuPreflight, ...]:
    return (
        {
            "index": 7,
            "name": "NVIDIA GeForce RTX 3090",
            "memory_used_mib": memory,
            "utilization_percent": utilization,
            "temperature_c": temperature,
            "driver_version": "535.261.03",
        },
    )


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


def test_supervisor_command_is_fixed_and_path_safe() -> None:
    config = GatewayConfig(config_id="m7-gateway-unit", gpu_index=4)
    supervisor = BackendSupervisor(
        config=config,
        resolved_model=_resolved(),
        python_executable=Path("/venv/bin/python"),
        gpu_inspector=lambda _indices: _gpu(),
        internal_token="internal-unit-token-that-is-longer-than-32-characters",
    )
    command = supervisor.command()
    assert command[:4] == (
        "/venv/bin/python",
        "-m",
        "tinyllm.serving.vllm_entrypoint",
        "--model",
    )
    assert "--disable-fastapi-docs" in command
    assert "--enable-auto-tool-choice" in command
    assert "tinyllm.serving.vllm_guard.VLLMBackendGuard" in command
    assert "hermes" in command
    assert ";" not in " ".join(command)

    lazy_config = GatewayConfig(config_id="m7-gateway-unit", gpu_index=4, enforce_eager=False)
    lazy_supervisor = BackendSupervisor(
        config=lazy_config,
        resolved_model=_resolved(),
        gpu_inspector=lambda _indices: _gpu(),
    )
    assert "--enforce-eager" not in lazy_supervisor.command()


def test_supervisor_rejects_short_internal_token() -> None:
    with pytest.raises(BackendSupervisorError, match="at least 32"):
        BackendSupervisor(
            config=GatewayConfig(config_id="m7-gateway-unit"),
            resolved_model=_resolved(),
            internal_token="too-short",
        )


def test_supervisor_gpu_inspection_requires_exact_physical_device() -> None:
    supervisor = BackendSupervisor(
        config=GatewayConfig(config_id="m7-gateway-unit", gpu_index=7),
        resolved_model=_resolved(),
        gpu_inspector=lambda _indices: (),
    )
    with pytest.raises(BackendSupervisorError, match="telemetry is invalid"):
        supervisor.inspect_selected_gpu()


def test_supervisor_start_timeout_stops_child(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        pid = 12345
        returncode: int | None = None
        terminated = False
        finished = asyncio.Event()

        async def wait(self) -> int:
            await self.finished.wait()
            return self.returncode or 0

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = 0
            self.finished.set()

        def kill(self) -> None:
            self.returncode = -9
            self.finished.set()

    process = Process()

    async def create(*_args: object, **_kwargs: object) -> Any:
        return process

    async def unavailable(_self: BackendSupervisor) -> None:
        raise BackendSupervisorError("offline")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(os, "killpg", lambda _pid, _signal: process.terminate())
    monkeypatch.setattr(BackendSupervisor, "_wait_until_healthy", unavailable)
    config = GatewayConfig(config_id="m7-gateway-unit", backend_startup_timeout_seconds=1)
    supervisor = BackendSupervisor(
        config=config, resolved_model=_resolved(), gpu_inspector=lambda _indices: _gpu()
    )

    with pytest.raises(BackendSupervisorError, match="did not become ready"):
        asyncio.run(supervisor.start())
    assert process.terminated


def test_supervisor_preflight_rejects_busy_gpu_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned = False

    async def create(*_args: object, **_kwargs: object) -> Any:
        nonlocal spawned
        spawned = True
        raise AssertionError("busy GPU must be rejected before process creation")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    config = GatewayConfig(config_id="m7-gateway-unit", gpu_index=7)
    supervisor = BackendSupervisor(
        config=config,
        resolved_model=_resolved(),
        gpu_inspector=lambda _indices: _gpu(memory=2048),
    )
    with pytest.raises(BackendSupervisorError, match="rejected"):
        asyncio.run(supervisor.start())
    assert not spawned


@pytest.mark.parametrize(
    "inspector,error",
    [
        (lambda _indices: (), "invalid data"),
        (lambda _indices: _gpu(utilization=100), "rejected"),
        (lambda _indices: _gpu(temperature=100), "rejected"),
    ],
)
def test_supervisor_preflight_rejects_invalid_or_unsafe_telemetry(
    inspector: Any, error: str
) -> None:
    supervisor = BackendSupervisor(
        config=GatewayConfig(config_id="m7-gateway-unit", gpu_index=7),
        resolved_model=_resolved(),
        gpu_inspector=inspector,
    )
    with pytest.raises(BackendSupervisorError, match=error):
        supervisor._run_preflight()


def test_supervisor_preflight_wraps_inspector_failure() -> None:
    def broken(_indices: tuple[int, ...]) -> tuple[GpuPreflight, ...]:
        raise RuntimeError("nvidia-smi failed")

    supervisor = BackendSupervisor(
        config=GatewayConfig(config_id="m7-gateway-unit"),
        resolved_model=_resolved(),
        gpu_inspector=broken,
    )
    with pytest.raises(BackendSupervisorError, match="preflight failed"):
        supervisor._run_preflight()


def test_supervisor_prepares_private_runtime_directories(tmp_path: Path) -> None:
    config = GatewayConfig(config_id="m7-gateway-unit", gpu_index=7)
    supervisor = BackendSupervisor(
        config=config,
        resolved_model=_resolved(),
        artifact_root=tmp_path,
        gpu_inspector=lambda _indices: _gpu(),
    )
    supervisor._prepare_private_runtime_directories()
    for relative in ("cache/vllm-m7", "cache/huggingface-m7", "deployments/vllm-config"):
        path = tmp_path / relative
        assert path.stat().st_mode & 0o777 == 0o700


def test_supervisor_runtime_directory_checks_fail_closed(tmp_path: Path) -> None:
    relative = BackendSupervisor(
        config=GatewayConfig(config_id="m7-gateway-unit"),
        resolved_model=_resolved(),
        artifact_root=Path("relative"),
    )
    with pytest.raises(BackendSupervisorError, match="root is unsafe"):
        relative._prepare_private_runtime_directories()

    unsafe_path = tmp_path / "cache" / "vllm-m7"
    unsafe_path.parent.mkdir()
    unsafe_path.write_text("not-a-directory", encoding="utf-8")
    unsafe = BackendSupervisor(
        config=GatewayConfig(config_id="m7-gateway-unit"),
        resolved_model=_resolved(),
        artifact_root=tmp_path,
    )
    with pytest.raises(BackendSupervisorError, match="directory is unsafe"):
        unsafe._prepare_private_runtime_directories()
