"""Bounded local vLLM process supervision for the M7 Gateway."""

from __future__ import annotations

import asyncio
import os
import secrets
import signal
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from tinyllm.deployment import ResolvedEvaluationSubject, ServingModel
from tinyllm.serving.config import GatewayConfig
from tinyllm.training.smoke_preflight import GpuPreflight, inspect_gpus


class BackendSupervisorError(RuntimeError):
    """Raised when the managed backend cannot become ready."""


class BackendSupervisor:
    """Start, monitor, restart, and stop one fixed loopback vLLM backend."""

    def __init__(
        self,
        *,
        config: GatewayConfig,
        resolved_model: ServingModel,
        python_executable: Path | None = None,
        gpu_inspector: Callable[[tuple[int, ...]], tuple[GpuPreflight, ...]] = inspect_gpus,
        artifact_root: Path | None = None,
        internal_token: str | None = None,
    ) -> None:
        self._config = config
        self._model = resolved_model
        self._python = python_executable or Path(sys.executable)
        self._gpu_inspector = gpu_inspector
        self._artifact_root = artifact_root
        self._internal_token = (
            internal_token
            or os.environ.get("TINYLLM_VLLM_INTERNAL_TOKEN")
            or secrets.token_urlsafe(48)
        )
        if len(self._internal_token) < 32:
            raise BackendSupervisorError(
                "managed backend internal token must contain at least 32 characters"
            )
        self._process: asyncio.subprocess.Process | None = None
        self._monitor: asyncio.Task[None] | None = None
        self._stopping = False
        self._ready = asyncio.Event()
        self._restarts = 0
        self.last_exit_code: int | None = None
        self.preflight: GpuPreflight | None = None

    @property
    def ready(self) -> bool:
        return (
            self._ready.is_set() and self._process is not None and self._process.returncode is None
        )

    @property
    def restart_count(self) -> int:
        return self._restarts

    @property
    def internal_token(self) -> str:
        return self._internal_token

    def inspect_selected_gpu(self) -> GpuPreflight:
        """Return current physical-GPU telemetry for private service metrics."""

        rows = self._gpu_inspector((self._config.gpu_index,))
        if len(rows) != 1 or rows[0]["index"] != self._config.gpu_index:
            raise BackendSupervisorError("managed backend GPU telemetry is invalid")
        return rows[0]

    def command(self) -> tuple[str, ...]:
        """Return the fixed path-safe vLLM invocation without a shell."""

        port = self._config.backend_base_url.rsplit(":", 1)[-1]
        adapter_dir = (
            self._model.adapter_dir if isinstance(self._model, ResolvedEvaluationSubject) else None
        )
        served_model_name = (
            f"{self._model.model_version}-base" if adapter_dir else self._model.model_version
        )
        command = [
            str(self._python),
            "-m",
            "tinyllm.serving.vllm_entrypoint",
            "--model",
            str(self._model.model_dir),
            "--tokenizer",
            str(self._model.tokenizer_dir),
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--served-model-name",
            served_model_name,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(self._config.model_max_length),
            "--gpu-memory-utilization",
            str(self._config.gpu_memory_utilization),
            "--generation-config",
            "vllm",
            "--disable-log-requests",
            "--disable-uvicorn-access-log",
            "--disable-fastapi-docs",
            "--middleware",
            "tinyllm.serving.vllm_guard.VLLMBackendGuard",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            self._config.tool_call_parser,
        ]
        if adapter_dir is not None:
            command.extend(
                [
                    "--enable-lora",
                    "--lora-modules",
                    f"{self._model.model_version}={adapter_dir}",
                    "--max-lora-rank",
                    "16",
                ]
            )
        if self._config.enforce_eager:
            command.append("--enforce-eager")
        return tuple(command)

    async def start(self) -> None:
        """Start and health-check the backend before the Gateway becomes ready."""

        self._run_preflight()
        self._prepare_private_runtime_directories()
        await self._spawn()
        self._monitor = asyncio.create_task(self._monitor_loop())
        try:
            await asyncio.wait_for(
                self._wait_until_healthy(),
                timeout=self._config.backend_startup_timeout_seconds,
            )
        except (TimeoutError, BackendSupervisorError):
            await self.stop()
            raise BackendSupervisorError("managed model backend did not become ready") from None

    def _run_preflight(self) -> None:
        """Fail closed when the selected physical GPU is busy or too hot."""

        try:
            rows = self._gpu_inspector((self._config.gpu_index,))
        except RuntimeError as exc:
            raise BackendSupervisorError("managed backend GPU preflight failed") from exc
        if len(rows) != 1 or rows[0]["index"] != self._config.gpu_index:
            raise BackendSupervisorError("managed backend GPU preflight returned invalid data")
        row = rows[0]
        if (
            row["memory_used_mib"] > self._config.preflight_max_memory_used_mib
            or row["utilization_percent"] > self._config.preflight_max_utilization_percent
            or row["temperature_c"] > self._config.preflight_max_temperature_c
        ):
            raise BackendSupervisorError("managed backend GPU preflight rejected the selected GPU")
        self.preflight = row

    def _prepare_private_runtime_directories(self) -> None:
        """Create same-user-only caches when a private Artifact Store is supplied."""

        if self._artifact_root is None:
            return
        if not self._artifact_root.is_absolute() or self._artifact_root.is_symlink():
            raise BackendSupervisorError("managed backend Artifact Store root is unsafe")
        for path in (
            self._artifact_root / "cache" / "vllm-m7",
            self._artifact_root / "cache" / "huggingface-m7",
            self._artifact_root / "deployments" / "vllm-config",
        ):
            if path.exists() and (not path.is_dir() or path.is_symlink()):
                raise BackendSupervisorError("managed backend runtime directory is unsafe")
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.chmod(0o700)

    async def stop(self) -> None:
        """Stop monitoring and terminate the child process with a bounded grace period."""

        self._stopping = True
        self._ready.clear()
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
            self._monitor = None
        process = self._process
        if process is not None:
            await self._terminate_process_group(process)
        self._process = None

    async def _terminate_process_group(self, process: asyncio.subprocess.Process) -> None:
        """Terminate the whole vLLM session, including detached EngineCore children."""

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            if process.returncode is None:
                process.terminate()
        if process.returncode is None:
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=self._config.backend_cleanup_timeout_seconds
                )
            except TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        await self._wait_until_gpu_released()

    async def _wait_until_gpu_released(self) -> None:
        started = asyncio.get_running_loop().time()
        while (
            asyncio.get_running_loop().time() - started
            <= self._config.backend_cleanup_timeout_seconds
        ):
            try:
                telemetry = await asyncio.to_thread(self.inspect_selected_gpu)
            except RuntimeError:
                telemetry = None
            if (
                telemetry is not None
                and telemetry["memory_used_mib"] <= self._config.preflight_max_memory_used_mib
            ):
                return
            await asyncio.sleep(0.25)
        raise BackendSupervisorError("managed backend process group did not release the GPU")

    async def _spawn(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(self._config.gpu_index),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "VLLM_NO_USAGE_STATS": "1",
                "TINYLLM_VLLM_INTERNAL_TOKEN": self._internal_token,
            }
        )
        if self._artifact_root is not None:
            environment.update(
                {
                    "HF_HOME": str(self._artifact_root / "cache" / "huggingface-m7"),
                    "VLLM_CACHE_ROOT": str(self._artifact_root / "cache" / "vllm-m7"),
                    "VLLM_CONFIG_ROOT": str(self._artifact_root / "deployments" / "vllm-config"),
                }
            )
        self._ready.clear()
        self._process = await asyncio.create_subprocess_exec(
            *self.command(),
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            start_new_session=True,
        )

    async def _wait_until_healthy(self) -> None:
        import httpx

        async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as client:
            while self._process is not None and self._process.returncode is None:
                try:
                    response = await client.get(
                        f"{self._config.backend_base_url}/health",
                        headers={"Authorization": f"Bearer {self._internal_token}"},
                        timeout=self._config.backend_health_timeout_seconds,
                    )
                    if response.status_code == 200:
                        self._ready.set()
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)
        raise BackendSupervisorError("managed model backend exited during startup")

    async def _monitor_loop(self) -> None:
        while not self._stopping and self._process is not None:
            exited_process = self._process
            self.last_exit_code = await exited_process.wait()
            self._ready.clear()
            if self._stopping or self._restarts >= self._config.backend_restart_limit:
                return
            self._restarts += 1
            try:
                await self._terminate_process_group(exited_process)
            except BackendSupervisorError:
                return
            await self._spawn()
            try:
                await asyncio.wait_for(
                    self._wait_until_healthy(),
                    timeout=self._config.backend_startup_timeout_seconds,
                )
            except (TimeoutError, BackendSupervisorError):
                continue
