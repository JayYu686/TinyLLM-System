"""M7 Gateway YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema


class ServingConfigError(RuntimeError):
    """Raised when a Gateway configuration is missing or unsafe."""


class GatewayConfig(StrictSchema):
    """Secure-by-default local Model Gateway configuration."""

    schema_version: Literal["1.0"] = "1.0"
    config_id: str = Field(pattern=r"^m[78]-gateway-[a-z0-9-]{1,64}$")
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=8000, ge=1024, le=65535)
    backend_base_url: str = "http://127.0.0.1:8001"
    manage_backend: bool = True
    gpu_index: int = Field(default=7, ge=0, le=9)
    preflight_max_memory_used_mib: int = Field(default=1024, ge=0, le=24_576)
    preflight_max_utilization_percent: int = Field(default=10, ge=0, le=100)
    preflight_max_temperature_c: int = Field(default=79, ge=20, le=95)
    backend_startup_timeout_seconds: float = Field(default=180.0, gt=0, le=600)
    backend_cleanup_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    backend_restart_limit: int = Field(default=2, ge=0, le=10)
    model_max_length: int = Field(default=4096, ge=512, le=40960)
    gpu_memory_utilization: float = Field(default=0.4, gt=0.1, le=0.95)
    tool_call_parser: Literal["hermes"] = "hermes"
    enforce_eager: bool = True
    bearer_token_env: str = Field(
        default="TINYLLM_GATEWAY_BEARER_TOKEN",
        pattern=r"^[A-Z][A-Z0-9_]{2,63}$",
    )
    request_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    backend_health_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    max_concurrency: int = Field(default=32, ge=1, le=1024)
    requests_per_minute: int = Field(default=120, ge=1, le=100_000)
    max_request_bytes: int = Field(default=1_048_576, ge=1024, le=8_388_608)
    trusted_hosts: tuple[str, ...] = ("127.0.0.1", "localhost")
    expose_docs: Literal[False] = False

    @field_validator("backend_base_url")
    @classmethod
    def require_local_backend(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("backend_base_url must use a loopback HTTP address")
        return normalized

    @field_validator("trusted_hosts", mode="before")
    @classmethod
    def freeze_trusted_hosts(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_hosts(self) -> GatewayConfig:
        if not self.trusted_hosts or any(
            host not in {"127.0.0.1", "localhost", "testserver"} for host in self.trusted_hosts
        ):
            raise ValueError("trusted_hosts may contain only explicit loopback hosts")
        return self


def load_gateway_config(path: Path) -> GatewayConfig:
    """Load a strict Gateway YAML file without environment expansion."""

    try:
        decoded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        return GatewayConfig.model_validate(decoded)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ServingConfigError("M7 Gateway config is invalid") from exc
