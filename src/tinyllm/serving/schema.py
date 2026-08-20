"""Strict request and service schemas for the M7 Model Gateway."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema


class ChatMessage(StrictSchema):
    """Text-only OpenAI-compatible chat message supported by the M7 model."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = Field(default=None, max_length=262_144)
    name: str | None = Field(default=None, min_length=1, max_length=64)
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=128)
    tool_calls: tuple[dict[str, Any], ...] | None = None

    @field_validator("tool_calls", mode="before")
    @classmethod
    def freeze_tool_calls(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_tool_message(self) -> ChatMessage:
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "assistant" and self.tool_calls is not None:
            raise ValueError("only assistant messages may contain tool_calls")
        if self.content is None and not self.tool_calls:
            raise ValueError("message requires content or tool_calls")
        return self


class FunctionDefinition(StrictSchema):
    """OpenAI function definition forwarded to a fixed vLLM parser."""

    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    description: str | None = Field(default=None, max_length=4096)
    parameters: dict[str, Any]

    @field_validator("parameters")
    @classmethod
    def bound_json_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        from tinyllm.serving.vllm_guard import _validate_json_schema

        try:
            _validate_json_schema(value)
        except ValueError as exc:
            raise ValueError("tool parameters exceed the safe JSON Schema subset") from exc
        return value


class ChatTool(StrictSchema):
    """Function tool definition."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class NamedFunctionChoice(StrictSchema):
    """Request one specific function by name."""

    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")


class NamedToolChoice(StrictSchema):
    """OpenAI named-tool selector."""

    type: Literal["function"] = "function"
    function: NamedFunctionChoice


class ResponseFormat(StrictSchema):
    """M7 response-format subset supported by vLLM."""

    type: Literal["text", "json_object"]


class StreamOptions(StrictSchema):
    """OpenAI stream options supported by the proxy."""

    include_usage: bool = False


class ChatCompletionRequest(StrictSchema):
    """Bounded OpenAI-compatible Chat Completions request."""

    model: str = Field(min_length=1, max_length=180)
    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=128)
    mode: Literal["thinking", "nonthinking"] = "nonthinking"
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=8192)
    n: Literal[1] = 1
    seed: int | None = None
    stop: str | tuple[str, ...] | None = None
    tools: tuple[ChatTool, ...] | None = Field(default=None, max_length=128)
    tool_choice: Literal["auto", "required", "none"] | NamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    response_format: ResponseFormat | None = None
    user: str | None = Field(default=None, max_length=128)

    @field_validator("messages", "tools", mode="before")
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("stop", mode="before")
    @classmethod
    def freeze_stop(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("temperature", "top_p")
    @classmethod
    def require_finite_sampling_values(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("sampling values must be finite")
        return value

    @model_validator(mode="after")
    def validate_request(self) -> ChatCompletionRequest:
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("max_tokens and max_completion_tokens are mutually exclusive")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        if self.tool_choice is not None and self.tool_choice != "none" and not self.tools:
            raise ValueError("tool_choice requires tools")
        if isinstance(self.tool_choice, NamedToolChoice):
            names = {tool.function.name for tool in self.tools or ()}
            if self.tool_choice.function.name not in names:
                raise ValueError("named tool_choice must reference a supplied tool")
        if isinstance(self.stop, tuple) and (not self.stop or len(self.stop) > 4):
            raise ValueError("stop must contain between one and four strings")
        return self


class ModelCard(StrictSchema):
    """One OpenAI-compatible model-list entry."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(ge=0)
    owned_by: Literal["tinyllm-system"] = "tinyllm-system"


class ModelList(StrictSchema):
    """OpenAI-compatible model-list response."""

    object: Literal["list"] = "list"
    data: tuple[ModelCard, ...]


class HealthResponse(StrictSchema):
    """Path-free liveness/readiness response."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["ok", "unavailable"]
    ready: bool
    model: str | None = None
    backend: Literal["vllm-http"] = "vllm-http"


class VersionResponse(StrictSchema):
    """Public version and active deployment identity."""

    schema_version: Literal["1.0"] = "1.0"
    service: Literal["tinyllm-gateway"] = "tinyllm-gateway"
    version: str
    model: str
    deployment_status: Literal["Candidate", "Production", "Evaluation"]
    candidate_model_version: str | None = None
    evaluation_subject_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_registry_identity(self) -> VersionResponse:
        if self.deployment_status == "Evaluation":
            if self.candidate_model_version is not None or self.evaluation_subject_sha256 is None:
                raise ValueError("Evaluation version requires only an Evaluation record identity")
        elif self.candidate_model_version is None or self.evaluation_subject_sha256 is not None:
            raise ValueError("M6/M7 version requires only a Candidate identity")
        return self
