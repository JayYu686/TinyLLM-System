"""Strict M8 Agent, MCP registration, approval, and retrieval schemas."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema

AGENT_RUN_PATTERN = r"^agent-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}-[0-9a-f]{4}$"
APPROVAL_PATTERN = r"^approval-[0-9a-f]{12}$"
IDEMPOTENCY_KEY_PATTERN = r"^[A-Za-z0-9._:-]{16,128}$"
MCP_SERVER_ID_PATTERN = r"^[a-z][a-z0-9-]{2,63}$"
TOOL_NAME_PATTERN = r"^[a-z][a-z0-9_]{1,63}$"

AgentRunStatus = Literal[
    "created",
    "running",
    "waiting_approval",
    "succeeded",
    "failed",
    "cancelled",
    "expired",
]
AgentEventType = Literal[
    "run.started",
    "model.delta",
    "tool.call.proposed",
    "approval.required",
    "tool.started",
    "tool.completed",
    "message.completed",
    "run.completed",
    "run.failed",
]


class AgentMessage(StrictSchema):
    """Text and tool-result message accepted by the bounded Agent API."""

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
    def validate_tool_message(self) -> AgentMessage:
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "assistant" and self.tool_calls is not None:
            raise ValueError("only assistant messages may contain tool_calls")
        if self.content is None and not self.tool_calls:
            raise ValueError("message requires content or tool_calls")
        return self


def _freeze_unique_strings(value: object, *, label: str) -> tuple[str, ...] | object:
    if not isinstance(value, (list, tuple)):
        return value
    frozen = tuple(value)
    if len(frozen) != len(set(frozen)):
        raise ValueError(f"{label} must be unique")
    return frozen


def _reject_private_reasoning(value: object) -> None:
    forbidden = {"reasoning_content", "chain_of_thought", "raw_cot", "raw_reasoning"}
    pending = [value]
    nodes = 0
    while pending:
        current = pending.pop()
        nodes += 1
        if nodes > 4096:
            raise ValueError("Agent event payload exceeds the structural limit")
        if isinstance(current, dict):
            if forbidden.intersection(str(key).lower() for key in current):
                raise ValueError("Agent event payload cannot expose private reasoning")
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend(current)


class AgentRunRequest(StrictSchema):
    """Public request contract for one bounded Agent execution."""

    schema_version: Literal["1.0"] = "1.0"
    model: str = Field(default="production", min_length=1, max_length=180)
    messages: tuple[AgentMessage, ...] = Field(min_length=1, max_length=64)
    mode: Literal["thinking", "nonthinking"] = "nonthinking"
    mcp_server_ids: tuple[str, ...] = Field(default=("tinyllm-devops",), min_length=1, max_length=8)
    max_steps: int = Field(default=8, ge=1, le=8)

    @field_validator("messages", mode="before")
    @classmethod
    def freeze_messages(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("mcp_server_ids", mode="before")
    @classmethod
    def freeze_servers(cls, value: object) -> object:
        return _freeze_unique_strings(value, label="MCP Server IDs")

    @field_validator("mcp_server_ids")
    @classmethod
    def validate_server_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        import re

        if not all(re.fullmatch(MCP_SERVER_ID_PATTERN, item) for item in value):
            raise ValueError("MCP Server ID is invalid")
        return value


class MCPToolPolicy(StrictSchema):
    """Authoritative local policy for one MCP tool."""

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    access: Literal["read", "sandbox_write"]
    approval_required: bool
    timeout_seconds: float = Field(default=10.0, gt=0, le=30)
    max_attempts: int = Field(ge=1, le=3)

    @model_validator(mode="after")
    def validate_policy(self) -> MCPToolPolicy:
        if self.access == "sandbox_write":
            if not self.approval_required or self.max_attempts != 1:
                raise ValueError("sandbox write tools require approval and exactly one attempt")
        elif self.approval_required:
            raise ValueError("read-only tools cannot require write approval")
        return self


class MCPServerConfig(StrictSchema):
    """One administrator-controlled MCP Server registration."""

    server_id: str = Field(pattern=MCP_SERVER_ID_PATTERN)
    transport: Literal["stdio", "streamable_http"]
    command: Path | None = None
    args: tuple[str, ...] = Field(default=(), max_length=16)
    url: str | None = Field(default=None, max_length=512)
    bearer_token_env: str | None = Field(default=None, pattern=r"^TINYLLM_[A-Z0-9_]{3,100}$")
    tools: tuple[MCPToolPolicy, ...] = Field(min_length=1, max_length=32)

    @field_validator("command", mode="before")
    @classmethod
    def normalize_command_path(cls, value: object) -> object:
        return Path(value) if isinstance(value, str) else value

    @field_validator("args", "tools", mode="before")
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("args")
    @classmethod
    def reject_unsafe_arguments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any("\x00" in item or len(item) > 512 for item in value):
            raise ValueError("MCP stdio argument is unsafe")
        return value

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerConfig:
        if self.transport == "stdio":
            if self.command is None or not self.command.is_absolute() or self.url is not None:
                raise ValueError("stdio registration requires one absolute command and no URL")
            if self.bearer_token_env is not None:
                raise ValueError("stdio registration cannot use HTTP Bearer authentication")
        else:
            if self.command is not None or self.args:
                raise ValueError("Streamable HTTP registration cannot launch a command")
            if self.url is None or not self.url.startswith("https://"):
                raise ValueError("Streamable HTTP registration requires an HTTPS URL")
            if self.bearer_token_env is None:
                raise ValueError("Streamable HTTP registration requires an environment secret")
        tool_names = tuple(tool.name for tool in self.tools)
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("MCP tool policies must be unique")
        return self


class AgentConfig(StrictSchema):
    """Frozen Agent Runtime and allowlisted MCP Server configuration."""

    schema_version: Literal["1.0"] = "1.0"
    config_id: str = Field(pattern=r"^m8-agent-[a-z0-9-]{3,80}$")
    default_model: str = Field(default="production", min_length=1, max_length=180)
    default_mode: Literal["thinking", "nonthinking"] = "nonthinking"
    max_steps: int = Field(default=8, ge=1, le=8)
    max_tool_calls: int = Field(default=12, ge=1, le=12)
    tool_timeout_seconds: float = Field(default=10.0, gt=0, le=30)
    run_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    read_retry_delays_ms: tuple[Literal[250, 500], ...] = (250, 500)
    same_tool_consecutive_limit: Literal[2] = 2
    evidence_top_k: int = Field(default=8, ge=1, le=20)
    evidence_excerpt_characters: int = Field(default=1200, ge=100, le=4000)
    mcp_servers: tuple[MCPServerConfig, ...] = Field(min_length=1, max_length=8)

    @field_validator("read_retry_delays_ms", "mcp_servers", mode="before")
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_servers(self) -> AgentConfig:
        server_ids = tuple(server.server_id for server in self.mcp_servers)
        if len(server_ids) != len(set(server_ids)):
            raise ValueError("MCP Server registrations must be unique")
        return self


class AgentApprovalDecision(StrictSchema):
    """Idempotent user decision for one proposed sandbox write."""

    schema_version: Literal["1.0"] = "1.0"
    approval_id: str = Field(pattern=APPROVAL_PATTERN)
    decision: Literal["approved", "rejected"]
    idempotency_key: str = Field(pattern=IDEMPOTENCY_KEY_PATTERN)
    decided_at: datetime

    @model_validator(mode="after")
    def validate_timestamp(self) -> AgentApprovalDecision:
        if self.decided_at.tzinfo is None:
            raise ValueError("Agent approval timestamp must be timezone-aware")
        return self


class AgentApprovalRequest(StrictSchema):
    """Public approval body; idempotency identity is supplied only by the HTTP header."""

    schema_version: Literal["1.0"] = "1.0"
    decision: Literal["approved", "rejected"]


class AgentEvent(StrictSchema):
    """Durable SSE event with a per-Run monotonic sequence."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(pattern=AGENT_RUN_PATTERN)
    sequence: int = Field(ge=1)
    event_type: AgentEventType
    created_at: datetime
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data")
    @classmethod
    def hide_private_reasoning(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_private_reasoning(value)
        return value

    @model_validator(mode="after")
    def validate_timestamp(self) -> AgentEvent:
        if self.created_at.tzinfo is None:
            raise ValueError("Agent event timestamp must be timezone-aware")
        return self


class AgentToolCall(StrictSchema):
    """One model-proposed tool call, pending local policy enforcement."""

    call_id: str = Field(pattern=r"^call_[A-Za-z0-9_-]{1,120}$")
    server_id: str = Field(pattern=MCP_SERVER_ID_PATTERN)
    tool_name: str = Field(pattern=TOOL_NAME_PATTERN)
    arguments: dict[str, Any]

    @field_validator("arguments")
    @classmethod
    def reject_private_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_private_reasoning(value)
        return value


class AgentModelDecision(StrictSchema):
    """Parsed model decision: tool proposals or one final answer."""

    message: str | None = Field(default=None, max_length=262_144)
    tool_calls: tuple[AgentToolCall, ...] = Field(default=(), max_length=8)

    @field_validator("tool_calls", mode="before")
    @classmethod
    def freeze_calls(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_decision(self) -> AgentModelDecision:
        if bool(self.message) == bool(self.tool_calls):
            raise ValueError("model decision requires either a message or tool calls")
        identifiers = tuple(item.call_id for item in self.tool_calls)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("model tool call identifiers must be unique")
        return self


class AgentRunRecord(StrictSchema):
    """Content-minimized state projection for one Agent Run."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(pattern=AGENT_RUN_PATTERN)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model: str = Field(min_length=1, max_length=180)
    mode: Literal["thinking", "nonthinking"]
    mcp_server_ids: tuple[str, ...]
    max_steps: int = Field(ge=1, le=8)
    status: AgentRunStatus
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    completed_at: datetime | None = None
    steps_completed: int = Field(ge=0, le=8)
    tool_calls_completed: int = Field(ge=0, le=12)
    last_event_sequence: int = Field(ge=0)
    pending_approval_id: str | None = Field(default=None, pattern=APPROVAL_PATTERN)
    pending_tool_call: AgentToolCall | None = None
    error_code: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{2,63}$")

    @field_validator("mcp_server_ids", mode="before")
    @classmethod
    def freeze_servers(cls, value: object) -> object:
        return _freeze_unique_strings(value, label="MCP Server IDs")

    @model_validator(mode="after")
    def validate_state(self) -> AgentRunRecord:
        timestamps = (self.created_at, self.updated_at, self.expires_at)
        if any(value.tzinfo is None for value in timestamps) or (
            self.completed_at is not None and self.completed_at.tzinfo is None
        ):
            raise ValueError("Agent Run timestamps must be timezone-aware")
        if not self.created_at <= self.updated_at <= self.expires_at:
            raise ValueError("Agent Run timestamps are out of order")
        terminal = self.status in {"succeeded", "failed", "cancelled", "expired"}
        if terminal != (self.completed_at is not None):
            raise ValueError("Agent terminal status and completion time differ")
        waiting = self.status == "waiting_approval"
        if waiting != (self.pending_approval_id is not None):
            raise ValueError("Agent approval state is inconsistent")
        if waiting != (self.pending_tool_call is not None):
            raise ValueError("Agent pending tool call state is inconsistent")
        if self.status == "failed" and self.error_code is None:
            raise ValueError("failed Agent Run requires an error code")
        if self.status != "failed" and self.error_code is not None:
            raise ValueError("only failed Agent Runs may expose an error code")
        return self


class EvidenceSearchResult(StrictSchema):
    """One line-addressable untrusted evidence search result."""

    document_id: str = Field(pattern=r"^doc-[0-9a-f]{16}$")
    source_kind: Literal["documentation", "report", "registry", "run_metadata"]
    relative_path: str = Field(min_length=1, max_length=500)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    relevance_score: float = Field(ge=0)
    excerpt: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_lines(self) -> EvidenceSearchResult:
        if self.end_line < self.start_line:
            raise ValueError("evidence line range is invalid")
        if self.relative_path.startswith(("/", "../")) or "/../" in self.relative_path:
            raise ValueError("evidence path must remain relative")
        return self


class EvidenceIndexManifest(StrictSchema):
    """Deterministic identity of one rebuilt SQLite FTS5 evidence index."""

    schema_version: Literal["1.0"] = "1.0"
    index_version: str = Field(pattern=r"^m8-evidence-[0-9a-f]{8}$")
    built_at: datetime
    source_root_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    documents: int = Field(ge=1)
    chunks: int = Field(ge=1)
    index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_timestamp(self) -> EvidenceIndexManifest:
        if self.built_at.tzinfo is None:
            raise ValueError("evidence index timestamp must be timezone-aware")
        return self
