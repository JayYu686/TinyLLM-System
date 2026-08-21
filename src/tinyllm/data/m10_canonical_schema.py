"""Strict contracts for canonical M10 external Agent training sources."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.m10_devops_schema import canonical_json_sha256
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M10CanonicalSourceId = Literal["toolace", "hermes_function_calling"]
M10CanonicalLanguage = Literal["en", "zh"]
M10CanonicalRejectReason = Literal[
    "invalid_row_shape",
    "invalid_role_path",
    "invalid_tool_schema",
    "malformed_tool_call",
    "unpaired_tool_result",
    "visible_reasoning",
]


class M10CanonicalToolDefinition(StrictSchema):
    """One normalized OpenAI function definition."""

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,127}$")
    description: str | None = Field(default=None, max_length=16_384)
    input_schema: dict[str, Any]
    tool_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_tool(self) -> M10CanonicalToolDefinition:
        if self.input_schema.get("type") != "object":
            raise ValueError("M10 canonical tool Schema must describe an object")
        canonical = self.to_dict()
        canonical.pop("tool_sha256")
        if self.tool_sha256 != canonical_json_sha256(canonical):
            raise ValueError("M10 canonical tool SHA256 differs")
        return self


class M10CanonicalToolCall(StrictSchema):
    """One normalized assistant function call."""

    id: str = Field(pattern=r"^call_[a-z0-9_]{3,160}$")
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,127}$")
    arguments: dict[str, Any]
    call_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_call(self) -> M10CanonicalToolCall:
        canonical = self.to_dict()
        canonical.pop("call_sha256")
        if self.call_sha256 != canonical_json_sha256(canonical):
            raise ValueError("M10 canonical tool-call SHA256 differs")
        return self


class M10CanonicalMessage(StrictSchema):
    """One message with an immutable assistant-only loss decision."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = Field(default=None, max_length=524_288)
    tool_calls: tuple[M10CanonicalToolCall, ...] = Field(default=(), max_length=32)
    tool_call_ids: tuple[str, ...] = Field(default=(), max_length=32)
    supervised: bool
    message_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("tool_calls", "tool_call_ids", mode="before")
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_message(self) -> M10CanonicalMessage:
        if self.supervised != (self.role == "assistant"):
            raise ValueError("only M10 canonical assistant messages may be supervised")
        if self.role == "tool":
            if self.content is None or not self.tool_call_ids or self.tool_calls:
                raise ValueError("M10 canonical tool results require call IDs and content")
        elif self.tool_call_ids:
            raise ValueError("only M10 canonical tool results may carry call IDs")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only M10 canonical assistant messages may carry tool calls")
        if self.content is None and not self.tool_calls:
            raise ValueError("M10 canonical messages require content or tool calls")
        if self.role == "assistant" and self.content and "<think>" in self.content.casefold():
            raise ValueError("M10 canonical external data cannot expose chain-of-thought")
        if len(self.tool_call_ids) != len(set(self.tool_call_ids)):
            raise ValueError("M10 canonical tool result call IDs must be unique")
        canonical = self.to_dict()
        canonical.pop("message_sha256")
        if self.message_sha256 != canonical_json_sha256(canonical):
            raise ValueError("M10 canonical message SHA256 differs")
        return self


class M10CanonicalTrainingSample(StrictSchema):
    """One canonical external Agent conversation."""

    schema_version: Literal["1.0"] = "1.0"
    source_id: M10CanonicalSourceId
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_record_id: str = Field(pattern=r"^(?:toolace|hermes)-[a-z0-9_-]{1,128}$")
    source_record_sha256: str = Field(pattern=SHA256_PATTERN)
    license: Literal["Apache-2.0"] = "Apache-2.0"
    language: M10CanonicalLanguage
    mode: Literal["nonthinking"] = "nonthinking"
    group_id: str = Field(pattern=r"^group-(?:toolace|hermes)-[0-9a-f]{16}$")
    tools: tuple[M10CanonicalToolDefinition, ...] = Field(default=(), max_length=256)
    messages: tuple[M10CanonicalMessage, ...] = Field(min_length=3, max_length=128)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    tool_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    content_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("tools", "messages", mode="before")
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_sample(self) -> M10CanonicalTrainingSample:
        expected_prefix = "toolace-" if self.source_id == "toolace" else "hermes-"
        if not self.source_record_id.startswith(expected_prefix):
            raise ValueError("M10 canonical record ID differs from its source")
        if self.messages[0].role != "system" or not any(
            item.role == "user" for item in self.messages
        ):
            raise ValueError("M10 canonical samples require system and user context")
        if self.messages[-1].role not in {"assistant", "tool"}:
            raise ValueError("M10 canonical samples must end with assistant or tool")
        tool_names = tuple(item.name for item in self.tools)
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("M10 canonical tools must use unique normalized names")
        proposed: dict[str, str] = {}
        observed: set[str] = set()
        for message in self.messages:
            for call in message.tool_calls:
                if call.id in proposed or call.name not in tool_names:
                    raise ValueError("M10 canonical tool call identity is invalid")
                proposed[call.id] = call.name
            for call_id in message.tool_call_ids:
                if call_id not in proposed or call_id in observed:
                    raise ValueError("M10 canonical tool result does not match a call")
                observed.add(call_id)
        prompt = [
            {"role": item.role, "content": item.content}
            for item in self.messages
            if item.role == "user"
        ]
        if self.prompt_sha256 != canonical_json_sha256(prompt):
            raise ValueError("M10 canonical prompt SHA256 differs")
        if self.tool_schema_sha256 != canonical_json_sha256(
            [item.to_dict() for item in self.tools]
        ):
            raise ValueError("M10 canonical tool Schema SHA256 differs")
        canonical = self.to_dict()
        canonical.pop("content_sha256")
        if self.content_sha256 != canonical_json_sha256(canonical):
            raise ValueError("M10 canonical sample SHA256 differs")
        return self


class M10ExternalRejectedRecord(StrictSchema):
    """One private rejected source row with content-free identity."""

    schema_version: Literal["1.0"] = "1.0"
    source_id: M10CanonicalSourceId
    row_index: int = Field(ge=0)
    source_record_sha256: str = Field(pattern=SHA256_PATTERN)
    reason: M10CanonicalRejectReason


class M10ExternalImportManifest(StrictSchema):
    """Committed identity and aggregate facts for one external canonical source."""

    schema_version: Literal["1.0"] = "1.0"
    import_version: str = Field(pattern=r"^m10-(?:toolace|hermes)-canonical-v1-[0-9a-f]{8}$")
    source_id: M10CanonicalSourceId
    dataset_id: str = Field(min_length=1)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    source_rows: int = Field(gt=0)
    accepted_rows: int = Field(ge=0)
    rejected_rows: int = Field(ge=0)
    rejection_counts: dict[M10CanonicalRejectReason, int]
    language_counts: dict[M10CanonicalLanguage, int]
    supervised_messages: int = Field(ge=0)
    masked_messages: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    items_sha256: str = Field(pattern=SHA256_PATTERN)
    rejected_sha256: str = Field(pattern=SHA256_PATTERN)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    status: Literal["committed"] = "committed"
    contains_evaluation_content: Literal[False] = False

    @model_validator(mode="after")
    def validate_manifest(self) -> M10ExternalImportManifest:
        if self.accepted_rows + self.rejected_rows != self.source_rows:
            raise ValueError("M10 external import rows do not sum to source rows")
        if sum(self.rejection_counts.values()) != self.rejected_rows:
            raise ValueError("M10 external rejection counts differ")
        if sum(self.language_counts.values()) != self.accepted_rows:
            raise ValueError("M10 external language counts differ")
        expected_prefix = "m10-toolace-" if self.source_id == "toolace" else "m10-hermes-"
        if not self.import_version.startswith(expected_prefix):
            raise ValueError("M10 external import version differs from source")
        return self


class M10ExternalImportSummary(StrictSchema):
    """Path-free public summary for one private canonical import."""

    source_id: M10CanonicalSourceId
    import_version: str = Field(pattern=r"^m10-(?:toolace|hermes)-canonical-v1-[0-9a-f]{8}$")
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    source_rows: int = Field(gt=0)
    accepted_rows: int = Field(ge=0)
    rejected_rows: int = Field(ge=0)
    rejection_counts: dict[M10CanonicalRejectReason, int]
    language_counts: dict[M10CanonicalLanguage, int]
    supervised_messages: int = Field(ge=0)
    masked_messages: int = Field(ge=0)
    tool_calls: int = Field(ge=0)


class M10ExternalImportReport(StrictSchema):
    """Combined content-free evidence for both frozen external sources."""

    schema_version: Literal["1.0"] = "1.0"
    report_version: Literal["m10-external-canonical-import-v1"] = "m10-external-canonical-import-v1"
    status: Literal["pass", "fail"]
    sources: tuple[M10ExternalImportSummary, M10ExternalImportSummary]
    total_source_rows: int = Field(gt=0)
    total_accepted_rows: int = Field(ge=0)
    total_rejected_rows: int = Field(ge=0)
    contains_source_content: Literal[False] = False

    @field_validator("sources", mode="before")
    @classmethod
    def freeze_sources(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_report(self) -> M10ExternalImportReport:
        if tuple(item.source_id for item in self.sources) != (
            "toolace",
            "hermes_function_calling",
        ):
            raise ValueError("M10 external imports must use frozen source order")
        if (
            self.total_source_rows != sum(item.source_rows for item in self.sources)
            or self.total_accepted_rows != sum(item.accepted_rows for item in self.sources)
            or self.total_rejected_rows != sum(item.rejected_rows for item in self.sources)
            or self.total_accepted_rows + self.total_rejected_rows != self.total_source_rows
        ):
            raise ValueError("M10 external import report totals differ")
        if self.status != ("pass" if self.total_accepted_rows else "fail"):
            raise ValueError("M10 external import report status differs")
        return self
