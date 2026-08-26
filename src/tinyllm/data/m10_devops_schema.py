"""Strict contracts for the authored M10 DevOps Agent training source."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.agent import AgentToolDefinition
from tinyllm.agent_eval.schema import AgentEvalCategory, AgentEvalLanguage
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M10DevOpsCategory = AgentEvalCategory
M10DevOpsLanguage = AgentEvalLanguage
M10DevOpsRevision = Literal["m10-devops-training-v1", "m10-devops-training-v2"]


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON-compatible value with stable UTF-8 rendering."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class M10DevOpsFunctionCall(StrictSchema):
    """One normalized OpenAI function invocation."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    arguments: dict[str, Any]


class M10DevOpsToolCall(StrictSchema):
    """One assistant tool call with a stable conversation-local identifier."""

    id: str = Field(pattern=r"^call_[a-z0-9_]{3,120}$")
    type: Literal["function"] = "function"
    function: M10DevOpsFunctionCall


class M10DevOpsTrainingMessage(StrictSchema):
    """One canonical message and its immutable supervision decision."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = Field(default=None, max_length=262_144)
    name: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{1,63}$")
    tool_call_id: str | None = Field(default=None, pattern=r"^call_[a-z0-9_]{3,120}$")
    tool_calls: tuple[M10DevOpsToolCall, ...] = Field(default=(), max_length=8)
    supervised: bool
    message_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("tool_calls", mode="before")
    @classmethod
    def freeze_tool_calls(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_message(self) -> M10DevOpsTrainingMessage:
        if self.supervised != (self.role == "assistant"):
            raise ValueError("only assistant messages may be supervised")
        if self.role == "tool":
            if self.tool_call_id is None or self.name is None or self.content is None:
                raise ValueError("tool messages require name, tool_call_id, and content")
        elif self.tool_call_id is not None or self.name is not None:
            raise ValueError("only tool messages may carry name or tool_call_id")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages may contain tool calls")
        if self.content is None and not self.tool_calls:
            raise ValueError("messages require content or tool calls")
        if self.role == "assistant" and self.content and "<think>" in self.content.lower():
            raise ValueError("M10 authored trajectories cannot contain visible chain-of-thought")
        canonical = {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "supervised": self.supervised,
        }
        if self.message_sha256 != canonical_json_sha256(canonical):
            raise ValueError("M10 message SHA256 does not match its canonical content")
        return self


class M10DevOpsTrainingSample(StrictSchema):
    """One self-authored, policy-safe DevOps Agent SFT trajectory."""

    schema_version: Literal["1.0"] = "1.0"
    sample_id: str = Field(pattern=r"^m10-devops-(?:en|zh)-[a-z0-9-]+-[0-9]{4}$")
    source_id: Literal["tinyllm_devops"] = "tinyllm_devops"
    source_revision: M10DevOpsRevision = "m10-devops-training-v1"
    license: Literal["Apache-2.0"] = "Apache-2.0"
    language: M10DevOpsLanguage
    category: M10DevOpsCategory
    template_family: str = Field(pattern=r"^family-[a-z0-9-]{3,80}$")
    group_id: str = Field(pattern=r"^group-[a-z0-9-]{3,80}$")
    mode: Literal["nonthinking"] = "nonthinking"
    available_tools: tuple[AgentToolDefinition, ...] = Field(min_length=7, max_length=7)
    messages: tuple[M10DevOpsTrainingMessage, ...] = Field(min_length=3, max_length=12)
    source_record_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    tool_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    content_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("available_tools", "messages", mode="before")
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_trajectory(self) -> M10DevOpsTrainingSample:
        if not self.sample_id.startswith(f"m10-devops-{self.language}-"):
            raise ValueError("M10 sample identity does not match language")
        if self.messages[0].role != "system" or self.messages[1].role != "user":
            raise ValueError("M10 trajectories must begin with system then user")
        if self.messages[-1].role != "assistant" or not self.messages[-1].content:
            raise ValueError("M10 trajectories must end with a supervised final assistant answer")
        tool_names = tuple(item.tool_name for item in self.available_tools)
        if len(set(tool_names)) != 7:
            raise ValueError("M10 tool catalog must contain seven unique tools")
        proposed: dict[str, str] = {}
        observed: set[str] = set()
        for message in self.messages:
            for call in message.tool_calls:
                if call.id in proposed:
                    raise ValueError("M10 tool call identifiers must be unique")
                if call.function.name not in tool_names:
                    raise ValueError("M10 trajectory references an unavailable tool")
                proposed[call.id] = call.function.name
            if message.role == "tool":
                assert message.tool_call_id is not None
                assert message.name is not None
                if proposed.get(message.tool_call_id) != message.name:
                    raise ValueError("M10 tool result does not match an earlier tool call")
                if message.tool_call_id in observed:
                    raise ValueError("M10 tool calls may have only one result")
                observed.add(message.tool_call_id)
        if set(proposed) != observed:
            raise ValueError("every M10 tool call requires exactly one tool result")
        no_call_categories = {
            "no_tool",
            "wrong_tool_irrelevance",
            "missing_argument_clarification",
        }
        if self.category in no_call_categories and proposed:
            raise ValueError("M10 no-call categories cannot contain tool invocations")
        if self.category not in no_call_categories and not proposed:
            raise ValueError("M10 tool-use categories require at least one tool invocation")
        if self.category == "parallel_independent_tools" and not any(
            len(message.tool_calls) >= 2 for message in self.messages
        ):
            raise ValueError("parallel trajectories require one multi-call assistant message")
        prompt = [
            {"role": message.role, "content": message.content}
            for message in self.messages
            if message.role == "user"
        ]
        if self.prompt_sha256 != canonical_json_sha256(prompt):
            raise ValueError("M10 prompt SHA256 does not match user messages")
        tools_hash = canonical_json_sha256([item.to_dict() for item in self.available_tools])
        if self.tool_schema_sha256 != tools_hash:
            raise ValueError("M10 tool Schema SHA256 does not match available tools")
        canonical = self.to_dict()
        canonical.pop("content_sha256")
        if self.content_sha256 != canonical_json_sha256(canonical):
            raise ValueError("M10 content SHA256 does not match the sample")
        return self


class M10DevOpsDatasetManifest(StrictSchema):
    """Immutable authored-source identity and aggregate validation evidence."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_version: str = Field(pattern=r"^m10-devops-training-v[12]-[0-9a-f]{8}$")
    source_revision: M10DevOpsRevision = "m10-devops-training-v1"
    license: Literal["Apache-2.0"] = "Apache-2.0"
    seed: Literal[20260820, 20260825] = 20260820
    item_count: Literal[2400] = 2400
    category_counts: dict[M10DevOpsCategory, int]
    language_counts: dict[M10DevOpsLanguage, int]
    supervised_message_count: int = Field(gt=0)
    masked_message_count: int = Field(gt=0)
    tool_call_count: int = Field(ge=0)
    unique_group_count: int = Field(gt=0)
    tool_catalog_sha256: str = Field(pattern=SHA256_PATTERN)
    items_sha256: str = Field(pattern=SHA256_PATTERN)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    generator_config_sha256: str = Field(pattern=SHA256_PATTERN)
    review_status: Literal["pending", "approved"]
    training_permitted: bool
    contains_evaluation_content: Literal[False] = False

    @model_validator(mode="after")
    def validate_manifest(self) -> M10DevOpsDatasetManifest:
        expected_categories = (
            {
                "single_tool": 360,
                "no_tool": 360,
                "wrong_tool_irrelevance": 360,
                "missing_argument_clarification": 360,
                "sequential_multi_step": 360,
                "parallel_independent_tools": 120,
                "tool_failure_recovery": 240,
                "grounding_approval_security": 240,
            }
            if self.source_revision == "m10-devops-training-v1"
            else {
                "single_tool": 360,
                "no_tool": 240,
                "wrong_tool_irrelevance": 480,
                "missing_argument_clarification": 240,
                "sequential_multi_step": 480,
                "parallel_independent_tools": 120,
                "tool_failure_recovery": 360,
                "grounding_approval_security": 120,
            }
        )
        if self.category_counts != expected_categories:
            raise ValueError("M10 authored category counts differ from the frozen design")
        if self.language_counts != {"en": 1680, "zh": 720}:
            raise ValueError("M10 authored language counts must be exactly 70/30")
        if sum(self.category_counts.values()) != self.item_count:
            raise ValueError("M10 authored category counts do not sum to item_count")
        if sum(self.language_counts.values()) != self.item_count:
            raise ValueError("M10 authored language counts do not sum to item_count")
        if self.training_permitted != (self.review_status == "approved"):
            raise ValueError("M10 authored training requires approved content review")
        return self


class M10DevOpsDuplicateReport(StrictSchema):
    """Content-free exact and near-duplicate scan result."""

    schema_version: Literal["1.0"] = "1.0"
    algorithm: Literal["minhash-5gram-lsh-v1"] = "minhash-5gram-lsh-v1"
    permutation_count: Literal[128] = 128
    threshold_basis_points: Literal[8500] = 8500
    item_count: Literal[2400] = 2400
    exact_duplicate_pairs: int = Field(ge=0)
    clustered_near_duplicate_pairs: int = Field(ge=0)
    cross_group_near_duplicate_pairs: int = Field(ge=0)
    maximum_candidate_prompt_similarity_basis_points: int = Field(ge=0, le=10_000)
    shared_tool_schema_alone_is_match: Literal[False] = False
    status: Literal["pass", "fail"]
    report_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_report(self) -> M10DevOpsDuplicateReport:
        expected = (
            "pass"
            if not self.exact_duplicate_pairs and not self.cross_group_near_duplicate_pairs
            else "fail"
        )
        if self.status != expected:
            raise ValueError("M10 duplicate report status is inconsistent")
        canonical = self.to_dict()
        canonical.pop("report_sha256")
        if self.report_sha256 != canonical_json_sha256(canonical):
            raise ValueError("M10 duplicate report SHA256 is inconsistent")
        return self


class M10ContaminationTargetResult(StrictSchema):
    """Content-free result for one immutable evaluation target."""

    target_id: Literal["m9_dev", "m9_release", "bfcl_core", "m6_domain"]
    target_version: str = Field(min_length=1, max_length=180)
    target_content_sha256: str = Field(pattern=SHA256_PATTERN)
    target_items: int = Field(gt=0)
    exact_matches: int = Field(ge=0)
    near_matches: int = Field(ge=0)
    maximum_candidate_prompt_similarity_basis_points: int = Field(ge=0, le=10_000)
    contains_target_content: Literal[False] = False


class M10DevOpsContaminationReport(StrictSchema):
    """Content-free four-boundary evaluation contamination report."""

    schema_version: Literal["1.0"] = "1.0"
    scan_version: Literal["m10-devops-contamination-v1", "m10-devops-contamination-v2"] = (
        "m10-devops-contamination-v1"
    )
    algorithm: Literal["minhash-5gram-lsh-v1"] = "minhash-5gram-lsh-v1"
    permutation_count: Literal[128] = 128
    threshold_basis_points: Literal[8500] = 8500
    source_dataset_version: str = Field(pattern=r"^m10-devops-training-v[12]-[0-9a-f]{8}$")
    source_content_sha256: str = Field(pattern=SHA256_PATTERN)
    source_items: Literal[2400] = 2400
    targets: tuple[
        M10ContaminationTargetResult,
        M10ContaminationTargetResult,
        M10ContaminationTargetResult,
        M10ContaminationTargetResult,
    ]
    shared_tool_schema_alone_is_match: Literal[False] = False
    status: Literal["pass", "fail"]
    report_sha256: str = Field(pattern=SHA256_PATTERN)
    contains_evaluation_content: Literal[False] = False

    @field_validator("targets", mode="before")
    @classmethod
    def freeze_targets(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_report(self) -> M10DevOpsContaminationReport:
        if tuple(item.target_id for item in self.targets) != (
            "m9_dev",
            "m9_release",
            "bfcl_core",
            "m6_domain",
        ):
            raise ValueError("M10 contamination targets must use the frozen order")
        matches = sum(item.exact_matches + item.near_matches for item in self.targets)
        if self.status != ("pass" if matches == 0 else "fail"):
            raise ValueError("M10 contamination report status is inconsistent")
        canonical = self.to_dict()
        canonical.pop("report_sha256")
        if self.report_sha256 != canonical_json_sha256(canonical):
            raise ValueError("M10 contamination report SHA256 is inconsistent")
        return self


class M10DevOpsBuildReport(StrictSchema):
    """Public, path-free summary of one authored-source build."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["review_pending", "ready"]
    dataset_version: str = Field(pattern=r"^m10-devops-training-v[12]-[0-9a-f]{8}$")
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    items_sha256: str = Field(pattern=SHA256_PATTERN)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    item_count: Literal[2400] = 2400
    category_counts: dict[M10DevOpsCategory, int]
    language_counts: dict[M10DevOpsLanguage, int]
    duplicate_report_sha256: str = Field(pattern=SHA256_PATTERN)
    contamination_report_sha256: str = Field(pattern=SHA256_PATTERN)
    duplicate_status: Literal["pass", "fail"]
    contamination_status: Literal["pass", "fail"]
    review_status: Literal["pending", "approved"]
    training_permitted: bool
    private_artifacts_only: Literal[True] = True
    contains_source_or_evaluation_content: Literal[False] = False

    @model_validator(mode="after")
    def validate_build_status(self) -> M10DevOpsBuildReport:
        ready = (
            self.duplicate_status == "pass"
            and self.contamination_status == "pass"
            and self.review_status == "approved"
        )
        if self.training_permitted != ready or self.status != (
            "ready" if ready else "review_pending"
        ):
            raise ValueError("M10 authored build status is inconsistent")
        return self


class M10DevOpsContentReviewResult(StrictSchema):
    """Path-free maintainer approval bound to one immutable review packet."""

    schema_version: Literal["1.0"] = "1.0"
    review_version: Literal["m10-devops-content-review-v1", "m10-devops-content-review-v2"] = (
        "m10-devops-content-review-v1"
    )
    reviewed_at: datetime
    status: Literal["approved"] = "approved"
    reviewer_role: Literal["maintainer"] = "maintainer"
    source_dataset_version: str = Field(pattern=r"^m10-devops-training-v[12]-[0-9a-f]{8}$")
    source_pending_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    source_items_sha256: str = Field(pattern=SHA256_PATTERN)
    source_content_sha256: str = Field(pattern=SHA256_PATTERN)
    source_review_packet_sha256: str = Field(pattern=SHA256_PATTERN)
    source_duplicate_report_sha256: str = Field(pattern=SHA256_PATTERN)
    source_contamination_report_sha256: str = Field(pattern=SHA256_PATTERN)
    approved_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    reviewed_items: Literal[80] = 80
    passed_items: Literal[80] = 80
    rejected_items: Literal[0] = 0
    category_counts: dict[M10DevOpsCategory, int]
    language_counts: dict[M10DevOpsLanguage, int]
    authored_source_authorized: Literal[True] = True
    full_m10_mixture_authorized: Literal[False] = False
    m10_training_authorized: Literal[False] = False

    @field_validator("reviewed_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("M10 DevOps review timestamp must use UTC")
        return value

    @model_validator(mode="after")
    def validate_review(self) -> M10DevOpsContentReviewResult:
        expected_categories = {
            "single_tool": 10,
            "no_tool": 10,
            "wrong_tool_irrelevance": 10,
            "missing_argument_clarification": 10,
            "sequential_multi_step": 10,
            "parallel_independent_tools": 10,
            "tool_failure_recovery": 10,
            "grounding_approval_security": 10,
        }
        if (
            self.passed_items + self.rejected_items != self.reviewed_items
            or self.category_counts != expected_categories
            or self.language_counts != {"en": 40, "zh": 40}
        ):
            raise ValueError("M10 DevOps content-review accounting is inconsistent")
        return self
