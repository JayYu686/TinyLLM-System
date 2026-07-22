"""Strict schemas for M5.1 reasoning tasks, teacher output, and dataset lineage."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema

ReasoningTaskFamily = Literal["config", "json", "linux", "log_diagnosis", "python"]
ReasoningLanguage = Literal["en", "zh"]
ReasoningSplit = Literal["pilot_train", "reasoning_dev"]
TeacherGenerationStatus = Literal["succeeded", "failed"]
TeacherFinishReason = Literal["stop", "length", "error"]
VerifierReason = Literal["accepted", "answer_mismatch", "invalid_final_json"]
ReasoningRejectionReason = Literal[
    "empty_final_answer",
    "empty_reasoning",
    "invalid_final_json",
    "missing_think_block",
    "multiple_think_blocks",
    "nested_think_tag",
    "no_candidate_passed",
    "sequence_too_long",
    "teacher_generation_failed",
    "teacher_length_limit",
    "verifier_failed",
]
ReasoningRejectionStage = Literal[
    "generation", "parsing", "tokenization", "verification", "selection"
]

REASONING_TASK_FAMILIES: tuple[ReasoningTaskFamily, ...] = (
    "config",
    "json",
    "linux",
    "log_diagnosis",
    "python",
)
REASONING_LANGUAGES: tuple[ReasoningLanguage, ...] = ("en", "zh")
REASONING_REJECTION_REASONS = frozenset(
    {
        "empty_final_answer",
        "empty_reasoning",
        "invalid_final_json",
        "missing_think_block",
        "multiple_think_blocks",
        "nested_think_tag",
        "no_candidate_passed",
        "sequence_too_long",
        "teacher_generation_failed",
        "teacher_length_limit",
        "verifier_failed",
    }
)


def canonical_json(value: object) -> str:
    """Render canonical UTF-8 JSON used by every M5.1 content identity."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def content_sha256(value: object) -> str:
    """Hash one canonical-JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class ReasoningTeacherIdentity(StrictSchema):
    """Pinned offline teacher identity used to produce visible reasoning traces."""

    repository: Literal["Qwen/Qwen3-8B"]
    revision: Literal["b968826d9c46dd6066d109eabc6255188de91218"]
    attention_architecture: Literal["gqa"]
    trust_remote_code: Literal[False]
    local_files_only: Literal[True]
    dtype: Literal["bfloat16"]
    mode: Literal["thinking"]


class ReasoningTeacherSampling(StrictSchema):
    """Frozen Qwen3 thinking-mode generation policy for the first M5 pilot."""

    do_sample: Literal[True]
    temperature: float = Field(gt=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: Literal[20]
    repetition_penalty: float = Field(ge=1.0, le=2.0)
    candidate_count: Literal[2]
    max_new_tokens: int = Field(ge=64, le=2048)
    base_seed: int = Field(ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def validate_frozen_sampling(self) -> ReasoningTeacherSampling:
        """Reject silent changes to the first pilot's Qwen3 sampling identity."""

        if (self.temperature, self.top_p, self.repetition_penalty) != (0.6, 0.95, 1.0):
            raise ValueError("M5.1 sampling must use temperature=0.6/top_p=0.95/repetition=1")
        return self


class ReasoningVerifierIdentity(StrictSchema):
    """Deterministic final-answer verifier that never executes generated code."""

    verifier_id: Literal["m5-json-exact-v1"]
    schema_version: Literal["1.0"] = "1.0"
    algorithm: Literal["canonical-json-object-equality"]
    execute_generated_code: Literal[False]


class ReasoningDevConfig(StrictSchema):
    """Frozen size, language mix, and family balance for the M5-only Dev set."""

    seed: int = Field(ge=0, le=2**32 - 1)
    total_tasks: Literal[200]
    task_family_counts: dict[str, int]
    language_counts_per_family: dict[str, int]

    @field_validator("task_family_counts")
    @classmethod
    def validate_family_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require five sorted families with exactly forty tasks each."""

        if list(value) != list(REASONING_TASK_FAMILIES) or any(
            count != 40 for count in value.values()
        ):
            raise ValueError("Dev task families must be sorted and contain exactly 40 each")
        return value

    @field_validator("language_counts_per_family")
    @classmethod
    def validate_language_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Freeze the per-family 70/30 item allocation used by the 200-item Dev set."""

        if value != {"en": 28, "zh": 12} or list(value) != list(REASONING_LANGUAGES):
            raise ValueError("Dev languages must contain sorted en=28 and zh=12 counts")
        return value


class M5ReasoningDataConfig(StrictSchema):
    """Complete M5.1 task, teacher, verifier, and sequence-length contract."""

    schema_version: Literal["1.0"] = "1.0"
    parent_dataset_version: Literal["m2-sft-v1-f82ff32e"]
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    thinking_template_sha256: Literal[
        "478614320637a49649c144d5bfef5d247344356900a615495eade36639845af9"
    ]
    max_sequence_length: Literal[1024]
    pilot_task_seed: int = Field(ge=0, le=2**32 - 1)
    dev: ReasoningDevConfig
    teacher: ReasoningTeacherIdentity
    sampling: ReasoningTeacherSampling
    verifier: ReasoningVerifierIdentity

    @model_validator(mode="after")
    def validate_seed_domains(self) -> M5ReasoningDataConfig:
        """Keep Pilot task identities separate from Dev and teacher sampling."""

        if len({self.pilot_task_seed, self.dev.seed, self.sampling.base_seed}) != 3:
            raise ValueError("Pilot, Dev, and teacher sampling seeds must be distinct")
        return self


class ReasoningTask(StrictSchema):
    """One deterministic reasoning prompt and its non-executable expected answer."""

    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^m5-reasoning:(pilot|dev):[a-z0-9][a-z0-9._-]{2,95}$")
    split: ReasoningSplit
    task_family: ReasoningTaskFamily
    language: ReasoningLanguage
    template_family: str = Field(pattern=r"^(pilot|dev)\.[a-z0-9][a-z0-9._-]+\.v1$")
    prompt: str = Field(min_length=1, max_length=8192)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_answer_json: str = Field(min_length=2, max_length=4096)
    expected_answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        """Reject whitespace-only prompts without silently rewriting content."""

        if not value.strip():
            raise ValueError("reasoning prompt cannot be blank")
        return value

    @field_validator("expected_answer_json")
    @classmethod
    def validate_expected_json(cls, value: str) -> str:
        """Require an already-canonical JSON object, never executable code."""

        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("expected answer must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("expected answer must be a JSON object")
        if canonical_json(decoded) != value:
            raise ValueError("expected answer JSON must use canonical formatting")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> ReasoningTask:
        """Bind split namespaces and hashes to the persisted task content."""

        namespace = "pilot" if self.split == "pilot_train" else "dev"
        if not self.id.startswith(f"m5-reasoning:{namespace}:"):
            raise ValueError("reasoning task ID must match its split namespace")
        if not self.template_family.startswith(f"{namespace}."):
            raise ValueError("template family must match its split namespace")
        if hashlib.sha256(self.prompt.encode("utf-8")).hexdigest() != self.prompt_sha256:
            raise ValueError("reasoning prompt hash does not match prompt content")
        answer_hash = hashlib.sha256(self.expected_answer_json.encode("utf-8")).hexdigest()
        if answer_hash != self.expected_answer_sha256:
            raise ValueError("expected answer hash does not match answer content")
        return self


class ReasoningTaskSetManifest(StrictSchema):
    """Content-addressed identity and distribution of one deterministic task set."""

    schema_version: Literal["1.0"] = "1.0"
    task_set_version: str = Field(pattern=r"^m5-reasoning-(pilot-tasks|dev)-v1-[0-9a-f]{8}$")
    split: ReasoningSplit
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tasks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(gt=0)
    task_family_counts: dict[str, int]
    language_counts: dict[str, int]
    template_family_counts: dict[str, int]

    @field_validator("task_family_counts", "language_counts", "template_family_counts")
    @classmethod
    def validate_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require deterministic sparse count maps."""

        if list(value) != sorted(value) or any(
            not key or count <= 0 for key, count in value.items()
        ):
            raise ValueError("task-set count mappings must use sorted keys and positive counts")
        return value

    @model_validator(mode="after")
    def validate_totals(self) -> ReasoningTaskSetManifest:
        """Keep version, split, and all distribution totals coherent."""

        namespace = "pilot-tasks" if self.split == "pilot_train" else "dev"
        if self.task_set_version != f"m5-reasoning-{namespace}-v1-{self.tasks_sha256[:8]}":
            raise ValueError("task-set version does not match task content hash")
        for counts in (
            self.task_family_counts,
            self.language_counts,
            self.template_family_counts,
        ):
            if sum(counts.values()) != self.task_count:
                raise ValueError("task-set distribution counts must equal task count")
        return self


class ReasoningContaminationMatch(StrictSchema):
    """Content-free identity for one Pilot/Dev prompt or template-family overlap."""

    kind: Literal["exact_prompt", "template_family"]
    pilot_identity: str = Field(min_length=1, max_length=160)
    dev_identity: str = Field(min_length=1, max_length=160)
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_evidence(self) -> ReasoningContaminationMatch:
        """Bind the evidence hash to match kind and both content-free identities."""

        expected = content_sha256(
            {
                "dev_identity": self.dev_identity,
                "kind": self.kind,
                "pilot_identity": self.pilot_identity,
            }
        )
        if self.evidence_sha256 != expected:
            raise ValueError("reasoning contamination match hash does not match identities")
        return self


class M5ReasoningContaminationReport(StrictSchema):
    """Deterministic Pilot/Dev isolation result required by every M5 pilot Manifest."""

    schema_version: Literal["1.0"] = "1.0"
    algorithm: Literal["exact-prompt-and-template-family-v1"]
    pilot_task_set_version: str = Field(pattern=r"^m5-reasoning-pilot-tasks-v1-[0-9a-f]{8}$")
    dev_task_set_version: str = Field(pattern=r"^m5-reasoning-dev-v1-[0-9a-f]{8}$")
    pilot_task_count: int = Field(gt=0)
    dev_task_count: Literal[200]
    exact_prompt_matches: int = Field(ge=0)
    template_family_overlaps: int = Field(ge=0)
    matches: tuple[ReasoningContaminationMatch, ...]
    status: Literal["pass", "fail"]

    @field_validator("matches")
    @classmethod
    def validate_match_order(
        cls, value: tuple[ReasoningContaminationMatch, ...]
    ) -> tuple[ReasoningContaminationMatch, ...]:
        """Require deterministic unique match ordering."""

        keys = tuple((match.kind, match.pilot_identity, match.dev_identity) for match in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("reasoning contamination matches must be unique and sorted")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> M5ReasoningContaminationReport:
        """Bind match counts and pass/fail status to retained evidence."""

        exact = sum(match.kind == "exact_prompt" for match in self.matches)
        template = sum(match.kind == "template_family" for match in self.matches)
        if exact != self.exact_prompt_matches or template != self.template_family_overlaps:
            raise ValueError("reasoning contamination counts do not match evidence")
        expected_status = "pass" if not self.matches else "fail"
        if self.status != expected_status:
            raise ValueError("reasoning contamination status does not match evidence")
        return self


class TeacherGenerationRecord(StrictSchema):
    """One auditable teacher attempt, including explicit failures and stop reasons."""

    schema_version: Literal["1.0"] = "1.0"
    generation_id: str = Field(
        pattern=r"^m5-reasoning:pilot:[a-z0-9][a-z0-9._-]{2,95}:candidate-[01]$"
    )
    task_id: str = Field(pattern=r"^m5-reasoning:pilot:[a-z0-9][a-z0-9._-]{2,95}$")
    candidate_index: int = Field(ge=0, le=1)
    seed: int = Field(ge=0, le=2**32 - 1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: TeacherGenerationStatus
    finish_reason: TeacherFinishReason
    raw_output: str | None = Field(default=None, max_length=32768)
    raw_output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    observed_token_count: int = Field(ge=0)
    error_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{2,63}$")

    @model_validator(mode="after")
    def validate_attempt(self) -> TeacherGenerationRecord:
        """Bind candidate identity, output hash, and success/failure evidence."""

        expected_id = f"{self.task_id}:candidate-{self.candidate_index}"
        if self.generation_id != expected_id:
            raise ValueError("teacher generation ID does not match task and candidate index")
        if self.status == "succeeded":
            if self.finish_reason == "error" or self.error_code is not None:
                raise ValueError("successful generation cannot carry an error")
            if self.raw_output is None or not self.raw_output.strip():
                raise ValueError("successful generation requires non-blank output")
            if self.raw_output_sha256 is None:
                raise ValueError("successful generation requires an output hash")
            actual_hash = hashlib.sha256(self.raw_output.encode("utf-8")).hexdigest()
            if actual_hash != self.raw_output_sha256:
                raise ValueError("teacher output hash does not match output content")
            if self.observed_token_count == 0:
                raise ValueError("successful generation requires observed tokens")
        else:
            if self.finish_reason != "error" or self.error_code is None:
                raise ValueError("failed generation requires error finish reason and code")
            if self.raw_output is not None or self.raw_output_sha256 is not None:
                raise ValueError("failed generation cannot retain output content")
            if self.observed_token_count != 0:
                raise ValueError("failed generation cannot report generated tokens")
        return self


class ReasoningVerifierResult(StrictSchema):
    """Content-free deterministic verification evidence for one parsed candidate."""

    schema_version: Literal["1.0"] = "1.0"
    verification_id: str = Field(
        pattern=r"^m5-reasoning:pilot:[a-z0-9][a-z0-9._-]{2,95}:verify-[01]$"
    )
    task_id: str = Field(pattern=r"^m5-reasoning:pilot:[a-z0-9][a-z0-9._-]{2,95}$")
    generation_id: str = Field(
        pattern=r"^m5-reasoning:pilot:[a-z0-9][a-z0-9._-]{2,95}:candidate-[01]$"
    )
    verifier_id: Literal["m5-json-exact-v1"]
    expected_answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool
    reason: VerifierReason

    @model_validator(mode="after")
    def validate_result(self) -> ReasoningVerifierResult:
        """Bind result identities and pass/fail semantics."""

        candidate_suffix = self.generation_id.rsplit("candidate-", maxsplit=1)[1]
        if self.verification_id != f"{self.task_id}:verify-{candidate_suffix}":
            raise ValueError("verification ID does not match generation candidate")
        if not self.generation_id.startswith(f"{self.task_id}:candidate-"):
            raise ValueError("verification generation must belong to its task")
        if self.passed != (self.reason == "accepted"):
            raise ValueError("only an accepted verifier result may pass")
        return self


class ReasoningSample(StrictSchema):
    """One accepted visible-reasoning SFT sample with complete private lineage."""

    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^m5-reasoning-sample:[a-z0-9][a-z0-9._-]{2,95}$")
    task_id: str = Field(pattern=r"^m5-reasoning:pilot:[a-z0-9][a-z0-9._-]{2,95}$")
    task_family: ReasoningTaskFamily
    language: ReasoningLanguage
    split: Literal["pilot_train"]
    template_family: str = Field(pattern=r"^pilot\.[a-z0-9][a-z0-9._-]+\.v1$")
    prompt: str = Field(min_length=1, max_length=8192)
    reasoning_content: str = Field(min_length=1, max_length=24576)
    final_answer: str = Field(min_length=1, max_length=4096)
    generation_id: str = Field(
        pattern=r"^m5-reasoning:pilot:[a-z0-9][a-z0-9._-]{2,95}:candidate-[01]$"
    )
    verification_id: str = Field(
        pattern=r"^m5-reasoning:pilot:[a-z0-9][a-z0-9._-]{2,95}:verify-[01]$"
    )
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_token_count: int = Field(gt=0, le=1024)

    @field_validator("prompt", "reasoning_content", "final_answer")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject whitespace-only accepted content."""

        if not value.strip():
            raise ValueError("accepted reasoning sample text cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_lineage(self) -> ReasoningSample:
        """Bind sample ID, generation, verification, and content hash."""

        suffix = self.task_id.removeprefix("m5-reasoning:pilot:")
        if self.id != f"m5-reasoning-sample:{suffix}":
            raise ValueError("reasoning sample ID does not match its task")
        if not self.generation_id.startswith(f"{self.task_id}:candidate-"):
            raise ValueError("reasoning sample generation must belong to its task")
        index = self.generation_id.rsplit("candidate-", maxsplit=1)[1]
        if self.verification_id != f"{self.task_id}:verify-{index}":
            raise ValueError("reasoning sample verification does not match generation")
        if hashlib.sha256(self.prompt.encode("utf-8")).hexdigest() != self.prompt_sha256:
            raise ValueError("reasoning sample prompt hash does not match prompt")
        payload = {
            "final_answer": self.final_answer,
            "prompt": self.prompt,
            "reasoning_content": self.reasoning_content,
        }
        if content_sha256(payload) != self.content_sha256:
            raise ValueError("reasoning sample content hash does not match content")
        return self


class ReasoningRejectedRecord(StrictSchema):
    """Content-free audit evidence for one rejected candidate or exhausted task."""

    schema_version: Literal["1.0"] = "1.0"
    task_id: str = Field(pattern=r"^m5-reasoning:pilot:[a-z0-9][a-z0-9._-]{2,95}$")
    generation_id: str | None = Field(
        default=None,
        pattern=r"^m5-reasoning:pilot:[a-z0-9][a-z0-9._-]{2,95}:candidate-[01]$",
    )
    stage: ReasoningRejectionStage
    reason: ReasoningRejectionReason
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    observed_token_count: int | None = Field(default=None, ge=0)
    max_sequence_length: int | None = Field(default=None, gt=1)
    verification_id: str | None = Field(
        default=None,
        pattern=r"^m5-reasoning:pilot:[a-z0-9][a-z0-9._-]{2,95}:verify-[01]$",
    )

    @model_validator(mode="after")
    def validate_rejection(self) -> ReasoningRejectedRecord:
        """Require only the evidence relevant to each failure stage."""

        if self.generation_id is not None and not self.generation_id.startswith(
            f"{self.task_id}:candidate-"
        ):
            raise ValueError("rejected generation must belong to its task")
        if self.reason == "no_candidate_passed":
            if self.stage != "selection" or self.generation_id is not None:
                raise ValueError("exhausted task rejection must be selection-level")
        elif self.generation_id is None:
            raise ValueError("candidate rejection requires a generation ID")
        if self.reason == "teacher_generation_failed" and (
            self.stage != "generation" or self.raw_output_sha256 is not None
        ):
            raise ValueError("teacher failure cannot retain an output hash")
        if self.reason == "sequence_too_long":
            if (
                self.stage != "tokenization"
                or self.observed_token_count is None
                or self.max_sequence_length is None
                or self.observed_token_count <= self.max_sequence_length
            ):
                raise ValueError("overlength rejection requires observed and maximum tokens")
        elif self.max_sequence_length is not None:
            raise ValueError("only overlength rejection may retain a maximum length")
        if self.reason in {"invalid_final_json", "verifier_failed"}:
            if self.stage != "verification" or self.verification_id is None:
                raise ValueError("verifier rejection requires verification evidence")
        elif self.verification_id is not None:
            raise ValueError("only verifier rejection may reference verification evidence")
        return self


class M5ReasoningDatasetManifest(StrictSchema):
    """Content-addressed M5 pilot lineage, counts, hashes, and selection evidence."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_name: Literal["m5-reasoning-pilot"]
    dataset_version: str = Field(pattern=r"^m5-reasoning-pilot-v1-[0-9a-f]{8}$")
    parent_dataset_version: Literal["m2-sft-v1-f82ff32e"]
    task_set_version: str = Field(pattern=r"^m5-reasoning-pilot-tasks-v1-[0-9a-f]{8}$")
    dev_task_set_version: str = Field(pattern=r"^m5-reasoning-dev-v1-[0-9a-f]{8}$")
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tasks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifications_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rejections_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contamination_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contamination_status: Literal["pass"]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_tasks: int = Field(gt=0)
    generation_attempts: int = Field(ge=0)
    successful_generations: int = Field(ge=0)
    failed_generations: int = Field(ge=0)
    verified_candidates: int = Field(ge=0)
    unused_candidates: int = Field(ge=0)
    accepted_samples: int = Field(ge=0)
    rejected_candidates: int = Field(ge=0)
    rejected_tasks: int = Field(ge=0)
    rejected_records: int = Field(ge=0)
    task_family_counts: dict[str, int]
    language_counts: dict[str, int]
    template_family_counts: dict[str, int]
    rejection_counts: dict[str, int]
    teacher: ReasoningTeacherIdentity
    sampling: ReasoningTeacherSampling
    verifier: ReasoningVerifierIdentity

    @field_validator(
        "task_family_counts", "language_counts", "template_family_counts", "rejection_counts"
    )
    @classmethod
    def validate_count_maps(cls, value: dict[str, int]) -> dict[str, int]:
        """Require stable positive sparse summaries."""

        if list(value) != sorted(value) or any(
            not key or count <= 0 for key, count in value.items()
        ):
            raise ValueError("reasoning manifest counts must use sorted keys and positive values")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> M5ReasoningDatasetManifest:
        """Keep every task, generation, verification, and rejection transition coherent."""

        if self.dataset_version != f"m5-reasoning-pilot-v1-{self.content_sha256[:8]}":
            raise ValueError("reasoning dataset version does not match content hash")
        if self.successful_generations + self.failed_generations != self.generation_attempts:
            raise ValueError("successful and failed generations must equal attempts")
        if self.accepted_samples + self.rejected_tasks != self.input_tasks:
            raise ValueError("accepted and rejected tasks must equal input tasks")
        if self.verified_candidates + self.unused_candidates > self.successful_generations:
            raise ValueError("verified and unused candidates exceed successful generations")
        if self.rejected_candidates + self.rejected_tasks != self.rejected_records:
            raise ValueError("candidate and task rejections must equal rejected records")
        if sum(self.rejection_counts.values()) != self.rejected_records:
            raise ValueError("rejection reason counts must equal rejected records")
        if not set(self.rejection_counts).issubset(REASONING_REJECTION_REASONS):
            raise ValueError("reasoning manifest contains an unknown rejection reason")
        for counts in (
            self.task_family_counts,
            self.language_counts,
            self.template_family_counts,
        ):
            if sum(counts.values()) != self.input_tasks:
                raise ValueError("reasoning task distribution must equal input tasks")
        return self


class M5TeacherSmokeResult(StrictSchema):
    """Public, path-free evidence from one real offline Qwen3 teacher invocation."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["pass", "fail"]
    generated_at: datetime
    model: ReasoningTeacherIdentity
    sampling: ReasoningTeacherSampling
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    physical_gpu_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1, max_length=128)
    torch_version: str = Field(min_length=1, max_length=64)
    transformers_version: str = Field(min_length=1, max_length=64)
    input_token_count: int = Field(gt=0)
    generated_token_counts: tuple[int, ...] = Field(min_length=1, max_length=2)
    generation_attempts: int = Field(ge=1, le=2)
    accepted_samples: int = Field(ge=0, le=1)
    rejection_counts: dict[str, int]
    dataset_version: str | None = Field(
        default=None, pattern=r"^m5-reasoning-pilot-v1-[0-9a-f]{8}$"
    )
    duration_seconds: float = Field(gt=0.0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    raw_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("generated_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        """Require timezone-aware UTC evidence timestamps."""

        offset = value.utcoffset()
        if value.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("teacher smoke timestamp must use UTC")
        return value

    @field_validator("rejection_counts")
    @classmethod
    def validate_rejection_summary(cls, value: dict[str, int]) -> dict[str, int]:
        """Require stable sparse rejection evidence."""

        if list(value) != sorted(value) or any(
            not key or count <= 0 for key, count in value.items()
        ):
            raise ValueError("teacher smoke rejection counts must be sorted and positive")
        if not set(value).issubset(REASONING_REJECTION_REASONS):
            raise ValueError("teacher smoke contains an unknown rejection reason")
        return value

    @model_validator(mode="after")
    def validate_smoke(self) -> M5TeacherSmokeResult:
        """Do not label an unverified or dirty run as a passing teacher smoke."""

        if len(self.generated_token_counts) != self.generation_attempts:
            raise ValueError("teacher smoke token counts must equal generation attempts")
        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("teacher smoke reserved memory cannot be below allocated memory")
        if self.status == "pass":
            if self.git_dirty or self.accepted_samples != 1 or self.dataset_version is None:
                raise ValueError("passing teacher smoke requires clean lineage and one sample")
        elif self.dataset_version is not None and self.accepted_samples == 0:
            raise ValueError("failed teacher smoke cannot claim an empty dataset version")
        return self
