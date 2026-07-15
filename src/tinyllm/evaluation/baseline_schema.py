"""Strict M2.4c model Baseline configuration and result schemas."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema

BaselineMode = Literal["formal", "smoke"]
BaselineScorerKind = Literal[
    "exact_match",
    "human_rubric",
    "json_object",
    "multiple_choice",
    "required_terms",
]


def _freeze_array(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("Baseline path must be safe and repository-relative")
    return value


class ModelFileIdentity(StrictSchema):
    """One required file in the immutable local Qwen model snapshot."""

    filename: str = Field(pattern=r"^[a-z0-9_.-]+$")
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BaselineModelIdentity(StrictSchema):
    """Pinned model and safe-loading policy for the pre-training Baseline."""

    repository: Literal["Qwen/Qwen3-0.6B"]
    revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    license: Literal["Apache-2.0"]
    files: tuple[ModelFileIdentity, ...]
    dtype: Literal["bfloat16"]
    attention_implementation: Literal["sdpa"]
    trust_remote_code: Literal[False]
    local_files_only: Literal[True]

    @field_validator("files", mode="before")
    @classmethod
    def freeze_files(cls, value: object) -> object:
        """Convert YAML arrays to an immutable artifact tuple."""

        return _freeze_array(value)

    @model_validator(mode="after")
    def validate_files(self) -> BaselineModelIdentity:
        """Require every runtime model file exactly once in stable order."""

        names = tuple(item.filename for item in self.files)
        expected = (
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        if names != expected:
            raise ValueError("Baseline model files must contain the fixed snapshot in name order")
        return self


class BaselineSoftwareIdentity(StrictSchema):
    """Direct package versions that define the Baseline implementation."""

    torch: Literal["2.7.1+cu118"]
    transformers: Literal["4.57.6"]
    tokenizers: Literal["0.22.2"]
    accelerate: Literal["1.12.0"]
    datasets: Literal["4.8.5"]
    lm_eval: Literal["0.4.12"]
    safetensors: Literal["0.6.2"]


class GenerationTemplateIdentity(StrictSchema):
    """Frozen prompt renderer used only for model generation."""

    template_id: Literal["qwen3-chatml-nonthinking-generation-v1"]
    template_sha256: Literal["b9a510e2f016a112860e47056f770b04e5c93131cc4a8ecd47fcc950cfdb6273"]
    thinking: Literal[False]
    add_generation_prompt: Literal[True]


class BaselineSeeds(StrictSchema):
    """All random seeds accepted by lm-eval and Transformers."""

    python: int = Field(ge=0, le=2**32 - 1)
    numpy: int = Field(ge=0, le=2**32 - 1)
    torch: int = Field(ge=0, le=2**32 - 1)
    fewshot: int = Field(ge=0, le=2**32 - 1)


class DomainBaselineProtocol(StrictSchema):
    """Frozen generation and scoring policy for the 300-item suite."""

    suite_version: Literal["tinyllm-domain-v1-83bdd8ef"]
    content_sha256: Literal["83bdd8ef24dfa2bae0a997570594e7243f81ec3891a420458dd29b10f5e7af27"]
    items_path: str
    expected_items: Literal[300]
    batch_size: int = Field(gt=0, le=64)
    max_sequence_length: Literal[1024]
    max_new_tokens: Literal[512]
    do_sample: Literal[False]
    scorer_policy: Literal["tinyllm-domain-scorer-v1"]
    limit: int | None = Field(default=None, gt=0, le=300)

    @field_validator("items_path")
    @classmethod
    def validate_items_path(cls, value: str) -> str:
        """Keep the content identity independent of checkout location."""

        return _validate_relative_path(value)


class GeneralTaskIdentity(StrictSchema):
    """One pinned lm-eval Adapter and Hub dataset revision."""

    task: Literal["tinyllm_arc_easy", "tinyllm_hellaswag", "tinyllm_piqa"]
    adapter_path: str
    adapter_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    split: Literal["test", "validation"]
    expected_samples: int = Field(gt=0)
    dataset_license: Literal["cc-by-sa-4.0", "not-declared-by-hub-mirror"]
    metrics: tuple[Literal["acc", "acc_norm"], ...]

    @field_validator("metrics", mode="before")
    @classmethod
    def freeze_metrics(cls, value: object) -> object:
        """Convert YAML arrays to an immutable metric tuple."""

        return _freeze_array(value)

    @field_validator("adapter_path")
    @classmethod
    def validate_adapter_path(cls, value: str) -> str:
        """Keep task definitions inside the repository."""

        return _validate_relative_path(value)

    @model_validator(mode="after")
    def validate_metrics(self) -> GeneralTaskIdentity:
        """Require the two published metrics in stable order."""

        if self.metrics != ("acc", "acc_norm"):
            raise ValueError("general task metrics must be acc and acc_norm in order")
        return self


class GeneralBaselineProtocol(StrictSchema):
    """Frozen lm-eval execution policy for general regression tasks."""

    harness_version: Literal["0.4.12"]
    include_path: str
    tasks: tuple[GeneralTaskIdentity, ...]
    task_validation: Literal[True]
    tokenizer_chat_template_sha256: Literal[
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    ]
    apply_chat_template: Literal[True]
    enable_thinking: Literal[False]
    num_fewshot: Literal[0]
    batch_size: int = Field(gt=0, le=64)
    max_length: Literal[1024]
    log_samples: Literal[True]
    limit: int | None = Field(default=None, gt=0)

    @field_validator("tasks", mode="before")
    @classmethod
    def freeze_tasks(cls, value: object) -> object:
        """Convert YAML arrays to immutable task identities."""

        return _freeze_array(value)

    @field_validator("include_path")
    @classmethod
    def validate_include_path(cls, value: str) -> str:
        """Keep external Task YAML loading repository-relative."""

        return _validate_relative_path(value)

    @model_validator(mode="after")
    def validate_tasks(self) -> GeneralBaselineProtocol:
        """Require all three tasks once in stable order."""

        names = tuple(item.task for item in self.tasks)
        expected = ("tinyllm_arc_easy", "tinyllm_hellaswag", "tinyllm_piqa")
        if names != expected:
            raise ValueError("general Baseline must contain all fixed tasks in name order")
        return self


class BaselineRunConfig(StrictSchema):
    """Complete formal or bounded-Smoke M2.4c Baseline contract."""

    schema_version: Literal["1.0"] = "1.0"
    mode: BaselineMode
    run_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    model: BaselineModelIdentity
    software: BaselineSoftwareIdentity
    generation_template: GenerationTemplateIdentity
    seeds: BaselineSeeds
    domain: DomainBaselineProtocol
    general: GeneralBaselineProtocol

    @model_validator(mode="after")
    def validate_mode(self) -> BaselineRunConfig:
        """Forbid accidental partial execution under the formal mode."""

        if self.mode == "formal" and (
            self.domain.limit is not None or self.general.limit is not None
        ):
            raise ValueError("formal Baseline cannot set Domain or general-task limits")
        if self.mode == "smoke" and (self.domain.limit is None or self.general.limit is None):
            raise ValueError("Smoke Baseline must bound Domain and general-task samples")
        return self


class DomainItemResult(StrictSchema):
    """Private raw model output plus deterministic automatic-scoring state."""

    schema_version: Literal["1.0"] = "1.0"
    item_id: str = Field(
        pattern=r"^domain-(config|json|linux|logs|python|refusal|short-code)-[0-9]{3}$"
    )
    scorer_kind: BaselineScorerKind
    response: str = Field(max_length=131_072)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_tokens: int = Field(gt=0)
    generated_tokens: int = Field(ge=0)
    finish_reason: Literal["eos", "length"]
    automatic_correct: bool | None
    json_valid: bool | None
    human_review_required: bool

    @model_validator(mode="after")
    def validate_result(self) -> DomainItemResult:
        """Bind response identity and scoring state to the scorer type."""

        if hashlib.sha256(self.response.encode()).hexdigest() != self.response_sha256:
            raise ValueError("Domain response SHA256 does not match response text")
        is_human = self.scorer_kind == "human_rubric"
        if self.human_review_required != is_human:
            raise ValueError("human-review flag does not match scorer kind")
        if is_human and self.automatic_correct is not None:
            raise ValueError("human-rubric item cannot have an automatic score")
        if not is_human and self.automatic_correct is None:
            raise ValueError("objective item must have an automatic score")
        if (self.scorer_kind == "json_object") != (self.json_valid is not None):
            raise ValueError("JSON-valid state must exist only for JSON-object items")
        return self


class HumanRubricJudgment(StrictSchema):
    """Maintainer decision for one evidence-grounded refusal output."""

    schema_version: Literal["1.0"] = "1.0"
    item_id: str = Field(pattern=r"^domain-refusal-[0-9]{3}$")
    criterion_results: tuple[bool, bool, bool]
    passed: bool
    rationale: str = Field(min_length=1, max_length=4096)
    reviewer_role: Literal["maintainer"]

    @field_validator("criterion_results", mode="before")
    @classmethod
    def freeze_results(cls, value: object) -> object:
        """Convert JSON arrays to an immutable criterion tuple."""

        return _freeze_array(value)

    @model_validator(mode="after")
    def validate_passed(self) -> HumanRubricJudgment:
        """The frozen three-of-three rubric passes only when every criterion passes."""

        if self.passed != all(self.criterion_results):
            raise ValueError("human-rubric passed state must equal all three criteria")
        return self


class HumanReviewCommit(StrictSchema):
    """Completion marker for one atomically published private Judgment set."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(
        pattern=(r"^\d{8}T\d{6}Z-[a-z0-9]+(?:-[a-z0-9]+)*-[0-9a-f]{8}-[0-9a-f]{4}$")
    )
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    committed_at: datetime
    judgment_count: int = Field(gt=0, le=40)
    judgments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_timestamp(self) -> HumanReviewCommit:
        """Require an unambiguous completion time."""

        if self.committed_at.tzinfo is None:
            raise ValueError("human-review commit timestamp must be timezone-aware")
        return self


class DomainBaselineSummary(StrictSchema):
    """Path-free aggregate over generated Domain responses."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["complete", "awaiting_human_review"]
    suite_version: Literal["tinyllm-domain-v1-83bdd8ef"]
    evaluated_items: int = Field(gt=0, le=300)
    objective_items: int = Field(ge=0, le=300)
    objective_correct: int = Field(ge=0, le=300)
    human_review_pending: int = Field(ge=0, le=40)
    human_reviewed: int = Field(ge=0, le=40)
    human_passed: int = Field(ge=0, le=40)
    json_items: int = Field(ge=0, le=80)
    json_valid: int = Field(ge=0, le=80)

    @model_validator(mode="after")
    def validate_counts(self) -> DomainBaselineSummary:
        """Keep every aggregate internally consistent."""

        if (
            self.objective_items + self.human_review_pending + self.human_reviewed
            != self.evaluated_items
        ):
            raise ValueError(
                "Domain objective, pending, and reviewed counts must equal evaluated items"
            )
        if self.objective_correct > self.objective_items:
            raise ValueError("Domain correct count exceeds objective items")
        if self.json_valid > self.json_items:
            raise ValueError("Domain valid-JSON count exceeds JSON items")
        if self.human_passed > self.human_reviewed:
            raise ValueError("Domain human-pass count exceeds reviewed items")
        expected_status = "awaiting_human_review" if self.human_review_pending else "complete"
        if self.status != expected_status:
            raise ValueError("Domain status does not match pending human review")
        return self


class GeneralTaskResult(StrictSchema):
    """Path-free lm-eval aggregate for one pinned general-regression task."""

    task: Literal["tinyllm_arc_easy", "tinyllm_hellaswag", "tinyllm_piqa"]
    samples: int = Field(gt=0)
    acc: float = Field(ge=0.0, le=1.0)
    acc_stderr: float = Field(ge=0.0, le=1.0)
    acc_norm: float = Field(ge=0.0, le=1.0)
    acc_norm_stderr: float = Field(ge=0.0, le=1.0)


class GeneralBaselineSummary(StrictSchema):
    """Verified aggregates parsed from a private lm-eval result directory."""

    schema_version: Literal["1.0"] = "1.0"
    harness_version: Literal["0.4.12"]
    model_parameters: int = Field(gt=0)
    tasks: tuple[GeneralTaskResult, GeneralTaskResult, GeneralTaskResult]
    evaluation_seconds: float = Field(gt=0.0)

    @field_validator("tasks", mode="before")
    @classmethod
    def freeze_results(cls, value: object) -> object:
        """Convert decoded JSON arrays to an immutable result tuple."""

        return _freeze_array(value)

    @model_validator(mode="after")
    def validate_task_order(self) -> GeneralBaselineSummary:
        """Require all three aggregates once in the frozen order."""

        expected = ("tinyllm_arc_easy", "tinyllm_hellaswag", "tinyllm_piqa")
        if tuple(result.task for result in self.tasks) != expected:
            raise ValueError("general Baseline results must contain all tasks in order")
        return self


class BaselineEvaluationResult(StrictSchema):
    """Stable path-free result returned by the Baseline CLI."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded", "awaiting_human_review"]
    mode: BaselineMode
    run_id: str = Field(
        pattern=(r"^\d{8}T\d{6}Z-[a-z0-9]+(?:-[a-z0-9]+)*-[0-9a-f]{8}-[0-9a-f]{4}$")
    )
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    model_repository: Literal["Qwen/Qwen3-0.6B"]
    model_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    domain: DomainBaselineSummary
    general: GeneralBaselineSummary

    @model_validator(mode="after")
    def validate_status(self) -> BaselineEvaluationResult:
        """Keep overall completion state aligned with the human-review queue."""

        expected = (
            "awaiting_human_review"
            if self.domain.status == "awaiting_human_review"
            else "succeeded"
        )
        if self.status != expected:
            raise ValueError("Baseline status does not match Domain status")
        return self
