"""Strict M6 release-evaluation, comparison, and promotion schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M6Mode = Literal["thinking", "nonthinking"]
M6Category = Literal["config", "json", "linux", "logs", "python", "refusal", "short_code"]
M6ScorerKind = Literal[
    "exact_match",
    "human_rubric",
    "json_object",
    "multiple_choice",
    "required_terms",
]
M6Task = Literal["tinyllm_arc_easy", "tinyllm_hellaswag", "tinyllm_piqa"]

EXPECTED_CATEGORY_COUNTS: dict[str, int] = {
    "config": 40,
    "json": 40,
    "linux": 45,
    "logs": 45,
    "python": 50,
    "refusal": 40,
    "short_code": 40,
}


def _freeze(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


class M6BootstrapConfig(StrictSchema):
    """Paired cluster-bootstrap policy fixed before release-set evaluation."""

    seed: Literal[20260809]
    replicates: Literal[10000]
    confidence_basis_points: Literal[9500]
    resampling_unit: Literal["bilingual-pair-or-english-singleton"]
    percentile_method: Literal["nearest-rank-v1"]


class M6GateConfig(StrictSchema):
    """Pre-registered Candidate thresholds expressed as integer basis points."""

    domain_min_delta_basis_points: Literal[300]
    domain_ci_lower_min_exclusive_basis_points: Literal[0]
    general_max_drop_basis_points: Literal[200]
    json_valid_min_basis_points: Literal[9800]
    thinking_format_min_basis_points: Literal[9900]
    thinking_forced_close_max_basis_points: Literal[1000]
    nonthinking_leakage_max_basis_points: Literal[0]
    require_both_domain_modes: Literal[True]
    require_complete_lineage: Literal[True]
    production_gate_enabled: Literal[False]


class M6ThinkingGenerationConfig(StrictSchema):
    """Pinned Qwen3 Thinking generation and bounded-controller policy."""

    template_id: Literal["qwen3-chatml-thinking-generation-v1"]
    template_sha256: Literal["7c37a1ab66f274f52208e50167e6cafbf00b6f5319207beca572e4b8cb1f8451"]
    add_generation_prompt: Literal[True]
    thinking_budget_tokens: Literal[1536]
    final_answer_max_new_tokens: Literal[512]
    do_sample: Literal[True]
    temperature: float = Field(gt=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: Literal[20]
    repetition_penalty: float = Field(ge=1.0, le=2.0)
    seed: Literal[20260809]
    early_stopping_text: Literal[
        "\n\n Considering the limited time by the user, I have to give the solution "
        "based on the thinking directly now.\n</think>\n\n"
    ]

    @model_validator(mode="after")
    def validate_sampler(self) -> M6ThinkingGenerationConfig:
        if (self.temperature, self.top_p, self.repetition_penalty) != (0.6, 0.95, 1.0):
            raise ValueError("M6 Thinking sampler differs from the frozen Qwen3 policy")
        return self


class M6NonthinkingGenerationConfig(StrictSchema):
    """Pinned Qwen3 Non-thinking greedy-generation policy."""

    template_id: Literal["qwen3-chatml-nonthinking-generation-v1"]
    template_sha256: Literal["b9a510e2f016a112860e47056f770b04e5c93131cc4a8ecd47fcc950cfdb6273"]
    add_generation_prompt: Literal[True]
    max_new_tokens: Literal[512]
    do_sample: Literal[False]


class M6DomainExecutionConfig(StrictSchema):
    """Complete generation/scoring policy for the frozen domain suite."""

    batch_size: Literal[4]
    max_sequence_length: Literal[1024]
    scorer_policy: Literal["tinyllm-domain-scorer-v1"]
    thinking: M6ThinkingGenerationConfig
    nonthinking: M6NonthinkingGenerationConfig


class M6GeneralTaskConfig(StrictSchema):
    """Pinned lm-eval Task Adapter and immutable dataset identity."""

    task: M6Task
    adapter_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    expected_samples: int = Field(gt=0)


class M6GeneralExecutionConfig(StrictSchema):
    """Pinned Non-thinking lm-eval execution policy."""

    harness_version: Literal["0.4.12"]
    tokenizer_chat_template_sha256: Literal[
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    ]
    apply_chat_template: Literal[True]
    enable_thinking: Literal[False]
    num_fewshot: Literal[0]
    batch_size: Literal[8]
    max_length: Literal[1024]
    log_samples: Literal[True]
    tasks: tuple[M6GeneralTaskConfig, M6GeneralTaskConfig, M6GeneralTaskConfig]

    @field_validator("tasks", mode="before")
    @classmethod
    def freeze_tasks(cls, value: object) -> object:
        return _freeze(value)

    @model_validator(mode="after")
    def validate_tasks(self) -> M6GeneralExecutionConfig:
        expected = (
            (
                "tinyllm_arc_easy",
                "e292c7593040de3e0997b0f9854f2feb9531aec13d16dddf017eb62a035bba69",
                "210d026faf9955653af8916fad021475a3f00453",
                2376,
            ),
            (
                "tinyllm_hellaswag",
                "9e325da13e9028d2d2a4fa0bd9e0ff46b6a1a7d4849e3d005be17a3265415971",
                "218ec52e09a7e7462a5400043bb9a69a41d06b76",
                10042,
            ),
            (
                "tinyllm_piqa",
                "56f8c1f15f5cb66d58a5c5c48a46a946b9ad55c8a8c1a608d996511946f3ac50",
                "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
                1838,
            ),
        )
        actual = tuple(
            (task.task, task.adapter_sha256, task.dataset_revision, task.expected_samples)
            for task in self.tasks
        )
        if actual != expected:
            raise ValueError("M6 general execution tasks differ from the frozen inputs")
        return self


class M6ReleaseConfig(StrictSchema):
    """Frozen M6 comparison policy, independent from model outputs."""

    schema_version: Literal["1.0"] = "1.0"
    protocol_version: Literal["m6-release-v1"]
    suite_version: Literal["tinyllm-domain-v1-83bdd8ef"]
    suite_content_sha256: Literal[
        "83bdd8ef24dfa2bae0a997570594e7243f81ec3891a420458dd29b10f5e7af27"
    ]
    expected_domain_items: Literal[300]
    expected_languages: Literal["en-210_zh-90"]
    compare_modes_separately: Literal[True]
    general_metric: Literal["acc_norm"]
    general_aggregation: Literal["equal-task-mean"]
    general_tasks: tuple[M6Task, M6Task, M6Task]
    domain_execution: M6DomainExecutionConfig
    general_execution: M6GeneralExecutionConfig
    bootstrap: M6BootstrapConfig
    gate: M6GateConfig

    @field_validator("general_tasks", mode="before")
    @classmethod
    def freeze_tasks(cls, value: object) -> object:
        return _freeze(value)

    @model_validator(mode="after")
    def validate_tasks(self) -> M6ReleaseConfig:
        expected = ("tinyllm_arc_easy", "tinyllm_hellaswag", "tinyllm_piqa")
        if self.general_tasks != expected:
            raise ValueError("M6 general tasks must use the fixed order")
        return self


class M6ModelIdentity(StrictSchema):
    """Base or trained-model identity used by one M6 evaluation."""

    role: Literal["base", "candidate"]
    repository: Literal["Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"]
    base_revision: Literal[
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "b968826d9c46dd6066d109eabc6255188de91218",
    ]
    attention_architecture: Literal["gqa"]
    adaptation: Literal["base", "full_sft", "lora"]
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    model_parameters: int = Field(gt=0)
    training_run_id: str | None = Field(default=None, min_length=1, max_length=180)
    training_checkpoint_id: str | None = Field(
        default=None,
        pattern=r"^checkpoint-tokens-[0-9]{10}$",
    )
    training_tokens: int | None = Field(default=None, gt=0, le=100_000_000)
    training_config_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    dataset_version: str | None = Field(default=None, min_length=1, max_length=128)
    dataset_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    adapter_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_identity(self) -> M6ModelIdentity:
        revisions = {
            "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
            "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
        }
        if revisions[self.repository] != self.base_revision:
            raise ValueError("M6 repository and base Revision differ")
        training_fields = (
            self.training_run_id,
            self.training_checkpoint_id,
            self.training_tokens,
            self.training_config_sha256,
            self.dataset_version,
            self.dataset_manifest_sha256,
        )
        if self.role == "base":
            if self.adaptation != "base" or any(value is not None for value in training_fields):
                raise ValueError("M6 Base cannot claim training lineage")
            if self.adapter_sha256 is not None:
                raise ValueError("M6 Base cannot claim an Adapter")
        else:
            if self.adaptation == "base" or any(value is None for value in training_fields):
                raise ValueError("M6 Candidate requires complete training lineage")
            if (self.adaptation == "lora") != (self.adapter_sha256 is not None):
                raise ValueError("M6 LoRA identity requires exactly one Adapter hash")
        return self


class M6DomainItemScore(StrictSchema):
    """Content-free private score for one frozen domain item."""

    item_id: str = Field(
        pattern=r"^domain-(config|json|linux|logs|python|refusal|short-code)-[0-9]{3}$"
    )
    cluster_id: str = Field(
        pattern=(
            r"^(pair:(config|json|linux|logs|python|refusal|short_code):[0-9]{3}|"
            r"singleton:domain-(config|json|linux|logs|python|refusal|short-code)-[0-9]{3})$"
        )
    )
    language: Literal["en", "zh"]
    category: M6Category
    scorer_kind: M6ScorerKind
    correct: bool
    json_valid: bool | None
    format_valid: bool
    visible_reasoning_leakage: bool

    @model_validator(mode="after")
    def validate_score(self) -> M6DomainItemScore:
        if not self.item_id.startswith(f"domain-{self.category.replace('_', '-')}-"):
            raise ValueError("M6 item ID and category differ")
        if (self.scorer_kind == "json_object") != (self.json_valid is not None):
            raise ValueError("M6 JSON validity must exist exactly for JSON-object scorers")
        if self.correct and not self.format_valid:
            raise ValueError("M6 correct items require a valid output format")
        if self.cluster_id.startswith("pair:"):
            cluster_category = self.cluster_id.split(":", 2)[1]
            if cluster_category != self.category:
                raise ValueError("M6 pair cluster and item category differ")
        elif self.cluster_id != f"singleton:{self.item_id}" or self.language != "en":
            raise ValueError("M6 singletons must be their English item identity")
        return self


class M6DomainModeResult(StrictSchema):
    """Exactly one Thinking or Non-thinking pass over the 300-item suite."""

    mode: M6Mode
    items: tuple[M6DomainItemScore, ...]
    evaluated_items: Literal[300]
    correct_items: int = Field(ge=0, le=300)
    score_basis_points: int = Field(ge=0, le=10000)
    format_valid_items: int = Field(ge=0, le=300)
    format_valid_basis_points: int = Field(ge=0, le=10000)
    json_items: Literal[80]
    json_valid_items: int = Field(ge=0, le=80)
    json_valid_basis_points: int = Field(ge=0, le=10000)
    visible_reasoning_leakage_items: int = Field(ge=0, le=300)
    visible_reasoning_leakage_basis_points: int = Field(ge=0, le=10000)
    natural_thinking_closed_items: int = Field(ge=0, le=300)
    budget_forced_close_items: int = Field(ge=0, le=300)
    forced_close_basis_points: int = Field(ge=0, le=10000)
    generated_tokens: int = Field(ge=0)
    injected_tokens: int = Field(ge=0)

    @field_validator("items", mode="before")
    @classmethod
    def freeze_items(cls, value: object) -> object:
        return _freeze(value)

    @model_validator(mode="after")
    def validate_mode_result(self) -> M6DomainModeResult:
        if len(self.items) != 300:
            raise ValueError("M6 mode result requires exactly 300 domain items")
        item_ids = tuple(item.item_id for item in self.items)
        if item_ids != tuple(sorted(item_ids)) or len(set(item_ids)) != 300:
            raise ValueError("M6 item scores must be unique and sorted by item ID")
        language_counts = {
            language: sum(item.language == language for item in self.items)
            for language in ("en", "zh")
        }
        if language_counts != {"en": 210, "zh": 90}:
            raise ValueError("M6 language counts differ from the frozen suite")
        category_counts = {
            category: sum(item.category == category for item in self.items)
            for category in EXPECTED_CATEGORY_COUNTS
        }
        if category_counts != EXPECTED_CATEGORY_COUNTS:
            raise ValueError("M6 category counts differ from the frozen suite")
        clusters: dict[str, list[M6DomainItemScore]] = {}
        for item in self.items:
            clusters.setdefault(item.cluster_id, []).append(item)
        pair_clusters = [items for key, items in clusters.items() if key.startswith("pair:")]
        singleton_clusters = [
            items for key, items in clusters.items() if key.startswith("singleton:")
        ]
        if len(pair_clusters) != 90 or len(singleton_clusters) != 120 or len(clusters) != 210:
            raise ValueError("M6 cluster counts differ from the bilingual-pair contract")
        if any(
            len(cluster) != 2
            or {item.language for item in cluster} != {"en", "zh"}
            or len({item.category for item in cluster}) != 1
            for cluster in pair_clusters
        ):
            raise ValueError("M6 bilingual clusters must contain one EN/ZH pair in one category")
        if any(len(cluster) != 1 for cluster in singleton_clusters):
            raise ValueError("M6 singleton clusters must contain exactly one item")

        correct = sum(item.correct for item in self.items)
        format_valid = sum(item.format_valid for item in self.items)
        json_items = [item for item in self.items if item.scorer_kind == "json_object"]
        json_valid = sum(item.json_valid is True for item in json_items)
        leakage = sum(item.visible_reasoning_leakage for item in self.items)
        if (
            self.correct_items != correct
            or self.score_basis_points != round(correct * 10000 / 300)
            or self.format_valid_items != format_valid
            or self.format_valid_basis_points != round(format_valid * 10000 / 300)
            or len(json_items) != self.json_items
            or self.json_valid_items != json_valid
            or self.json_valid_basis_points != round(json_valid * 10000 / 80)
            or self.visible_reasoning_leakage_items != leakage
            or self.visible_reasoning_leakage_basis_points != round(leakage * 10000 / 300)
            or self.forced_close_basis_points != round(self.budget_forced_close_items * 10000 / 300)
        ):
            raise ValueError("M6 mode aggregates differ from item-level evidence")
        if self.mode == "thinking":
            if self.natural_thinking_closed_items + self.budget_forced_close_items != 300:
                raise ValueError("M6 Thinking close paths must cover every item")
        elif (
            self.natural_thinking_closed_items
            or self.budget_forced_close_items
            or self.forced_close_basis_points
            or self.injected_tokens
        ):
            raise ValueError("M6 Non-thinking mode cannot claim controller activity")
        return self


class M6GeneralTaskResult(StrictSchema):
    """One full pinned lm-eval aggregate used for M6 regression checks."""

    task: M6Task
    samples: int = Field(gt=0)
    acc: float = Field(ge=0.0, le=1.0)
    acc_stderr: float = Field(ge=0.0, le=1.0)
    acc_norm: float = Field(ge=0.0, le=1.0)
    acc_norm_stderr: float = Field(ge=0.0, le=1.0)


class M6GeneralResult(StrictSchema):
    """Fixed equal-task mean over ARC-Easy, HellaSwag, and PIQA."""

    harness_version: Literal["0.4.12"]
    metric: Literal["acc_norm"]
    aggregation: Literal["equal-task-mean"]
    tasks: tuple[M6GeneralTaskResult, M6GeneralTaskResult, M6GeneralTaskResult]
    aggregate_basis_points: int = Field(ge=0, le=10000)

    @field_validator("tasks", mode="before")
    @classmethod
    def freeze_results(cls, value: object) -> object:
        return _freeze(value)

    @model_validator(mode="after")
    def validate_general(self) -> M6GeneralResult:
        expected = ("tinyllm_arc_easy", "tinyllm_hellaswag", "tinyllm_piqa")
        if tuple(task.task for task in self.tasks) != expected:
            raise ValueError("M6 general results must use the fixed task order")
        expected_samples = (2376, 10042, 1838)
        if tuple(task.samples for task in self.tasks) != expected_samples:
            raise ValueError("M6 general sample counts differ from the frozen tasks")
        aggregate = round(sum(task.acc_norm for task in self.tasks) * 10000 / 3)
        if self.aggregate_basis_points != aggregate:
            raise ValueError("M6 general aggregate differs from equal-task acc_norm mean")
        return self


class M6EvaluationResult(StrictSchema):
    """Complete private M6 evaluation for one Base or Candidate."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded"]
    evaluation_id: str = Field(min_length=1, max_length=180)
    protocol_version: Literal["m6-release-v1"]
    suite_version: Literal["tinyllm-domain-v1-83bdd8ef"]
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    model: M6ModelIdentity
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    thinking_template_sha256: str = Field(pattern=SHA256_PATTERN)
    nonthinking_template_sha256: str = Field(pattern=SHA256_PATTERN)
    general_chat_template_sha256: str = Field(pattern=SHA256_PATTERN)
    software_environment_sha256: str = Field(pattern=SHA256_PATTERN)
    hardware_sha256: str = Field(pattern=SHA256_PATTERN)
    domain_modes: tuple[M6DomainModeResult, M6DomainModeResult]
    general: M6GeneralResult
    human_review_complete: bool
    thinking_human_review_sha256: str = Field(pattern=SHA256_PATTERN)
    nonthinking_human_review_sha256: str = Field(pattern=SHA256_PATTERN)
    lineage_complete: bool
    raw_domain_results_sha256: str = Field(pattern=SHA256_PATTERN)
    raw_general_results_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("domain_modes", mode="before")
    @classmethod
    def freeze_modes(cls, value: object) -> object:
        return _freeze(value)

    @model_validator(mode="after")
    def validate_evaluation(self) -> M6EvaluationResult:
        if tuple(mode.mode for mode in self.domain_modes) != ("thinking", "nonthinking"):
            raise ValueError("M6 domain modes must be Thinking then Non-thinking")
        thinking_identity = tuple(
            (item.item_id, item.cluster_id) for item in self.domain_modes[0].items
        )
        nonthinking_identity = tuple(
            (item.item_id, item.cluster_id) for item in self.domain_modes[1].items
        )
        if thinking_identity != nonthinking_identity:
            raise ValueError("M6 modes must evaluate identical item and cluster identities")
        return self


class M6BootstrapInterval(StrictSchema):
    """Deterministic paired cluster-bootstrap interval in basis points."""

    replicates: Literal[10000]
    confidence_basis_points: Literal[9500]
    point_delta_basis_points: int = Field(ge=-10000, le=10000)
    lower_basis_points: int = Field(ge=-10000, le=10000)
    upper_basis_points: int = Field(ge=-10000, le=10000)

    @model_validator(mode="after")
    def validate_interval(self) -> M6BootstrapInterval:
        if self.lower_basis_points > self.upper_basis_points:
            raise ValueError("M6 Bootstrap interval bounds are reversed")
        return self


class M6ModeComparison(StrictSchema):
    """Base/Candidate domain comparison for one explicit generation mode."""

    mode: M6Mode
    base_score_basis_points: int = Field(ge=0, le=10000)
    candidate_score_basis_points: int = Field(ge=0, le=10000)
    delta_basis_points: int = Field(ge=-10000, le=10000)
    bootstrap: M6BootstrapInterval

    @model_validator(mode="after")
    def validate_delta(self) -> M6ModeComparison:
        if self.delta_basis_points != (
            self.candidate_score_basis_points - self.base_score_basis_points
        ):
            raise ValueError("M6 mode delta differs from aggregate scores")
        if self.bootstrap.point_delta_basis_points != self.delta_basis_points:
            raise ValueError("M6 Bootstrap point estimate differs from mode delta")
        return self


class M6GeneralComparison(StrictSchema):
    """Equal-task general-regression comparison."""

    metric: Literal["acc_norm"]
    aggregation: Literal["equal-task-mean"]
    base_basis_points: int = Field(ge=0, le=10000)
    candidate_basis_points: int = Field(ge=0, le=10000)
    delta_basis_points: int = Field(ge=-10000, le=10000)

    @model_validator(mode="after")
    def validate_delta(self) -> M6GeneralComparison:
        if self.delta_basis_points != self.candidate_basis_points - self.base_basis_points:
            raise ValueError("M6 general delta differs from aggregate scores")
        return self


M6GateName = Literal[
    "thinking_domain_delta",
    "thinking_domain_ci",
    "nonthinking_domain_delta",
    "nonthinking_domain_ci",
    "general_regression",
    "json_valid_rate",
    "thinking_format",
    "thinking_forced_close",
    "nonthinking_leakage",
    "evaluation_integrity",
    "lineage",
]


class M6GateCheck(StrictSchema):
    """One transparent Promotion Gate predicate."""

    name: M6GateName
    passed: bool
    actual_basis_points: int | None = Field(default=None, ge=-10000, le=10000)
    threshold_basis_points: int | None = Field(default=None, ge=-10000, le=10000)
    comparison: Literal["gte", "gt", "lte", "boolean"]
    detail: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_check(self) -> M6GateCheck:
        numeric = self.comparison != "boolean"
        if numeric != (self.actual_basis_points is not None):
            raise ValueError("M6 numeric Gate checks require an actual value")
        if numeric != (self.threshold_basis_points is not None):
            raise ValueError("M6 numeric Gate checks require a threshold")
        return self


class M6ComparisonResult(StrictSchema):
    """Path-free Base/Candidate comparison and Candidate decision."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["accepted", "rejected"]
    protocol_version: Literal["m6-release-v1"]
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    base_evaluation_id: str = Field(min_length=1, max_length=180)
    base_evaluation_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_evaluation_id: str = Field(min_length=1, max_length=180)
    candidate_evaluation_sha256: str = Field(pattern=SHA256_PATTERN)
    base_model: M6ModelIdentity
    candidate_model: M6ModelIdentity
    mode_comparisons: tuple[M6ModeComparison, M6ModeComparison]
    general_comparison: M6GeneralComparison
    checks: tuple[M6GateCheck, ...]
    candidate_eligible: bool
    production_eligible: Literal[False]

    @field_validator("mode_comparisons", "checks", mode="before")
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return _freeze(value)

    @model_validator(mode="after")
    def validate_decision(self) -> M6ComparisonResult:
        if self.base_model.role != "base" or self.candidate_model.role != "candidate":
            raise ValueError("M6 comparison requires Base then Candidate identities")
        if tuple(result.mode for result in self.mode_comparisons) != (
            "thinking",
            "nonthinking",
        ):
            raise ValueError("M6 comparisons must preserve mode order")
        expected_checks = (
            "thinking_domain_delta",
            "thinking_domain_ci",
            "nonthinking_domain_delta",
            "nonthinking_domain_ci",
            "general_regression",
            "json_valid_rate",
            "thinking_format",
            "thinking_forced_close",
            "nonthinking_leakage",
            "evaluation_integrity",
            "lineage",
        )
        if tuple(check.name for check in self.checks) != expected_checks:
            raise ValueError("M6 Gate checks must use the frozen order")
        accepted = all(check.passed for check in self.checks)
        if self.candidate_eligible != accepted or self.status != (
            "accepted" if accepted else "rejected"
        ):
            raise ValueError("M6 Candidate decision differs from Gate checks")
        return self


class M6PromotionRecord(StrictSchema):
    """Immutable registry record for an M6 Candidate, never Production."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["Candidate"]
    model_version: str = Field(pattern=r"^qwen3-(0-6b|8b)-m6-[0-9a-f]{8}$")
    promoted_at: datetime
    comparison_sha256: str = Field(pattern=SHA256_PATTERN)
    comparison_config_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_evaluation_id: str = Field(min_length=1, max_length=180)
    candidate_evaluation_sha256: str = Field(pattern=SHA256_PATTERN)
    model: M6ModelIdentity
    production_eligible: Literal[False]

    @model_validator(mode="after")
    def validate_record(self) -> M6PromotionRecord:
        if self.promoted_at.tzinfo is None:
            raise ValueError("M6 promotion timestamp must be timezone-aware")
        if self.model.role != "candidate":
            raise ValueError("M6 promotion requires a Candidate model identity")
        expected_size = "0-6b" if self.model.repository.endswith("0.6B") else "8b"
        if not self.model_version.startswith(f"qwen3-{expected_size}-m6-"):
            raise ValueError("M6 model version and repository differ")
        return self
