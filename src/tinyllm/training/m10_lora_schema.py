"""Strict contracts for staged M10 Qwen3-8B Agent LoRA."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.evaluation.m6_schema import M6GeneralResult, M6ModelIdentity
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN
from tinyllm.training.m10_sft_schema import (
    M10_DATASET_MANIFEST_SHA256,
    M10_DATASET_VERSION,
    M10_STAGE_TOKENS,
)

M10_LORA_MODEL_REVISION: Literal["b968826d9c46dd6066d109eabc6255188de91218"] = (
    "b968826d9c46dd6066d109eabc6255188de91218"
)
M10_LORA_PARENT_SUBJECT: Literal["qwen3-8b-m9-base-90587dd6"] = "qwen3-8b-m9-base-90587dd6"
M10_LORA_PARENT_RECORD_SHA256: Literal[
    "9f72bba28bcfaed45f116080033cb9bc83be1632570e71623f2a5684350261d8"
] = "9f72bba28bcfaed45f116080033cb9bc83be1632570e71623f2a5684350261d8"
M10_LORA_PARENT_MODEL_SHA256: Literal[
    "81fec43ab8b1f03a158e39e50ec23d99cf8701144e8678aea3ca656d12d08de0"
] = "81fec43ab8b1f03a158e39e50ec23d99cf8701144e8678aea3ca656d12d08de0"
M10_LORA_PARENT_TOKENIZER_SHA256: Literal[
    "99d4d297ece6cc43fa551987701e4ded4fa5c860d9448965d212508519bfc382"
] = "99d4d297ece6cc43fa551987701e4ded4fa5c860d9448965d212508519bfc382"
M10_LORA_TARGET_MODULES: tuple[
    Literal["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
    ...,
] = (
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
)


class M10LoRARunConfig(StrictSchema):
    """Stable identity for the single seeded 8B Agent LoRA campaign."""

    name: Literal[
        "m10-agent-lora-qwen3-8b-seed42",
        "m10-5-agent-repair-lora-qwen3-8b-seed42",
        "m10-5-agent-repair-v3-lora-qwen3-8b-seed42",
    ]
    seed: Literal[42]
    purpose: Literal["agent_lora"]


class M10LoRAAdapterConfig(StrictSchema):
    """Frozen PEFT topology."""

    rank: Literal[16]
    alpha: Literal[32]
    dropout: float = Field(ge=0.0, lt=1.0)
    bias: Literal["none"]
    target_modules: tuple[
        Literal["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
        ...,
    ]

    @field_validator("target_modules", mode="before")
    @classmethod
    def freeze_modules(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_modules(self) -> M10LoRAAdapterConfig:
        if self.target_modules != M10_LORA_TARGET_MODULES or self.dropout != 0.05:
            raise ValueError("M10 LoRA Adapter constants differ from the frozen topology")
        return self


class M10LoRAModelConfig(StrictSchema):
    """Pinned Qwen3-8B Base evaluation subject and LoRA identity."""

    repository: Literal["Qwen/Qwen3-8B"]
    revision: Literal["b968826d9c46dd6066d109eabc6255188de91218"]
    model_type: Literal["qwen3"]
    license: Literal["Apache-2.0"]
    attention_architecture: Literal["gqa"]
    adaptation: Literal["lora"]
    trust_remote_code: Literal[False]
    parent_evaluation_subject: Literal["qwen3-8b-m9-base-90587dd6"]
    parent_evaluation_subject_sha256: Literal[
        "9f72bba28bcfaed45f116080033cb9bc83be1632570e71623f2a5684350261d8"
    ]
    parent_model_artifact_sha256: Literal[
        "81fec43ab8b1f03a158e39e50ec23d99cf8701144e8678aea3ca656d12d08de0"
    ]
    parent_tokenizer_artifact_sha256: Literal[
        "99d4d297ece6cc43fa551987701e4ded4fa5c860d9448965d212508519bfc382"
    ]
    lora: M10LoRAAdapterConfig


class M10LoRADataConfig(StrictSchema):
    """Immutable M10 Agent training array identity."""

    dataset_version: str = Field(pattern=r"^m10-agent-sft-v[12]-[0-9a-f]{8}$")
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    sequence_length: Literal[2048]
    target_supervised_tokens_per_epoch: Literal[1_000_000]
    assistant_only_loss: Literal[True]
    language_target: Literal["en70-zh30"]
    mode_target: Literal["nonthinking94-thinking6"]


class M10LoRAOptimizationConfig(StrictSchema):
    """Single-GPU BF16 LoRA optimization contract."""

    max_train_tokens: Literal[10_000_000]
    stage_tokens: tuple[Literal[1_000_000], Literal[5_000_000], Literal[10_000_000]]
    micro_batch_size: Literal[1]
    gradient_accumulation_steps: Literal[8]
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    warmup_tokens: Literal[100_000]
    gradient_checkpointing: Literal[True]
    max_job_duration_seconds: Literal[43_200]

    @field_validator("stage_tokens", mode="before")
    @classmethod
    def freeze_stages(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_stages(self) -> M10LoRAOptimizationConfig:
        if self.stage_tokens != M10_STAGE_TOKENS:
            raise ValueError("M10 LoRA stages must remain exactly 1M/5M/10M Tokens")
        if self.weight_decay != 0.01 or self.max_grad_norm != 1.0:
            raise ValueError("M10 LoRA optimizer safety constants differ from the frozen contract")
        return self


class M10LoRAPrecisionConfig(StrictSchema):
    dtype: Literal["bf16"]
    allow_tf32: Literal[True]
    use_grad_scaler: Literal[False]
    qlora_fallback: Literal["only_after_verified_bf16_oom"]


class M10LoRAParallelConfig(StrictSchema):
    strategy: Literal["single"]
    device_type: Literal["cuda"]
    world_size: Literal[1]


class M10LoRACheckpointConfig(StrictSchema):
    save_interval_tokens: Literal[1_000_000]
    keep_last: Literal[2]
    resume: Literal["auto"]
    pin_stage_checkpoints: Literal[True]


class M10LoRAEvaluationConfig(StrictSchema):
    agent_dev_version: Literal["tinyllm-devops-agent-dev-v1-f958bcc6"]
    parent_task_success_basis_points: int = Field(ge=0, le=10_000)
    stage_min_improvement_basis_points: Literal[100]
    m6_max_regression_basis_points: Literal[200]
    release_set_access: Literal["m10_final_gate_only"]
    scoring_protocol: Literal["m10-agent-scoring-v2", "m10-agent-scoring-v3"] | None = None


class M10LoRAMemoryProbeConfig(StrictSchema):
    optimizer_steps: Literal[10]
    required_before_fresh_training: Literal[True]


class M10LoRAConfig(StrictSchema):
    """Complete staged Qwen3-8B Agent LoRA configuration."""

    schema_version: Literal["1.0"]
    config_kind: Literal["m10_agent_lora"]
    run: M10LoRARunConfig
    model: M10LoRAModelConfig
    data: M10LoRADataConfig
    optimization: M10LoRAOptimizationConfig
    precision: M10LoRAPrecisionConfig
    parallel: M10LoRAParallelConfig
    checkpoint: M10LoRACheckpointConfig
    evaluation: M10LoRAEvaluationConfig
    memory_probe: M10LoRAMemoryProbeConfig

    @model_validator(mode="after")
    def validate_campaign_identity(self) -> M10LoRAConfig:
        repair_v3 = self.run.name.startswith("m10-5-agent-repair-v3-")
        repair_v2 = self.run.name.startswith("m10-5-agent-repair-") and not repair_v3
        if repair_v3:
            if (
                not self.data.dataset_version.startswith("m10-agent-sft-v2-")
                or self.optimization.learning_rate != 1e-5
                or self.evaluation.scoring_protocol != "m10-agent-scoring-v3"
                or self.evaluation.parent_task_success_basis_points != 4875
            ):
                raise ValueError(
                    "M10.5 v3 repair requires Dataset v2, LR 1e-5, scoring v3, and its "
                    "frozen parent baseline"
                )
        elif repair_v2:
            if (
                not self.data.dataset_version.startswith("m10-agent-sft-v2-")
                or self.optimization.learning_rate != 5e-5
                or self.evaluation.scoring_protocol != "m10-agent-scoring-v2"
                or self.evaluation.parent_task_success_basis_points != 4750
            ):
                raise ValueError("M10.5 repair requires Dataset v2, LR 5e-5, and scoring v2")
        elif (
            self.data.dataset_version != M10_DATASET_VERSION
            or self.data.manifest_sha256 != M10_DATASET_MANIFEST_SHA256
            or self.optimization.learning_rate != 2e-4
            or self.evaluation.scoring_protocol is not None
            or self.evaluation.parent_task_success_basis_points != 4500
        ):
            raise ValueError(
                "legacy M10 LoRA optimizer constants or data identity differ from the frozen "
                "v1 campaign"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class M10LoRACheckpointFile(StrictSchema):
    path: Literal["training_state.pt"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M10LoRACheckpointManifest(StrictSchema):
    """Atomic Adapter, optimizer, RNG, cursor, and lineage state."""

    schema_version: Literal["1.0"] = "1.0"
    checkpoint_id: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    run_id: str = Field(min_length=1, max_length=180)
    resume_capability: Literal["exact"] = "exact"
    strategy: Literal["single_gpu_bf16_lora"] = "single_gpu_bf16_lora"
    world_size: Literal[1] = 1
    global_step: int = Field(gt=0)
    completed_epochs: int = Field(ge=1, le=10)
    sequence_cursor: Literal[0]
    supervised_tokens: int = Field(ge=1_000_000, le=10_000_000)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_version: str = Field(pattern=r"^m10-agent-sft-v[12]-[0-9a-f]{8}$")
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_evaluation_subject: Literal["qwen3-8b-m9-base-90587dd6"]
    parent_evaluation_subject_sha256: Literal[
        "9f72bba28bcfaed45f116080033cb9bc83be1632570e71623f2a5684350261d8"
    ]
    parent_model_artifact_sha256: Literal[
        "81fec43ab8b1f03a158e39e50ec23d99cf8701144e8678aea3ca656d12d08de0"
    ]
    peft_version: Literal["0.19.1"]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    hardware_sha256: str = Field(pattern=SHA256_PATTERN)
    memory_probe_sha256: str = Field(pattern=SHA256_PATTERN)
    file: M10LoRACheckpointFile
    pinned: bool
    pin_reason: Literal["stage", "final"] | None = None

    @model_validator(mode="after")
    def validate_progress(self) -> M10LoRACheckpointManifest:
        if self.checkpoint_id != f"checkpoint-tokens-{self.supervised_tokens:010d}":
            raise ValueError("M10 LoRA Checkpoint ID differs from token progress")
        if self.supervised_tokens != self.completed_epochs * 1_000_000:
            raise ValueError("M10 LoRA Checkpoint must be committed at an epoch boundary")
        if self.pinned != (self.pin_reason is not None):
            raise ValueError("M10 LoRA Checkpoint pin flag and reason differ")
        if self.pin_reason == "stage" and self.supervised_tokens not in M10_STAGE_TOKENS[:-1]:
            raise ValueError("only M10 LoRA 1M/5M boundaries may be pinned as stages")
        if self.pin_reason == "final" and self.supervised_tokens != M10_STAGE_TOKENS[-1]:
            raise ValueError("only the M10 LoRA 10M boundary may be pinned as final")
        return self


class M10LoRAStageExport(StrictSchema):
    """Immutable Adapter export for one evaluated stage."""

    checkpoint_id: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    supervised_tokens: Literal[1_000_000, 5_000_000, 10_000_000]
    adapter_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    adapter_files: tuple[Literal["adapter_config.json", "adapter_model.safetensors"], ...]

    @field_validator("adapter_files", mode="before")
    @classmethod
    def freeze_files(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_export(self) -> M10LoRAStageExport:
        if self.checkpoint_id != f"checkpoint-tokens-{self.supervised_tokens:010d}":
            raise ValueError("M10 LoRA stage export differs from its Checkpoint")
        if self.adapter_files != ("adapter_config.json", "adapter_model.safetensors"):
            raise ValueError("M10 LoRA Adapter file set differs")
        return self


class M10LoRAContinuationGate(StrictSchema):
    """Evidence authorizing 1M→5M or 5M→10M continuation."""

    schema_version: Literal["1.0"] = "1.0"
    gate_version: Literal["m10-agent-lora-continuation-v1"] = "m10-agent-lora-continuation-v1"
    scoring_protocol: Literal[
        "m9-agent-scoring-v1",
        "m10-agent-scoring-v2",
        "m10-agent-scoring-v3",
    ] = "m9-agent-scoring-v1"
    evaluated_at: datetime
    decision: Literal["accepted", "rejected"]
    run_id: str = Field(min_length=1, max_length=180)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    source_stage_tokens: Literal[1_000_000, 5_000_000]
    target_stage_tokens: Literal[5_000_000, 10_000_000]
    source_checkpoint_id: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    source_adapter_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    agent_dev_version: Literal["tinyllm-devops-agent-dev-v1-f958bcc6"]
    parent_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_task_success_basis_points: int = Field(ge=0, le=10_000)
    candidate_task_success_basis_points: int = Field(ge=0, le=10_000)
    improvement_basis_points: int = Field(ge=-10_000, le=10_000)
    m6_evidence_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    m6_regression_basis_points: int | None = Field(default=None, ge=-10_000, le=10_000)

    @model_validator(mode="after")
    def validate_decision(self) -> M10LoRAContinuationGate:
        if self.evaluated_at.tzinfo is None:
            raise ValueError("M10 LoRA Gate timestamp must be timezone-aware")
        if (
            self.target_stage_tokens
            != {1_000_000: 5_000_000, 5_000_000: 10_000_000}[self.source_stage_tokens]
        ):
            raise ValueError("M10 LoRA Gate stage transition differs")
        if self.source_checkpoint_id != f"checkpoint-tokens-{self.source_stage_tokens:010d}":
            raise ValueError("M10 LoRA Gate Checkpoint differs from source stage")
        observed = self.candidate_task_success_basis_points - self.parent_task_success_basis_points
        if self.improvement_basis_points != observed:
            raise ValueError("M10 LoRA Gate improvement differs from evidence")
        expected_parent = {
            "m9-agent-scoring-v1": 4500,
            "m10-agent-scoring-v2": 4750,
            "m10-agent-scoring-v3": 4875,
        }[self.scoring_protocol]
        if self.parent_task_success_basis_points != expected_parent:
            raise ValueError("M10 LoRA Gate parent score differs from its scoring protocol")
        if self.source_stage_tokens == 1_000_000:
            if self.m6_evidence_sha256 is not None or self.m6_regression_basis_points is not None:
                raise ValueError("M10 LoRA 1M Gate must not consume M6 evidence")
            accepted = observed >= 100
        else:
            if self.m6_evidence_sha256 is None or self.m6_regression_basis_points is None:
                raise ValueError("M10 LoRA 5M Gate requires M6 evidence")
            accepted = observed >= 100 and self.m6_regression_basis_points <= 200
        if (self.decision == "accepted") != accepted:
            raise ValueError("M10 LoRA Gate decision differs from frozen thresholds")
        return self


class M10LoRAM6RegressionEvidence(StrictSchema):
    """Paired M6 evidence required only by the 5M→10M LoRA Gate."""

    schema_version: Literal["1.0"] = "1.0"
    evidence_version: Literal["m10-agent-lora-m6-regression-v1"] = "m10-agent-lora-m6-regression-v1"
    evaluated_at: datetime
    protocol_version: Literal["m6-release-v7"]
    parent_subject_id: Literal["qwen3-8b-m9-base-90587dd6"]
    parent_evaluation_subject_sha256: Literal[
        "9f72bba28bcfaed45f116080033cb9bc83be1632570e71623f2a5684350261d8"
    ]
    parent_model_artifact_sha256: Literal[
        "81fec43ab8b1f03a158e39e50ec23d99cf8701144e8678aea3ca656d12d08de0"
    ]
    parent_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_aggregate_basis_points: int = Field(ge=0, le=10_000)
    candidate_subject_id: str = Field(pattern=r"^qwen3-8b-m10-agent-lora-5m-[0-9a-f]{8}$")
    candidate_evaluation_subject_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_aggregate_basis_points: int = Field(ge=0, le=10_000)
    regression_basis_points: int = Field(ge=-10_000, le=10_000)

    @model_validator(mode="after")
    def validate_regression(self) -> M10LoRAM6RegressionEvidence:
        if self.evaluated_at.tzinfo is None:
            raise ValueError("M10 Agent LoRA M6 evidence timestamp must be timezone-aware")
        observed = self.parent_aggregate_basis_points - self.candidate_aggregate_basis_points
        if self.regression_basis_points != observed:
            raise ValueError("M10 Agent LoRA M6 regression differs from paired summaries")
        return self


class M10LoRAGeneralPassSummary(StrictSchema):
    """One M6-v7 general pass for the 8B Base parent or a 5M LoRA stage."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded"]
    evaluation_id: str = Field(pattern=r"^m10-lora-m6-general-(?:parent|candidate)-[0-9a-f]{8}$")
    protocol_version: Literal["m6-release-v7"]
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    evaluation_subject_id: str = Field(
        pattern=(
            r"^(?:qwen3-8b-m9-base-90587dd6|"
            r"qwen3-8b-m10-agent-lora-5m-[0-9a-f]{8})$"
        )
    )
    evaluation_subject_sha256: str = Field(pattern=SHA256_PATTERN)
    model: M6ModelIdentity
    general: M6GeneralResult
    physical_gpu_index: int = Field(ge=0)
    gpu_name: Literal["NVIDIA GeForce RTX 3090"]
    duration_seconds: float = Field(gt=0.0)
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    hardware_sha256: str = Field(pattern=SHA256_PATTERN)
    raw_results_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_subject(self) -> M10LoRAGeneralPassSummary:
        if self.evaluation_subject_id == M10_LORA_PARENT_SUBJECT:
            if (
                self.evaluation_subject_sha256 != M10_LORA_PARENT_RECORD_SHA256
                or self.model.role != "base"
                or self.model.adaptation != "base"
                or self.model.model_artifact_sha256 != M10_LORA_PARENT_MODEL_SHA256
            ):
                raise ValueError("M10 LoRA parent M6 identity differs from the frozen Base")
        elif (
            self.model.role != "candidate"
            or self.model.adaptation != "lora"
            or self.model.adapter_sha256 is None
            or self.model.training_tokens != 5_000_000
        ):
            raise ValueError("M10 LoRA Candidate M6 identity must be the 5M Adapter stage")
        if (
            self.model.repository != "Qwen/Qwen3-8B"
            or self.model.base_revision != M10_LORA_MODEL_REVISION
        ):
            raise ValueError("M10 LoRA M6 identity must use the frozen Qwen3-8B Base")
        return self


class M10LoRAMemoryProbeResult(StrictSchema):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded"] = "succeeded"
    probe_version: Literal["m10-agent-lora-memory-v1"] = "m10-agent-lora-memory-v1"
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    dataset_version: str = Field(pattern=r"^m10-agent-sft-v[12]-[0-9a-f]{8}$")
    parent_evaluation_subject: Literal["qwen3-8b-m9-base-90587dd6"]
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    hardware_compatibility_sha256: str = Field(pattern=SHA256_PATTERN)
    physical_gpu_index: int = Field(ge=0)
    gpu_name: Literal["NVIDIA GeForce RTX 3090"]
    optimizer_steps: Literal[10]
    supervised_tokens: int = Field(gt=0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_memory(self) -> M10LoRAMemoryProbeResult:
        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("M10 LoRA Probe reserved memory cannot be below allocated memory")
        return self


class M10LoRARunResult(StrictSchema):
    """Path-free result for one fresh or Exact-Resume stage."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["stage_completed", "succeeded"]
    mode: Literal["fresh", "exact_resume"]
    run_id: str = Field(min_length=1, max_length=180)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    dataset_version: str = Field(pattern=r"^m10-agent-sft-v[12]-[0-9a-f]{8}$")
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_evaluation_subject: Literal["qwen3-8b-m9-base-90587dd6"]
    parent_evaluation_subject_sha256: Literal[
        "9f72bba28bcfaed45f116080033cb9bc83be1632570e71623f2a5684350261d8"
    ]
    parent_model_artifact_sha256: Literal[
        "81fec43ab8b1f03a158e39e50ec23d99cf8701144e8678aea3ca656d12d08de0"
    ]
    attention_architecture: Literal["gqa"]
    adaptation: Literal["lora"]
    peft_version: Literal["0.19.1"]
    seed: Literal[42]
    physical_gpu_index: int = Field(ge=0)
    gpu_name: Literal["NVIDIA GeForce RTX 3090"]
    trainable_parameters: int = Field(gt=0)
    total_parameters: int = Field(gt=0)
    global_step: int = Field(gt=0)
    completed_epochs: int = Field(ge=1, le=10)
    supervised_tokens: Literal[1_000_000, 5_000_000, 10_000_000]
    initial_loss: float = Field(gt=0)
    final_loss: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    memory_probe_sha256: str = Field(pattern=SHA256_PATTERN)
    latest_checkpoint: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    resumed_from_tokens: int | None = Field(default=None, ge=1_000_000, lt=10_000_000)
    continuation_gate_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    stage_export: M10LoRAStageExport

    @model_validator(mode="after")
    def validate_result(self) -> M10LoRARunResult:
        expected_status = "succeeded" if self.supervised_tokens == 10_000_000 else "stage_completed"
        if self.status != expected_status:
            raise ValueError("M10 LoRA result status differs from stage progress")
        if self.completed_epochs * 1_000_000 != self.supervised_tokens:
            raise ValueError("M10 LoRA result epoch and token progress differ")
        if self.latest_checkpoint != self.stage_export.checkpoint_id:
            raise ValueError("M10 LoRA result Checkpoint and export differ")
        if self.trainable_parameters >= self.total_parameters:
            raise ValueError("M10 LoRA must train fewer parameters than the Base model")
        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("M10 LoRA reserved memory cannot be below allocated memory")
        if self.mode == "fresh" and (
            self.resumed_from_tokens is not None or self.continuation_gate_sha256 is not None
        ):
            raise ValueError("fresh M10 LoRA training cannot claim Resume evidence")
        if self.mode == "exact_resume" and (
            self.resumed_from_tokens is None or self.continuation_gate_sha256 is None
        ):
            raise ValueError("M10 LoRA Exact Resume requires an accepted continuation Gate")
        return self


__all__ = [
    "M10_DATASET_MANIFEST_SHA256",
    "M10_DATASET_VERSION",
    "M10_LORA_MODEL_REVISION",
    "M10_LORA_PARENT_MODEL_SHA256",
    "M10_LORA_PARENT_RECORD_SHA256",
    "M10_LORA_PARENT_SUBJECT",
    "M10_LORA_PARENT_TOKENIZER_SHA256",
    "M10_LORA_TARGET_MODULES",
    "M10LoRACheckpointFile",
    "M10LoRACheckpointManifest",
    "M10LoRAConfig",
    "M10LoRAContinuationGate",
    "M10LoRAGeneralPassSummary",
    "M10LoRAMemoryProbeResult",
    "M10LoRAM6RegressionEvidence",
    "M10LoRARunResult",
    "M10LoRAStageExport",
]
