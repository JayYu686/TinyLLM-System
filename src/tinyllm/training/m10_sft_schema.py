"""Strict contracts for staged M10 Qwen3-0.6B Agent Full SFT."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN

M10_MODEL_REVISION: Literal["c1899de289a04d12100db370d81485cdf75e47ca"] = (
    "c1899de289a04d12100db370d81485cdf75e47ca"
)
M10_DATASET_VERSION: Literal["m10-agent-sft-v1-4655d3e3"] = "m10-agent-sft-v1-4655d3e3"
M10_DATASET_MANIFEST_SHA256: Literal[
    "6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490"
] = "6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490"
M10_PARENT_VERSION: Literal["qwen3-0-6b-m7-fa678d92"] = "qwen3-0-6b-m7-fa678d92"
M10_PARENT_RECORD_SHA256: Literal[
    "a83eaff5e9b91be403bddb7a4ac23926465e0a78872921429dad72b6ee7d0ca5"
] = "a83eaff5e9b91be403bddb7a4ac23926465e0a78872921429dad72b6ee7d0ca5"
M10_PARENT_MODEL_SHA256: Literal[
    "63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6"
] = "63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6"
M10_STAGE_TOKENS: tuple[Literal[1_000_000], Literal[5_000_000], Literal[10_000_000]] = (
    1_000_000,
    5_000_000,
    10_000_000,
)


class M10RunConfig(StrictSchema):
    """Stable identity for the single seeded M10 Full-SFT campaign."""

    name: Literal["m10-agent-full-sft-qwen3-0-6b-seed42"]
    seed: Literal[42]
    purpose: Literal["agent_full_sft"]


class M10ModelConfig(StrictSchema):
    """Pinned M7 Production parent and reviewed Qwen3 architecture."""

    repository: Literal["Qwen/Qwen3-0.6B"]
    revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    model_type: Literal["qwen3"]
    license: Literal["Apache-2.0"]
    attention_architecture: Literal["gqa"]
    adaptation: Literal["full_sft"]
    trust_remote_code: Literal[False]
    parent_model_ref: Literal["production"]
    parent_production_version: Literal["qwen3-0-6b-m7-fa678d92"]
    parent_production_record_sha256: Literal[
        "a83eaff5e9b91be403bddb7a4ac23926465e0a78872921429dad72b6ee7d0ca5"
    ]
    parent_model_artifact_sha256: Literal[
        "63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6"
    ]


class M10DataConfig(StrictSchema):
    """The immutable M10.1 training array identity."""

    dataset_version: Literal["m10-agent-sft-v1-4655d3e3"]
    manifest_sha256: Literal["6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490"]
    sequence_length: Literal[2048]
    target_supervised_tokens_per_epoch: Literal[1_000_000]
    assistant_only_loss: Literal[True]
    language_target: Literal["en70-zh30"]
    mode_target: Literal["nonthinking94-thinking6"]


class M10OptimizationConfig(StrictSchema):
    """One-GPU BF16 optimization contract shared by all three stages."""

    max_train_tokens: Literal[10_000_000]
    stage_tokens: tuple[Literal[1_000_000], Literal[5_000_000], Literal[10_000_000]]
    micro_batch_size: Literal[2]
    gradient_accumulation_steps: Literal[4]
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
    def validate_stages(self) -> M10OptimizationConfig:
        if self.stage_tokens != M10_STAGE_TOKENS:
            raise ValueError("M10 Full SFT stages must remain exactly 1M/5M/10M Tokens")
        if (self.learning_rate, self.weight_decay, self.max_grad_norm) != (1e-5, 0.01, 1.0):
            raise ValueError("M10 Full SFT optimizer constants differ from the frozen contract")
        return self


class M10PrecisionConfig(StrictSchema):
    """RTX 3090 precision identity."""

    dtype: Literal["bf16"]
    allow_tf32: Literal[True]
    use_grad_scaler: Literal[False]


class M10ParallelConfig(StrictSchema):
    """M10.2 intentionally uses one independently available RTX 3090."""

    strategy: Literal["single"]
    device_type: Literal["cuda"]
    world_size: Literal[1]


class M10CheckpointConfig(StrictSchema):
    """Token-indexed Exact Resume and retention policy."""

    save_interval_tokens: Literal[1_000_000]
    keep_last: Literal[2]
    resume: Literal["auto"]
    pin_stage_checkpoints: Literal[True]


class M10EvaluationConfig(StrictSchema):
    """Pre-registered stage decision without opening the sealed Release set."""

    agent_dev_version: Literal["tinyllm-devops-agent-dev-v1-f958bcc6"]
    stage_5m_to_10m_min_improvement_basis_points: Literal[100]
    m6_max_regression_basis_points: Literal[200]
    release_set_access: Literal["m10_final_gate_only"]


class M10FullSFTConfig(StrictSchema):
    """Complete staged Full-SFT configuration."""

    schema_version: Literal["1.0"]
    config_kind: Literal["m10_agent_full_sft"]
    run: M10RunConfig
    model: M10ModelConfig
    data: M10DataConfig
    optimization: M10OptimizationConfig
    precision: M10PrecisionConfig
    parallel: M10ParallelConfig
    checkpoint: M10CheckpointConfig
    evaluation: M10EvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class M10CheckpointFile(StrictSchema):
    """One integrity-checked PyTorch training-state payload."""

    path: Literal["training_state.pt"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M10CheckpointManifest(StrictSchema):
    """Atomic single-GPU Exact Resume identity at an epoch boundary."""

    schema_version: Literal["1.0"] = "1.0"
    checkpoint_id: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    run_id: str = Field(min_length=1, max_length=180)
    resume_capability: Literal["exact"] = "exact"
    strategy: Literal["single"] = "single"
    world_size: Literal[1] = 1
    global_step: int = Field(gt=0)
    completed_epochs: int = Field(ge=1, le=10)
    sequence_cursor: Literal[0]
    supervised_tokens: int = Field(ge=1_000_000, le=10_000_000)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_version: Literal["m10-agent-sft-v1-4655d3e3"]
    dataset_manifest_sha256: Literal[
        "6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490"
    ]
    parent_production_version: Literal["qwen3-0-6b-m7-fa678d92"]
    parent_production_record_sha256: Literal[
        "a83eaff5e9b91be403bddb7a4ac23926465e0a78872921429dad72b6ee7d0ca5"
    ]
    parent_model_artifact_sha256: Literal[
        "63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6"
    ]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    file: M10CheckpointFile
    pinned: bool
    pin_reason: Literal["stage", "final"] | None = None

    @model_validator(mode="after")
    def validate_progress(self) -> M10CheckpointManifest:
        if self.checkpoint_id != f"checkpoint-tokens-{self.supervised_tokens:010d}":
            raise ValueError("M10 Checkpoint ID differs from token progress")
        if self.supervised_tokens != self.completed_epochs * 1_000_000:
            raise ValueError("M10 Checkpoint must be committed at a logical epoch boundary")
        if self.pinned != (self.pin_reason is not None):
            raise ValueError("M10 Checkpoint pin flag and reason differ")
        if self.pin_reason == "stage" and self.supervised_tokens not in M10_STAGE_TOKENS[:-1]:
            raise ValueError("only M10 1M/5M boundaries may be pinned as stages")
        if self.pin_reason == "final" and self.supervised_tokens != M10_STAGE_TOKENS[-1]:
            raise ValueError("only the M10 10M boundary may be pinned as final")
        return self


class M10StageExport(StrictSchema):
    """Immutable model export for one evaluated M10 stage."""

    checkpoint_id: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    supervised_tokens: Literal[1_000_000, 5_000_000, 10_000_000]
    export_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_checkpoint(self) -> M10StageExport:
        if self.checkpoint_id != f"checkpoint-tokens-{self.supervised_tokens:010d}":
            raise ValueError("M10 stage export differs from its Checkpoint")
        return self


class M10ContinuationGate(StrictSchema):
    """Immutable evidence authorizing the otherwise blocked 5M-to-10M transition."""

    schema_version: Literal["1.0"] = "1.0"
    gate_version: Literal["m10-full-sft-continuation-v1"] = "m10-full-sft-continuation-v1"
    evaluated_at: datetime
    decision: Literal["accepted", "rejected"]
    run_id: str = Field(min_length=1, max_length=180)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    source_checkpoint_id: Literal["checkpoint-tokens-0005000000"]
    source_stage_export_sha256: str = Field(pattern=SHA256_PATTERN)
    agent_dev_version: Literal["tinyllm-devops-agent-dev-v1-f958bcc6"]
    parent_agent_dev_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_agent_dev_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_task_success_basis_points: int = Field(ge=0, le=10_000)
    candidate_task_success_basis_points: int = Field(ge=0, le=10_000)
    agent_dev_improvement_basis_points: int = Field(ge=-10_000, le=10_000)
    m6_evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    m6_regression_basis_points: int = Field(ge=-10_000, le=10_000)

    @model_validator(mode="after")
    def validate_decision(self) -> M10ContinuationGate:
        if self.evaluated_at.tzinfo is None:
            raise ValueError("M10 continuation gate timestamp must be timezone-aware")
        observed = self.candidate_task_success_basis_points - self.parent_task_success_basis_points
        if self.agent_dev_improvement_basis_points != observed:
            raise ValueError("M10 continuation improvement differs from Agent Dev evidence")
        accepted = observed >= 100 and self.m6_regression_basis_points <= 200
        if (self.decision == "accepted") != accepted:
            raise ValueError("M10 continuation decision differs from frozen thresholds")
        return self


class M10M6RegressionEvidence(StrictSchema):
    """Paired M6 general-capability evidence used only by the 5M continuation gate."""

    schema_version: Literal["1.0"] = "1.0"
    evidence_version: Literal["m10-m6-general-regression-v1"] = "m10-m6-general-regression-v1"
    evaluated_at: datetime
    protocol_version: Literal["m6-release-v7"]
    parent_model_version: Literal["qwen3-0-6b-m7-fa678d92"]
    parent_model_artifact_sha256: Literal[
        "63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6"
    ]
    parent_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_aggregate_basis_points: int = Field(ge=0, le=10_000)
    candidate_subject_id: str = Field(pattern=r"^qwen3-0-6b-m10-full-sft-5m-[0-9a-f]{8}$")
    candidate_evaluation_subject_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_aggregate_basis_points: int = Field(ge=0, le=10_000)
    regression_basis_points: int = Field(ge=-10_000, le=10_000)

    @model_validator(mode="after")
    def validate_regression(self) -> M10M6RegressionEvidence:
        if self.evaluated_at.tzinfo is None:
            raise ValueError("M10 M6 regression timestamp must be timezone-aware")
        observed = self.parent_aggregate_basis_points - self.candidate_aggregate_basis_points
        if self.regression_basis_points != observed:
            raise ValueError("M10 M6 regression differs from paired summaries")
        return self


class M10FullSFTRunResult(StrictSchema):
    """Path-free result for one fresh or resumed M10 stage attempt."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["stage_completed", "succeeded"]
    mode: Literal["fresh", "exact_resume"]
    run_id: str = Field(min_length=1, max_length=180)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    dataset_version: Literal["m10-agent-sft-v1-4655d3e3"]
    dataset_manifest_sha256: Literal[
        "6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490"
    ]
    parent_production_version: Literal["qwen3-0-6b-m7-fa678d92"]
    parent_production_record_sha256: Literal[
        "a83eaff5e9b91be403bddb7a4ac23926465e0a78872921429dad72b6ee7d0ca5"
    ]
    parent_model_artifact_sha256: Literal[
        "63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6"
    ]
    attention_architecture: Literal["gqa"]
    seed: Literal[42]
    physical_gpu_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1, max_length=128)
    global_step: int = Field(gt=0)
    completed_epochs: int = Field(ge=1, le=10)
    supervised_tokens: Literal[1_000_000, 5_000_000, 10_000_000]
    initial_loss: float = Field(gt=0)
    final_loss: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)
    latest_checkpoint: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    resumed_from_tokens: int | None = Field(default=None, ge=1_000_000, le=5_000_000)
    continuation_gate_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    stage_export: M10StageExport

    @model_validator(mode="after")
    def validate_result(self) -> M10FullSFTRunResult:
        expected_status = "succeeded" if self.supervised_tokens == 10_000_000 else "stage_completed"
        if self.status != expected_status:
            raise ValueError("M10 result status differs from stage progress")
        if self.latest_checkpoint != self.stage_export.checkpoint_id:
            raise ValueError("M10 result Checkpoint and export differ")
        if self.completed_epochs * 1_000_000 != self.supervised_tokens:
            raise ValueError("M10 result epoch and token progress differ")
        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("M10 reserved memory cannot be below allocated memory")
        if self.mode == "fresh" and self.resumed_from_tokens is not None:
            raise ValueError("fresh M10 training cannot claim Resume progress")
        if self.mode == "exact_resume" and self.resumed_from_tokens is None:
            raise ValueError("M10 Exact Resume must identify its source stage")
        requires_gate = self.resumed_from_tokens == 5_000_000
        if requires_gate != (self.continuation_gate_sha256 is not None):
            raise ValueError("M10 5M-to-10M Resume must bind one accepted continuation gate")
        return self
