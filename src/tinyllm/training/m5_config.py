"""Strict configuration contract for M5 Qwen3 dual-mode SFT."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from tinyllm.data import (
    QWEN3_NONTHINKING_TEMPLATE_SHA256,
    QWEN3_THINKING_TEMPLATE_SHA256,
)
from tinyllm.schemas.base import StrictSchema

QWEN3_0_6B_REPOSITORY = "Qwen/Qwen3-0.6B"
QWEN3_0_6B_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
QWEN3_8B_REPOSITORY = "Qwen/Qwen3-8B"
QWEN3_8B_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
M2_PARENT_DATASET_VERSION = "m2-sft-v1-f82ff32e"


class M5ConfigError(ValueError):
    """Raised when an M5 SFT configuration violates the public contract."""


class M5RunConfig(StrictSchema):
    """Run identity and declared evidence purpose."""

    name: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0, le=2**32 - 1)
    purpose: Literal["smoke", "ablation", "formal"]


class M5LoRAConfig(StrictSchema):
    """Frozen LoRA topology for the Qwen3-8B route."""

    rank: Literal[16]
    alpha: Literal[32]
    dropout: float = Field(ge=0.0, le=1.0)
    target_scope: Literal["attention_and_mlp_linear"]
    bias: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_dropout(self) -> M5LoRAConfig:
        """Keep the accepted LoRA dropout at the preregistered value."""

        if self.dropout != 0.05:
            raise ValueError("M5 LoRA dropout must be 0.05")
        return self


class M5ModelConfig(StrictSchema):
    """Pinned Qwen3 identity, GQA architecture, and adaptation strategy."""

    repository: Literal["Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"]
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    model_type: Literal["qwen3"]
    license: Literal["Apache-2.0"]
    attention_architecture: Literal["gqa"]
    adaptation: Literal["full_sft", "lora", "qlora"]
    trust_remote_code: Literal[False] = False
    lora: M5LoRAConfig | None = None
    bf16_lora_oom_evidence_run_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_model_route(self) -> M5ModelConfig:
        """Bind each immutable checkpoint to its only accepted M5 adaptation route."""

        expected_revision = {
            QWEN3_0_6B_REPOSITORY: QWEN3_0_6B_REVISION,
            QWEN3_8B_REPOSITORY: QWEN3_8B_REVISION,
        }[self.repository]
        if self.revision != expected_revision:
            raise ValueError("model revision does not match the pinned repository")
        if self.adaptation == "full_sft":
            if self.repository != QWEN3_0_6B_REPOSITORY:
                raise ValueError("M5 Full SFT is restricted to Qwen3-0.6B")
            if self.lora is not None or self.bf16_lora_oom_evidence_run_id is not None:
                raise ValueError("Full SFT cannot define LoRA or QLoRA fields")
            return self
        if self.repository != QWEN3_8B_REPOSITORY or self.lora is None:
            raise ValueError("M5 LoRA and QLoRA require Qwen3-8B and the frozen LoRA policy")
        if self.adaptation == "lora" and self.bf16_lora_oom_evidence_run_id is not None:
            raise ValueError("BF16 LoRA cannot claim a QLoRA fallback evidence Run")
        if self.adaptation == "qlora" and self.bf16_lora_oom_evidence_run_id is None:
            raise ValueError("QLoRA requires a retained BF16 LoRA OOM evidence Run")
        return self


class M5DataConfig(StrictSchema):
    """Immutable dual-mode dataset and token-mixture identity."""

    dataset_version: str = Field(pattern=r"^m5-(reasoning-pilot|dual-sft)-v[0-9]+-[a-f0-9]{8}$")
    parent_dataset_version: Literal["m2-sft-v1-f82ff32e"]
    split: Literal["train"]
    sequence_length: Literal[1024]
    assistant_only_loss: Literal[True]
    mode: Literal["dual"]
    thinking_token_fraction: float = Field(ge=0.0, le=0.5)
    mix_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_mixture_fraction(self) -> M5DataConfig:
        """Restrict M5.2 to the preregistered ablation and formal candidates."""

        if self.thinking_token_fraction not in {0.0, 0.3, 0.5}:
            raise ValueError("thinking_token_fraction must be 0.0, 0.3, or 0.5")
        return self


class M5ReasoningConfig(StrictSchema):
    """Frozen native Qwen3 dual-mode rendering and supervision policy."""

    explicit_mode_selection: Literal[True]
    supervise_visible_reasoning: Literal[True]
    nonthinking_template_id: Literal["qwen3-chatml-nonthinking-v1"]
    nonthinking_template_sha256: Literal[
        "d41161e0416a1047b0f31cce1497e610a4050fbe4d3fb7bda19cc56a1523cb33"
    ]
    thinking_template_id: Literal["qwen3-chatml-thinking-v1"]
    thinking_template_sha256: Literal[
        "4786143dbb7adb72a922d5efdcbe6596f2d65dcdc35d7bbf1b22830b795c2af9"
    ]

    @model_validator(mode="after")
    def validate_template_implementation(self) -> M5ReasoningConfig:
        """Detect drift between the public config and built-in renderers."""

        if self.nonthinking_template_sha256 != QWEN3_NONTHINKING_TEMPLATE_SHA256:
            raise ValueError("Non-thinking Template hash does not match the implementation")
        if self.thinking_template_sha256 != QWEN3_THINKING_TEMPLATE_SHA256:
            raise ValueError("Thinking Template hash does not match the implementation")
        return self


class M5TrainingLoopConfig(StrictSchema):
    """Token-budgeted optimizer loop contract for M5."""

    max_train_tokens: int = Field(gt=0)
    evaluation_interval_tokens: int = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    warmup_tokens: int = Field(ge=0)
    gradient_checkpointing: Literal[True]
    max_job_duration_seconds: int = Field(gt=0, le=43_200)

    @model_validator(mode="after")
    def validate_token_intervals(self) -> M5TrainingLoopConfig:
        """Keep warmup and evaluation boundaries inside the requested token budget."""

        if self.warmup_tokens > self.max_train_tokens:
            raise ValueError("warmup_tokens cannot exceed max_train_tokens")
        if self.evaluation_interval_tokens > self.max_train_tokens:
            raise ValueError("evaluation interval cannot exceed max_train_tokens")
        return self


class M5PrecisionConfig(StrictSchema):
    """RTX 3090 numerical policy for both M5 routes."""

    dtype: Literal["bf16"]
    allow_tf32: bool
    use_grad_scaler: Literal[False]


class M5ParallelConfig(StrictSchema):
    """Single-GPU or four-GPU DDP launch identity."""

    strategy: Literal["single", "ddp"]
    backend: Literal["nccl"] | None
    device_type: Literal["cuda"]
    world_size: Literal[1, 4]
    timeout_seconds: int = Field(ge=10, le=1800)

    @model_validator(mode="after")
    def validate_launch(self) -> M5ParallelConfig:
        """Reject ambiguous single-process and DDP combinations."""

        if self.strategy == "single" and (self.world_size != 1 or self.backend is not None):
            raise ValueError("single strategy requires world_size=1 and backend=null")
        if self.strategy == "ddp" and (self.world_size != 4 or self.backend != "nccl"):
            raise ValueError("M5 DDP requires world_size=4 and backend=nccl")
        return self


class M5CheckpointConfig(StrictSchema):
    """Token-based rolling Checkpoint and exact-resume contract."""

    save_interval_tokens: int = Field(gt=0)
    keep_last: Literal[2]
    resume: Literal["none", "auto", "exact"]


class M5EvaluationConfig(StrictSchema):
    """Development-only selection suite kept separate from M6 tests."""

    reasoning_dev_version: Literal["m5-reasoning-dev-v1"]
    compare_modes_separately: Literal[True]
    consume_m6_frozen_results: Literal[False]


class M5SFTConfig(StrictSchema):
    """Complete validated configuration for one M5 SFT run."""

    config_kind: Literal["qwen_sft"]
    schema_version: Literal["1.0"]
    run: M5RunConfig
    model: M5ModelConfig
    data: M5DataConfig
    reasoning: M5ReasoningConfig
    training: M5TrainingLoopConfig
    precision: M5PrecisionConfig
    parallel: M5ParallelConfig
    checkpoint: M5CheckpointConfig
    evaluation: M5EvaluationConfig

    @model_validator(mode="after")
    def validate_milestone_route(self) -> M5SFTConfig:
        """Bind Smoke, ablation, and formal configs to the accepted M5 protocol."""

        if self.checkpoint.save_interval_tokens > self.training.max_train_tokens:
            raise ValueError("Checkpoint interval cannot exceed max_train_tokens")
        if self.run.purpose == "ablation":
            if self.model.adaptation != "full_sft":
                raise ValueError("M5 mixture ablation is restricted to Qwen3-0.6B Full SFT")
            if not self.data.dataset_version.startswith("m5-reasoning-pilot-"):
                raise ValueError("ablation requires an m5-reasoning-pilot Dataset Version")
            if self.training.max_train_tokens != 1_000_000:
                raise ValueError("each M5 ablation arm requires exactly 1M Tokens")
            if self.parallel.strategy != "single":
                raise ValueError("M5 ablation runs use one GPU")
            return self
        if self.run.purpose == "smoke":
            if self.training.max_train_tokens > 100_000:
                raise ValueError("M5 Smoke runs are bounded to at most 100K Tokens")
            if self.parallel.strategy != "single":
                raise ValueError("M5 Smoke runs use one GPU")
            return self

        if not self.data.dataset_version.startswith("m5-dual-sft-"):
            raise ValueError("formal M5 runs require an m5-dual-sft Dataset Version")
        if self.data.thinking_token_fraction == 0.0:
            raise ValueError("formal dual-mode training requires a non-zero Thinking fraction")
        if self.model.adaptation == "full_sft":
            if self.parallel.strategy != "ddp" or self.parallel.world_size != 4:
                raise ValueError("formal Qwen3-0.6B Full SFT requires four-GPU DDP")
            if not 50_000_000 <= self.training.max_train_tokens <= 100_000_000:
                raise ValueError("formal Full SFT requires a 50M–100M Token budget")
            if self.training.evaluation_interval_tokens != 10_000_000:
                raise ValueError("formal Full SFT evaluates every 10M Tokens")
            if self.checkpoint.save_interval_tokens != 2_000_000:
                raise ValueError("formal Full SFT checkpoints every 2M Tokens")
            return self
        if self.parallel.strategy != "single":
            raise ValueError("formal Qwen3-8B LoRA/QLoRA requires one GPU")
        if not 10_000_000 <= self.training.max_train_tokens <= 30_000_000:
            raise ValueError("formal LoRA/QLoRA requires a 10M–30M Token budget")
        if self.training.evaluation_interval_tokens != 2_000_000:
            raise ValueError("formal LoRA/QLoRA evaluates every 2M Tokens")
        if self.checkpoint.save_interval_tokens != 1_000_000:
            raise ValueError("formal LoRA/QLoRA checkpoints every 1M Tokens")
        return self

    @property
    def global_batch_size(self) -> int:
        """Return sequence batches per optimizer step across all data-parallel Ranks."""

        return (
            self.training.micro_batch_size
            * self.training.gradient_accumulation_steps
            * self.parallel.world_size
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible resolved configuration."""

        return self.model_dump(mode="json")


def m5_sft_config_from_mapping(raw: object) -> M5SFTConfig:
    """Validate one decoded YAML object as an M5 SFT configuration."""

    try:
        return M5SFTConfig.model_validate(raw)
    except ValidationError as exc:
        messages: list[str] = []
        for error in exc.errors(include_url=False, include_context=False):
            location = ".".join(str(part) for part in error["loc"])
            if error["type"] == "extra_forbidden":
                messages.append(f"unknown config field: {location}")
            elif location:
                messages.append(f"{location}: {error['msg']}")
            else:
                messages.append(str(error["msg"]))
        raise M5ConfigError("; ".join(messages)) from exc


def load_m5_sft_config(path: Path) -> M5SFTConfig:
    """Load and validate one strict M5 SFT YAML configuration."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise M5ConfigError("M5 SFT config must use a .yaml or .yml extension")
    try:
        decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise M5ConfigError(f"cannot read M5 SFT config: {path}") from exc
    except yaml.YAMLError as exc:
        raise M5ConfigError(f"invalid M5 SFT YAML: {path}") from exc
    return m5_sft_config_from_mapping(decoded)
