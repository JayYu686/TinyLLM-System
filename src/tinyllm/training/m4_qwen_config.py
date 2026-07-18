"""Frozen public configuration for the formal M4 Qwen3-8B FSDP2 gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.training.config import PrecisionConfig, RunConfig, TrainingLoopConfig
from tinyllm.training.fsdp2_config import FSDP2CheckpointConfig, FSDP2PolicyConfig

QWEN3_8B_REPOSITORY = "Qwen/Qwen3-8B"
QWEN3_8B_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
M4_DATASET_VERSION = "m2-sft-v1-f82ff32e"


class M4QwenConfigError(ValueError):
    """Raised when a formal M4 Qwen configuration violates the frozen contract."""


class M4QwenModelConfig(StrictSchema):
    """Immutable model identity and safe local-loading policy."""

    repository: Literal["Qwen/Qwen3-8B"]
    revision: Literal["b968826d9c46dd6066d109eabc6255188de91218"]
    model_type: Literal["qwen3"]
    license: Literal["Apache-2.0"]
    trust_remote_code: Literal[False] = False


class M4QwenDataConfig(StrictSchema):
    """Deterministic registered-data view used by the bounded four-GPU run."""

    dataset_version: Literal["m2-sft-v1-f82ff32e"]
    split: Literal["train"]
    sequence_length: Literal[512]
    max_sequences: int = Field(ge=200, le=4096)
    pad_token_id: Literal[151643]
    assistant_only_loss: Literal[True]
    slice_position_ids_reset: Literal[True]


class M4QwenFSDP2Config(StrictSchema):
    """Complete four-GPU Qwen3-8B M4.3 configuration."""

    schema_version: Literal["1.0"]
    run: RunConfig
    model: M4QwenModelConfig
    data: M4QwenDataConfig
    training: TrainingLoopConfig
    precision: PrecisionConfig
    distributed: FSDP2PolicyConfig
    checkpoint: FSDP2CheckpointConfig

    @model_validator(mode="after")
    def validate_formal_gate(self) -> M4QwenFSDP2Config:
        """Reject any silent weakening or expansion of the M4 acceptance run."""

        if self.training.max_steps != 50:
            raise ValueError("formal M4 requires exactly 50 optimizer steps")
        if self.training.micro_batch_size != 1:
            raise ValueError("formal M4 requires micro_batch_size=1 per Rank")
        if self.training.gradient_accumulation_steps != 1:
            raise ValueError("formal M4 does not use gradient accumulation")
        if self.precision.dtype != "bf16" or self.precision.use_grad_scaler:
            raise ValueError("formal M4 requires BF16 without GradScaler")
        if (
            self.distributed.backend != "nccl"
            or self.distributed.device_type != "cuda"
            or self.distributed.world_size != 4
        ):
            raise ValueError("formal M4 requires four-GPU CUDA/NCCL FSDP2")
        if not self.distributed.activation_checkpointing:
            raise ValueError("formal M4 requires Activation Checkpointing")
        if self.checkpoint.save_steps != 25:
            raise ValueError("formal M4 requires the recovery boundary at Step 25")
        required_samples = self.training.max_steps * self.distributed.world_size
        if self.data.max_sequences < required_samples:
            raise ValueError("formal M4 data view cannot cover 50 steps without an Epoch wrap")
        return self

    @property
    def global_batch_size(self) -> int:
        """Return the frozen data-parallel global batch size."""

        return self.training.micro_batch_size * self.distributed.world_size

    def to_dict(self) -> dict[str, Any]:
        """Return the complete resolved config as canonical JSON values."""

        return self.model_dump(mode="json")


def m4_qwen_config_from_mapping(raw: object) -> M4QwenFSDP2Config:
    """Validate one decoded YAML object as the formal M4 configuration."""

    try:
        return M4QwenFSDP2Config.model_validate(raw)
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
        raise M4QwenConfigError("; ".join(messages)) from exc


def load_m4_qwen_config(path: Path) -> M4QwenFSDP2Config:
    """Load and validate the formal M4 YAML configuration."""

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise M4QwenConfigError("M4 Qwen config must use a .yaml or .yml extension")
    try:
        decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise M4QwenConfigError(f"cannot read M4 Qwen config: {path}") from exc
    except yaml.YAMLError as exc:
        raise M4QwenConfigError(f"invalid YAML in M4 Qwen config: {path}") from exc
    return m4_qwen_config_from_mapping(decoded)
