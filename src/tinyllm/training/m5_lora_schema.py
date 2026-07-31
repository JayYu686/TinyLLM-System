"""Strict artifacts for M5 Qwen3-8B single-GPU BF16 LoRA."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN
from tinyllm.training.m5_formal_schema import M5FormalPackage


class M5LoRAEnvironment(StrictSchema):
    """Software identity including the reviewed PEFT dependency."""

    schema_version: Literal["1.0"] = "1.0"
    python_version: str = Field(min_length=1, max_length=200)
    python_implementation: str = Field(min_length=1, max_length=100)
    python_executable: str = Field(min_length=1)
    torch_version: Literal["2.7.1+cu118"]
    cuda_runtime: Literal["11.8"]
    transformers_version: Literal["4.57.6"]
    peft_version: Literal["0.19.1"]
    packages: tuple[M5FormalPackage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_packages(self) -> M5LoRAEnvironment:
        """Require the explicit PEFT package and deterministic ordering."""

        names = tuple(item.name for item in self.packages)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("M5 LoRA packages must be unique and ordered")
        versions = {item.name: item.version for item in self.packages}
        if versions.get("peft") != self.peft_version:
            raise ValueError("M5 LoRA PEFT package differs from the explicit version")
        return self


class M5LoRAGPU(StrictSchema):
    """Stable identity for the selected physical RTX 3090."""

    physical_gpu_index: int = Field(ge=0)
    uuid: str = Field(pattern=r"^GPU-[0-9a-f-]+$")
    name: Literal["NVIDIA GeForce RTX 3090"]
    memory_total_mib: int = Field(ge=24_000, le=25_000)
    pci_bus_id: str = Field(pattern=r"^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$")


class M5LoRAHardware(StrictSchema):
    """Stable single-GPU host and topology identity."""

    schema_version: Literal["1.0"] = "1.0"
    hostname: str = Field(min_length=1, max_length=255)
    platform: str = Field(min_length=1)
    machine: str = Field(min_length=1, max_length=100)
    cpu_count: int = Field(gt=0)
    cuda_driver: str = Field(pattern=r"^[0-9]+\.[0-9]+(\.[0-9]+)?$")
    selected_gpu: M5LoRAGPU
    gpu_topology: str = Field(min_length=1)


class M5LoRACheckpointFile(StrictSchema):
    """One integrity-checked LoRA training-state file."""

    path: Literal["training_state.pt"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M5LoRACheckpointManifest(StrictSchema):
    """Atomic adapter, optimizer, RNG, and cursor recovery point."""

    schema_version: Literal["1.0"] = "1.0"
    checkpoint_id: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    run_id: str = Field(min_length=1, max_length=180)
    resume_capability: Literal["exact"] = "exact"
    strategy: Literal["single_gpu_bf16_lora"] = "single_gpu_bf16_lora"
    world_size: Literal[1] = 1
    global_step: int = Field(ge=0)
    sequence_cursor: int = Field(ge=0)
    supervised_tokens: int = Field(ge=0, le=10_000_000)
    dataset_epoch: float = Field(ge=0.0, le=10.0)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_version: Literal["m5-dual-sft-v1-b5b9e839"]
    dataset_manifest_sha256: Literal[
        "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
    ]
    model_revision: Literal["b968826d9c46dd6066d109eabc6255188de91218"]
    peft_version: Literal["0.19.1"]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    hardware_sha256: str = Field(pattern=SHA256_PATTERN)
    file: M5LoRACheckpointFile
    pinned: bool
    pin_reason: Literal["interruption", "evaluation", "final"] | None = None

    @model_validator(mode="after")
    def validate_checkpoint(self) -> M5LoRACheckpointManifest:
        """Bind Token identity, pin semantics, and final completion."""

        if self.checkpoint_id != f"checkpoint-tokens-{self.supervised_tokens:010d}":
            raise ValueError("M5 LoRA Checkpoint ID differs from Token progress")
        if self.pinned != (self.pin_reason is not None):
            raise ValueError("M5 LoRA Checkpoint pin flag and reason differ")
        if self.pin_reason == "final" and (
            self.supervised_tokens != 10_000_000 or self.dataset_epoch != 10.0
        ):
            raise ValueError("M5 LoRA final Checkpoint requires 10M Tokens and 10 epochs")
        return self


class M5LoRAMemory(StrictSchema):
    """Peak CUDA memory from the LoRA process."""

    physical_gpu_index: int = Field(ge=0)
    gpu_name: Literal["NVIDIA GeForce RTX 3090"]
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_memory(self) -> M5LoRAMemory:
        """Reject inverted CUDA memory accounting."""

        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("M5 LoRA reserved memory cannot be below allocated memory")
        return self


class M5LoRARunResult(StrictSchema):
    """Path-free result for one fresh or resumed LoRA attempt."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded", "interrupted"]
    mode: Literal["fresh", "exact_resume"]
    run_id: str = Field(min_length=1, max_length=180)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    hardware_sha256: str = Field(pattern=SHA256_PATTERN)
    model_revision: Literal["b968826d9c46dd6066d109eabc6255188de91218"]
    attention_architecture: Literal["gqa"]
    adaptation: Literal["lora"]
    peft_version: Literal["0.19.1"]
    dataset_version: Literal["m5-dual-sft-v1-b5b9e839"]
    dataset_manifest_sha256: Literal[
        "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
    ]
    thinking_fraction_basis_points: Literal[3000]
    seed: int = Field(ge=0, le=2**32 - 1)
    world_size: Literal[1]
    trainable_parameters: int = Field(gt=0)
    total_parameters: int = Field(gt=0)
    global_step: int = Field(gt=0)
    sequence_cursor: int = Field(gt=0)
    supervised_tokens: int = Field(gt=0, le=10_000_000)
    completed_dataset_epochs: float = Field(gt=0.0, le=10.0)
    initial_loss: float = Field(gt=0)
    final_loss: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    memory: M5LoRAMemory
    latest_checkpoint: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    evaluation_checkpoints: tuple[str, ...]
    resumed_from_tokens: int | None = Field(default=None, ge=0, lt=10_000_000)
    adapter_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_result(self) -> M5LoRARunResult:
        """Bind adaptation size, recovery mode, and staged completion."""

        if self.trainable_parameters >= self.total_parameters:
            raise ValueError("M5 LoRA must train fewer parameters than the Base model")
        if self.mode == "fresh" and self.resumed_from_tokens is not None:
            raise ValueError("fresh M5 LoRA run cannot claim resumed progress")
        if self.mode == "exact_resume" and self.resumed_from_tokens is None:
            raise ValueError("M5 LoRA Exact Resume requires source progress")
        if self.status == "succeeded":
            evaluation_tokens = tuple(
                int(checkpoint.removeprefix("checkpoint-tokens-"))
                for checkpoint in self.evaluation_checkpoints
            )
            if (
                self.supervised_tokens != 10_000_000
                or self.completed_dataset_epochs != 10.0
                or self.adapter_sha256 is None
                or len(evaluation_tokens) != 5
                or evaluation_tokens[-1] != 10_000_000
                or self.latest_checkpoint != self.evaluation_checkpoints[-1]
                or any(
                    not boundary <= tokens < boundary + 100_000
                    for boundary, tokens in zip(
                        range(2_000_000, 10_000_001, 2_000_000),
                        evaluation_tokens,
                        strict=True,
                    )
                )
            ):
                raise ValueError("successful M5 LoRA run requires staged 10M completion")
        elif self.supervised_tokens >= 10_000_000 or self.adapter_sha256 is not None:
            raise ValueError("interrupted M5 LoRA run cannot claim completion or Adapter")
        return self
