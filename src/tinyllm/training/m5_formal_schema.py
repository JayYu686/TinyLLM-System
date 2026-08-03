"""Strict Checkpoint and Run results for M5.3 four-GPU Full SFT."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN


class M5FormalPackage(StrictSchema):
    """One installed Python distribution in the immutable Run environment."""

    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=200)


class M5FormalEnvironment(StrictSchema):
    """Stable software identity used by a formal Full-SFT Run."""

    schema_version: Literal["1.0"] = "1.0"
    python_version: str = Field(min_length=1, max_length=200)
    python_implementation: str = Field(min_length=1, max_length=100)
    python_executable: str = Field(min_length=1)
    torch_version: Literal["2.7.1+cu118"]
    cuda_runtime: Literal["11.8"]
    transformers_version: Literal["4.57.6"]
    packages: tuple[M5FormalPackage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_packages(self) -> M5FormalEnvironment:
        """Require a normalized, unique, ordered package snapshot."""

        names = tuple(item.name for item in self.packages)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("formal M5 packages must be unique and ordered")
        return self


class M5FormalGPU(StrictSchema):
    """Stable identity for one selected physical RTX 3090."""

    local_rank: int = Field(ge=0, le=3)
    physical_gpu_index: int = Field(ge=0)
    uuid: str = Field(pattern=r"^GPU-[0-9a-f-]+$")
    name: Literal["NVIDIA GeForce RTX 3090"]
    memory_total_mib: int = Field(ge=24_000, le=25_000)
    pci_bus_id: str = Field(pattern=r"^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$")


class M5FormalHardware(StrictSchema):
    """Stable host, driver, selected-GPU, and topology identity."""

    schema_version: Literal["1.0"] = "1.0"
    hostname: str = Field(min_length=1, max_length=255)
    platform: str = Field(min_length=1)
    machine: str = Field(min_length=1, max_length=100)
    cpu_count: int = Field(gt=0)
    cuda_driver: str = Field(pattern=r"^[0-9]+\.[0-9]+(\.[0-9]+)?$")
    selected_gpus: tuple[M5FormalGPU, M5FormalGPU, M5FormalGPU, M5FormalGPU]
    gpu_topology: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_selected_gpus(self) -> M5FormalHardware:
        """Require ordered Ranks and four distinct physical devices."""

        if tuple(item.local_rank for item in self.selected_gpus) != (0, 1, 2, 3):
            raise ValueError("formal M5 selected GPUs must be ordered by local Rank")
        if len({item.physical_gpu_index for item in self.selected_gpus}) != 4:
            raise ValueError("formal M5 selected GPUs must be physically distinct")
        if len({item.uuid for item in self.selected_gpus}) != 4:
            raise ValueError("formal M5 selected GPU UUIDs must be distinct")
        return self


class M5FormalCheckpointFile(StrictSchema):
    """One integrity-checked complete DDP training state."""

    path: Literal["training_state.pt"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M5FormalCheckpointManifest(StrictSchema):
    """Atomic exact-resume point at an optimizer boundary."""

    schema_version: Literal["1.0"] = "1.0"
    checkpoint_id: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    run_id: str = Field(min_length=1, max_length=180)
    resume_capability: Literal["exact"] = "exact"
    strategy: Literal["ddp"] = "ddp"
    world_size: Literal[4] = 4
    global_step: int = Field(ge=0)
    local_sequence_cursor: int = Field(ge=0)
    supervised_tokens: int = Field(ge=0, le=50_000_000)
    dataset_epoch: float = Field(ge=0.0, le=50.0)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_version: Literal["m5-dual-sft-v1-b5b9e839"]
    dataset_manifest_sha256: Literal[
        "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
    ]
    model_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    hardware_sha256: str = Field(pattern=SHA256_PATTERN)
    file: M5FormalCheckpointFile
    pinned: bool
    pin_reason: Literal["interruption", "evaluation", "final"] | None = None

    @model_validator(mode="after")
    def validate_checkpoint(self) -> M5FormalCheckpointManifest:
        """Bind checkpoint identity, progress, and retention pin."""

        if self.checkpoint_id != f"checkpoint-tokens-{self.supervised_tokens:010d}":
            raise ValueError("formal M5 Checkpoint ID differs from Token progress")
        if self.pinned != (self.pin_reason is not None):
            raise ValueError("formal M5 Checkpoint pin flag and reason differ")
        if self.pin_reason == "final" and self.supervised_tokens != 50_000_000:
            raise ValueError("formal M5 final Checkpoint requires exactly 50M Tokens")
        if self.pin_reason == "final" and self.dataset_epoch != 50.0:
            raise ValueError("formal M5 final Checkpoint requires 50 Dataset epochs")
        return self


class M5FormalRankMemory(StrictSchema):
    """Peak CUDA memory from one training Rank."""

    rank: int = Field(ge=0, le=3)
    physical_gpu_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1, max_length=128)
    peak_allocated_bytes: int = Field(gt=0)
    peak_reserved_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_memory(self) -> M5FormalRankMemory:
        """Reject inverted CUDA memory accounting."""

        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("formal M5 reserved memory cannot be below allocated memory")
        return self


class M5FormalRunResult(StrictSchema):
    """Path-free result for one fresh or Exact-Resume Full-SFT attempt."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded", "interrupted"]
    mode: Literal["fresh", "exact_resume"]
    run_id: str = Field(min_length=1, max_length=180)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False]
    environment_sha256: str = Field(pattern=SHA256_PATTERN)
    hardware_sha256: str = Field(pattern=SHA256_PATTERN)
    model_revision: Literal["c1899de289a04d12100db370d81485cdf75e47ca"]
    attention_architecture: Literal["gqa"]
    dataset_version: Literal["m5-dual-sft-v1-b5b9e839"]
    dataset_manifest_sha256: Literal[
        "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
    ]
    thinking_fraction_basis_points: Literal[3000]
    seed: int = Field(ge=0, le=2**32 - 1)
    world_size: Literal[4]
    global_step: int = Field(gt=0)
    local_sequence_cursor: int = Field(gt=0)
    supervised_tokens: int = Field(gt=0, le=50_000_000)
    completed_dataset_epochs: float = Field(gt=0.0, le=50.0)
    initial_loss: float = Field(gt=0)
    final_loss: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    rank_memory: tuple[
        M5FormalRankMemory,
        M5FormalRankMemory,
        M5FormalRankMemory,
        M5FormalRankMemory,
    ]
    latest_checkpoint: str = Field(pattern=r"^checkpoint-tokens-[0-9]{10}$")
    evaluation_checkpoints: tuple[str, ...]
    resumed_from_tokens: int | None = Field(default=None, ge=0, lt=50_000_000)
    export_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_result(self) -> M5FormalRunResult:
        """Bind ordered Ranks, attempt mode, completion, and exported model."""

        if tuple(item.rank for item in self.rank_memory) != (0, 1, 2, 3):
            raise ValueError("formal M5 Rank memory must be ordered 0–3")
        if len({item.physical_gpu_index for item in self.rank_memory}) != 4:
            raise ValueError("formal M5 DDP requires four distinct physical GPUs")
        if self.mode == "fresh" and self.resumed_from_tokens is not None:
            raise ValueError("fresh formal M5 run cannot claim resumed progress")
        if self.mode == "exact_resume" and self.resumed_from_tokens is None:
            raise ValueError("formal M5 Exact Resume requires source progress")
        if self.status == "succeeded":
            if (
                self.supervised_tokens != 50_000_000
                or self.completed_dataset_epochs != 50.0
                or self.export_sha256 is None
            ):
                raise ValueError("successful formal M5 run requires 50M Tokens and export")
            evaluation_tokens = tuple(
                int(checkpoint.removeprefix("checkpoint-tokens-"))
                for checkpoint in self.evaluation_checkpoints
            )
            if (
                len(evaluation_tokens) != 5
                or evaluation_tokens[-1] != 50_000_000
                or self.latest_checkpoint != self.evaluation_checkpoints[-1]
                or any(
                    not boundary <= tokens < boundary + 100_000
                    for boundary, tokens in zip(
                        range(10_000_000, 50_000_001, 10_000_000),
                        evaluation_tokens,
                        strict=True,
                    )
                )
            ):
                raise ValueError(
                    "successful formal M5 run requires five staged evaluation Checkpoints"
                )
        elif self.supervised_tokens >= 50_000_000 or self.export_sha256 is not None:
            raise ValueError("interrupted formal M5 run cannot claim completion or export")
        return self


class M5FormalCampaignResult(StrictSchema):
    """Path-free result for the two-segment four-GPU Full-SFT campaign."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["succeeded"]
    campaign_id: str = Field(pattern=r"^[0-9]{8}T[0-9]{6}Z-m5-formal-campaign$")
    run_id: str = Field(min_length=1, max_length=180)
    physical_gpu_indices: tuple[int, int, int, int]
    segment_count: Literal[2]
    interruption_tokens: int = Field(ge=2_000_000, lt=2_100_000)
    resumed_from_tokens: int = Field(ge=2_000_000, lt=2_100_000)
    final_tokens: Literal[50_000_000]
    export_sha256: str = Field(pattern=SHA256_PATTERN)
    interrupted_result_sha256: str = Field(pattern=SHA256_PATTERN)
    final_result_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluation_id: str = Field(min_length=1, max_length=180)
    evaluation_summary_sha256: str = Field(pattern=SHA256_PATTERN)
    thinking_controlled_format_basis_points: int = Field(ge=0, le=10_000)
    thinking_natural_close_basis_points: int = Field(ge=0, le=10_000)
    thinking_forced_close_basis_points: int = Field(ge=0, le=10_000)
    thinking_score_basis_points: int = Field(ge=0, le=10_000)
    nonthinking_score_basis_points: int = Field(ge=0, le=10_000)
    thermal_events_sha256: str = Field(pattern=SHA256_PATTERN)
    thermal_pause_count: int = Field(ge=0)
    max_observed_temperature_c: int = Field(ge=0, le=100)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")

    @model_validator(mode="after")
    def validate_campaign(self) -> M5FormalCampaignResult:
        """Bind distinct GPUs and the Exact-Resume boundary."""

        if len(set(self.physical_gpu_indices)) != 4:
            raise ValueError("formal M5 campaign requires four distinct GPUs")
        if self.resumed_from_tokens != self.interruption_tokens:
            raise ValueError("formal M5 campaign Resume point differs from interruption")
        return self
