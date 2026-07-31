from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from tinyllm.training.m5_config import load_m5_sft_config
from tinyllm.training.m5_lora import (
    M5LoRACheckpointStore,
    M5LoRAError,
    M5LoRAProgress,
)
from tinyllm.training.m5_lora_schema import (
    M5LoRACampaignResult,
    M5LoRAMemory,
    M5LoRARunResult,
)


def _result_mapping() -> dict[str, object]:
    return {
        "status": "succeeded",
        "mode": "exact_resume",
        "run_id": "20260731T000000Z-m5-lora-a1b2c3d4-cafe",
        "config_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "environment_sha256": "c" * 64,
        "hardware_sha256": "d" * 64,
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "attention_architecture": "gqa",
        "adaptation": "lora",
        "peft_version": "0.19.1",
        "dataset_version": "m5-dual-sft-v1-b5b9e839",
        "dataset_manifest_sha256": (
            "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
        ),
        "thinking_fraction_basis_points": 3000,
        "seed": 42,
        "world_size": 1,
        "trainable_parameters": 10,
        "total_parameters": 100,
        "global_step": 100,
        "sequence_cursor": 41_500,
        "supervised_tokens": 10_000_000,
        "completed_dataset_epochs": 10.0,
        "initial_loss": 2.0,
        "final_loss": 1.0,
        "duration_seconds": 60.0,
        "memory": M5LoRAMemory(
            physical_gpu_index=7,
            gpu_name="NVIDIA GeForce RTX 3090",
            peak_allocated_bytes=10,
            peak_reserved_bytes=20,
        ),
        "latest_checkpoint": "checkpoint-tokens-0010000000",
        "evaluation_checkpoints": (
            "checkpoint-tokens-0002000000",
            "checkpoint-tokens-0004000000",
            "checkpoint-tokens-0006000000",
            "checkpoint-tokens-0008000000",
            "checkpoint-tokens-0010000000",
        ),
        "resumed_from_tokens": 1_000_000,
        "adapter_sha256": "e" * 64,
    }


def test_lora_config_freezes_single_gpu_ten_million_route() -> None:
    config = load_m5_sft_config(Path("configs/sft/m5_formal_qwen3_8b_lora.yaml"))

    assert config.model.adaptation == "lora"
    assert config.model.lora is not None
    assert config.model.lora.rank == 16
    assert config.model.lora.alpha == 32
    assert config.parallel.strategy == "single"
    assert config.training.max_train_tokens == 10_000_000
    assert config.global_batch_size == 8


def test_lora_result_requires_parameter_efficiency_and_staged_completion() -> None:
    result = M5LoRARunResult.model_validate(_result_mapping())

    assert result.adapter_sha256 == "e" * 64
    with pytest.raises(ValidationError, match="fewer parameters"):
        M5LoRARunResult.model_validate(_result_mapping() | {"trainable_parameters": 100})
    with pytest.raises(ValidationError, match="staged 10M completion"):
        M5LoRARunResult.model_validate(
            _result_mapping() | {"evaluation_checkpoints": ("checkpoint-tokens-0010000000",)}
        )


def test_lora_campaign_binds_interruption_and_resume() -> None:
    payload = {
        "status": "succeeded",
        "campaign_id": "20260731T000000Z-m5-lora-campaign-gpu4",
        "run_id": "formal-run",
        "physical_gpu_index": 4,
        "segment_count": 2,
        "interruption_tokens": 5_000_123,
        "resumed_from_tokens": 5_000_123,
        "final_tokens": 10_000_000,
        "adapter_sha256": "a" * 64,
        "interrupted_result_sha256": "b" * 64,
        "final_result_sha256": "c" * 64,
        "thermal_events_sha256": "d" * 64,
        "thermal_pause_count": 2,
        "max_observed_temperature_c": 84,
        "git_commit": "e" * 40,
    }

    assert M5LoRACampaignResult.model_validate(payload).segment_count == 2
    with pytest.raises(ValidationError, match="Resume point"):
        M5LoRACampaignResult.model_validate(payload | {"resumed_from_tokens": 5_000_124})


def test_lora_checkpoint_store_detects_payload_corruption(tmp_path: Path) -> None:
    config = load_m5_sft_config(Path("configs/sft/m5_formal_qwen3_8b_lora.yaml"))
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = torch.optim.AdamW((parameter,), lr=2e-4)
    store = M5LoRACheckpointStore(tmp_path / "checkpoints", keep_last=2)
    manifest = store.save(
        adapter_state={"adapter": parameter.detach()},
        optimizer=optimizer,
        progress=M5LoRAProgress(
            global_step=1,
            sequence_cursor=10,
            supervised_tokens=1_000_000,
            initial_loss=2.0,
            final_loss=1.5,
            evaluation_checkpoints=(),
        ),
        rng={"rank": 0},
        config=config,
        config_sha256="f" * 64,
        dataset_manifest_sha256=(
            "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
        ),
        source_sequence_count=4_150,
        environment_sha256="a" * 64,
        hardware_sha256="b" * 64,
        run_id="unit-lora-run",
        git_commit="c" * 40,
        pin_reason="interruption",
    )

    assert store.latest_valid() == manifest
    state = tmp_path / "checkpoints" / manifest.checkpoint_id / "training_state.pt"
    with state.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(M5LoRAError, match="integrity"):
        store.validate(manifest.checkpoint_id)
