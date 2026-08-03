from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from tinyllm.training.m5_config import load_m5_sft_config
from tinyllm.training.m5_formal import (
    M5FormalCheckpointStore,
    M5FormalProgress,
    M5FormalTrainingError,
)
from tinyllm.training.m5_formal_schema import (
    M5FormalCampaignResult,
    M5FormalEnvironment,
    M5FormalHardware,
    M5FormalRankMemory,
    M5FormalRunResult,
)


def test_formal_campaign_binds_four_gpus_and_resume_boundary() -> None:
    staged_evaluations = tuple(
        {
            "checkpoint_id": f"checkpoint-tokens-{tokens:010d}",
            "supervised_tokens": tokens,
            "snapshot_export_sha256": str(index) * 64,
            "evaluation_id": f"evaluation-{tokens}",
            "summary_sha256": str(index + 1) * 64,
            "thinking_controlled_format_basis_points": 10_000,
            "thinking_natural_close_basis_points": 9_500,
            "thinking_forced_close_basis_points": 500,
            "thinking_score_basis_points": 9_500,
            "nonthinking_score_basis_points": 6_500,
        }
        for index, tokens in enumerate(
            (10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000),
            start=1,
        )
    )
    payload = {
        "status": "succeeded",
        "campaign_id": "20260803T000000Z-m5-formal-campaign",
        "run_id": "formal-run",
        "physical_gpu_indices": (4, 5, 6, 7),
        "segment_count": 2,
        "interruption_tokens": 2_000_123,
        "resumed_from_tokens": 2_000_123,
        "final_tokens": 50_000_000,
        "export_sha256": "a" * 64,
        "interrupted_result_sha256": "b" * 64,
        "final_result_sha256": "c" * 64,
        "evaluation_id": "evaluation-50000000",
        "evaluation_summary_sha256": "6" * 64,
        "thinking_controlled_format_basis_points": 10_000,
        "thinking_natural_close_basis_points": 9_500,
        "thinking_forced_close_basis_points": 500,
        "thinking_score_basis_points": 9_500,
        "nonthinking_score_basis_points": 6_500,
        "staged_evaluations": staged_evaluations,
        "thermal_events_sha256": "d" * 64,
        "thermal_pause_count": 1,
        "max_observed_temperature_c": 84,
        "git_commit": "e" * 40,
    }

    assert M5FormalCampaignResult.model_validate(payload).segment_count == 2
    with pytest.raises(ValidationError, match="Resume point"):
        M5FormalCampaignResult.model_validate(payload | {"resumed_from_tokens": 2_000_124})


def _rank_memory() -> tuple[
    M5FormalRankMemory,
    M5FormalRankMemory,
    M5FormalRankMemory,
    M5FormalRankMemory,
]:
    return tuple(
        M5FormalRankMemory(
            rank=rank,
            physical_gpu_index=rank + 4,
            gpu_name="NVIDIA GeForce RTX 3090",
            peak_allocated_bytes=10,
            peak_reserved_bytes=20,
        )
        for rank in range(4)
    )  # type: ignore[return-value]


def _result_mapping() -> dict[str, object]:
    return {
        "status": "succeeded",
        "mode": "exact_resume",
        "run_id": "20260731T000000Z-m5-formal-a1b2c3d4-cafe",
        "config_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "environment_sha256": "e" * 64,
        "hardware_sha256": "f" * 64,
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "attention_architecture": "gqa",
        "dataset_version": "m5-dual-sft-v1-b5b9e839",
        "dataset_manifest_sha256": (
            "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
        ),
        "thinking_fraction_basis_points": 3000,
        "seed": 42,
        "world_size": 4,
        "global_step": 100,
        "local_sequence_cursor": 51_875,
        "supervised_tokens": 50_000_000,
        "completed_dataset_epochs": 50.0,
        "initial_loss": 2.0,
        "final_loss": 1.0,
        "duration_seconds": 60.0,
        "rank_memory": _rank_memory(),
        "latest_checkpoint": "checkpoint-tokens-0050000000",
        "evaluation_checkpoints": (
            "checkpoint-tokens-0010000000",
            "checkpoint-tokens-0020000000",
            "checkpoint-tokens-0030000000",
            "checkpoint-tokens-0040000000",
            "checkpoint-tokens-0050000000",
        ),
        "evaluation_export_sha256s": tuple(str(index) * 64 for index in range(1, 6)),
        "resumed_from_tokens": 2_000_000,
        "export_sha256": "c" * 64,
    }


def test_formal_environment_and_hardware_require_stable_ordered_identity() -> None:
    environment = M5FormalEnvironment.model_validate(
        {
            "python_version": "3.11.14",
            "python_implementation": "CPython",
            "python_executable": "/private/venv/bin/python",
            "torch_version": "2.7.1+cu118",
            "cuda_runtime": "11.8",
            "transformers_version": "4.57.6",
            "packages": (
                {"name": "torch", "version": "2.7.1+cu118"},
                {"name": "transformers", "version": "4.57.6"},
            ),
        }
    )
    assert environment.packages[0].name == "torch"
    with pytest.raises(ValidationError, match="unique and ordered"):
        M5FormalEnvironment.model_validate(
            environment.to_dict()
            | {
                "packages": (
                    {"name": "transformers", "version": "4.57.6"},
                    {"name": "torch", "version": "2.7.1+cu118"},
                )
            }
        )

    hardware = M5FormalHardware.model_validate(
        {
            "hostname": "private-host",
            "platform": "Linux",
            "machine": "x86_64",
            "cpu_count": 64,
            "cuda_driver": "535.183.06",
            "selected_gpus": tuple(
                {
                    "local_rank": rank,
                    "physical_gpu_index": rank,
                    "uuid": f"GPU-00000000-0000-0000-0000-00000000000{rank}",
                    "name": "NVIDIA GeForce RTX 3090",
                    "memory_total_mib": 24576,
                    "pci_bus_id": f"00000000:{rank + 1:02X}:00.0",
                }
                for rank in range(4)
            ),
            "gpu_topology": "GPU0 GPU1 CPU Affinity",
        }
    )
    assert tuple(item.local_rank for item in hardware.selected_gpus) == (0, 1, 2, 3)


def test_formal_config_freezes_four_gpu_fifty_million_route() -> None:
    config = load_m5_sft_config(Path("configs/sft/m5_formal_qwen3_0_6b.yaml"))

    assert config.run.purpose == "formal"
    assert config.parallel.world_size == 4
    assert config.training.max_train_tokens == 50_000_000
    assert config.checkpoint.save_interval_tokens == 2_000_000
    assert config.training.micro_batch_size == 4
    assert config.global_batch_size == 32


def test_formal_result_requires_completion_export_and_distinct_ranks() -> None:
    result = M5FormalRunResult.model_validate(_result_mapping())

    assert result.status == "succeeded"
    assert result.resumed_from_tokens == 2_000_000

    with pytest.raises(ValidationError, match="50M Tokens and export"):
        M5FormalRunResult.model_validate(_result_mapping() | {"supervised_tokens": 49_000_000})
    duplicated = list(_rank_memory())
    duplicated[3] = duplicated[2]
    with pytest.raises(ValidationError, match="ordered 0–3"):
        M5FormalRunResult.model_validate(_result_mapping() | {"rank_memory": tuple(duplicated)})


def test_formal_checkpoint_store_detects_payload_corruption(tmp_path: Path) -> None:
    config = load_m5_sft_config(Path("configs/sft/m5_formal_qwen3_0_6b.yaml"))
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    store = M5FormalCheckpointStore(tmp_path / "checkpoints", keep_last=2)
    manifest = store.save(
        model=model,
        optimizer=optimizer,
        progress=M5FormalProgress(
            global_step=1,
            local_sequence_cursor=10,
            supervised_tokens=2_000_000,
            initial_loss=2.0,
            final_loss=1.5,
            evaluation_checkpoints=(),
        ),
        rank_rng=({"rank": 0}, {"rank": 1}, {"rank": 2}, {"rank": 3}),
        config=config,
        config_sha256="d" * 64,
        dataset_manifest_sha256=(
            "607b3b1a73ae03d5f183f11d8c4824b04243ed30f6352567ce7bd7a972c962f6"
        ),
        source_sequence_count=4,
        environment_sha256="a" * 64,
        hardware_sha256="b" * 64,
        run_id="unit-formal-run",
        git_commit="e" * 40,
        pin_reason="interruption",
    )

    assert store.latest_valid() == manifest
    state = tmp_path / "checkpoints" / manifest.checkpoint_id / "training_state.pt"
    with state.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(M5FormalTrainingError, match="integrity"):
        store.validate(manifest.checkpoint_id)
