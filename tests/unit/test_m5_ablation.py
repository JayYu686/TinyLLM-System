from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tinyllm.training.m5_ablation import (
    _capture_rng,
    _record_attempt_result,
    _restore_rng,
    group_loss_scale,
    model_export_sha256,
    token_learning_rate,
    validate_m5_initial_model,
)
from tinyllm.training.m5_ablation_schema import M5AblationRunResult
from tinyllm.training.m5_config import M5SFTConfig, load_m5_sft_config


def _result_mapping() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "status": "succeeded",
        "mode": "fresh",
        "run_id": "20260722T000000Z-m5-ablation-a1b2c3d4-cafe",
        "config_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "attention_architecture": "gqa",
        "mixture_version": "m5-ablation-mixture-v1-a1b2c3d4",
        "mixture_manifest_sha256": "c" * 64,
        "thinking_fraction_basis_points": 3000,
        "seed": 42,
        "physical_gpu_index": 9,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "global_step": 100,
        "supervised_tokens": 1_000_000,
        "sequence_cursor": 1500,
        "initial_loss": 2.0,
        "final_loss": 1.5,
        "duration_seconds": 60.0,
        "peak_allocated_bytes": 10,
        "peak_reserved_bytes": 20,
        "latest_checkpoint": "checkpoint-tokens-0001000000",
        "resumed_from_tokens": None,
        "export_sha256": "d" * 64,
    }


def test_token_scheduler_warms_up_and_cosine_decays() -> None:
    start = token_learning_rate(
        base_learning_rate=2e-5,
        tokens=0,
        warmup_tokens=100,
        total_tokens=1000,
    )
    warm = token_learning_rate(
        base_learning_rate=2e-5,
        tokens=100,
        warmup_tokens=100,
        total_tokens=1000,
    )
    end = token_learning_rate(
        base_learning_rate=2e-5,
        tokens=1000,
        warmup_tokens=100,
        total_tokens=1000,
    )

    assert 0 < start < warm
    assert warm == pytest.approx(2e-5)
    assert end == pytest.approx(0.0)


def test_accumulation_loss_scale_is_token_weighted() -> None:
    assert group_loss_scale(25, 100) == 0.25
    with pytest.raises(ValueError, match="invalid"):
        group_loss_scale(0, 100)


def test_warm_start_requires_the_configured_model_export_identity(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"warm-start")
    identity = model_export_sha256(model_dir)
    raw = load_m5_sft_config(Path("configs/sft/m6_gate_replay_r3_seed42.yaml")).to_dict()
    raw["model"].update(
        {
            "initialization": "m5_formal_snapshot",
            "initial_model_artifact_sha256": identity,
            "initial_training_run_id": "formal-run",
            "initial_checkpoint_id": "checkpoint-tokens-0010000532",
        }
    )
    config = M5SFTConfig.model_validate(raw)

    assert validate_m5_initial_model(config, model_dir) == identity
    (model_dir / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="differs from the configured snapshot"):
        validate_m5_initial_model(config, model_dir)


def test_rng_restore_returns_default_generator_state_to_cpu() -> None:
    state = _capture_rng()
    torch.manual_seed(123456)

    _restore_rng(state)

    expected = state["torch"]
    assert isinstance(expected, torch.Tensor)
    assert torch.equal(torch.get_rng_state(), expected.cpu())


def test_success_result_requires_exact_budget_and_export() -> None:
    result = M5AblationRunResult.model_validate(_result_mapping())

    assert result.supervised_tokens == 1_000_000

    mapping = _result_mapping()
    mapping["export_sha256"] = None
    with pytest.raises(ValueError, match="export"):
        M5AblationRunResult.model_validate(mapping)


def test_interrupted_and_exact_resume_result_are_explicit() -> None:
    mapping = _result_mapping()
    mapping.update(
        {
            "status": "interrupted",
            "mode": "exact_resume",
            "supervised_tokens": 100_000,
            "latest_checkpoint": "checkpoint-tokens-0000100000",
            "resumed_from_tokens": 50_000,
            "export_sha256": None,
        }
    )

    result = M5AblationRunResult.model_validate(mapping)

    assert result.status == "interrupted"
    assert result.resumed_from_tokens == 50_000


def test_attempt_results_preserve_interruption_before_exact_resume(tmp_path: Path) -> None:
    interrupted_mapping = _result_mapping()
    interrupted_mapping.update(
        {
            "status": "interrupted",
            "supervised_tokens": 100_000,
            "latest_checkpoint": "checkpoint-tokens-0000100000",
            "export_sha256": None,
        }
    )
    interrupted = M5AblationRunResult.model_validate(interrupted_mapping)
    _record_attempt_result(tmp_path, interrupted)

    resumed_mapping = _result_mapping()
    resumed_mapping.update({"mode": "exact_resume", "resumed_from_tokens": 100_000})
    resumed = M5AblationRunResult.model_validate(resumed_mapping)
    _record_attempt_result(tmp_path, resumed)

    assert (tmp_path / "attempts/fresh-interrupted-tokens-0000100000.json").is_file()
    assert (tmp_path / "attempts/exact_resume-succeeded-tokens-0001000000.json").is_file()
    assert (
        M5AblationRunResult.model_validate_json(
            (tmp_path / "result.json").read_text(encoding="utf-8")
        )
        == resumed
    )
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 2
