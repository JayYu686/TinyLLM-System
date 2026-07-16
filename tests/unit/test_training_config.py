from __future__ import annotations

from pathlib import Path

import pytest

from tinyllm.schemas import canonical_config_hash
from tinyllm.training.config import (
    TrainingConfigError,
    load_training_config,
    training_config_from_mapping,
)


def valid_mapping() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "run": {"name": "unit-test", "seed": 42},
        "model": {
            "vocab_size": 32,
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 4,
            "intermediate_size": 64,
            "max_sequence_length": 16,
            "rope_theta": 10_000.0,
            "rms_norm_epsilon": 1.0e-6,
            "dropout": 0.0,
            "tie_word_embeddings": True,
        },
        "data": {
            "kind": "toy",
            "vocab_size": 32,
            "sequence_length": 16,
            "num_samples": 16,
        },
        "training": {
            "max_steps": 10,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 3.0e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "warmup_steps": 2,
        },
        "precision": {"dtype": "fp32", "allow_tf32": False, "use_grad_scaler": False},
        "checkpoint": {
            "output_dir": "runs/unit-test",
            "save_steps": 5,
            "keep_last": 2,
            "resume": "none",
        },
    }


def test_load_m1_example_config() -> None:
    config = load_training_config(Path("configs/pretrain/tinygpt_debug.yaml"))

    assert config.schema_version == "1.0"
    assert config.model.head_dimension == 32
    assert config.training.global_batch_size == 8
    assert config.global_batch_size == 8
    assert config.distributed.strategy == "single"
    assert "distributed" not in config.to_dict()
    assert config.data.vocab_size == config.model.vocab_size

    cpu_smoke_config = load_training_config(Path("configs/pretrain/tinygpt_debug_cpu_smoke.yaml"))
    assert canonical_config_hash(cpu_smoke_config) == (
        "1dc537638ea9984943a4423c38400073735d331c39885fd2d657bd822160fbd7"
    )

    gpu_config = load_training_config(
        Path("configs/pretrain/tinygpt_debug_rtx3090_bf16_smoke.yaml")
    )
    assert gpu_config.precision.dtype == "bf16"
    assert gpu_config.precision.use_grad_scaler is False
    assert gpu_config.training.max_steps == 40

    ddp_config = load_training_config(Path("configs/pretrain/tinygpt_debug_ddp_cpu_smoke.yaml"))
    assert ddp_config.distributed.strategy == "ddp"
    assert ddp_config.distributed.backend == "gloo"
    assert ddp_config.distributed.world_size == 2
    assert ddp_config.global_batch_size == 8
    assert ddp_config.to_dict()["distributed"] == ddp_config.distributed.to_dict()


def test_training_config_rejects_unknown_fields() -> None:
    mapping = valid_mapping()
    mapping["surprise"] = True

    with pytest.raises(TrainingConfigError, match="unknown config field"):
        training_config_from_mapping(mapping)


def test_training_config_rejects_cross_field_mismatch() -> None:
    mapping = valid_mapping()
    data = mapping["data"]
    assert isinstance(data, dict)
    data["vocab_size"] = 31

    with pytest.raises(TrainingConfigError, match="model.vocab_size"):
        training_config_from_mapping(mapping)


def test_training_config_rejects_non_yaml_extension(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(TrainingConfigError, match=".yaml"):
        load_training_config(path)


def test_training_config_schema_rejects_coercion() -> None:
    mapping = valid_mapping()
    training = mapping["training"]
    assert isinstance(training, dict)
    training["max_steps"] = "10"

    with pytest.raises(TrainingConfigError, match="max_steps"):
        training_config_from_mapping(mapping)


def test_ddp_config_rejects_ambiguous_batch_sampler_and_resume_contracts() -> None:
    mapping = valid_mapping()
    training = mapping["training"]
    checkpoint = mapping["checkpoint"]
    assert isinstance(training, dict)
    assert isinstance(checkpoint, dict)
    training["max_steps"] = 1
    training["gradient_accumulation_steps"] = 1
    training["warmup_steps"] = 0
    mapping["distributed"] = {
        "strategy": "ddp",
        "backend": "gloo",
        "world_size": 2,
        "timeout_seconds": 120,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
    }
    config = training_config_from_mapping(mapping)
    assert config.global_batch_size == 4

    checkpoint["resume"] = "auto"
    assert training_config_from_mapping(mapping).checkpoint.resume == "auto"
    checkpoint["resume"] = "warm"
    with pytest.raises(TrainingConfigError, match="none, auto, or exact"):
        training_config_from_mapping(mapping)
    checkpoint["resume"] = "none"

    data = mapping["data"]
    assert isinstance(data, dict)
    data["num_samples"] = 15
    with pytest.raises(TrainingConfigError, match="divisible by world_size"):
        training_config_from_mapping(mapping)
