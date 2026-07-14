from __future__ import annotations

from pathlib import Path

import pytest

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
    assert config.data.vocab_size == config.model.vocab_size


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
