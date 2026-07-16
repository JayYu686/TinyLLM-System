from __future__ import annotations

from pathlib import Path

import pytest

from tinyllm.training.fsdp2_config import (
    FSDP2ConfigError,
    fsdp2_config_from_mapping,
    load_fsdp2_config,
)


def valid_mapping() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "run": {"name": "fsdp2-unit", "seed": 42},
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
            "max_steps": 2,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 3.0e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "warmup_steps": 1,
        },
        "precision": {"dtype": "fp32", "allow_tf32": False, "use_grad_scaler": False},
        "distributed": {
            "strategy": "fsdp2",
            "backend": "gloo",
            "device_type": "cpu",
            "world_size": 2,
            "timeout_seconds": 120,
            "reshard_after_forward": True,
            "cpu_offload": False,
            "activation_checkpointing": False,
        },
    }


def test_load_fsdp2_gloo_smoke_config() -> None:
    config = load_fsdp2_config(Path("configs/fsdp2/tinygpt_debug_gloo_smoke.yaml"))

    assert config.distributed.strategy == "fsdp2"
    assert config.distributed.backend == "gloo"
    assert config.distributed.device_type == "cpu"
    assert config.distributed.world_size == 2
    assert config.global_batch_size == 4
    assert config.training.max_steps == 2
    assert config.to_dict()["distributed"]["reshard_after_forward"] is True


def test_fsdp2_config_rejects_unknown_and_coerced_fields() -> None:
    mapping = valid_mapping()
    mapping["surprise"] = True
    with pytest.raises(FSDP2ConfigError, match="unknown config field"):
        fsdp2_config_from_mapping(mapping)

    mapping = valid_mapping()
    distributed = mapping["distributed"]
    assert isinstance(distributed, dict)
    distributed["world_size"] = "2"
    with pytest.raises(FSDP2ConfigError, match="world_size"):
        fsdp2_config_from_mapping(mapping)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("distributed", "device_type", "cuda", "gloo FSDP2"),
        ("precision", "dtype", "bf16", "CPU/Gloo"),
        ("training", "gradient_accumulation_steps", 2, "gradient accumulation"),
        ("training", "max_steps", 11, "at most 10"),
        ("distributed", "world_size", 5, "less than or equal to 4"),
    ],
)
def test_fsdp2_config_rejects_unvalidated_runtime_combinations(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    mapping = valid_mapping()
    target = mapping[section]
    assert isinstance(target, dict)
    target[field] = value

    with pytest.raises(FSDP2ConfigError, match=message):
        fsdp2_config_from_mapping(mapping)


def test_fsdp2_config_rejects_non_yaml_extension(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(FSDP2ConfigError, match=".yaml"):
        load_fsdp2_config(path)
