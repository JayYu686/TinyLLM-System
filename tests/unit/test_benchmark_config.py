from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinyllm.benchmark import (
    DDPBenchmarkConfig,
    DDPBenchmarkConfigError,
    load_ddp_benchmark_config,
    resolve_benchmark_profile,
    validate_formal_m3_config,
)
from tinyllm.models.tinygpt import TinyGPT


def test_formal_m3_config_resolves_exact_strong_and_weak_batches() -> None:
    config = load_ddp_benchmark_config(Path("configs/benchmark/m3_tinygpt_120m_ddp.yaml"))
    validate_formal_m3_config(config)

    model = TinyGPT(config.model)
    assert model.parameter_count() == 117_197_568
    assert [
        resolve_benchmark_profile(
            config,
            profile="strong",
            world_size=world_size,
            repeat=1,
        ).gradient_accumulation_steps
        for world_size in (1, 2, 4, 8)
    ] == [8, 4, 2, 1]
    assert [
        resolve_benchmark_profile(
            config,
            profile="weak",
            world_size=world_size,
            repeat=2,
        ).global_batch_size
        for world_size in (1, 2, 4, 8)
    ] == [1, 2, 4, 8]
    first = resolve_benchmark_profile(
        config,
        profile="weak",
        world_size=2,
        repeat=1,
    )
    second = resolve_benchmark_profile(
        config,
        profile="weak",
        world_size=2,
        repeat=2,
    )
    assert first.profiler_steps == 5
    assert second.profiler_steps == 0
    assert second.seed == first.seed + 1


def test_formal_config_rejects_unsupported_strong_world_size() -> None:
    config = load_ddp_benchmark_config(Path("configs/benchmark/m3_tinygpt_120m_ddp.yaml"))
    with pytest.raises(DDPBenchmarkConfigError, match="divisible"):
        resolve_benchmark_profile(
            config,
            profile="strong",
            world_size=3,
            repeat=1,
        )


def test_benchmark_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        Path("configs/benchmark/m3_ddp_cpu_test.yaml").read_text(encoding="utf-8")
        + "\nunknown: true\n",
        encoding="utf-8",
    )
    with pytest.raises(DDPBenchmarkConfigError, match="Extra inputs"):
        load_ddp_benchmark_config(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw["model"].update(vocab_size=65), "vocab_size"),
        (lambda raw: raw["data"].update(sequence_length=17), "max_sequence_length"),
        (
            lambda raw: raw["training"].update(weak_per_rank_batch_size=3, micro_batch_size=2),
            "divisible",
        ),
        (lambda raw: raw["training"].update(learning_rate_warmup_steps=4), "total benchmark"),
        (lambda raw: raw["precision"].update(dtype="bf16"), "Gloo"),
        (
            lambda raw: (
                raw["distributed"].update(backend="nccl"),
                raw["precision"].update(dtype="fp32"),
            ),
            "NCCL",
        ),
        (lambda raw: raw["benchmark"].update(profiler_steps=3), "measurement_steps"),
        (lambda raw: raw["benchmark"].update(profiler_repeat=4), "repetitions"),
    ],
)
def test_benchmark_config_rejects_cross_field_drift(
    mutate: object,
    message: str,
) -> None:
    config = load_ddp_benchmark_config(Path("configs/benchmark/m3_ddp_cpu_test.yaml"))
    raw = config.to_dict()
    assert callable(mutate)
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        DDPBenchmarkConfig.model_validate_json(json.dumps(raw))


def test_profile_resolution_and_formal_validation_reject_runtime_drift() -> None:
    config = load_ddp_benchmark_config(Path("configs/benchmark/m3_tinygpt_120m_ddp.yaml"))
    with pytest.raises(DDPBenchmarkConfigError, match="world_size"):
        resolve_benchmark_profile(config, profile="weak", world_size=0, repeat=1)
    with pytest.raises(DDPBenchmarkConfigError, match="repeat"):
        resolve_benchmark_profile(config, profile="weak", world_size=1, repeat=4)

    raw = config.to_dict()
    raw["model"]["hidden_size"] = 384
    drifted = DDPBenchmarkConfig.model_validate_json(json.dumps(raw))
    with pytest.raises(DDPBenchmarkConfigError, match="formal M3"):
        validate_formal_m3_config(drifted)


def test_benchmark_loader_rejects_extension_missing_and_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(DDPBenchmarkConfigError, match="yaml"):
        load_ddp_benchmark_config(tmp_path / "config.json")
    with pytest.raises(DDPBenchmarkConfigError, match="cannot read"):
        load_ddp_benchmark_config(tmp_path / "missing.yaml")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("run: [", encoding="utf-8")
    with pytest.raises(DDPBenchmarkConfigError, match="invalid benchmark YAML"):
        load_ddp_benchmark_config(invalid)
