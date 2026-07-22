from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import yaml

from tinyllm.training.m5_config import (
    M5ConfigError,
    load_m5_sft_config,
    m5_sft_config_from_mapping,
)

AnyMutation = Callable[[dict[str, object]], object]


def _full_formal_mapping() -> dict[str, object]:
    return {
        "config_kind": "qwen_sft",
        "schema_version": "1.0",
        "run": {"name": "m5-qwen3-0-6b-full-sft", "seed": 42, "purpose": "formal"},
        "model": {
            "repository": "Qwen/Qwen3-0.6B",
            "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "model_type": "qwen3",
            "license": "Apache-2.0",
            "attention_architecture": "gqa",
            "adaptation": "full_sft",
            "trust_remote_code": False,
        },
        "data": {
            "dataset_version": "m5-dual-sft-v1-a1b2c3d4",
            "parent_dataset_version": "m2-sft-v1-f82ff32e",
            "split": "train",
            "sequence_length": 1024,
            "assistant_only_loss": True,
            "mode": "dual",
            "thinking_token_fraction": 0.3,
            "mix_manifest_sha256": "a" * 64,
        },
        "reasoning": {
            "explicit_mode_selection": True,
            "supervise_visible_reasoning": True,
            "nonthinking_template_id": "qwen3-chatml-nonthinking-v1",
            "nonthinking_template_sha256": (
                "d41161e0416a1047b0f31cce1497e610a4050fbe4d3fb7bda19cc56a1523cb33"
            ),
            "thinking_template_id": "qwen3-chatml-thinking-v1",
            "thinking_template_sha256": (
                "4786143dbb7adb72a922d5efdcbe6596f2d65dcdc35d7bbf1b22830b795c2af9"
            ),
        },
        "training": {
            "max_train_tokens": 50_000_000,
            "evaluation_interval_tokens": 10_000_000,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2.0e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "warmup_tokens": 1_000_000,
            "gradient_checkpointing": True,
            "max_job_duration_seconds": 43_200,
        },
        "precision": {"dtype": "bf16", "allow_tf32": True, "use_grad_scaler": False},
        "parallel": {
            "strategy": "ddp",
            "backend": "nccl",
            "device_type": "cuda",
            "world_size": 4,
            "timeout_seconds": 1800,
        },
        "checkpoint": {
            "save_interval_tokens": 2_000_000,
            "keep_last": 2,
            "resume": "auto",
        },
        "evaluation": {
            "reasoning_dev_version": "m5-reasoning-dev-v1",
            "compare_modes_separately": True,
            "consume_m6_frozen_results": False,
        },
    }


def _section(mapping: dict[str, object], name: str) -> dict[str, object]:
    return cast(dict[str, object], mapping[name])


def _lora_formal_mapping(*, adaptation: str = "lora") -> dict[str, object]:
    mapping = _full_formal_mapping()
    _section(mapping, "model").update(
        {
            "repository": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "adaptation": adaptation,
            "lora": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_scope": "attention_and_mlp_linear",
                "bias": "none",
            },
        }
    )
    _section(mapping, "training").update(
        {"max_train_tokens": 10_000_000, "evaluation_interval_tokens": 2_000_000}
    )
    _section(mapping, "parallel").update({"strategy": "single", "backend": None, "world_size": 1})
    _section(mapping, "checkpoint")["save_interval_tokens"] = 1_000_000
    return mapping


def _lora_policy(mapping: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], _section(mapping, "model")["lora"])


def test_formal_full_sft_contract_freezes_gqa_dual_mode_and_four_gpu_ddp() -> None:
    config = m5_sft_config_from_mapping(_full_formal_mapping())

    assert config.config_kind == "qwen_sft"
    assert config.model.attention_architecture == "gqa"
    assert config.model.adaptation == "full_sft"
    assert config.data.mode == "dual"
    assert config.data.thinking_token_fraction == 0.3
    assert config.parallel.world_size == 4
    assert config.global_batch_size == 8
    assert config.to_dict()["config_kind"] == "qwen_sft"


def test_formal_qwen3_8b_lora_contract_is_single_gpu_and_fixed_policy() -> None:
    mapping = _full_formal_mapping()
    _section(mapping, "model").update(
        {
            "repository": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "adaptation": "lora",
            "lora": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_scope": "attention_and_mlp_linear",
                "bias": "none",
            },
        }
    )
    _section(mapping, "training").update(
        {"max_train_tokens": 10_000_000, "evaluation_interval_tokens": 2_000_000}
    )
    _section(mapping, "parallel").update({"strategy": "single", "backend": None, "world_size": 1})
    _section(mapping, "checkpoint")["save_interval_tokens"] = 1_000_000

    config = m5_sft_config_from_mapping(mapping)

    assert config.model.lora is not None
    assert config.model.lora.rank == 16
    assert config.parallel.strategy == "single"


def test_ablation_contract_uses_pilot_data_one_gpu_and_exactly_one_million_tokens() -> None:
    mapping = _full_formal_mapping()
    _section(mapping, "run")["purpose"] = "ablation"
    _section(mapping, "data").update(
        {
            "dataset_version": "m5-reasoning-pilot-v1-a1b2c3d4",
            "thinking_token_fraction": 0.0,
        }
    )
    _section(mapping, "training").update(
        {"max_train_tokens": 1_000_000, "evaluation_interval_tokens": 1_000_000}
    )
    _section(mapping, "parallel").update({"strategy": "single", "backend": None, "world_size": 1})
    _section(mapping, "checkpoint")["save_interval_tokens"] = 1_000_000

    config = m5_sft_config_from_mapping(mapping)

    assert config.run.purpose == "ablation"
    assert config.data.thinking_token_fraction == 0.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: _section(raw, "model").update({"attention_architecture": "mla"}), "gqa"),
        (lambda raw: _section(raw, "model").update({"revision": "f" * 40}), "pinned"),
        (lambda raw: _section(raw, "data").update({"thinking_token_fraction": 0.0}), "non-zero"),
        (
            lambda raw: _section(raw, "parallel").update(
                {"strategy": "single", "backend": None, "world_size": 1}
            ),
            "four-GPU",
        ),
        (lambda raw: raw.update({"unknown": True}), "unknown config field"),
    ],
)
def test_formal_config_rejects_scope_or_identity_drift(
    mutate: AnyMutation,
    message: str,
) -> None:
    mapping = _full_formal_mapping()
    mutate(mapping)

    with pytest.raises(M5ConfigError, match=message):
        m5_sft_config_from_mapping(mapping)


def test_qlora_requires_retained_bf16_oom_evidence() -> None:
    mapping = _lora_formal_mapping(adaptation="qlora")

    with pytest.raises(M5ConfigError, match="OOM evidence"):
        m5_sft_config_from_mapping(mapping)

    _section(mapping, "model")["bf16_lora_oom_evidence_run_id"] = (
        "20260722T000000Z-qwen3-8b-lora-oom-a1b2c3d4-cafe"
    )
    assert m5_sft_config_from_mapping(mapping).model.adaptation == "qlora"


@pytest.mark.parametrize(
    ("mapping_factory", "mutate", "message"),
    [
        (
            _lora_formal_mapping,
            lambda raw: _lora_policy(raw).update({"dropout": 0.1}),
            "dropout",
        ),
        (
            _full_formal_mapping,
            lambda raw: _section(raw, "model").update(
                {
                    "repository": "Qwen/Qwen3-8B",
                    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                }
            ),
            "restricted",
        ),
        (
            _full_formal_mapping,
            lambda raw: _section(raw, "model").update(
                {
                    "lora": {
                        "rank": 16,
                        "alpha": 32,
                        "dropout": 0.05,
                        "target_scope": "attention_and_mlp_linear",
                        "bias": "none",
                    }
                }
            ),
            "cannot define",
        ),
        (
            _full_formal_mapping,
            lambda raw: _section(raw, "model").update({"adaptation": "lora"}),
            "require Qwen3-8B",
        ),
        (
            _lora_formal_mapping,
            lambda raw: _section(raw, "model").update(
                {"bf16_lora_oom_evidence_run_id": "unexpected"}
            ),
            "cannot claim",
        ),
    ],
)
def test_model_route_rejects_invalid_adaptation_combinations(
    mapping_factory: Callable[[], dict[str, object]],
    mutate: AnyMutation,
    message: str,
) -> None:
    mapping = mapping_factory()
    mutate(mapping)

    with pytest.raises(M5ConfigError, match=message):
        m5_sft_config_from_mapping(mapping)


@pytest.mark.parametrize(
    ("section", "updates", "message"),
    [
        ("data", {"thinking_token_fraction": 0.4}, "0.0, 0.3, or 0.5"),
        ("training", {"warmup_tokens": 60_000_000}, "warmup_tokens"),
        ("training", {"evaluation_interval_tokens": 60_000_000}, "evaluation interval"),
        (
            "parallel",
            {"strategy": "single", "backend": "nccl", "world_size": 4},
            "single strategy",
        ),
        ("parallel", {"strategy": "ddp", "backend": "nccl", "world_size": 1}, "M5 DDP"),
        ("checkpoint", {"save_interval_tokens": 60_000_000}, "Checkpoint interval"),
    ],
)
def test_nested_contracts_reject_invalid_intervals_or_launches(
    section: str,
    updates: dict[str, object],
    message: str,
) -> None:
    mapping = _full_formal_mapping()
    _section(mapping, section).update(updates)

    with pytest.raises(M5ConfigError, match=message):
        m5_sft_config_from_mapping(mapping)


def test_smoke_contract_is_bounded_and_single_gpu() -> None:
    mapping = _full_formal_mapping()
    _section(mapping, "run")["purpose"] = "smoke"
    _section(mapping, "training").update(
        {
            "max_train_tokens": 100_000,
            "evaluation_interval_tokens": 100_000,
            "warmup_tokens": 0,
        }
    )
    _section(mapping, "parallel").update({"strategy": "single", "backend": None, "world_size": 1})
    _section(mapping, "checkpoint")["save_interval_tokens"] = 100_000
    assert m5_sft_config_from_mapping(mapping).run.purpose == "smoke"

    _section(mapping, "training")["max_train_tokens"] = 100_001
    with pytest.raises(M5ConfigError, match="100K"):
        m5_sft_config_from_mapping(mapping)

    _section(mapping, "training")["max_train_tokens"] = 100_000
    _section(mapping, "parallel").update({"strategy": "ddp", "backend": "nccl", "world_size": 4})
    with pytest.raises(M5ConfigError, match="one GPU"):
        m5_sft_config_from_mapping(mapping)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: _section(raw, "model").update(
                {
                    "repository": "Qwen/Qwen3-8B",
                    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                }
            ),
            "restricted",
        ),
        (
            lambda raw: _section(raw, "data").update(
                {"dataset_version": "m5-dual-sft-v1-a1b2c3d4"}
            ),
            "reasoning-pilot",
        ),
        (
            lambda raw: (
                _section(raw, "training").update(
                    {
                        "max_train_tokens": 999_999,
                        "evaluation_interval_tokens": 999_999,
                        "warmup_tokens": 0,
                    }
                ),
                _section(raw, "checkpoint").update({"save_interval_tokens": 999_999}),
            ),
            "exactly 1M",
        ),
        (
            lambda raw: _section(raw, "parallel").update(
                {"strategy": "ddp", "backend": "nccl", "world_size": 4}
            ),
            "one GPU",
        ),
    ],
)
def test_ablation_rejects_protocol_drift(mutate: AnyMutation, message: str) -> None:
    mapping = _full_formal_mapping()
    _section(mapping, "run")["purpose"] = "ablation"
    _section(mapping, "data")["dataset_version"] = "m5-reasoning-pilot-v1-a1b2c3d4"
    _section(mapping, "training").update(
        {
            "max_train_tokens": 1_000_000,
            "evaluation_interval_tokens": 1_000_000,
            "warmup_tokens": 0,
        }
    )
    _section(mapping, "parallel").update({"strategy": "single", "backend": None, "world_size": 1})
    _section(mapping, "checkpoint")["save_interval_tokens"] = 1_000_000
    mutate(mapping)

    with pytest.raises(M5ConfigError, match=message):
        m5_sft_config_from_mapping(mapping)


@pytest.mark.parametrize(
    ("mapping_factory", "section", "updates", "message"),
    [
        (
            _full_formal_mapping,
            "data",
            {"dataset_version": "m5-reasoning-pilot-v1-a1b2c3d4"},
            "dual-sft",
        ),
        (_full_formal_mapping, "training", {"max_train_tokens": 49_000_000}, "50M–100M"),
        (
            _full_formal_mapping,
            "training",
            {"evaluation_interval_tokens": 5_000_000},
            "every 10M",
        ),
        (
            _full_formal_mapping,
            "checkpoint",
            {"save_interval_tokens": 1_000_000},
            "every 2M",
        ),
        (
            _lora_formal_mapping,
            "parallel",
            {"strategy": "ddp", "backend": "nccl", "world_size": 4},
            "one GPU",
        ),
        (_lora_formal_mapping, "training", {"max_train_tokens": 9_000_000}, "10M–30M"),
        (
            _lora_formal_mapping,
            "training",
            {"evaluation_interval_tokens": 1_000_000},
            "every 2M",
        ),
        (
            _lora_formal_mapping,
            "checkpoint",
            {"save_interval_tokens": 2_000_000},
            "every 1M",
        ),
    ],
)
def test_formal_routes_reject_budget_or_interval_drift(
    mapping_factory: Callable[[], dict[str, object]],
    section: str,
    updates: dict[str, object],
    message: str,
) -> None:
    mapping = mapping_factory()
    _section(mapping, section).update(updates)

    with pytest.raises(M5ConfigError, match=message):
        m5_sft_config_from_mapping(mapping)


def test_m5_yaml_loader_reports_extension_parse_and_schema_failures(tmp_path: Path) -> None:
    with pytest.raises(M5ConfigError, match="extension"):
        load_m5_sft_config(tmp_path / "config.json")
    with pytest.raises(M5ConfigError, match="cannot read"):
        load_m5_sft_config(tmp_path / "missing.yaml")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_version: [", encoding="utf-8")
    with pytest.raises(M5ConfigError, match="invalid"):
        load_m5_sft_config(invalid)

    valid = tmp_path / "valid.yaml"
    valid.write_text(yaml.safe_dump(_full_formal_mapping(), sort_keys=False), encoding="utf-8")
    assert load_m5_sft_config(valid).model.adaptation == "full_sft"
