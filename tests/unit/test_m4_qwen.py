from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import yaml
from pydantic import ValidationError

from tinyllm.schemas import canonical_config_hash, generate_run_id
from tinyllm.training.m4_dataset import M4RegisteredDatasetView
from tinyllm.training.m4_qwen_config import (
    M4QwenConfigError,
    load_m4_qwen_config,
    m4_qwen_config_from_mapping,
)
from tinyllm.training.m4_qwen_schema import M4QwenRankMemory, M4QwenRunResult

CONFIG = Path("configs/fsdp2/qwen3_8b_four_gpu_formal.yaml")


def _mapping() -> dict[str, object]:
    return cast(dict[str, object], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


def test_formal_m4_qwen_config_freezes_acceptance_boundary() -> None:
    config = load_m4_qwen_config(CONFIG)

    assert config.model.repository == "Qwen/Qwen3-8B"
    assert config.model.revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert config.data.dataset_version == "m2-sft-v1-f82ff32e"
    assert config.data.sequence_length == 512
    assert config.training.max_steps == 50
    assert config.distributed.world_size == 4
    assert config.distributed.activation_checkpointing is True
    assert config.checkpoint.save_steps == 25
    assert config.global_batch_size == 4


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("training", "max_steps", 49, "exactly 50"),
        ("training", "micro_batch_size", 2, "micro_batch_size=1"),
        ("distributed", "world_size", 2, "four-GPU"),
        ("distributed", "activation_checkpointing", False, "Activation Checkpointing"),
        ("checkpoint", "save_steps", 20, "Step 25"),
    ],
)
def test_formal_m4_qwen_config_rejects_silent_contract_drift(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    mapping = _mapping()
    target = mapping[section]
    assert isinstance(target, dict)
    target[field] = value

    with pytest.raises(M4QwenConfigError, match=message):
        m4_qwen_config_from_mapping(mapping)


def test_registered_m4_view_is_deterministic_and_preserves_assistant_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_mapping = _mapping()
    data = config_mapping["data"]
    assert isinstance(data, dict)
    data["max_sequences"] = 200
    config = m4_qwen_config_from_mapping(config_mapping)
    packs = tuple(
        SimpleNamespace(
            split="train",
            sample_token_counts=(3,),
            input_ids=(151644, index + 1, index + 2),
            labels=(-100, index + 1, index + 2),
        )
        for index in range(200)
    )
    registered = SimpleNamespace(
        manifest=SimpleNamespace(
            dataset_version="m2-sft-v1-f82ff32e",
            content_sha256="f82ff32ee98cb852fe6779774d9cce75a71e9430da72a6e5e1f4e3f7c2efd108",
        ),
        iter_packs=lambda: iter(packs),
    )
    monkeypatch.setattr(
        "tinyllm.training.m4_dataset.open_registered_dataset",
        lambda **_: registered,
    )

    first = M4RegisteredDatasetView(artifact_root=Path("/data/test"), config=config.data)
    second = M4RegisteredDatasetView(artifact_root=Path("/data/test"), config=config.data)

    assert first.manifest == second.manifest
    assert first.manifest.sequence_count == 200
    assert first.manifest.supervised_token_count == 400
    assert len(first) == 200
    item = first[0]
    assert item["input_ids"].shape == (512,)
    assert item["labels"][:4].tolist() == [-100, 1, 2, -100]
    assert item["attention_mask"][:4].tolist() == [1, 1, 1, 0]
    assert item["position_ids"][:4].tolist() == [0, 1, 2, 0]


def _memory() -> tuple[M4QwenRankMemory, ...]:
    return tuple(
        M4QwenRankMemory(
            rank=rank,
            physical_gpu_index=rank + 5,
            peak_allocated_bytes=100,
            peak_reserved_bytes=200,
            final_allocated_bytes=90,
            final_reserved_bytes=200,
        )
        for rank in range(4)
    )


def _result_values() -> dict[str, object]:
    config_hash = canonical_config_hash({"formal": "m4"})
    return {
        "status": "probe_succeeded",
        "mode": "probe",
        "run_id": generate_run_id("m4-qwen", config_hash, nonce="cafe"),
        "artifact_dir": Path("/data/test/run"),
        "config_sha256": config_hash,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "model_artifact_sha256": "b" * 64,
        "data_view_sha256": "c" * 64,
        "world_size": 4,
        "global_step": 1,
        "durable_metric_records": 1,
        "activation_checkpointed_layers": 36,
        "rank_memory": _memory(),
    }


def test_m4_qwen_result_distinguishes_probe_and_exact_resume() -> None:
    probe = M4QwenRunResult.model_validate(_result_values())
    assert probe.checkpoint_id is None

    values = _result_values()
    values.update(
        {
            "status": "succeeded",
            "mode": "exact_resume",
            "global_step": 50,
            "checkpoint_id": "checkpoint-step-00000050",
            "model_parameter_sha256": "d" * 64,
            "resumed_from_step": 25,
            "durable_metric_records": 50,
            "export_sha256": "e" * 64,
        }
    )
    resumed = M4QwenRunResult.model_validate(values)
    assert resumed.resumed_from_step == 25


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"artifact_dir": Path("relative")}, "absolute"),
        ({"checkpoint_id": "checkpoint-step-00000001"}, "Probe cannot claim"),
        ({"status": "succeeded"}, "Probe must succeed"),
        (
            {
                "status": "succeeded",
                "mode": "fresh",
                "global_step": 50,
                "checkpoint_id": "checkpoint-step-00000049",
                "model_parameter_sha256": "d" * 64,
                "durable_metric_records": 50,
            },
            "describe global_step",
        ),
    ],
)
def test_m4_qwen_result_rejects_invalid_claims(
    updates: dict[str, object],
    message: str,
) -> None:
    values = _result_values()
    values.update(updates)
    with pytest.raises(ValidationError, match=message):
        M4QwenRunResult.model_validate(values)
