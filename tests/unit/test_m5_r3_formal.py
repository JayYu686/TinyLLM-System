from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts.run_m5_r3_formal_source_cpu_smoke import run_smoke
from tinyllm.data.m5_r3_formal import (
    generate_m5_r3_formal_contexts,
    load_m5_r3_formal_source_config,
    m5_r3_formal_source_config_sha256,
)
from tinyllm.data.m5_r3_formal_schema import M5R3FormalCPUSmoke
from tinyllm.data.m5_r3_p1_schema import M5R3P1TaskContext

CONFIG = Path("configs/data/m5_r3_formal_source.yaml")


def test_m5_r3_formal_config_and_tasks_are_frozen_and_balanced() -> None:
    config = load_m5_r3_formal_source_config(CONFIG)
    contexts = generate_m5_r3_formal_contexts(config)

    assert (
        m5_r3_formal_source_config_sha256(config)
        == "08ccc14ca01173df853b60065aad978833dd617fc5ae38c01263e2023f5d8eba"
    )
    assert len(contexts) == len({item.task.id for item in contexts}) == 240
    assert len({item.task.prompt_sha256 for item in contexts}) == 240
    assert len({item.evidence_anchor for item in contexts}) == 240
    assert Counter((item.task.task_family, item.task.language) for item in contexts) == {
        ("config", "en"): 84,
        ("config", "zh"): 36,
        ("log_diagnosis", "en"): 84,
        ("log_diagnosis", "zh"): 36,
    }


def test_m5_r3_formal_labels_are_balanced_per_family() -> None:
    contexts = generate_m5_r3_formal_contexts(load_m5_r3_formal_source_config(CONFIG))

    for family in ("config", "log_diagnosis"):
        assert Counter(
            item.expected_label for item in contexts if item.task.task_family == family
        ) == {
            label: 30
            for label in (
                (
                    "forbidden_truncation",
                    "missing_checkpoint",
                    "unsupported_precision",
                    "world_size_mismatch",
                )
                if family == "config"
                else (
                    "collective_timeout",
                    "cuda_oom",
                    "disk_full",
                    "non_finite_gradient",
                )
            )
        }


def test_m5_r3_formal_context_survives_json_round_trip() -> None:
    context = generate_m5_r3_formal_contexts(load_m5_r3_formal_source_config(CONFIG))[0]

    restored = M5R3P1TaskContext.model_validate_json(context.model_dump_json())

    assert restored == context
    assert isinstance(restored.allowed_labels, tuple)


def test_m5_r3_formal_cpu_smoke_matches_committed_evidence() -> None:
    committed = M5R3FormalCPUSmoke.model_validate_json(
        Path("reports/m5/raw/m5_r3_formal_source_cpu_smoke.json").read_text(encoding="utf-8")
    )
    result = run_smoke()

    assert result == committed
    assert result.model_generated is False
    assert result.quality_metric is False
    assert result.accepted_samples == 240
    assert result.selected_samples == 160
    assert all(item.gate_passed for item in result.stratum_results)
    assert result.contamination.status == "pass"
    assert result.gpu_expansion_authorized is True
    assert result.r3_mixture_authorized is False
    assert result.r3_training_authorized is False
