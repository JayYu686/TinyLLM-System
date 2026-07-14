from __future__ import annotations

import pytest
import torch
from torch import nn

from tinyllm.training.config import TrainingLoopConfig
from tinyllm.training.scheduler import (
    build_adamw,
    build_warmup_cosine_scheduler,
    warmup_cosine_multiplier,
)


def loop_config(**overrides: object) -> TrainingLoopConfig:
    values: dict[str, object] = {
        "max_steps": 4,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "warmup_steps": 2,
    }
    values.update(overrides)
    return TrainingLoopConfig.model_validate(values)


def test_warmup_cosine_multiplier_has_explicit_boundaries() -> None:
    actual = [warmup_cosine_multiplier(index, max_steps=4, warmup_steps=2) for index in range(5)]

    assert actual == pytest.approx([0.5, 1.0, 1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        warmup_cosine_multiplier(-1, max_steps=4, warmup_steps=2)


def test_adamw_separates_matrix_and_vector_weight_decay() -> None:
    model = nn.Sequential(nn.Linear(4, 3, bias=True), nn.LayerNorm(3))
    optimizer = build_adamw(model, loop_config())

    groups = {float(group["weight_decay"]): group["params"] for group in optimizer.param_groups}
    assert set(groups) == {0.0, 0.1}
    assert len(groups[0.1]) == 1
    assert len(groups[0.0]) == 3


def test_scheduler_exposes_learning_rate_used_by_each_step() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    scheduler = build_warmup_cosine_scheduler(optimizer, loop_config())
    used: list[float] = []

    for _ in range(4):
        parameter.grad = torch.ones_like(parameter)
        used.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    assert used == pytest.approx([0.005, 0.01, 0.01, 0.005])
