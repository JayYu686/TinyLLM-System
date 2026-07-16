from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from tinyllm.training import TrainingError, TrainingErrorCode
from tinyllm.training.fsdp2 import (
    _gradient_norm,
    _require_finite_scalar_loss,
    full_fsdp2_state_sha256,
    local_fsdp2_shard_evidence,
    run_fsdp2_correctness,
)


def test_fsdp2_runtime_rejects_relative_output_before_torchrun() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        run_fsdp2_correctness(
            config_path=Path("configs/fsdp2/tinygpt_debug_gloo_smoke.yaml"),
            output_root=Path("relative-runs"),
        )


def test_fsdp2_runtime_rejects_unsharded_parameters() -> None:
    model = nn.Linear(2, 2)

    with pytest.raises(TrainingError) as error:
        local_fsdp2_shard_evidence(model, rank=0, device_type="cpu")
    assert error.value.code == TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH
    assert len(full_fsdp2_state_sha256(model)) == 64


@pytest.mark.parametrize("loss", [None, torch.tensor([1.0]), torch.tensor(float("nan"))])
def test_fsdp2_runtime_rejects_missing_nonscalar_or_nonfinite_loss(
    loss: torch.Tensor | None,
) -> None:
    with pytest.raises(TrainingError) as error:
        _require_finite_scalar_loss(loss, global_step=1)
    assert error.value.code == TrainingErrorCode.NON_FINITE_LOSS
    assert error.value.context["global_step"] == 1


def test_fsdp2_gradient_norm_rejects_missing_and_nonfinite_gradients() -> None:
    model = nn.Linear(2, 1)
    with pytest.raises(TrainingError) as missing:
        _gradient_norm(model, max_norm=1.0)
    assert missing.value.code == TrainingErrorCode.NON_FINITE_GRADIENT

    model(torch.full((1, 2), float("nan"))).sum().backward()
    with pytest.raises(TrainingError) as nonfinite:
        _gradient_norm(model, max_norm=1.0)
    assert nonfinite.value.code == TrainingErrorCode.NON_FINITE_GRADIENT
