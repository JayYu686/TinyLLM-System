from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from tinyllm.models.tinygpt import TinyGPT
from tinyllm.training import TrainingError, TrainingErrorCode
from tinyllm.training.fsdp2 import (
    _apply_activation_checkpointing,
    _gradient_norm,
    _require_finite_scalar_loss,
    _validate_failure_injection,
    full_fsdp2_state_sha256,
    local_fsdp2_shard_evidence,
    run_fsdp2_correctness,
)
from tinyllm.training.fsdp2_config import load_fsdp2_config


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


def test_fsdp2_activation_checkpointing_wraps_every_transformer_block() -> None:
    config = load_fsdp2_config(
        Path("configs/fsdp2/tinygpt_debug_gloo_activation_checkpointing_smoke.yaml")
    )
    model = TinyGPT(config.model)

    block_type, block_count = _apply_activation_checkpointing(model, config=config)
    batch = torch.randint(0, config.model.vocab_size, (2, 16))
    output = model(batch, labels=batch)
    assert output.loss is not None
    output.loss.backward()

    assert block_type == "TransformerBlock"
    assert block_count == config.model.num_layers
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_fsdp2_failure_injection_rejects_partial_zero_or_terminal_requests() -> None:
    _validate_failure_injection(
        fail_rank=1,
        fail_after_step=1,
        world_size=2,
        max_steps=2,
    )

    with pytest.raises(ValueError, match="provided together"):
        _validate_failure_injection(
            fail_rank=1,
            fail_after_step=None,
            world_size=2,
            max_steps=2,
        )
    with pytest.raises(ValueError, match="nonzero Rank"):
        _validate_failure_injection(
            fail_rank=0,
            fail_after_step=1,
            world_size=2,
            max_steps=2,
        )
    with pytest.raises(ValueError, match="before training.max_steps"):
        _validate_failure_injection(
            fail_rank=1,
            fail_after_step=2,
            world_size=2,
            max_steps=2,
        )
