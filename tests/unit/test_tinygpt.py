from __future__ import annotations

import pytest
import torch

from tinyllm.models.tinygpt import TinyGPT, TinyGPTConfig
from tinyllm.models.tinygpt.config import TinyGPTConfigError
from tinyllm.models.tinygpt.layers import RMSNorm


def small_config() -> TinyGPTConfig:
    return TinyGPTConfig(
        vocab_size=32,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_sequence_length=16,
        dropout=0.0,
    )


def test_default_tinygpt_is_debug_scale() -> None:
    model = TinyGPT(TinyGPTConfig())

    assert 1_000_000 <= model.parameter_count() <= 5_000_000


def test_tinygpt_forward_backward_and_weight_tying() -> None:
    torch.manual_seed(7)
    model = TinyGPT(small_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 12))

    output = model(input_ids, labels=input_ids)

    assert output.logits.shape == (2, 12, model.config.vocab_size)
    assert output.loss is not None
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.token_embeddings.weight.grad is not None
    assert model.lm_head.weight.data_ptr() == model.token_embeddings.weight.data_ptr()


def test_tinygpt_attention_is_causal() -> None:
    torch.manual_seed(11)
    model = TinyGPT(small_config()).eval()
    original = torch.randint(0, model.config.vocab_size, (1, 8))
    changed = original.clone()
    changed[:, 5] = (changed[:, 5] + 1).remainder(model.config.vocab_size)

    with torch.no_grad():
        original_logits = model(original).logits
        changed_logits = model(changed).logits

    torch.testing.assert_close(original_logits[:, :5], changed_logits[:, :5])


def test_tinygpt_rejects_invalid_token_dtype() -> None:
    model = TinyGPT(small_config())

    with pytest.raises(ValueError, match="torch.long"):
        model(torch.ones((1, 4), dtype=torch.float32))


def test_tinygpt_config_rejects_invalid_head_partition() -> None:
    with pytest.raises(TinyGPTConfigError, match="divisible"):
        TinyGPTConfig(hidden_size=30, num_heads=4)


def test_rms_norm_matches_reference() -> None:
    layer = RMSNorm(hidden_size=4, epsilon=1.0e-6)
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)

    actual = layer(inputs)
    expected = inputs * torch.rsqrt(inputs.pow(2).mean(dim=-1, keepdim=True) + 1.0e-6)

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
