"""Neural network layers used by TinyGPT."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from tinyllm.models.tinygpt.config import TinyGPTConfig


class RMSNorm(nn.Module):
    """Root mean square layer normalization with a learned scale."""

    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, inputs: Tensor) -> Tensor:
        """Normalize the final dimension in float32 for numerical stability."""

        normalized = inputs.float()
        normalized = normalized * torch.rsqrt(
            normalized.pow(2).mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized.to(dtype=inputs.dtype) * self.weight


def _rotate_half(inputs: Tensor) -> Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding for one attention head dimension."""

    cosine: Tensor
    sine: Tensor

    def __init__(self, dimension: int, max_sequence_length: int, theta: float) -> None:
        super().__init__()
        if dimension % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        frequencies = 1.0 / (
            theta ** (torch.arange(0, dimension, 2, dtype=torch.float32) / dimension)
        )
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        embeddings = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cosine", embeddings.cos(), persistent=False)
        self.register_buffer("sine", embeddings.sin(), persistent=False)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the precomputed rotary embedding to query and key tensors."""

        sequence_length = query.shape[-2]
        cosine = self.cosine[:sequence_length].to(dtype=query.dtype)[None, None, :, :]
        sine = self.sine[:sequence_length].to(dtype=query.dtype)[None, None, :, :]
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention backed by PyTorch SDPA."""

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dimension = config.head_dimension
        self.dropout = config.dropout
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(
            config.head_dimension,
            config.max_sequence_length,
            config.rope_theta,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return causal attention outputs for a batch-first hidden state."""

        batch_size, sequence_length, hidden_size = inputs.shape
        projected = self.query_key_value(inputs)
        projected = projected.view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dimension,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = projected.unbind(dim=0)
        query, key = self.rotary(query, key)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_size)
        )
        return cast(Tensor, self.output(attended))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the gated feed-forward transformation."""

        return cast(Tensor, self.down(functional.silu(self.gate(inputs)) * self.up(inputs)))


class TransformerBlock(nn.Module):
    """Pre-normalized TinyGPT transformer block."""

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.feed_forward = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply attention and feed-forward residual updates."""

        hidden_states = inputs + self.residual_dropout(self.attention(self.attention_norm(inputs)))
        return cast(
            Tensor,
            hidden_states
            + self.residual_dropout(self.feed_forward(self.feed_forward_norm(hidden_states))),
        )
