"""Deterministic synthetic token data for M1 correctness tests."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ToyTokenDataset(Dataset[Tensor]):
    """Generate deterministic modular arithmetic token sequences."""

    def __init__(
        self,
        *,
        vocab_size: int,
        sequence_length: int,
        num_samples: int,
        seed: int,
    ) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be between 0 and 2**63 - 1")

        generator = torch.Generator(device="cpu").manual_seed(seed)
        starts = torch.randint(0, vocab_size, (num_samples, 1), generator=generator)
        increments = torch.randint(1, min(vocab_size, 5), (num_samples, 1), generator=generator)
        positions = torch.arange(sequence_length).unsqueeze(0)
        self._tokens = (starts + increments * positions).remainder(vocab_size).long()

    def __len__(self) -> int:
        return self._tokens.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        return self._tokens[index].clone()
