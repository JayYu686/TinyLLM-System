from __future__ import annotations

import random

import numpy
import pytest
import torch

from tinyllm.data import ToyTokenDataset
from tinyllm.training.seed import seed_everything


def test_toy_dataset_is_deterministic_and_bounded() -> None:
    first = ToyTokenDataset(vocab_size=32, sequence_length=12, num_samples=8, seed=42)
    second = ToyTokenDataset(vocab_size=32, sequence_length=12, num_samples=8, seed=42)
    different = ToyTokenDataset(vocab_size=32, sequence_length=12, num_samples=8, seed=43)

    assert len(first) == 8
    torch.testing.assert_close(first[0], second[0])
    assert not torch.equal(first[0], different[0])
    assert first[0].dtype == torch.long
    assert int(first[0].min()) >= 0
    assert int(first[0].max()) < 32


def test_toy_dataset_returns_a_copy() -> None:
    dataset = ToyTokenDataset(vocab_size=16, sequence_length=8, num_samples=2, seed=1)
    sample = dataset[0]
    expected = dataset[0]

    sample[0] = 15

    torch.testing.assert_close(dataset[0], expected)


def test_seed_everything_resets_all_cpu_generators() -> None:
    seed_everything(123)
    first = (random.random(), float(numpy.random.random()), float(torch.rand(())))
    seed_everything(123)
    second = (random.random(), float(numpy.random.random()), float(torch.rand(())))

    assert first == second


def test_seed_everything_rejects_out_of_range_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        seed_everything(-1)
