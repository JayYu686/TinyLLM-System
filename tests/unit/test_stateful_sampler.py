from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from tinyllm.data import StatefulSequentialSampler, ToyTokenDataset


def test_sampler_state_restores_the_next_batch() -> None:
    dataset = ToyTokenDataset(vocab_size=16, sequence_length=8, num_samples=6, seed=7)
    first_sampler = StatefulSequentialSampler(dataset)
    first_loader = DataLoader(dataset, batch_size=2, sampler=first_sampler)
    first_iterator = iter(first_loader)
    next(first_iterator)
    state = first_sampler.state_dict()
    expected_next_batch = next(first_iterator)

    restored_sampler = StatefulSequentialSampler(dataset)
    restored_sampler.load_state_dict(state)
    restored_loader = DataLoader(dataset, batch_size=2, sampler=restored_sampler)

    torch.testing.assert_close(next(iter(restored_loader)), expected_next_batch)
    assert restored_sampler.cursor == 4
    assert restored_sampler.epoch == 0


def test_sampler_rolls_epoch_and_rejects_incompatible_state() -> None:
    dataset = ToyTokenDataset(vocab_size=16, sequence_length=8, num_samples=3, seed=7)
    sampler = StatefulSequentialSampler(dataset)

    assert list(sampler) == [0, 1, 2]
    assert list(sampler) == [0, 1, 2]
    assert sampler.epoch == 1

    incompatible = {**sampler.state_dict(), "num_samples": 4}
    with pytest.raises(ValueError, match="num_samples"):
        StatefulSequentialSampler(dataset).load_state_dict(incompatible)
