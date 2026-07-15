from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from tinyllm.data import StatefulDistributedSampler, StatefulSequentialSampler, ToyTokenDataset


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


def test_distributed_sampler_partitions_and_restores_the_next_local_sample() -> None:
    dataset = ToyTokenDataset(vocab_size=16, sequence_length=8, num_samples=12, seed=7)
    samplers = [
        StatefulDistributedSampler(dataset, num_replicas=2, rank=rank, seed=42) for rank in range(2)
    ]
    partitions = [list(sampler) for sampler in samplers]

    assert len(partitions[0]) == len(partitions[1]) == 6
    assert set(partitions[0]).isdisjoint(partitions[1])
    assert set(partitions[0] + partitions[1]) == set(range(12))

    source = StatefulDistributedSampler(dataset, num_replicas=2, rank=1, seed=42)
    iterator = iter(source)
    next(iterator)
    next(iterator)
    state = source.state_dict()
    expected_next = next(iterator)

    restored = StatefulDistributedSampler(dataset, num_replicas=2, rank=1, seed=42)
    restored.load_state_dict(state)
    assert next(iter(restored)) == expected_next


def test_distributed_sampler_rolls_epoch_and_rejects_rank_or_cursor_drift() -> None:
    dataset = ToyTokenDataset(vocab_size=16, sequence_length=8, num_samples=4, seed=7)
    sampler = StatefulDistributedSampler(dataset, num_replicas=2, rank=0, seed=42)
    epoch_zero = list(sampler)
    epoch_one = list(sampler)

    assert sampler.epoch == 1
    assert epoch_zero != epoch_one

    wrong_rank = {**sampler.state_dict(), "rank": 1}
    with pytest.raises(ValueError, match="identity"):
        StatefulDistributedSampler(dataset, num_replicas=2, rank=0, seed=42).load_state_dict(
            wrong_rank
        )

    invalid_cursor = {**sampler.state_dict(), "cursor": 3}
    with pytest.raises(ValueError, match="cursor"):
        StatefulDistributedSampler(dataset, num_replicas=2, rank=0, seed=42).load_state_dict(
            invalid_cursor
        )
