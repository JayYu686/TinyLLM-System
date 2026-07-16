"""Stateful deterministic sampler used by M1 checkpoint and resume tests."""

from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Any, Literal, Protocol

import torch
from pydantic import Field, model_validator
from torch.utils.data import Sampler

from tinyllm.schemas.base import StrictSchema


class SamplerState(StrictSchema):
    """Serializable position of a deterministic sequential sampler."""

    schema_version: Literal["1.0"] = "1.0"
    num_samples: int = Field(gt=0)
    epoch: int = Field(ge=0)
    cursor: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_cursor(self) -> SamplerState:
        """Allow the exhausted sentinel but never a position beyond the dataset."""

        if self.cursor > self.num_samples:
            raise ValueError("sampler cursor cannot exceed num_samples")
        return self


class DistributedSamplerState(StrictSchema):
    """Serializable cursor in one deterministic distributed partition."""

    schema_version: Literal["1.0"] = "1.0"
    num_samples: int = Field(gt=0)
    num_replicas: int = Field(gt=0)
    rank: int = Field(ge=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    shuffle: Literal[True] = True
    drop_last: Literal[True] = True
    epoch: int = Field(ge=0)
    cursor: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_partition_cursor(self) -> DistributedSamplerState:
        """Require an exact divisible partition and a cursor within this Rank."""

        if self.rank >= self.num_replicas:
            raise ValueError("distributed sampler rank must be smaller than num_replicas")
        if self.num_samples % self.num_replicas != 0:
            raise ValueError("distributed sampler requires data divisible by num_replicas")
        if self.cursor > self.num_samples // self.num_replicas:
            raise ValueError("distributed sampler cursor exceeds the Rank partition")
        return self


class StatefulSampler(Protocol):
    """Minimal progress interface consumed by the native Trainer."""

    @property
    def epoch(self) -> int:
        """Return the epoch containing the next sample."""

    @property
    def cursor(self) -> int:
        """Return the next sampler-local position."""

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible state snapshot."""

    def load_state_dict(self, raw: object) -> None:
        """Restore a compatible state snapshot."""


class StatefulSequentialSampler(Sampler[int]):
    """Yield sequential indices while exposing the next index and epoch."""

    def __init__(self, data_source: Sized) -> None:
        super().__init__()
        self._num_samples = len(data_source)
        if self._num_samples <= 0:
            raise ValueError("stateful sampler requires a non-empty data source")
        self._epoch = 0
        self._cursor = 0

    def __iter__(self) -> Iterator[int]:
        if self._cursor == self._num_samples:
            self._epoch += 1
            self._cursor = 0
        while self._cursor < self._num_samples:
            index = self._cursor
            self._cursor += 1
            yield index

    def __len__(self) -> int:
        return self._num_samples - self._cursor

    @property
    def epoch(self) -> int:
        """Return the epoch containing the next index."""

        return self._epoch

    @property
    def cursor(self) -> int:
        """Return the next sample index, or ``num_samples`` when exhausted."""

        return self._cursor

    def state_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible state snapshot."""

        return SamplerState(
            num_samples=self._num_samples,
            epoch=self._epoch,
            cursor=self._cursor,
        ).to_dict()

    def load_state_dict(self, raw: object) -> None:
        """Restore state only when it belongs to the same dataset cardinality."""

        state = SamplerState.model_validate(raw)
        if state.num_samples != self._num_samples:
            raise ValueError("sampler num_samples does not match the current dataset")
        self._epoch = state.epoch
        self._cursor = state.cursor


class StatefulDistributedSampler(Sampler[int]):
    """Deterministically partition shuffled samples while preserving the next local cursor."""

    def __init__(
        self,
        data_source: Sized,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        super().__init__()
        self._num_samples = len(data_source)
        if self._num_samples <= 0:
            raise ValueError("stateful distributed sampler requires a non-empty data source")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler Rank coordinates")
        if self._num_samples % num_replicas != 0:
            raise ValueError("distributed sampler requires data divisible by num_replicas")
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("distributed sampler seed is outside the supported range")
        self._num_replicas = num_replicas
        self._rank = rank
        self._seed = seed
        self._epoch = 0
        self._cursor = 0

    @property
    def _partition_size(self) -> int:
        return self._num_samples // self._num_replicas

    def _partition(self) -> tuple[int, ...]:
        generator = torch.Generator()
        generator.manual_seed(self._seed + self._epoch)
        indices = torch.randperm(self._num_samples, generator=generator).tolist()
        return tuple(indices[self._rank :: self._num_replicas])

    def __iter__(self) -> Iterator[int]:
        if self._cursor == self._partition_size:
            self._epoch += 1
            self._cursor = 0
        partition = self._partition()
        while self._cursor < self._partition_size:
            index = partition[self._cursor]
            self._cursor += 1
            yield index

    def __len__(self) -> int:
        return self._partition_size - self._cursor

    @property
    def epoch(self) -> int:
        """Return the epoch containing the next local sample."""

        return self._epoch

    @property
    def cursor(self) -> int:
        """Return the next position within this Rank's partition."""

        return self._cursor

    @property
    def rank(self) -> int:
        """Return the immutable global Rank identity."""

        return self._rank

    def state_dict(self) -> dict[str, Any]:
        """Return the complete deterministic partition and cursor identity."""

        return DistributedSamplerState(
            num_samples=self._num_samples,
            num_replicas=self._num_replicas,
            rank=self._rank,
            seed=self._seed,
            epoch=self._epoch,
            cursor=self._cursor,
        ).to_dict()

    def load_state_dict(self, raw: object) -> None:
        """Restore only an exact state for this dataset, Rank, World Size, and Seed."""

        state = DistributedSamplerState.model_validate(raw)
        expected = (
            self._num_samples,
            self._num_replicas,
            self._rank,
            self._seed,
        )
        actual = (state.num_samples, state.num_replicas, state.rank, state.seed)
        if actual != expected:
            raise ValueError("distributed sampler state identity does not match this Rank")
        self._epoch = state.epoch
        self._cursor = state.cursor
