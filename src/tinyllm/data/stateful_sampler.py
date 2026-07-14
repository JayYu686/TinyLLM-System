"""Stateful deterministic sampler used by M1 checkpoint and resume tests."""

from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Any, Literal

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
