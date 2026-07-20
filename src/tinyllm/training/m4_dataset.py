"""Verified deterministic 512-token training view over the registered M2 data product."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import Field
from torch import Tensor
from torch.utils.data import Dataset

from tinyllm.data import open_registered_dataset
from tinyllm.schemas.base import StrictSchema
from tinyllm.schemas.run import SHA256_PATTERN
from tinyllm.training.m4_qwen_config import M4QwenDataConfig


class M4DatasetViewManifest(StrictSchema):
    """Content and transformation identity for the bounded M4 Train view."""

    schema_version: Literal["1.0"] = "1.0"
    parent_dataset_version: str
    parent_content_sha256: str = Field(pattern=SHA256_PATTERN)
    split: Literal["train"]
    sequence_length: Literal[512]
    pad_token_id: Literal[151643]
    assistant_only_loss: Literal[True]
    slice_position_ids_reset: Literal[True]
    max_sequences: int = Field(gt=0)
    sequence_count: int = Field(gt=0)
    supervised_token_count: int = Field(gt=0)
    rejected_unsupervised_slices: int = Field(ge=0)
    content_sha256: str = Field(pattern=SHA256_PATTERN)

    @property
    def view_version(self) -> str:
        """Return a parent-bound stable version used in Run and Checkpoint lineage."""

        return f"{self.parent_dataset_version}-m4view-{self.content_sha256[:8]}"


@dataclass(frozen=True, slots=True)
class M4TrainingExample:
    """One fixed-length, single-conversation-slice Assistant-only example."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    position_ids: tuple[int, ...]


def _update_array_hash(digest: hashlib._Hash, values: tuple[int, ...]) -> None:
    digest.update(np.asarray(values, dtype="<i4").tobytes())


class M4RegisteredDatasetView(Dataset[dict[str, Tensor]]):
    """In-memory bounded view built only after Registry hashes and Schemas pass."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        config: M4QwenDataConfig,
    ) -> None:
        registered = open_registered_dataset(
            artifact_root=artifact_root,
            dataset_version=config.dataset_version,
        )
        examples: list[M4TrainingExample] = []
        rejected_unsupervised = 0
        supervised_tokens = 0
        digest = hashlib.sha256()
        digest.update(registered.manifest.content_sha256.encode("ascii"))
        digest.update(
            json.dumps(config.to_dict(), separators=(",", ":"), sort_keys=True).encode("utf-8")
        )

        for pack in registered.iter_packs():
            if str(pack.split) != config.split:
                continue
            cursor = 0
            for sample_tokens in pack.sample_token_counts:
                sample_end = cursor + sample_tokens
                sample_input = pack.input_ids[cursor:sample_end]
                sample_labels = pack.labels[cursor:sample_end]
                for start in range(0, sample_tokens, config.sequence_length):
                    end = min(start + config.sequence_length, sample_tokens)
                    token_slice = sample_input[start:end]
                    label_slice = sample_labels[start:end]
                    supervised = sum(label != -100 for label in label_slice)
                    if supervised == 0:
                        rejected_unsupervised += 1
                        continue
                    valid = len(token_slice)
                    padding = config.sequence_length - valid
                    example = M4TrainingExample(
                        input_ids=token_slice + (config.pad_token_id,) * padding,
                        labels=label_slice + (-100,) * padding,
                        attention_mask=(1,) * valid + (0,) * padding,
                        position_ids=tuple(range(valid)) + (0,) * padding,
                    )
                    for array in (
                        example.input_ids,
                        example.labels,
                        example.attention_mask,
                        example.position_ids,
                    ):
                        _update_array_hash(digest, array)
                    examples.append(example)
                    supervised_tokens += supervised
                    if len(examples) == config.max_sequences:
                        break
                cursor = sample_end
                if len(examples) == config.max_sequences:
                    break
            if len(examples) == config.max_sequences:
                break
        if len(examples) != config.max_sequences:
            raise ValueError("registered M2 Train split cannot supply the configured M4 view")
        self._examples = tuple(examples)
        self.manifest = M4DatasetViewManifest(
            parent_dataset_version=registered.manifest.dataset_version,
            parent_content_sha256=registered.manifest.content_sha256,
            split=config.split,
            sequence_length=config.sequence_length,
            pad_token_id=config.pad_token_id,
            assistant_only_loss=config.assistant_only_loss,
            slice_position_ids_reset=config.slice_position_ids_reset,
            max_sequences=config.max_sequences,
            sequence_count=len(examples),
            supervised_token_count=supervised_tokens,
            rejected_unsupervised_slices=rejected_unsupervised,
            content_sha256=digest.hexdigest(),
        )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        example = self._examples[index]
        return {
            "input_ids": torch.tensor(example.input_ids, dtype=torch.long),
            "labels": torch.tensor(example.labels, dtype=torch.long),
            "attention_mask": torch.tensor(example.attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(example.position_ids, dtype=torch.long),
        }
