"""Pinned local preprocessing used by external lm-eval task YAML files."""

from __future__ import annotations

import re
from typing import Any, Protocol


class MappableDataset(Protocol):
    """Minimal Hugging Face Dataset surface needed by task preprocessing."""

    def map(self, function: Any) -> MappableDataset:
        """Apply a deterministic document transform."""


def _clean_hellaswag_text(text: str) -> str:
    cleaned = text.strip().replace(" [title]", ". ")
    cleaned = re.sub(r"\[.*?\]", "", cleaned)
    return cleaned.replace("  ", " ")


def process_hellaswag(dataset: MappableDataset) -> MappableDataset:
    """Match the lm-eval v0.4.12 HellaSwag document transformation."""

    def transform(document: dict[str, Any]) -> dict[str, Any]:
        context_b = str(document["ctx_b"])
        context = str(document["ctx_a"]) + " " + context_b.capitalize()
        return {
            "query": _clean_hellaswag_text(str(document["activity_label"]) + ": " + context),
            "choices": [_clean_hellaswag_text(str(ending)) for ending in document["endings"]],
            "gold": int(document["label"]),
        }

    return dataset.map(transform)
