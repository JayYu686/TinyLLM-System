"""Random seed utilities for deterministic training tests."""

from __future__ import annotations

import random

import numpy
import torch


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed Python, NumPy, PyTorch, and all visible CUDA devices."""

    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
