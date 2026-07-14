"""Optimizer and learning-rate construction for deterministic M1 training."""

from __future__ import annotations

import math

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from tinyllm.training.config import TrainingLoopConfig


def warmup_cosine_multiplier(step_index: int, *, max_steps: int, warmup_steps: int) -> float:
    """Return the LR multiplier used by a zero-based optimizer step."""

    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0 <= warmup_steps <= max_steps:
        raise ValueError("warmup_steps must be in [0, max_steps]")
    if step_index >= max_steps:
        return 0.0
    if warmup_steps and step_index < warmup_steps:
        return (step_index + 1) / warmup_steps

    decay_steps = max_steps - warmup_steps
    progress = (step_index - warmup_steps) / decay_steps
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_adamw(model: nn.Module, config: TrainingLoopConfig) -> AdamW:
    """Build AdamW with explicit decay and no-decay parameter groups."""

    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        target = decay_parameters if parameter.ndim >= 2 else no_decay_parameters
        target.append(parameter)
    if not decay_parameters and not no_decay_parameters:
        raise ValueError("model has no trainable parameters")

    parameter_groups: list[dict[str, object]] = []
    if decay_parameters:
        parameter_groups.append({"params": decay_parameters, "weight_decay": config.weight_decay})
    if no_decay_parameters:
        parameter_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})
    return AdamW(
        parameter_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )


def build_warmup_cosine_scheduler(optimizer: Optimizer, config: TrainingLoopConfig) -> LRScheduler:
    """Build an optimizer-step scheduler with deterministic warmup and cosine decay."""

    def multiplier(step_index: int) -> float:
        return warmup_cosine_multiplier(
            step_index,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
        )

    return LambdaLR(optimizer, lr_lambda=multiplier)
