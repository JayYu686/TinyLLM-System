"""Native PyTorch single-device training loop for M1 correctness work."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from tinyllm.data import StatefulSequentialSampler, ToyTokenDataset
from tinyllm.models.tinygpt import TinyGPT
from tinyllm.training.config import M1TrainingConfig
from tinyllm.training.errors import TrainingError, TrainingErrorCode
from tinyllm.training.metrics import MetricSink, TrainerState, TrainingStepMetrics
from tinyllm.training.scheduler import build_adamw, build_warmup_cosine_scheduler
from tinyllm.training.seed import seed_everything


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Progress and metrics produced by one call to ``train``."""

    state: TrainerState
    metrics: tuple[TrainingStepMetrics, ...]


class SingleDeviceTrainer:
    """Train a causal LM on one device with explicit optimizer-step semantics."""

    def __init__(
        self,
        *,
        model: nn.Module,
        dataloader: DataLoader[Tensor],
        optimizer: Optimizer,
        scheduler: LRScheduler,
        config: M1TrainingConfig,
        device: torch.device,
        metric_sink: MetricSink | None = None,
        sampler: StatefulSequentialSampler | None = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.metric_sink = metric_sink
        self.sampler = sampler
        self.state = TrainerState()
        self._iterator: Iterator[Tensor] | None = None

    @property
    def is_pristine(self) -> bool:
        """Return whether no training or sampler progress has occurred."""

        sampler_pristine = self.sampler is None or (
            self.sampler.epoch == 0 and self.sampler.cursor == 0
        )
        return self.state == TrainerState() and not self.optimizer.state and sampler_pristine

    def restore_progress(self, state: TrainerState) -> None:
        """Install optimizer-boundary progress and reset transient iteration state."""

        accumulation_steps = self.config.training.gradient_accumulation_steps
        if state.global_step > self.config.training.max_steps:
            raise ValueError("restored global_step exceeds configured max_steps")
        if state.micro_step != state.global_step * accumulation_steps:
            raise ValueError("restored state is not at an optimizer-step boundary")
        if self.sampler is not None and self.sampler.epoch != state.epoch:
            raise ValueError("restored sampler epoch does not match trainer state")
        self.state = state
        self._iterator = None
        self.optimizer.zero_grad(set_to_none=True)

    def _next_batch(self) -> Tensor:
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        try:
            return cast(Tensor, next(self._iterator))
        except StopIteration:
            self._iterator = iter(self.dataloader)
            if self.sampler is None:
                self.state = TrainerState(
                    global_step=self.state.global_step,
                    micro_step=self.state.micro_step,
                    epoch=self.state.epoch + 1,
                    tokens_seen=self.state.tokens_seen,
                )
            try:
                return cast(Tensor, next(self._iterator))
            except StopIteration as exc:
                raise TrainingError(
                    TrainingErrorCode.EMPTY_DATALOADER,
                    "dataloader cannot form a complete micro batch",
                    context={"micro_batch_size": self.config.training.micro_batch_size},
                ) from exc

    def _require_scalar_finite_loss(self, output: object, *, micro_step: int) -> Tensor:
        loss = getattr(output, "loss", None)
        if not isinstance(loss, Tensor) or loss.ndim != 0:
            raise TrainingError(
                TrainingErrorCode.TRAIN_OUTPUT_INVALID,
                "model output must contain a scalar loss tensor",
                context={"global_step": self.state.global_step, "micro_step": micro_step},
            )
        if not bool(torch.isfinite(loss).item()):
            raise TrainingError(
                TrainingErrorCode.NON_FINITE_LOSS,
                "non-finite loss detected before backward",
                context={"global_step": self.state.global_step, "micro_step": micro_step},
            )
        return loss

    def _clip_gradients(self) -> tuple[float, bool]:
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        gradients = [cast(Tensor, parameter.grad) for parameter in parameters]
        if not gradients or any(
            not bool(torch.isfinite(gradient).all().item()) for gradient in gradients
        ):
            raise TrainingError(
                TrainingErrorCode.NON_FINITE_GRADIENT,
                "missing or non-finite gradient detected before optimizer step",
                context={
                    "global_step": self.state.global_step,
                    "micro_step": self.state.micro_step,
                },
            )
        norm = nn.utils.clip_grad_norm_(parameters, self.config.training.max_grad_norm)
        if not bool(torch.isfinite(norm).item()):
            raise TrainingError(
                TrainingErrorCode.NON_FINITE_GRADIENT,
                "non-finite gradient norm detected before optimizer step",
                context={
                    "global_step": self.state.global_step,
                    "micro_step": self.state.micro_step,
                },
            )
        value = float(norm.item())
        return value, value > self.config.training.max_grad_norm

    def train(self, *, target_global_step: int | None = None) -> TrainingResult:
        """Train through a target successful optimizer step and return validated metrics."""

        target = (
            self.config.training.max_steps if target_global_step is None else target_global_step
        )
        if not self.state.global_step <= target <= self.config.training.max_steps:
            raise ValueError("target_global_step must be between current state and max_steps")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        emitted: list[TrainingStepMetrics] = []
        accumulation_steps = self.config.training.gradient_accumulation_steps

        while self.state.global_step < target:
            accumulated_loss = 0.0
            for _ in range(accumulation_steps):
                batch = self._next_batch().to(self.device)
                attempted_micro_step = self.state.micro_step + 1
                output = self.model(batch, labels=batch)
                loss = self._require_scalar_finite_loss(
                    output,
                    micro_step=attempted_micro_step,
                )
                torch.autograd.backward(loss / accumulation_steps)
                predicted_tokens = batch.shape[0] * max(batch.shape[1] - 1, 0)
                self.state = TrainerState(
                    global_step=self.state.global_step,
                    micro_step=attempted_micro_step,
                    epoch=self.sampler.epoch if self.sampler is not None else self.state.epoch,
                    tokens_seen=self.state.tokens_seen + predicted_tokens,
                )
                accumulated_loss += float(loss.detach().float().item())

            gradient_norm, gradient_clipped = self._clip_gradients()
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.state = TrainerState(
                global_step=self.state.global_step + 1,
                micro_step=self.state.micro_step,
                epoch=self.state.epoch,
                tokens_seen=self.state.tokens_seen,
            )
            metric = TrainingStepMetrics(
                global_step=self.state.global_step,
                micro_step=self.state.micro_step,
                epoch=self.state.epoch,
                loss=accumulated_loss / accumulation_steps,
                learning_rate=learning_rate,
                gradient_norm=gradient_norm,
                gradient_clipped=gradient_clipped,
                tokens_seen=self.state.tokens_seen,
            )
            emitted.append(metric)
            if self.metric_sink is not None:
                self.metric_sink.emit(metric)

        return TrainingResult(state=self.state, metrics=tuple(emitted))


def build_m1_cpu_trainer(
    config: M1TrainingConfig,
    *,
    metric_sink: MetricSink | None = None,
) -> SingleDeviceTrainer:
    """Construct the deterministic CPU/FP32 trainer used by M1.1 tests and smoke runs."""

    if config.precision.dtype != "fp32" or config.precision.use_grad_scaler:
        raise TrainingError(
            TrainingErrorCode.UNSUPPORTED_PRECISION,
            "M1.1 CPU trainer only supports fp32 without GradScaler",
            context={"dtype": config.precision.dtype},
        )
    if config.precision.allow_tf32:
        raise TrainingError(
            TrainingErrorCode.UNSUPPORTED_PRECISION,
            "TF32 cannot be enabled for the M1.1 CPU trainer",
            context={"allow_tf32": True},
        )

    seed_everything(config.run.seed, deterministic_algorithms=True)
    dataset = ToyTokenDataset(
        vocab_size=config.data.vocab_size,
        sequence_length=config.data.sequence_length,
        num_samples=config.data.num_samples,
        seed=config.run.seed,
    )
    sampler = StatefulSequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.micro_batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
    )
    model = TinyGPT(config.model).to(torch.device("cpu"))
    optimizer = build_adamw(model, config.training)
    scheduler = build_warmup_cosine_scheduler(optimizer, config.training)
    return SingleDeviceTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=torch.device("cpu"),
        metric_sink=metric_sink,
        sampler=sampler,
    )
