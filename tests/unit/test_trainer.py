from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor, nn
from torch.utils.data import DataLoader

from tinyllm.data import ToyTokenDataset
from tinyllm.training import (
    InMemoryMetricSink,
    TrainingError,
    TrainingErrorCode,
    build_m1_cpu_trainer,
)
from tinyllm.training.config import M1TrainingConfig, training_config_from_mapping
from tinyllm.training.metrics import TrainingStepMetrics
from tinyllm.training.scheduler import build_adamw, build_warmup_cosine_scheduler
from tinyllm.training.trainer import SingleDeviceTrainer


def trainer_config(**training_overrides: object) -> M1TrainingConfig:
    training: dict[str, object] = {
        "max_steps": 4,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_steps": 2,
    }
    training.update(training_overrides)
    return training_config_from_mapping(
        {
            "schema_version": "1.0",
            "run": {"name": "trainer-unit", "seed": 17},
            "model": {
                "vocab_size": 32,
                "hidden_size": 32,
                "num_layers": 1,
                "num_heads": 4,
                "intermediate_size": 64,
                "max_sequence_length": 16,
                "rope_theta": 10_000.0,
                "rms_norm_epsilon": 1.0e-6,
                "dropout": 0.0,
                "tie_word_embeddings": True,
            },
            "data": {
                "kind": "toy",
                "vocab_size": 32,
                "sequence_length": 12,
                "num_samples": 16,
            },
            "training": training,
            "precision": {
                "dtype": "fp32",
                "allow_tf32": False,
                "use_grad_scaler": False,
            },
            "checkpoint": {
                "output_dir": "runs/trainer-unit",
                "save_steps": 2,
                "keep_last": 2,
                "resume": "none",
            },
        }
    )


def test_trainer_applies_accumulation_schedule_and_metrics() -> None:
    config = trainer_config()
    sink = InMemoryMetricSink()
    trainer = build_m1_cpu_trainer(config, metric_sink=sink)
    initial = [parameter.detach().clone() for parameter in trainer.model.parameters()]

    result = trainer.train()

    assert result.state.global_step == 4
    assert result.state.micro_step == 8
    assert result.state.tokens_seen == 2 * 11 * 8
    assert [metric.learning_rate for metric in result.metrics] == pytest.approx(
        [0.005, 0.01, 0.01, 0.005]
    )
    assert tuple(sink.metrics) == result.metrics
    assert all(metric.gradient_norm >= 0 for metric in result.metrics)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial, trainer.model.parameters(), strict=True)
    )


def test_training_is_step_deterministic_in_the_same_environment() -> None:
    first = build_m1_cpu_trainer(trainer_config())
    second = build_m1_cpu_trainer(trainer_config())

    first_result = first.train()
    second_result = second.train()

    assert first_result == second_result
    for first_parameter, second_parameter in zip(
        first.model.parameters(), second.model.parameters(), strict=True
    ):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)


class InvalidLossModel(nn.Module):
    def __init__(self, *, failure: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.failure = failure

    def forward(self, inputs: Tensor, *, labels: Tensor) -> object:
        del labels
        loss = self.weight * inputs.float().mean()
        if self.failure == "vector":
            loss = loss.repeat(2)
        elif self.failure == "nan":
            loss = loss * torch.tensor(float("nan"))
        elif self.failure == "gradient":
            self.weight.register_hook(  # type: ignore[no-untyped-call]
                lambda gradient: torch.full_like(gradient, float("nan"))
            )
        return SimpleNamespace(loss=loss)


def custom_trainer(model: nn.Module, config: M1TrainingConfig) -> SingleDeviceTrainer:
    dataset = ToyTokenDataset(
        vocab_size=config.data.vocab_size,
        sequence_length=config.data.sequence_length,
        num_samples=config.data.num_samples,
        seed=config.run.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.micro_batch_size,
        drop_last=True,
        shuffle=False,
    )
    optimizer = build_adamw(model, config.training)
    scheduler = build_warmup_cosine_scheduler(optimizer, config.training)
    return SingleDeviceTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        ("vector", TrainingErrorCode.TRAIN_OUTPUT_INVALID),
        ("nan", TrainingErrorCode.NON_FINITE_LOSS),
        ("gradient", TrainingErrorCode.NON_FINITE_GRADIENT),
    ],
)
def test_trainer_fails_closed_on_invalid_numerics(failure: str, code: TrainingErrorCode) -> None:
    trainer = custom_trainer(InvalidLossModel(failure=failure), trainer_config())

    with pytest.raises(TrainingError) as caught:
        trainer.train(target_global_step=1)

    assert caught.value.code == code
    assert "micro_step" in caught.value.context


def test_empty_dataloader_and_unsupported_precision_fail_preflight() -> None:
    empty_config = trainer_config(micro_batch_size=32)
    trainer = build_m1_cpu_trainer(empty_config)
    with pytest.raises(TrainingError) as caught:
        trainer.train(target_global_step=1)
    assert caught.value.code == TrainingErrorCode.EMPTY_DATALOADER

    mapping = trainer_config().to_dict()
    precision = mapping["precision"]
    assert isinstance(precision, dict)
    precision["dtype"] = "bf16"
    with pytest.raises(TrainingError) as caught:
        build_m1_cpu_trainer(M1TrainingConfig.model_validate(mapping))
    assert caught.value.code == TrainingErrorCode.UNSUPPORTED_PRECISION


def test_metrics_schema_rejects_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        TrainingStepMetrics(
            global_step=1,
            micro_step=1,
            epoch=0,
            loss=float("nan"),
            learning_rate=0.01,
            gradient_norm=1.0,
            gradient_clipped=False,
            tokens_seen=10,
        )
