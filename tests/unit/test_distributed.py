from __future__ import annotations

import pytest

from tinyllm.training import TrainingError, TrainingErrorCode
from tinyllm.training.ddp_schema import (
    GRADIENT_NORM_ATOL,
    LOSS_REDUCTION_ATOL,
    DDPCorrectnessSummary,
)
from tinyllm.training.distributed import torchrun_environment, validate_sampler_partitions


def test_torchrun_environment_requires_explicit_valid_rank_coordinates() -> None:
    environment = torchrun_environment(
        {
            "RANK": "1",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
        }
    )
    assert environment.rank == environment.local_rank == 1
    assert environment.world_size == environment.local_world_size == 2

    with pytest.raises(TrainingError) as missing:
        torchrun_environment({})
    assert missing.value.code == TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID
    assert missing.value.context["variable"] == "RANK"

    with pytest.raises(TrainingError) as invalid:
        torchrun_environment(
            {
                "RANK": "2",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "2",
                "LOCAL_WORLD_SIZE": "2",
            }
        )
    assert invalid.value.code == TrainingErrorCode.DISTRIBUTED_LAUNCH_INVALID


def test_sampler_partition_validation_requires_exact_non_overlapping_coverage() -> None:
    evidence = validate_sampler_partitions(((0, 2, 4), (1, 3, 5)), num_samples=6)
    assert [item.rank for item in evidence] == [0, 1]
    assert [item.sample_count for item in evidence] == [3, 3]
    assert all(len(item.sample_ids_sha256) == 64 for item in evidence)

    with pytest.raises(TrainingError) as overlap:
        validate_sampler_partitions(((0, 1, 2), (2, 3, 4)), num_samples=5)
    assert overlap.value.code == TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH

    with pytest.raises(TrainingError) as incomplete:
        validate_sampler_partitions(((0, 2), (1, 3)), num_samples=5)
    assert incomplete.value.code == TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH


def test_ddp_summary_binds_world_size_sampler_and_durable_metrics() -> None:
    partitions = validate_sampler_partitions(((0, 2), (1, 3)), num_samples=4)
    summary = DDPCorrectnessSummary(
        backend="gloo",
        world_size=2,
        global_batch_size=4,
        optimizer_steps=2,
        durable_metric_records=2,
        model_parameter_count=100,
        initial_parameter_sha256="a" * 64,
        final_parameter_sha256="b" * 64,
        sampler_num_samples=4,
        sampler_union_samples=4,
        sampler_no_overlap=True,
        partitions=partitions,
        max_loss_reduction_abs_diff=0.0,
        max_gradient_norm_abs_diff=0.0,
    )
    assert summary.durable_writer_rank == 0

    with pytest.raises(ValueError, match="exactly one row per step"):
        DDPCorrectnessSummary.model_validate(
            {**summary.model_dump(mode="python"), "durable_metric_records": 1}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_loss_reduction_abs_diff", LOSS_REDUCTION_ATOL * 2),
        ("max_gradient_norm_abs_diff", GRADIENT_NORM_ATOL * 2),
    ],
)
def test_ddp_summary_rejects_cross_rank_difference_above_tolerance(
    field: str,
    value: float,
) -> None:
    partitions = validate_sampler_partitions(((0, 2), (1, 3)), num_samples=4)
    raw = {
        "backend": "gloo",
        "world_size": 2,
        "global_batch_size": 4,
        "optimizer_steps": 2,
        "durable_metric_records": 2,
        "model_parameter_count": 100,
        "initial_parameter_sha256": "a" * 64,
        "final_parameter_sha256": "b" * 64,
        "sampler_num_samples": 4,
        "sampler_union_samples": 4,
        "partitions": partitions,
        "max_loss_reduction_abs_diff": 0.0,
        "max_gradient_norm_abs_diff": 0.0,
    }
    raw[field] = value

    with pytest.raises(ValueError, match="exceeds the fixed tolerance"):
        DDPCorrectnessSummary.model_validate(raw)
