from __future__ import annotations

from pathlib import Path

import pytest

from tinyllm.training.fsdp2_schema import (
    FSDP2CorrectnessSummary,
    FSDP2RankEvidence,
    FSDP2RankFailureEvidence,
    FSDP2TrainingResult,
)


def rank_evidence(rank: int, *, local_numel: int = 50) -> FSDP2RankEvidence:
    return FSDP2RankEvidence(
        rank=rank,
        device_type="cpu",
        local_shard_numel=local_numel,
        local_shard_sha256=f"{rank + 1:064x}",
    )


def valid_summary() -> FSDP2CorrectnessSummary:
    return FSDP2CorrectnessSummary(
        backend="gloo",
        device_type="cpu",
        world_size=2,
        global_batch_size=4,
        optimizer_steps=2,
        durable_metric_records=2,
        logical_parameter_count=100,
        local_shard_parameter_sum=100,
        initial_full_parameter_sha256="a" * 64,
        final_full_parameter_sha256="b" * 64,
        rank_evidence=(rank_evidence(0), rank_evidence(1)),
        max_loss_reduction_abs_diff=0.0,
        max_gradient_norm_abs_diff=0.0,
    )


def test_fsdp2_summary_binds_rank_shards_and_steps() -> None:
    summary = valid_summary()

    assert summary.durable_writer_rank == 0
    assert summary.checkpoint_status == "not_evaluated_m4_1"
    assert summary.local_shard_parameter_sum == summary.logical_parameter_count

    with pytest.raises(ValueError, match="contiguous and ordered"):
        FSDP2CorrectnessSummary.model_validate(
            {
                **summary.model_dump(mode="python"),
                "rank_evidence": (rank_evidence(1), rank_evidence(0)),
            }
        )
    with pytest.raises(ValueError, match="exactly one row per step"):
        FSDP2CorrectnessSummary.model_validate(
            {**summary.model_dump(mode="python"), "durable_metric_records": 1}
        )
    with pytest.raises(ValueError, match="cover each logical parameter"):
        FSDP2CorrectnessSummary.model_validate(
            {**summary.model_dump(mode="python"), "logical_parameter_count": 101}
        )
    with pytest.raises(ValueError, match="must change"):
        FSDP2CorrectnessSummary.model_validate(
            {
                **summary.model_dump(mode="python"),
                "final_full_parameter_sha256": "a" * 64,
            }
        )


def test_fsdp2_training_result_binds_run_id_to_config_hash() -> None:
    result = FSDP2TrainingResult(
        run_id="20260716T000000Z-fsdp2-test-aaaaaaaa-beef",
        artifact_dir=Path("/tmp/fsdp2-test"),
        config_sha256="a" * 64,
        git_commit="b" * 40,
        git_dirty=False,
        summary=valid_summary(),
    )
    assert result.status == "succeeded"

    with pytest.raises(ValueError, match="config hash"):
        FSDP2TrainingResult.model_validate(
            {**result.model_dump(mode="python"), "config_sha256": "c" * 64}
        )


def test_fsdp2_summary_binds_activation_checkpointing_evidence() -> None:
    summary = valid_summary()
    activated = FSDP2CorrectnessSummary.model_validate(
        {
            **summary.model_dump(mode="python"),
            "activation_checkpointing": True,
            "activation_checkpointed_block_type": "TransformerBlock",
            "activation_checkpointed_block_count": 2,
        }
    )
    assert activated.activation_checkpointed_block_count == 2

    with pytest.raises(ValueError, match="cannot report wrapped blocks"):
        FSDP2CorrectnessSummary.model_validate(
            {
                **summary.model_dump(mode="python"),
                "activation_checkpointed_block_count": 2,
            }
        )


def test_fsdp2_rank_failure_evidence_rejects_rank_zero_and_out_of_range() -> None:
    evidence = FSDP2RankFailureEvidence(
        run_id="20260717T000000Z-fsdp2-rank-failure-aaaaaaaa-beef",
        config_sha256="a" * 64,
        git_commit="b" * 40,
        world_size=2,
        rank=1,
        global_step=1,
    )
    assert evidence.resumable is False
    assert evidence.checkpoint_status == "not_evaluated_m4_1"

    with pytest.raises(ValueError):
        FSDP2RankFailureEvidence.model_validate({**evidence.model_dump(mode="python"), "rank": 0})
    with pytest.raises(ValueError, match="member of world_size"):
        FSDP2RankFailureEvidence.model_validate({**evidence.model_dump(mode="python"), "rank": 2})
