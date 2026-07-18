from __future__ import annotations

import json
from pathlib import Path

from tinyllm.schemas import canonical_config_hash
from tinyllm.training import load_fsdp2_config
from tinyllm.training.fsdp2_schema import (
    FSDP2CorrectnessSummary,
    FSDP2RankFailureEvidence,
)


def test_m4_multigpu_activation_and_rank_failure_evidence_is_strict() -> None:
    evidence = json.loads(
        Path("reports/m4/raw/fsdp2_multigpu_activation_failure.json").read_text(encoding="utf-8")
    )
    config = load_fsdp2_config(Path(evidence["run"]["config_path"]))
    summary = FSDP2CorrectnessSummary.model_validate_json(
        json.dumps(evidence["success_run"]["summary"])
    )
    diagnostic = FSDP2RankFailureEvidence.model_validate_json(
        json.dumps(
            {
                **evidence["rank_failure_run"]["diagnostic"],
                "run_id": evidence["rank_failure_run"]["run_id"],
            }
        )
    )

    assert evidence["status"] == "pass"
    assert evidence["run"]["config_sha256"] == canonical_config_hash(config)
    assert evidence["run"]["git_dirty"] is False
    assert evidence["environment"]["physical_gpu_indices"] == [6, 7]
    assert evidence["environment"]["pair_topology"] == "PIX"
    assert summary.backend == "nccl"
    assert summary.device_type == "cuda"
    assert summary.world_size == 2
    assert summary.activation_checkpointing is True
    assert summary.activation_checkpointed_block_count == config.model.num_layers
    assert summary.local_shard_parameter_sum == summary.logical_parameter_count
    assert diagnostic.rank == 1
    assert diagnostic.exit_code == 17
    assert diagnostic.resumable is False
    assert evidence["rank_failure_run"]["torchrun_exit_code"] != 0
    assert evidence["rank_failure_run"]["correctness_published"] is False
    assert evidence["rank_failure_run"]["checkpoint_file_count"] == 0
