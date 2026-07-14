from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.export_schemas import SCHEMAS
from tinyllm.schemas import (
    ArtifactRoots,
    CheckpointFile,
    CheckpointManifest,
    CheckpointStateCoverage,
    ResumeResult,
    RunManifest,
    RunStatus,
    canonical_config_hash,
    generate_run_id,
)


def full_state_coverage(**overrides: bool) -> CheckpointStateCoverage:
    values = {
        "model": True,
        "optimizer": True,
        "scheduler": True,
        "grad_scaler": True,
        "python_rng": True,
        "numpy_rng": True,
        "torch_rng": True,
        "cuda_rng": True,
        "sampler": True,
        "config_snapshot": True,
        "environment": True,
    }
    values.update(overrides)
    return CheckpointStateCoverage.model_validate(values)


def test_canonical_config_hash_and_run_id_are_stable() -> None:
    first = canonical_config_hash({"b": 2, "a": 1})
    second = canonical_config_hash({"a": 1, "b": 2})

    assert first == second
    assert (
        generate_run_id(
            "TinyGPT Debug",
            first,
            now=datetime(2026, 7, 14, tzinfo=UTC),
            nonce="cafe",
        )
        == f"20260714T000000Z-tinygpt-debug-{first[:8]}-cafe"
    )


def test_artifact_roots_reject_relative_and_traversal_paths() -> None:
    with pytest.raises(ValidationError, match="absolute"):
        ArtifactRoots(root=Path("runs"))

    roots = ArtifactRoots(root=Path("/data/test/tinyllm"))
    assert roots.cache == Path("/data/test/tinyllm/cache")
    assert roots.datasets == Path("/data/test/tinyllm/datasets")
    assert roots.models == Path("/data/test/tinyllm/models")
    assert roots.registry == Path("/data/test/tinyllm/registry")
    with pytest.raises(ValueError, match="run_id"):
        roots.run_directory("../../escape")


@pytest.mark.parametrize(
    ("name", "config_hash", "nonce", "expected"),
    [
        ("---", "a" * 64, "cafe", "ASCII letter"),
        ("valid", "not-a-hash", "cafe", "SHA256"),
        ("valid", "a" * 64, "xyz1", "hexadecimal"),
    ],
)
def test_run_id_rejects_invalid_identity_inputs(
    name: str, config_hash: str, nonce: str, expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        generate_run_id(
            name,
            config_hash,
            now=datetime(2026, 7, 14, tzinfo=UTC),
            nonce=nonce,
        )


def test_config_hash_rejects_non_json_values() -> None:
    with pytest.raises(ValueError, match="canonical-JSON"):
        canonical_config_hash({"unsupported": object()})


def test_run_manifest_binds_id_to_config_hash() -> None:
    config_hash = canonical_config_hash({"seed": 42})
    now = datetime(2026, 7, 14, tzinfo=UTC)
    run_id = generate_run_id("unit test", config_hash, now=now, nonce="1234")

    manifest = RunManifest(
        run_id=run_id,
        name="unit test",
        status=RunStatus.CREATED,
        created_at=now,
        updated_at=now + timedelta(seconds=1),
        config_hash=config_hash,
        git_commit="a" * 40,
        git_dirty=False,
        artifact_root=Path("/data/test/tinyllm"),
        strategy="single",
        world_size=1,
    )

    assert manifest.to_dict()["status"] == "created"
    with pytest.raises(ValidationError, match="config hash"):
        RunManifest.model_validate({**manifest.model_dump(), "config_hash": "b" * 64})
    with pytest.raises(ValidationError, match="timezone-aware"):
        RunManifest.model_validate(
            {
                **manifest.model_dump(),
                "created_at": datetime(2026, 7, 14),
                "updated_at": datetime(2026, 7, 14),
            }
        )


def test_checkpoint_rejects_incomplete_exact_resume_and_unsafe_files() -> None:
    config_hash = canonical_config_hash({"seed": 42})
    run_id = generate_run_id(
        "unit test",
        config_hash,
        now=datetime(2026, 7, 14, tzinfo=UTC),
        nonce="1234",
    )
    state_file = CheckpointFile(
        path="state.pt",
        role="training_state",
        size_bytes=128,
        sha256="a" * 64,
    )

    with pytest.raises(ValidationError, match="exact resume"):
        CheckpointManifest(
            checkpoint_id="checkpoint-step-00000025",
            run_id=run_id,
            created_at=datetime(2026, 7, 14, tzinfo=UTC),
            strategy="single",
            resume_capability="exact",
            world_size=1,
            global_step=25,
            micro_step=25,
            epoch=0,
            config_hash=config_hash,
            dataset_version="toy-v1",
            git_commit="a" * 40,
            state=full_state_coverage(sampler=False),
            files=(state_file,),
        )

    with pytest.raises(ValidationError, match="global_step"):
        CheckpointManifest(
            checkpoint_id="checkpoint-step-00000026",
            run_id=run_id,
            created_at=datetime(2026, 7, 14, tzinfo=UTC),
            strategy="single",
            resume_capability="exact",
            world_size=1,
            global_step=25,
            micro_step=25,
            epoch=0,
            config_hash=config_hash,
            dataset_version="toy-v1",
            git_commit="a" * 40,
            state=full_state_coverage(),
            files=(state_file,),
        )

    with pytest.raises(ValidationError, match="safe relative"):
        CheckpointFile(
            path="../state.pt",
            role="training_state",
            size_bytes=128,
            sha256="a" * 64,
        )

    with pytest.raises(ValidationError, match="config hash"):
        CheckpointManifest(
            checkpoint_id="checkpoint-step-00000025",
            run_id=run_id,
            created_at=datetime(2026, 7, 14, tzinfo=UTC),
            strategy="single",
            resume_capability="exact",
            world_size=1,
            global_step=25,
            micro_step=25,
            epoch=0,
            config_hash="b" * 64,
            dataset_version="toy-v1",
            git_commit="a" * 40,
            state=full_state_coverage(),
            files=(state_file,),
        )


def test_committed_json_schema_snapshots_match_models() -> None:
    schema_root = Path("schemas")
    for filename, model in SCHEMAS.items():
        committed = json.loads((schema_root / filename).read_text(encoding="utf-8"))
        assert committed == model.model_json_schema(), filename


def test_resume_result_cannot_mislabel_partial_state_as_exact() -> None:
    with pytest.raises(ValidationError, match="complete model state"):
        ResumeResult(
            mode="exact",
            checkpoint_id="checkpoint-step-00000003",
            source_run_id="20260714T000000Z-source-run-aaaaaaaa-beef",
            source_global_step=3,
            target_global_step=3,
            optimizer_restored=True,
            scheduler_restored=True,
            scaler_restored=True,
            sampler_restored=True,
            rng_restored=True,
            loaded_model_keys=("weight",),
            missing_model_keys=("bias",),
        )
