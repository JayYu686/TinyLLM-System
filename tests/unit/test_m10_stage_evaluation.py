from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from tinyllm.deployment import (
    DeploymentError,
    M10StageEvaluationSubjectRecord,
    evaluation_artifact_sha256,
    m10_stage_evaluation_subject_id,
    publish_m10_stage_evaluation_subject,
    resolve_m10_stage_evaluation_subject,
    resolve_serving_model,
)
from tinyllm.deployment.m10_stage import (
    MODEL_FILES,
    TOKENIZER_FILES,
    build_m10_stage_evaluation_subject,
    register_m10_stage_evaluation_subject,
)
from tinyllm.deployment.schema import ResolvedModel
from tinyllm.evaluation import M6ModelIdentity
from tinyllm.schemas import canonical_config_hash
from tinyllm.training.m10_sft import load_m10_full_sft_config
from tinyllm.training.m10_sft_schema import (
    M10CheckpointFile,
    M10CheckpointManifest,
    M10FullSFTRunResult,
    M10StageExport,
)

NOW = datetime(2026, 8, 25, tzinfo=UTC)
RUN_ID = "20260824T011335Z-m10-agent-full-sft-qwen3-0-6b-seed42-1ac1cad4-7b63"
REVISION: Literal["c1899de289a04d12100db370d81485cdf75e47ca"] = (
    "c1899de289a04d12100db370d81485cdf75e47ca"
)
PARENT_SHA: Literal["63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6"] = (
    "63db3b5f6ce0f12224167635fc8d67c53c32d7922caba3d65ac20c950f07dde6"
)
PARENT_RECORD: Literal["a83eaff5e9b91be403bddb7a4ac23926465e0a78872921429dad72b6ee7d0ca5"] = (
    "a83eaff5e9b91be403bddb7a4ac23926465e0a78872921429dad72b6ee7d0ca5"
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_model(root: Path) -> tuple[Path, str]:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        '{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}',
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    return root, evaluation_artifact_sha256(root, MODEL_FILES)


def _write_tokenizer(root: Path) -> tuple[Path, str]:
    root.mkdir(parents=True)
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return root, evaluation_artifact_sha256(root, TOKENIZER_FILES)


def _model(model_sha: str) -> M6ModelIdentity:
    return M6ModelIdentity(
        role="candidate",
        repository="Qwen/Qwen3-0.6B",
        base_revision=REVISION,
        attention_architecture="gqa",
        adaptation="full_sft",
        model_artifact_sha256=model_sha,
        model_parameters=596_049_920,
        training_run_id=RUN_ID,
        training_checkpoint_id="checkpoint-tokens-0005000000",
        training_tokens=5_000_000,
        training_config_sha256="1" * 64,
        dataset_version="m10-agent-sft-v1-4655d3e3",
        dataset_manifest_sha256=(
            "6ec41bdee5d6509cbd04c6f4a0c1c7af2fa0ef116de185d9c43e2076d1ee0490"
        ),
    )


def _record(root: Path) -> M10StageEvaluationSubjectRecord:
    run = root / "runs" / RUN_ID
    model_dir, model_sha = _write_model(run / "exports" / "checkpoint" / "model")
    tokenizer_dir, tokenizer_sha = _write_tokenizer(root / "cache" / "tokenizer")
    model = _model(model_sha)
    subject_id = m10_stage_evaluation_subject_id(
        model=model,
        tokenizer_artifact_sha256=tokenizer_sha,
        source_result_sha256="2" * 64,
        checkpoint_manifest_sha256="3" * 64,
        environment_sha256="4" * 64,
    )
    return M10StageEvaluationSubjectRecord(
        subject_id=subject_id,
        created_at=NOW,
        model=model,
        model_dir=model_dir,
        model_files=MODEL_FILES,
        model_artifact_sha256=model_sha,
        tokenizer_dir=tokenizer_dir,
        tokenizer_files=TOKENIZER_FILES,
        tokenizer_artifact_sha256=tokenizer_sha,
        source_run_dir=run,
        source_result_sha256="2" * 64,
        checkpoint_manifest_sha256="3" * 64,
        checkpoint_payload_sha256="5" * 64,
        environment_sha256="4" * 64,
    )


def test_publish_and_resolve_m10_stage_subject(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    record = _record(root)
    stored, record_sha = publish_m10_stage_evaluation_subject(root, record)
    resolved = resolve_m10_stage_evaluation_subject(root, record.subject_id, now=NOW)

    assert stored.production_eligible is False
    assert resolved.model == record.model
    assert resolved.evaluation_subject_sha256 == record_sha
    assert resolve_serving_model(root, record.subject_id, now=NOW) == resolved


def test_m10_stage_subject_rejects_artifact_drift(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    record = _record(root)
    publish_m10_stage_evaluation_subject(root, record)
    (record.model_dir / "model.safetensors").write_bytes(b"drift")

    with pytest.raises(DeploymentError, match="Artifact hash differs"):
        resolve_m10_stage_evaluation_subject(root, record.subject_id)


def test_build_m10_stage_subject_validates_run_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    run = root / "runs" / RUN_ID
    model_dir, model_sha = _write_model(run / "exports" / "checkpoint-tokens-0005000000" / "model")
    tokenizer_dir, tokenizer_sha = _write_tokenizer(root / "cache" / "tokenizer")
    config_source = Path("configs/sft/m10_agent_full_sft_qwen3_0_6b.yaml")
    run.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_source, run / "config.original.yaml")
    config = load_m10_full_sft_config(config_source)
    config_sha = canonical_config_hash(config)
    (run / "environment.json").write_text('{"schema_version":"1.0"}', encoding="utf-8")

    checkpoint_dir = run / "checkpoints" / "checkpoint-tokens-0005000000"
    checkpoint_dir.mkdir(parents=True)
    state = checkpoint_dir / "training_state.pt"
    state.write_bytes(b"state")
    checkpoint = M10CheckpointManifest(
        checkpoint_id="checkpoint-tokens-0005000000",
        run_id=RUN_ID,
        global_step=5040,
        completed_epochs=5,
        sequence_cursor=0,
        supervised_tokens=5_000_000,
        config_sha256=config_sha,
        dataset_version=config.data.dataset_version,
        dataset_manifest_sha256=config.data.manifest_sha256,
        parent_production_version=config.model.parent_production_version,
        parent_production_record_sha256=config.model.parent_production_record_sha256,
        parent_model_artifact_sha256=config.model.parent_model_artifact_sha256,
        git_commit="b" * 40,
        file=M10CheckpointFile(
            path="training_state.pt",
            size_bytes=state.stat().st_size,
            sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
        ),
        pinned=True,
        pin_reason="stage",
    )
    checkpoint_bytes = _json_bytes(checkpoint.to_dict())
    (checkpoint_dir / "manifest.json").write_bytes(checkpoint_bytes)
    (checkpoint_dir / "COMMITTED").write_bytes(
        _json_bytes({"manifest_sha256": hashlib.sha256(checkpoint_bytes).hexdigest()})
    )

    export_dir = model_dir.parent
    export = M10StageExport(
        checkpoint_id="checkpoint-tokens-0005000000",
        supervised_tokens=5_000_000,
        export_sha256=model_sha,
    )
    export_bytes = _json_bytes(export.to_dict())
    (export_dir / "stage_export.json").write_bytes(export_bytes)
    (export_dir / "COMMITTED").write_bytes(
        _json_bytes({"manifest_sha256": hashlib.sha256(export_bytes).hexdigest()})
    )
    result = M10FullSFTRunResult(
        status="stage_completed",
        mode="exact_resume",
        run_id=RUN_ID,
        config_sha256=config_sha,
        git_commit="b" * 40,
        git_dirty=False,
        dataset_version=config.data.dataset_version,
        dataset_manifest_sha256=config.data.manifest_sha256,
        parent_production_version=config.model.parent_production_version,
        parent_production_record_sha256=config.model.parent_production_record_sha256,
        parent_model_artifact_sha256=config.model.parent_model_artifact_sha256,
        attention_architecture="gqa",
        seed=42,
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        global_step=5040,
        completed_epochs=5,
        supervised_tokens=5_000_000,
        initial_loss=1.0,
        final_loss=0.2,
        duration_seconds=1.0,
        peak_allocated_bytes=2,
        peak_reserved_bytes=3,
        latest_checkpoint="checkpoint-tokens-0005000000",
        resumed_from_tokens=1_000_000,
        stage_export=export,
    )
    (run / "result.json").write_bytes(_json_bytes(result.to_dict()))
    production_model = M6ModelIdentity(
        role="candidate",
        repository="Qwen/Qwen3-0.6B",
        base_revision=REVISION,
        attention_architecture="gqa",
        adaptation="full_sft",
        model_artifact_sha256=PARENT_SHA,
        model_parameters=596_049_920,
        training_run_id="parent-run",
        training_checkpoint_id="checkpoint-tokens-0001000000",
        training_tokens=1_000_000,
        training_config_sha256="6" * 64,
        dataset_version="parent-data",
        dataset_manifest_sha256="7" * 64,
    )
    resolved = ResolvedModel(
        requested_ref="production",
        status="Production",
        model_version="qwen3-0-6b-m7-fa678d92",
        candidate_model_version="qwen3-0-6b-m6-d16c2357",
        candidate_record_sha256="8" * 64,
        production_record_sha256=PARENT_RECORD,
        model=production_model,
        model_dir=root / "parent-model",
        model_artifact_sha256=PARENT_SHA,
        tokenizer_dir=tokenizer_dir,
        tokenizer_artifact_sha256=tokenizer_sha,
        verified_at=NOW,
    )
    monkeypatch.setattr("tinyllm.deployment.m10_stage.resolve_model", lambda *_args: resolved)

    record = build_m10_stage_evaluation_subject(artifact_root=root, source_run=run)

    assert record.model_dir == model_dir
    assert record.model.model_artifact_sha256 == model_sha
    assert record.checkpoint_payload_sha256 == checkpoint.file.sha256

    published, record_sha = register_m10_stage_evaluation_subject(
        artifact_root=root,
        source_run=run,
    )
    repeated, repeated_sha = register_m10_stage_evaluation_subject(
        artifact_root=root,
        source_run=run,
    )
    assert published.subject_id == record.subject_id
    assert repeated == published
    assert repeated_sha == record_sha

    one_million_model_dir, one_million_model_sha = _write_model(
        run / "exports" / "checkpoint-tokens-0001000000" / "model"
    )
    one_million_checkpoint_dir = run / "checkpoints" / "checkpoint-tokens-0001000000"
    one_million_checkpoint_dir.mkdir(parents=True)
    one_million_state = one_million_checkpoint_dir / "training_state.pt"
    one_million_state.write_bytes(b"one-million-state")
    one_million_checkpoint = M10CheckpointManifest(
        checkpoint_id="checkpoint-tokens-0001000000",
        run_id=RUN_ID,
        global_step=1008,
        completed_epochs=1,
        sequence_cursor=0,
        supervised_tokens=1_000_000,
        config_sha256=config_sha,
        dataset_version=config.data.dataset_version,
        dataset_manifest_sha256=config.data.manifest_sha256,
        parent_production_version=config.model.parent_production_version,
        parent_production_record_sha256=config.model.parent_production_record_sha256,
        parent_model_artifact_sha256=config.model.parent_model_artifact_sha256,
        git_commit="b" * 40,
        file=M10CheckpointFile(
            path="training_state.pt",
            size_bytes=one_million_state.stat().st_size,
            sha256=hashlib.sha256(one_million_state.read_bytes()).hexdigest(),
        ),
        pinned=True,
        pin_reason="stage",
    )
    one_million_checkpoint_bytes = _json_bytes(one_million_checkpoint.to_dict())
    (one_million_checkpoint_dir / "manifest.json").write_bytes(one_million_checkpoint_bytes)
    (one_million_checkpoint_dir / "COMMITTED").write_bytes(
        _json_bytes({"manifest_sha256": hashlib.sha256(one_million_checkpoint_bytes).hexdigest()})
    )
    one_million_export = M10StageExport(
        checkpoint_id="checkpoint-tokens-0001000000",
        supervised_tokens=1_000_000,
        export_sha256=one_million_model_sha,
    )
    one_million_export_dir = one_million_model_dir.parent
    one_million_export_bytes = _json_bytes(one_million_export.to_dict())
    (one_million_export_dir / "stage_export.json").write_bytes(one_million_export_bytes)
    (one_million_export_dir / "COMMITTED").write_bytes(
        _json_bytes({"manifest_sha256": hashlib.sha256(one_million_export_bytes).hexdigest()})
    )
    one_million_result = M10FullSFTRunResult(
        status="stage_completed",
        mode="fresh",
        run_id=RUN_ID,
        config_sha256=config_sha,
        git_commit="b" * 40,
        git_dirty=False,
        dataset_version=config.data.dataset_version,
        dataset_manifest_sha256=config.data.manifest_sha256,
        parent_production_version=config.model.parent_production_version,
        parent_production_record_sha256=config.model.parent_production_record_sha256,
        parent_model_artifact_sha256=config.model.parent_model_artifact_sha256,
        attention_architecture="gqa",
        seed=42,
        physical_gpu_index=4,
        gpu_name="NVIDIA GeForce RTX 3090",
        global_step=1008,
        completed_epochs=1,
        supervised_tokens=1_000_000,
        initial_loss=1.0,
        final_loss=0.9,
        duration_seconds=1.0,
        peak_allocated_bytes=2,
        peak_reserved_bytes=3,
        latest_checkpoint="checkpoint-tokens-0001000000",
        stage_export=one_million_export,
    )
    attempt = run / "attempts" / "fresh-stage_completed-tokens-0001000000.json"
    attempt.parent.mkdir()
    attempt.write_bytes(_json_bytes(one_million_result.to_dict()))

    one_million_record = build_m10_stage_evaluation_subject(
        artifact_root=root,
        source_run=run,
        stage_tokens=1_000_000,
    )

    assert one_million_record.kind == "m10_full_sft_1m"
    assert one_million_record.model_dir == one_million_model_dir
    assert one_million_record.model.training_tokens == 1_000_000
    assert one_million_record.subject_id.startswith("qwen3-0-6b-m10-full-sft-1m-")
