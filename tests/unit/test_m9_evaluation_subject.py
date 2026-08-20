from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from tinyllm.deployment import (
    DeploymentError,
    M9EvaluationSubjectRecord,
    effective_artifact_sha256,
    evaluation_artifact_sha256,
    evaluation_subject_id,
    publish_evaluation_subject,
    resolve_evaluation_subject,
    resolve_serving_model,
)
from tinyllm.evaluation import M6ModelIdentity

NOW = datetime(2026, 8, 20, tzinfo=UTC)
REVISION: Literal["b968826d9c46dd6066d109eabc6255188de91218"] = (
    "b968826d9c46dd6066d109eabc6255188de91218"
)


def _files(root: Path) -> tuple[Path, Path, tuple[str, ...], tuple[str, ...]]:
    model_dir = root / "cache" / "qwen3-8b"
    adapter_dir = root / "runs" / "historical" / "adapter"
    model_dir.mkdir(parents=True)
    adapter_dir.mkdir(parents=True)
    model_files = ("config.json", "model.safetensors")
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
    (model_dir / "config.json").write_text(
        '{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}',
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    return model_dir, adapter_dir, model_files, tokenizer_files


def _record(root: Path, *, historical: bool = False) -> M9EvaluationSubjectRecord:
    model_dir, adapter_dir, model_files, tokenizer_files = _files(root)
    base_sha = evaluation_artifact_sha256(model_dir, model_files)
    tokenizer_sha = evaluation_artifact_sha256(model_dir, tokenizer_files)
    adapter_files = ("adapter_config.json", "adapter_model.safetensors")
    adapter_sha = evaluation_artifact_sha256(adapter_dir, adapter_files) if historical else None
    effective_sha = effective_artifact_sha256(base_sha, adapter_sha)
    model = M6ModelIdentity(
        role="candidate" if historical else "base",
        repository="Qwen/Qwen3-8B",
        base_revision=REVISION,
        attention_architecture="gqa",
        adaptation="lora" if historical else "base",
        model_artifact_sha256=effective_sha,
        model_parameters=8_234_382_336,
        training_run_id="m5-historical-run" if historical else None,
        training_checkpoint_id=("checkpoint-tokens-0010000000" if historical else None),
        training_tokens=10_000_000 if historical else None,
        training_config_sha256="c" * 64 if historical else None,
        dataset_version="m5-data-v1" if historical else None,
        dataset_manifest_sha256="d" * 64 if historical else None,
        adapter_sha256=adapter_sha,
    )
    kind: Literal["base", "historical_lora"] = "historical_lora" if historical else "base"
    source_sha = "f" * 64
    subject_id = evaluation_subject_id(
        kind=kind,
        model=model,
        base_model_artifact_sha256=base_sha,
        tokenizer_artifact_sha256=tokenizer_sha,
        adapter_artifact_sha256=adapter_sha,
        source_evidence_sha256=source_sha,
    )
    return M9EvaluationSubjectRecord(
        subject_id=subject_id,
        kind=kind,
        created_at=NOW,
        model=model,
        model_dir=model_dir,
        model_files=model_files,
        base_model_artifact_sha256=base_sha,
        tokenizer_dir=model_dir,
        tokenizer_files=tokenizer_files,
        tokenizer_artifact_sha256=tokenizer_sha,
        adapter_dir=adapter_dir if historical else None,
        adapter_files=adapter_files if historical else (),
        adapter_artifact_sha256=adapter_sha,
        effective_artifact_sha256=effective_sha,
        source_evidence_sha256=source_sha,
    )


@pytest.mark.parametrize("historical", [False, True])
def test_publish_and_resolve_evaluation_subject(tmp_path: Path, historical: bool) -> None:
    root = tmp_path.resolve()
    record = _record(root, historical=historical)
    stored, record_sha = publish_evaluation_subject(root, record)
    resolved = resolve_evaluation_subject(root, record.subject_id, now=NOW)

    assert stored.production_eligible is False
    assert resolved.status == "Evaluation"
    assert resolved.evaluation_subject_sha256 == record_sha
    assert resolved.model_artifact_sha256 == record.effective_artifact_sha256
    assert (resolved.adapter_dir is not None) is historical
    assert resolve_serving_model(root, record.subject_id, now=NOW) == resolved


def test_publish_is_idempotent_across_invocation_timestamps(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    record = _record(root)
    first, first_sha = publish_evaluation_subject(root, record)
    later = record.model_copy(update={"created_at": NOW + timedelta(minutes=1)})
    second, second_sha = publish_evaluation_subject(root, later)

    assert second == first
    assert second_sha == first_sha


def test_resolve_rejects_artifact_drift(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    record = _record(root, historical=True)
    publish_evaluation_subject(root, record)
    assert record.adapter_dir is not None
    (record.adapter_dir / "adapter_model.safetensors").write_bytes(b"drift")

    with pytest.raises(DeploymentError, match="Artifact hash differs"):
        resolve_evaluation_subject(root, record.subject_id)


def test_publish_rejects_artifact_outside_store(tmp_path: Path) -> None:
    root = (tmp_path / "root").resolve()
    root.mkdir()
    outside = (tmp_path / "outside").resolve()
    record = _record(outside)

    with pytest.raises(DeploymentError, match="escapes the Artifact Store"):
        publish_evaluation_subject(root, record)


def test_resolve_rejects_path_like_subject_id(tmp_path: Path) -> None:
    root = tmp_path.resolve()

    with pytest.raises(DeploymentError, match="subject ID is invalid"):
        resolve_evaluation_subject(root, "qwen3-8b-m9-../../production")


def test_base_record_rejects_adapter_claim(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    record = _record(root)
    with pytest.raises(ValueError, match="cannot contain an Adapter"):
        M9EvaluationSubjectRecord.model_validate(
            {
                **record.model_dump(mode="python"),
                "adapter_dir": root / "adapter",
            }
        )
