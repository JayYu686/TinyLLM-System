from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import tinyllm.training.m10_lora_general as general_module
from tinyllm.deployment import ResolvedEvaluationSubject
from tinyllm.evaluation import GeneralBaselineSummary, GeneralTaskResult, M6ModelIdentity
from tinyllm.training.m10_lora_general import run_m10_lora_general_pass
from tinyllm.training.m10_lora_schema import (
    M10_LORA_PARENT_MODEL_SHA256,
    M10_LORA_PARENT_RECORD_SHA256,
    M10_LORA_PARENT_SUBJECT,
    M10LoRAGeneralPassSummary,
)


def _general_summary() -> GeneralBaselineSummary:
    tasks = tuple(
        GeneralTaskResult(
            task=task,  # type: ignore[arg-type]
            samples=samples,
            acc=0.5,
            acc_stderr=0.01,
            acc_norm=0.6,
            acc_norm_stderr=0.01,
        )
        for task, samples in (
            ("tinyllm_arc_easy", 2376),
            ("tinyllm_hellaswag", 10042),
            ("tinyllm_piqa", 1838),
        )
    )
    return GeneralBaselineSummary(
        harness_version="0.4.12",
        model_parameters=8_234_382_336,
        tasks=tasks,  # type: ignore[arg-type]
        evaluation_seconds=10.0,
    )


def _resolved(tmp_path: Path, *, candidate: bool) -> ResolvedEvaluationSubject:
    model_dir = (tmp_path / ("candidate-model" if candidate else "parent-model")).resolve()
    tokenizer_dir = (tmp_path / "tokenizer").resolve()
    adapter_dir = (tmp_path / "adapter").resolve() if candidate else None
    model_dir.mkdir(exist_ok=True)
    tokenizer_dir.mkdir(exist_ok=True)
    if adapter_dir is not None:
        adapter_dir.mkdir(exist_ok=True)
    subject_id = "qwen3-8b-m10-agent-lora-5m-1234abcd" if candidate else M10_LORA_PARENT_SUBJECT
    model_hash = "9" * 64 if candidate else M10_LORA_PARENT_MODEL_SHA256
    model = M6ModelIdentity(
        role="candidate" if candidate else "base",
        repository="Qwen/Qwen3-8B",
        base_revision="b968826d9c46dd6066d109eabc6255188de91218",
        attention_architecture="gqa",
        adaptation="lora" if candidate else "base",
        model_artifact_sha256=model_hash,
        model_parameters=8_234_382_336,
        training_run_id="m10-lora-test-run" if candidate else None,
        training_checkpoint_id="checkpoint-tokens-0005000000" if candidate else None,
        training_tokens=5_000_000 if candidate else None,
        training_config_sha256="7" * 64 if candidate else None,
        dataset_version="m10-agent-sft-v2-1234abcd" if candidate else None,
        dataset_manifest_sha256="6" * 64 if candidate else None,
        adapter_sha256="8" * 64 if candidate else None,
    )
    return ResolvedEvaluationSubject(
        requested_ref=subject_id,
        model_version=subject_id,
        evaluation_subject_sha256="5" * 64 if candidate else M10_LORA_PARENT_RECORD_SHA256,
        model=model,
        model_dir=model_dir,
        model_artifact_sha256=model_hash,
        tokenizer_dir=tokenizer_dir,
        tokenizer_artifact_sha256="4" * 64,
        adapter_dir=adapter_dir,
        adapter_artifact_sha256="8" * 64 if candidate else None,
        verified_at=datetime(2026, 8, 31, tzinfo=UTC),
    )


@pytest.mark.parametrize("candidate", [False, True])
def test_m10_lora_general_pass_binds_parent_and_candidate_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate: bool
) -> None:
    subject = _resolved(tmp_path, candidate=candidate)
    output = (tmp_path / ("candidate-output" if candidate else "parent-output")).resolve()
    artifact_root = (tmp_path / "artifacts").resolve()
    artifact_root.mkdir()

    def run_general(*_args: object, **kwargs: object) -> GeneralBaselineSummary:
        assert kwargs["adapter_path"] == subject.adapter_dir
        output_path = kwargs["output_path"]
        assert isinstance(output_path, Path)
        raw = output_path / "raw/model"
        raw.mkdir(parents=True)
        (raw / "results.json").write_text("{}\n", encoding="utf-8")
        return _general_summary()

    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda _: "NVIDIA GeForce RTX 3090",
    )
    monkeypatch.setattr("tinyllm.training.m10_lora_general.torch.cuda", cuda)
    monkeypatch.setattr(general_module, "read_git_identity", lambda _: ("a" * 40, False))
    monkeypatch.setattr(general_module, "_resolve_subject", lambda *_: subject)
    monkeypatch.setattr(general_module, "run_general_evaluation", run_general)
    monkeypatch.setattr(
        general_module,
        "_environment_payload",
        lambda **_: {"schema_version": "1.0", "environment": "test"},
    )
    monkeypatch.setattr(
        general_module,
        "_hardware_payload",
        lambda _: {"schema_version": "1.0", "hardware": "test"},
    )

    result = run_m10_lora_general_pass(
        artifact_root=artifact_root,
        subject_id=subject.requested_ref,
        output_dir=output,
        project_root=Path.cwd(),
        release_config_path=Path("configs/eval/m6_release_v7.yaml"),
        physical_gpu_index=6,
    )

    kind = "candidate" if candidate else "parent"
    assert result.evaluation_id.startswith(f"m10-lora-m6-general-{kind}-")
    assert result.evaluation_subject_sha256 == subject.evaluation_subject_sha256
    assert result.general.aggregate_basis_points == 6000
    assert (
        M10LoRAGeneralPassSummary.model_validate_json((output / "summary.json").read_bytes())
        == result
    )
