from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import pytest
import torch

from tinyllm.deployment import (
    DeploymentError,
    M10LoRAStageEvaluationSubjectRecord,
    effective_artifact_sha256,
    evaluation_artifact_sha256,
    m10_lora_stage_evaluation_subject_id,
    publish_m10_lora_stage_evaluation_subject,
    resolve_m10_lora_stage_evaluation_subject,
    resolve_serving_model,
)
from tinyllm.deployment.m10_lora_stage import (
    M10LoRAAdapterCalibrationEvidence,
    M10LoRAAdapterInterpolationEvidence,
    M10LoRAAdapterModuleProfileEvidence,
    M10LoRAStageRegistrationError,
    ModuleName,
    ModuleScale,
    _interpolate_adapter_states,
    _module_profile_scales,
    _profile_adapter_state,
)
from tinyllm.evaluation import M6ModelIdentity

NOW = datetime(2026, 8, 25, tzinfo=UTC)
REVISION: Literal["b968826d9c46dd6066d109eabc6255188de91218"] = (
    "b968826d9c46dd6066d109eabc6255188de91218"
)
MODEL_FILES = (
    "config.json",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
    "model.safetensors.index.json",
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def test_lora_calibration_evidence_requires_unchanged_weights_and_exact_scale() -> None:
    payload = {
        "source_subject_id": "qwen3-8b-m10-agent-lora-5m-1234abcd",
        "source_evaluation_subject_sha256": "a" * 64,
        "source_adapter_artifact_sha256": "b" * 64,
        "source_adapter_weights_sha256": "c" * 64,
        "calibrated_adapter_artifact_sha256": "d" * 64,
        "calibrated_adapter_weights_sha256": "c" * 64,
        "calibrated_alpha": 16,
        "relative_scale_basis_points": 5000,
    }
    evidence = M10LoRAAdapterCalibrationEvidence.model_validate(payload)
    assert evidence.expected_scale_basis_points == 5000
    quarter_strength = M10LoRAAdapterCalibrationEvidence.model_validate(
        {**payload, "calibrated_alpha": 8, "relative_scale_basis_points": 2500}
    )
    assert quarter_strength.expected_scale_basis_points == 2500
    eighth_strength = M10LoRAAdapterCalibrationEvidence.model_validate(
        {**payload, "calibrated_alpha": 4, "relative_scale_basis_points": 1250}
    )
    assert eighth_strength.expected_scale_basis_points == 1250
    with pytest.raises(ValueError, match="must not change Adapter weights"):
        M10LoRAAdapterCalibrationEvidence.model_validate(
            {**payload, "calibrated_adapter_weights_sha256": "e" * 64}
        )
    with pytest.raises(ValueError, match="differs from Alpha ratio"):
        M10LoRAAdapterCalibrationEvidence.model_validate(
            {**payload, "relative_scale_basis_points": 2500}
        )


def test_lora_interpolation_evidence_and_tensor_math_are_exact() -> None:
    payload = {
        "early_subject_id": "qwen3-8b-m10-agent-lora-1m-1234abcd",
        "early_evaluation_subject_sha256": "a" * 64,
        "early_adapter_artifact_sha256": "b" * 64,
        "late_subject_id": "qwen3-8b-m10-agent-lora-5m-2345bcde",
        "late_evaluation_subject_sha256": "c" * 64,
        "late_adapter_artifact_sha256": "d" * 64,
        "training_run_id": "m10-lora-unit-run",
        "early_weight_basis_points": 2500,
        "late_weight_basis_points": 7500,
        "interpolated_adapter_artifact_sha256": "e" * 64,
    }
    evidence = M10LoRAAdapterInterpolationEvidence.model_validate(payload)
    assert evidence.early_weight_basis_points + evidence.late_weight_basis_points == 10_000
    with pytest.raises(ValueError, match="must sum to 10000"):
        M10LoRAAdapterInterpolationEvidence.model_validate(
            {**payload, "late_weight_basis_points": 2500}
        )

    early = {"layer": torch.tensor([0.0, 4.0], dtype=torch.bfloat16)}
    late = {"layer": torch.tensor([4.0, 8.0], dtype=torch.bfloat16)}
    interpolated = _interpolate_adapter_states(early, late, late_weight_basis_points=7500)
    assert torch.equal(interpolated["layer"], torch.tensor([3.0, 7.0], dtype=torch.bfloat16))
    with pytest.raises(M10LoRAStageRegistrationError, match="inputs are incompatible"):
        _interpolate_adapter_states(early, {}, late_weight_basis_points=7500)


def test_lora_module_profile_evidence_and_tensor_math_are_exact() -> None:
    scales: dict[ModuleName, ModuleScale] = {
        "down_proj": 1250,
        "gate_proj": 1250,
        "k_proj": 10000,
        "o_proj": 10000,
        "q_proj": 10000,
        "up_proj": 1250,
        "v_proj": 10000,
    }
    evidence = M10LoRAAdapterModuleProfileEvidence(
        source_subject_id="qwen3-8b-m10-agent-lora-5m-1234abcd",
        source_evaluation_subject_sha256="a" * 64,
        source_adapter_artifact_sha256="b" * 64,
        source_adapter_weights_sha256="c" * 64,
        derived_adapter_artifact_sha256="d" * 64,
        derived_adapter_weights_sha256="e" * 64,
        profile="attention_full_mlp_eighth",
        module_relative_scale_basis_points=scales,
    )
    assert evidence.module_relative_scale_basis_points == scales
    with pytest.raises(ValueError, match="differ from the named profile"):
        M10LoRAAdapterModuleProfileEvidence(
            **evidence.model_dump(
                mode="python",
                exclude={"module_relative_scale_basis_points"},
            ),
            module_relative_scale_basis_points={**scales, "q_proj": 1250},
        )

    source = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor(
            [2.0], dtype=torch.bfloat16
        ),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.tensor(
            [2.0], dtype=torch.bfloat16
        ),
        "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": torch.tensor(
            [2.0], dtype=torch.bfloat16
        ),
    }
    derived = _profile_adapter_state(source, profile="attention_full_mlp_eighth")
    assert torch.equal(
        derived["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
        torch.tensor([2.0], dtype=torch.bfloat16),
    )
    assert torch.equal(
        derived["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"],
        torch.tensor([16.0], dtype=torch.bfloat16),
    )
    assert torch.equal(
        derived["base_model.model.model.layers.0.mlp.up_proj.lora_B.weight"],
        torch.tensor([2.0], dtype=torch.bfloat16),
    )
    assert _module_profile_scales("mlp_half_attention_eighth")["up_proj"] == 5000
    assert _module_profile_scales("mlp_quarter_attention_eighth")["up_proj"] == 2500
    assert _module_profile_scales("mlp_half_attention_eighth")["q_proj"] == 1250


def _record(root: Path, *, stage_tokens: int = 1_000_000) -> M10LoRAStageEvaluationSubjectRecord:
    model_dir = root / "cache" / "qwen3-8b"
    adapter_dir = root / "runs" / "m10-lora" / "adapter"
    source_run = root / "runs" / "m10-lora"
    model_dir.mkdir(parents=True)
    adapter_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        '{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}',
        encoding="utf-8",
    )
    for name in MODEL_FILES[1:]:
        (model_dir / name).write_bytes(name.encode())
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    base_sha = evaluation_artifact_sha256(model_dir, MODEL_FILES)
    tokenizer_sha = evaluation_artifact_sha256(model_dir, TOKENIZER_FILES)
    adapter_sha = evaluation_artifact_sha256(adapter_dir, ADAPTER_FILES)
    effective_sha = effective_artifact_sha256(base_sha, adapter_sha)
    checkpoint_id = f"checkpoint-tokens-{stage_tokens:010d}"
    model = M6ModelIdentity(
        role="candidate",
        repository="Qwen/Qwen3-8B",
        base_revision=REVISION,
        attention_architecture="gqa",
        adaptation="lora",
        model_artifact_sha256=effective_sha,
        model_parameters=8_234_382_336,
        training_run_id="m10-lora-unit-run",
        training_checkpoint_id=checkpoint_id,
        training_tokens=stage_tokens,
        training_config_sha256="a" * 64,
        dataset_version="m10-agent-sft-v1-4655d3e3",
        dataset_manifest_sha256="b" * 64,
        adapter_sha256=adapter_sha,
    )
    source_result_sha256 = "c" * 64
    checkpoint_manifest_sha256 = "d" * 64
    memory_probe_sha256 = "e" * 64
    subject_id = m10_lora_stage_evaluation_subject_id(
        model=model,
        base_model_artifact_sha256=base_sha,
        tokenizer_artifact_sha256=tokenizer_sha,
        adapter_artifact_sha256=adapter_sha,
        source_result_sha256=source_result_sha256,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        memory_probe_sha256=memory_probe_sha256,
        checkpoint_export_evidence_sha256=(
            "1" * 64 if stage_tokens in {3_000_000, 4_000_000} else None
        ),
    )
    kind = cast(
        Literal[
            "m10_agent_lora_1m",
            "m10_agent_lora_3m",
            "m10_agent_lora_4m",
            "m10_agent_lora_5m",
            "m10_agent_lora_10m",
        ],
        {
            1_000_000: "m10_agent_lora_1m",
            3_000_000: "m10_agent_lora_3m",
            4_000_000: "m10_agent_lora_4m",
            5_000_000: "m10_agent_lora_5m",
            10_000_000: "m10_agent_lora_10m",
        }[stage_tokens],
    )
    return M10LoRAStageEvaluationSubjectRecord(
        subject_id=subject_id,
        kind=kind,
        created_at=NOW,
        model=model,
        model_dir=model_dir,
        model_files=MODEL_FILES,
        base_model_artifact_sha256=base_sha,
        tokenizer_dir=model_dir,
        tokenizer_files=TOKENIZER_FILES,
        tokenizer_artifact_sha256=tokenizer_sha,
        adapter_dir=adapter_dir,
        adapter_files=ADAPTER_FILES,
        adapter_artifact_sha256=adapter_sha,
        effective_artifact_sha256=effective_sha,
        source_run_dir=source_run,
        source_result_sha256=source_result_sha256,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        checkpoint_payload_sha256="f" * 64,
        memory_probe_sha256=memory_probe_sha256,
        checkpoint_export_evidence_sha256=(
            "1" * 64 if stage_tokens in {3_000_000, 4_000_000} else None
        ),
        parent_evaluation_subject="qwen3-8b-m9-base-90587dd6",
        parent_evaluation_subject_sha256=(
            "9f72bba28bcfaed45f116080033cb9bc83be1632570e71623f2a5684350261d8"
        ),
    )


@pytest.mark.parametrize("stage_tokens", [1_000_000, 3_000_000, 4_000_000, 5_000_000, 10_000_000])
def test_publish_resolve_and_route_lora_subject(tmp_path: Path, stage_tokens: int) -> None:
    root = tmp_path.resolve()
    record = _record(root, stage_tokens=stage_tokens)
    stored, record_sha = publish_m10_lora_stage_evaluation_subject(root, record)
    resolved = resolve_m10_lora_stage_evaluation_subject(root, record.subject_id, now=NOW)

    assert stored.production_eligible is False
    assert resolved.evaluation_subject_sha256 == record_sha
    assert resolved.adapter_artifact_sha256 == record.adapter_artifact_sha256
    assert resolved.model_artifact_sha256 == record.effective_artifact_sha256
    assert resolve_serving_model(root, record.subject_id, now=NOW) == resolved


def test_lora_subject_rejects_identity_and_artifact_drift(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    record = _record(root)
    payload = record.model_dump(mode="python")
    payload["subject_id"] = "qwen3-8b-m10-agent-lora-1m-aaaaaaaa"
    with pytest.raises(ValueError, match="ID differs"):
        M10LoRAStageEvaluationSubjectRecord.model_validate(payload)

    publish_m10_lora_stage_evaluation_subject(root, record)
    (record.adapter_dir / "adapter_model.safetensors").write_bytes(b"drift")
    with pytest.raises(DeploymentError, match="Artifact hash differs"):
        resolve_m10_lora_stage_evaluation_subject(root, record.subject_id)


def test_lora_subject_rejects_paths_outside_store(tmp_path: Path) -> None:
    root = (tmp_path / "root").resolve()
    root.mkdir()
    record = _record((tmp_path / "outside").resolve())
    with pytest.raises(DeploymentError, match="escapes the Artifact Store"):
        publish_m10_lora_stage_evaluation_subject(root, record)
