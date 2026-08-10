from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import tinyllm.data.m6_gate_replay as replay_module
from tinyllm.data import (
    M5DualModeCorrectionMixtureManifest,
    M5MixtureSequence,
    M6GateRepairMixtureManifest,
    M6GateReplayMixtureManifest,
    OpenM5Mixture,
    build_m6_gate_replay_mixture,
    open_m5_ablation_mixture,
)
from tinyllm.training.m5_config import load_m5_sft_config


def _manifest_mapping() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset_name": "m6-gate-replay-mixture",
        "mixture_version": "m6-gate-replay-mixture-v1-6c169970",
        "parent_dataset_version": "m2-sft-v1-f82ff32e",
        "parent_content_sha256": "f" * 64,
        "diagnostic_protocol_version": "m6-release-v2",
        "source_consumed_evaluation_content": False,
        "evaluation_prompt_overlap_count": 0,
        "correction_mixture_version": "m5-dual-mode-correction-mixture-v1-4bc342d4",
        "correction_manifest_sha256": (
            "db66ce847fac4bd2966666d125f1bb4e21dd0fd3bb608a1a384806c206f8945c"
        ),
        "repair_mixture_version": "m6-gate-repair-mixture-v1-be2aa7fa",
        "repair_manifest_sha256": (
            "13826d120bdbfc3db38ba035f243ddd4e9e85e8f49aec25e8e7ff20f451c7fc1"
        ),
        "tokenizer_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "nonthinking_template_id": "qwen3-chatml-nonthinking-sft-v2",
        "nonthinking_template_sha256": (
            "fba6724bd16200356794105a2273bbd42e777c8311ef1760059c6f0766171ca2"
        ),
        "thinking_template_id": "qwen3-chatml-thinking-v1",
        "thinking_template_sha256": (
            "4786143dbb7adb72a922d5efdcbe6596f2d65dcdc35d7bbf1b22830b795c2af9"
        ),
        "sequence_length": 1024,
        "pad_token_id": 151643,
        "target_supervised_tokens": 1_000_000,
        "thinking_fraction_basis_points": 3000,
        "nonthinking_supervised_tokens": 700_000,
        "thinking_supervised_tokens": 300_000,
        "correction_nonthinking_supervised_tokens": 400_000,
        "repair_nonthinking_supervised_tokens": 300_000,
        "correction_thinking_supervised_tokens": 150_000,
        "repair_thinking_supervised_tokens": 150_000,
        "sequence_count": 20,
        "nonthinking_sequence_count": 14,
        "thinking_sequence_count": 6,
        "correction_nonthinking_source_sequences": 1,
        "repair_nonthinking_source_sequences": 1,
        "correction_thinking_source_sequences": 1,
        "repair_thinking_source_sequences": 1,
        "correction_nonthinking_reuse_count": 0,
        "repair_nonthinking_reuse_count": 0,
        "correction_thinking_reuse_count": 0,
        "repair_thinking_reuse_count": 0,
        "partially_masked_sequences": 4,
        "build_seed": 20260812,
        "content_sha256": "6c169970" + "0" * 56,
        "artifact": {"path": "sequences.npz", "size_bytes": 1, "sha256": "a" * 64},
    }


def test_gate_replay_manifest_binds_exact_replay_strata() -> None:
    manifest = M6GateReplayMixtureManifest.model_validate(_manifest_mapping())

    assert manifest.correction_nonthinking_supervised_tokens == 400_000
    assert manifest.repair_nonthinking_supervised_tokens == 300_000
    assert manifest.correction_thinking_supervised_tokens == 150_000
    assert manifest.repair_thinking_supervised_tokens == 150_000

    with pytest.raises(ValueError, match="version"):
        M6GateReplayMixtureManifest.model_validate(
            _manifest_mapping() | {"mixture_version": "m6-gate-replay-mixture-v1-deadbeef"}
        )
    with pytest.raises(ValueError, match="strata"):
        invalid_strata = manifest.model_copy(
            update={"repair_nonthinking_supervised_tokens": 299_999}
        )
        invalid_strata.validate_counts_and_identity()  # type: ignore[operator]
    with pytest.raises(ValueError, match="sequence counts"):
        invalid_sequences = manifest.model_copy(update={"sequence_count": 21})
        invalid_sequences.validate_counts_and_identity()  # type: ignore[operator]


def test_gate_replay_training_configs_bind_the_private_manifest() -> None:
    seed42 = load_m5_sft_config(Path("configs/sft/m6_gate_replay_r3_seed42.yaml"))
    stability = load_m5_sft_config(Path("configs/sft/m6_gate_replay_r3_seed20260812.yaml"))

    assert seed42.data.dataset_version == "m6-gate-replay-mixture-v1-6c169970"
    assert seed42.data.mix_manifest_sha256 == stability.data.mix_manifest_sha256
    assert seed42.run.seed == 42
    assert stability.run.seed == 20260812


def _sequence(mode: int) -> M5MixtureSequence:
    return M5MixtureSequence(
        input_ids=(1,) * 1024,
        labels=(-100,) + (1,) * 1000 + (-100,) * 23,
        attention_mask=(1,) * 1001 + (0,) * 23,
        mode=mode,
    )


def test_gate_replay_builder_commits_reopens_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correction_root = tmp_path / "correction"
    repair_root = tmp_path / "repair"
    correction_root.mkdir()
    repair_root.mkdir()
    (correction_root / "manifest.json").write_bytes(b"synthetic-correction\n")
    (repair_root / "manifest.json").write_bytes(b"synthetic-repair\n")
    correction_sha256 = hashlib.sha256(b"synthetic-correction\n").hexdigest()
    repair_sha256 = hashlib.sha256(b"synthetic-repair\n").hexdigest()
    parent_sha256 = "f" * 64
    correction_manifest = M5DualModeCorrectionMixtureManifest.model_construct(
        mixture_version="m5-dual-mode-correction-mixture-v1-4bc342d4",
        parent_content_sha256=parent_sha256,
    )
    repair_manifest = M6GateRepairMixtureManifest.model_construct(
        mixture_version="m6-gate-repair-mixture-v1-be2aa7fa",
        parent_content_sha256=parent_sha256,
    )
    real_open = open_m5_ablation_mixture

    def fake_open(root: Path) -> OpenM5Mixture:
        if root == correction_root:
            return OpenM5Mixture(root, correction_manifest)
        if root == repair_root:
            return OpenM5Mixture(root, repair_manifest)
        return real_open(root)

    monkeypatch.setattr(replay_module, "open_m5_ablation_mixture", fake_open)
    monkeypatch.setattr(
        replay_module,
        "_source_sequences",
        lambda _: (_sequence(0), _sequence(1)),
    )
    output_root = tmp_path / "output"
    manifest = build_m6_gate_replay_mixture(
        correction_root=correction_root,
        repair_root=repair_root,
        output_root=output_root,
        build_seed=20260812,
        expected_correction_manifest_sha256=correction_sha256,
        expected_repair_manifest_sha256=repair_sha256,
    )
    destination = output_root / manifest.mixture_version
    reopened = real_open(destination)
    repeated = build_m6_gate_replay_mixture(
        correction_root=correction_root,
        repair_root=repair_root,
        output_root=output_root,
        build_seed=20260812,
        expected_correction_manifest_sha256=correction_sha256,
        expected_repair_manifest_sha256=repair_sha256,
    )

    assert reopened.manifest == manifest
    assert repeated == manifest
    assert manifest.nonthinking_supervised_tokens == 700_000
    assert manifest.thinking_supervised_tokens == 300_000
    assert (destination / "COMMITTED").is_file()


def test_gate_replay_builder_rejects_wrong_source_kind_identity_and_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correction_root = tmp_path / "correction"
    repair_root = tmp_path / "repair"
    correction_root.mkdir()
    repair_root.mkdir()
    (correction_root / "manifest.json").write_bytes(b"correction\n")
    (repair_root / "manifest.json").write_bytes(b"repair\n")
    correction_sha256 = hashlib.sha256(b"correction\n").hexdigest()
    repair_sha256 = hashlib.sha256(b"repair\n").hexdigest()
    correction = M5DualModeCorrectionMixtureManifest.model_construct(
        mixture_version="m5-dual-mode-correction-mixture-v1-4bc342d4",
        parent_content_sha256="a" * 64,
    )
    repair = M6GateRepairMixtureManifest.model_construct(
        mixture_version="m6-gate-repair-mixture-v1-be2aa7fa",
        parent_content_sha256="a" * 64,
    )

    monkeypatch.setattr(
        replay_module,
        "open_m5_ablation_mixture",
        lambda root: OpenM5Mixture(root, repair),
    )
    with pytest.raises(ValueError, match="correction source"):
        build_m6_gate_replay_mixture(
            correction_root=correction_root,
            repair_root=repair_root,
            output_root=tmp_path / "output",
            build_seed=1,
        )

    monkeypatch.setattr(
        replay_module,
        "open_m5_ablation_mixture",
        lambda root: OpenM5Mixture(root, correction),
    )
    with pytest.raises(ValueError, match="repair source"):
        build_m6_gate_replay_mixture(
            correction_root=correction_root,
            repair_root=repair_root,
            output_root=tmp_path / "output",
            build_seed=1,
        )

    monkeypatch.setattr(
        replay_module,
        "open_m5_ablation_mixture",
        lambda root: OpenM5Mixture(root, correction if root == correction_root else repair),
    )
    with pytest.raises(ValueError, match="identity"):
        build_m6_gate_replay_mixture(
            correction_root=correction_root,
            repair_root=repair_root,
            output_root=tmp_path / "output",
            build_seed=1,
        )

    mismatched_repair = repair.model_copy(update={"parent_content_sha256": "b" * 64})
    monkeypatch.setattr(
        replay_module,
        "open_m5_ablation_mixture",
        lambda root: OpenM5Mixture(
            root, correction if root == correction_root else mismatched_repair
        ),
    )
    with pytest.raises(ValueError, match="parent"):
        build_m6_gate_replay_mixture(
            correction_root=correction_root,
            repair_root=repair_root,
            output_root=tmp_path / "output",
            build_seed=1,
            expected_correction_manifest_sha256=correction_sha256,
            expected_repair_manifest_sha256=repair_sha256,
        )
