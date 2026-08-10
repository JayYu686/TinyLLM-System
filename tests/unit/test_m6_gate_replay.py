from __future__ import annotations

from pathlib import Path

from tinyllm.data import M6GateReplayMixtureManifest
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


def test_gate_replay_training_configs_bind_the_private_manifest() -> None:
    seed42 = load_m5_sft_config(Path("configs/sft/m6_gate_replay_r3_seed42.yaml"))
    stability = load_m5_sft_config(Path("configs/sft/m6_gate_replay_r3_seed20260812.yaml"))

    assert seed42.data.dataset_version == "m6-gate-replay-mixture-v1-6c169970"
    assert seed42.data.mix_manifest_sha256 == stability.data.mix_manifest_sha256
    assert seed42.run.seed == 42
    assert stability.run.seed == 20260812
