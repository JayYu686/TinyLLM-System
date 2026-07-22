from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from tinyllm.data import (
    M5AblationDataset,
    M5MixtureError,
    M5MixtureSequence,
    build_m5_ablation_mixture,
    open_m5_ablation_mixture,
    select_exact_supervised_tokens,
)
from tinyllm.data import m5_mixture as mixture_module
from tinyllm.data.m5_mixture import M5PilotInput
from tinyllm.data.m5_mixture_schema import M5MixtureArtifactFile, M5MixtureManifest
from tinyllm.data.tokenization import TokenizersBackend


def _sequence(supervised: int, *, mode: int) -> M5MixtureSequence:
    labels = [-100] * 1024
    for index in range(1, supervised + 1):
        labels[index] = index
    return M5MixtureSequence(
        input_ids=tuple(range(1024)),
        labels=tuple(labels),
        attention_mask=(1,) * 1024,
        mode=mode,
    )


def _manifest_mapping() -> dict[str, object]:
    content_hash = hashlib.sha256(b"mixture").hexdigest()
    return {
        "schema_version": "1.0",
        "dataset_name": "m5-ablation-mixture",
        "mixture_version": f"m5-ablation-mixture-v1-{content_hash[:8]}",
        "parent_dataset_version": "m2-sft-v1-f82ff32e",
        "parent_content_sha256": "a" * 64,
        "pilot_dataset_version": "m5-reasoning-pilot-v1-a1b2c3d4",
        "pilot_content_sha256": "b" * 64,
        "tokenizer_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "nonthinking_template_id": "qwen3-chatml-nonthinking-v1",
        "thinking_template_id": "qwen3-chatml-thinking-v1",
        "sequence_length": 1024,
        "pad_token_id": 151643,
        "target_supervised_tokens": 1_000_000,
        "thinking_fraction_basis_points": 3000,
        "nonthinking_supervised_tokens": 700_000,
        "thinking_supervised_tokens": 300_000,
        "sequence_count": 20,
        "nonthinking_sequence_count": 12,
        "thinking_sequence_count": 8,
        "nonthinking_source_sequences": 4597,
        "thinking_source_sequences": 80,
        "nonthinking_reuse_count": 0,
        "thinking_reuse_count": 720,
        "partially_masked_sequences": 2,
        "build_seed": 20260725,
        "content_sha256": content_hash,
        "artifact": M5MixtureArtifactFile(
            path="sequences.npz",
            size_bytes=10,
            sha256="c" * 64,
        ).to_dict(),
    }


def test_exact_token_selector_cycles_and_masks_only_the_last_sequence() -> None:
    selected, reuse_count, partial = select_exact_supervised_tokens(
        (_sequence(7, mode=1), _sequence(5, mode=1)),
        target=20,
        seed=42,
    )

    assert sum(item.supervised_tokens for item in selected) == 20
    assert reuse_count == 2
    assert partial == 1
    assert selected[-1].supervised_tokens in {1, 3}


def test_exact_token_selector_is_deterministic() -> None:
    candidates = (_sequence(7, mode=0), _sequence(5, mode=0), _sequence(3, mode=0))

    first = select_exact_supervised_tokens(candidates, target=31, seed=7)
    second = select_exact_supervised_tokens(candidates, target=31, seed=7)

    assert first == second


def test_exact_token_selector_rejects_empty_or_invalid_budget() -> None:
    with pytest.raises(M5MixtureError, match="positive"):
        select_exact_supervised_tokens((), target=1, seed=7)
    with pytest.raises(M5MixtureError, match="positive"):
        select_exact_supervised_tokens((_sequence(3, mode=0),), target=-1, seed=7)
    assert select_exact_supervised_tokens((_sequence(3, mode=0),), target=0, seed=7) == (
        (),
        0,
        0,
    )


def test_mixture_manifest_freezes_exact_ratio_and_version() -> None:
    manifest = M5MixtureManifest.model_validate(_manifest_mapping())

    assert manifest.thinking_supervised_tokens == 300_000
    assert manifest.nonthinking_supervised_tokens == 700_000


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("thinking_supervised_tokens", 299_999, "exactly 1M"),
        ("mixture_version", "m5-ablation-mixture-v1-deadbeef", "content hash"),
        ("thinking_sequence_count", 0, "do not equal total"),
    ],
)
def test_mixture_manifest_rejects_identity_or_count_drift(
    field: str, value: object, message: str
) -> None:
    mapping = _manifest_mapping()
    mapping[field] = value

    with pytest.raises(ValidationError, match=message):
        M5MixtureManifest.model_validate(mapping)


def test_private_mixture_build_open_and_corruption_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nonthinking = (_sequence(1000, mode=0), _sequence(997, mode=0))
    thinking = (_sequence(1000, mode=1), _sequence(991, mode=1))
    pilot_manifest = SimpleNamespace(
        dataset_version="m5-reasoning-pilot-v1-a1b2c3d4",
        content_sha256="b" * 64,
    )
    pilot = SimpleNamespace(manifest=pilot_manifest, samples=(object(),))
    monkeypatch.setattr(
        mixture_module,
        "load_verified_reasoning_pilot",
        lambda **_kwargs: pilot,
    )
    monkeypatch.setattr(
        mixture_module,
        "_nonthinking_candidates",
        lambda **_kwargs: (nonthinking, "a" * 64),
    )
    monkeypatch.setattr(
        mixture_module,
        "_thinking_candidates",
        lambda **_kwargs: thinking,
    )
    output_root = tmp_path / "mixtures"

    manifest = build_m5_ablation_mixture(
        artifact_root=tmp_path,
        raw_pilot_artifact=tmp_path / "raw.json",
        reasoning_config_path=tmp_path / "reasoning.yaml",
        tokenizer_config_path=tmp_path / "tokenization.yaml",
        model_dir=tmp_path / "model",
        output_root=output_root,
        thinking_fraction=0.3,
        build_seed=20260725,
    )
    mixture_root = output_root / manifest.mixture_version
    opened = open_m5_ablation_mixture(mixture_root)
    dataset = M5AblationDataset(mixture_root)

    assert opened.manifest.thinking_supervised_tokens == 300_000
    assert opened.manifest.nonthinking_supervised_tokens == 700_000
    assert len(dataset) == manifest.sequence_count
    assert dataset[0]["input_ids"].shape == (1024,)
    assert dataset[0]["attention_mask"].sum() == 1024
    assert (
        build_m5_ablation_mixture(
            artifact_root=tmp_path,
            raw_pilot_artifact=tmp_path / "raw.json",
            reasoning_config_path=tmp_path / "reasoning.yaml",
            tokenizer_config_path=tmp_path / "tokenization.yaml",
            model_dir=tmp_path / "model",
            output_root=output_root,
            thinking_fraction=0.3,
            build_seed=20260725,
        )
        == manifest
    )

    payload = mixture_root / "sequences.npz"
    corrupted = bytearray(payload.read_bytes())
    corrupted[-1] ^= 1
    payload.write_bytes(corrupted)
    with pytest.raises(M5MixtureError, match="SHA256"):
        open_m5_ablation_mixture(mixture_root)


def test_mixture_builder_rejects_unregistered_ratio(tmp_path: Path) -> None:
    with pytest.raises(M5MixtureError, match="0.0, 0.3, or 0.5"):
        build_m5_ablation_mixture(
            artifact_root=tmp_path,
            raw_pilot_artifact=tmp_path / "raw.json",
            reasoning_config_path=tmp_path / "reasoning.yaml",
            tokenizer_config_path=tmp_path / "tokenization.yaml",
            model_dir=tmp_path / "model",
            output_root=tmp_path / "output",
            thinking_fraction=0.4,
            build_seed=1,
        )


def test_padding_and_partial_mask_helpers_fail_closed() -> None:
    padded = mixture_module._pad_sequence((1, 2), (-100, 2), mode=0)

    assert len(padded.input_ids) == 1024
    assert sum(padded.attention_mask) == 2
    assert padded.supervised_tokens == 1
    with pytest.raises(M5MixtureError, match="invalid length"):
        mixture_module._pad_sequence((1,), (-100,), mode=0)
    with pytest.raises(M5MixtureError, match="inside"):
        mixture_module._trim_supervision(_sequence(3, mode=0), 0)


def test_invalid_private_pilot_is_rejected_before_training(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text("not-json", encoding="utf-8")

    with pytest.raises(M5MixtureError, match="cannot be parsed"):
        mixture_module.load_verified_reasoning_pilot(
            raw_artifact=raw,
            reasoning_config=tmp_path / "reasoning.yaml",
        )


def test_expanded_pilot_gate_rejects_low_acceptance_or_missing_family() -> None:
    families = ("config", "json", "linux", "log_diagnosis", "python")
    samples = tuple(
        SimpleNamespace(task_family=families[index % len(families)]) for index in range(80)
    )
    manifest = SimpleNamespace(
        input_tasks=100,
        task_family_counts={family: 20 for family in families},
        language_counts={"en": 70, "zh": 30},
        accepted_samples=80,
    )
    pilot = cast(M5PilotInput, SimpleNamespace(manifest=manifest, samples=samples))

    mixture_module._validate_expanded_pilot_gate(pilot)
    low_acceptance = cast(
        M5PilotInput,
        SimpleNamespace(
            manifest=SimpleNamespace(**{**vars(manifest), "accepted_samples": 79}),
            samples=samples[:-1],
        ),
    )
    with pytest.raises(M5MixtureError, match="80%"):
        mixture_module._validate_expanded_pilot_gate(low_acceptance)
    missing_family = cast(
        M5PilotInput,
        SimpleNamespace(
            manifest=manifest,
            samples=tuple(SimpleNamespace(task_family="json") for _ in range(80)),
        ),
    )
    with pytest.raises(M5MixtureError, match="five-family"):
        mixture_module._validate_expanded_pilot_gate(missing_family)


def test_nonthinking_candidate_view_uses_only_supervised_train_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_pack = SimpleNamespace(
        split="train",
        sample_token_counts=(3, 2),
        input_ids=(1, 2, 3, 4, 5),
        labels=(-100, 2, 3, -100, -100),
    )
    test_pack = SimpleNamespace(
        split="test",
        sample_token_counts=(2,),
        input_ids=(6, 7),
        labels=(-100, 7),
    )
    registered = SimpleNamespace(
        manifest=SimpleNamespace(content_sha256="a" * 64),
        iter_packs=lambda: iter((test_pack, train_pack)),
    )
    monkeypatch.setattr(
        mixture_module,
        "open_registered_dataset",
        lambda **_kwargs: registered,
    )

    candidates, content_hash = mixture_module._nonthinking_candidates(artifact_root=tmp_path)

    assert content_hash == "a" * 64
    assert len(candidates) == 1
    assert candidates[0].supervised_tokens == 2


def test_thinking_candidate_view_tokenizes_accepted_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer_identity = SimpleNamespace(
        tokenizer_file="tokenizer.json",
        tokenizer_config_file="tokenizer_config.json",
    )
    monkeypatch.setattr(
        mixture_module,
        "load_m2_tokenization_config",
        lambda _path: SimpleNamespace(tokenizer=tokenizer_identity),
    )
    monkeypatch.setattr(
        TokenizersBackend,
        "from_files",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        mixture_module,
        "tokenize_thinking_messages",
        lambda *_args, **_kwargs: SimpleNamespace(
            input_ids=(1, 2, 3),
            labels=(-100, 2, 3),
        ),
    )
    sample = SimpleNamespace(
        prompt="prompt",
        final_answer='{"value":1}',
        reasoning_content="reason",
    )
    pilot = SimpleNamespace(samples=(sample,))

    candidates = mixture_module._thinking_candidates(
        pilot=cast(M5PilotInput, pilot),
        tokenizer_config_path=tmp_path / "tokenization.yaml",
        model_dir=tmp_path / "model",
    )

    assert len(candidates) == 1
    assert candidates[0].mode == 1
    assert candidates[0].supervised_tokens == 2
