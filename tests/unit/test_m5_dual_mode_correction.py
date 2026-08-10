from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tinyllm.data.m5_dual_mode_correction as correction
from tinyllm.data import (
    M5DualModeCorrectionMixtureManifest,
    M5MixtureError,
    M5MixtureSequence,
    M5R3MixtureManifest,
    align_legacy_nonthinking_sequence_v2,
    pair_thinking_sequence_as_nonthinking_v2,
)
from tinyllm.data.m5_mixture import open_m5_ablation_mixture

PAD = 151643
IM_START = 151644
IM_END = 151645
THINK_START = 151667
THINK_END = 151668
ASSISTANT = 77091
NEWLINE = 198
DOUBLE_NEWLINE = 271


def padded(ids: list[int], labels: list[int], *, mode: int) -> M5MixtureSequence:
    padding = 1024 - len(ids)
    return M5MixtureSequence(
        input_ids=tuple(ids) + (PAD,) * padding,
        labels=tuple(labels) + (-100,) * padding,
        attention_mask=(1,) * len(ids) + (0,) * padding,
        mode=mode,
    )


def active(sequence: M5MixtureSequence) -> tuple[tuple[int, ...], tuple[int, ...]]:
    count = sum(sequence.attention_mask)
    return sequence.input_ids[:count], sequence.labels[:count]


def test_legacy_nonthinking_alignment_inserts_masked_hard_switch_context() -> None:
    source_ids = [42, IM_START, ASSISTANT, NEWLINE, 77, IM_END, NEWLINE]
    source_labels = [-100, -100, -100, -100, 77, IM_END, -100]
    source = padded(source_ids, source_labels, mode=0)

    corrected = align_legacy_nonthinking_sequence_v2(source)
    ids, labels = active(corrected)

    assert ids == (
        42,
        IM_START,
        ASSISTANT,
        NEWLINE,
        THINK_START,
        DOUBLE_NEWLINE,
        THINK_END,
        DOUBLE_NEWLINE,
        77,
        IM_END,
        NEWLINE,
    )
    assert labels == (-100,) * 8 + (77, IM_END, -100)
    assert corrected.supervised_tokens == source.supervised_tokens


def test_thinking_source_becomes_paired_nonthinking_final_answer() -> None:
    source_ids = [
        42,
        IM_START,
        ASSISTANT,
        NEWLINE,
        THINK_START,
        NEWLINE,
        55,
        NEWLINE,
        THINK_END,
        DOUBLE_NEWLINE,
        77,
        IM_END,
        NEWLINE,
    ]
    source_labels = [-100] * 4 + source_ids[4:12] + [-100]
    source = padded(source_ids, source_labels, mode=1)

    corrected = pair_thinking_sequence_as_nonthinking_v2(source)
    ids, labels = active(corrected)

    assert ids == (
        42,
        IM_START,
        ASSISTANT,
        NEWLINE,
        THINK_START,
        DOUBLE_NEWLINE,
        THINK_END,
        DOUBLE_NEWLINE,
        77,
        IM_END,
        NEWLINE,
    )
    assert labels == (-100,) * 8 + (77, IM_END, -100)
    assert corrected.mode == 0


def test_correction_rejects_ambiguous_or_partial_sources() -> None:
    bad_header = padded([42, 77, IM_END], [-100, 77, IM_END], mode=0)
    partial_thinking = padded(
        [IM_START, ASSISTANT, NEWLINE, THINK_START, NEWLINE, 55],
        [-100, -100, -100, THINK_START, NEWLINE, 55],
        mode=1,
    )

    with pytest.raises(M5MixtureError, match="Assistant header"):
        align_legacy_nonthinking_sequence_v2(bad_header)
    with pytest.raises(M5MixtureError, match="complete Think block"):
        pair_thinking_sequence_as_nonthinking_v2(partial_thinking)


def test_correction_rejects_invalid_padding_missing_labels_and_bad_chatml() -> None:
    no_supervision = padded(
        [IM_START, ASSISTANT, NEWLINE, 77],
        [-100, -100, -100, -100],
        mode=0,
    )
    valid = padded([42, 43], [-100, -100], mode=0)
    invalid_padding = M5MixtureSequence(
        input_ids=valid.input_ids,
        labels=valid.labels,
        attention_mask=(1, 0, 1) + (0,) * 1021,
        mode=0,
    )
    bad_separator = thinking_sequence(500)
    bad_separator_ids = list(bad_separator.input_ids)
    bad_separator_ids[9] = NEWLINE
    bad_separator = M5MixtureSequence(
        input_ids=tuple(bad_separator_ids),
        labels=bad_separator.labels,
        attention_mask=bad_separator.attention_mask,
        mode=1,
    )
    partial_final = thinking_sequence(600)
    partial_labels = list(partial_final.labels)
    partial_labels[11] = -100
    partial_final = M5MixtureSequence(
        input_ids=partial_final.input_ids,
        labels=tuple(partial_labels),
        attention_mask=partial_final.attention_mask,
        mode=1,
    )

    with pytest.raises(M5MixtureError, match="no Assistant supervision"):
        align_legacy_nonthinking_sequence_v2(no_supervision)
    with pytest.raises(M5MixtureError, match="invalid padding"):
        align_legacy_nonthinking_sequence_v2(invalid_padding)
    with pytest.raises(M5MixtureError, match="does not match Qwen3"):
        pair_thinking_sequence_as_nonthinking_v2(bad_separator)
    with pytest.raises(M5MixtureError, match="partial final answer"):
        pair_thinking_sequence_as_nonthinking_v2(partial_final)


def dense_sequence(*, mode: int, marker: int) -> M5MixtureSequence:
    ids = [marker] * 1001
    labels = [-100] + [marker] * 999 + [IM_END]
    return padded(ids, labels, mode=mode)


def thinking_sequence(marker: int) -> M5MixtureSequence:
    ids = [
        marker,
        IM_START,
        ASSISTANT,
        NEWLINE,
        THINK_START,
        NEWLINE,
        marker + 1,
        NEWLINE,
        THINK_END,
        DOUBLE_NEWLINE,
        marker + 2,
        IM_END,
        NEWLINE,
    ]
    return padded(ids, [-100] * 4 + ids[4:12] + [-100], mode=1)


def test_general_sources_align_only_train_packs(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = (42, IM_START, ASSISTANT, NEWLINE, 77, IM_END, NEWLINE)
    labels = (-100, -100, -100, -100, 77, IM_END, -100)
    registered = SimpleNamespace(
        iter_packs=lambda: iter(
            (
                SimpleNamespace(split="validation"),
                SimpleNamespace(
                    split="train",
                    sample_token_counts=(len(ids),),
                    input_ids=ids,
                    labels=labels,
                ),
            )
        )
    )
    monkeypatch.setattr(correction, "open_registered_dataset", lambda **_kwargs: registered)

    sources = correction._general_nonthinking_sources(artifact_root=Path("/unused"))

    assert len(sources) == 1
    aligned_ids, aligned_labels = active(sources[0])
    assert aligned_ids[4:8] == (THINK_START, DOUBLE_NEWLINE, THINK_END, DOUBLE_NEWLINE)
    assert aligned_labels[4:8] == (-100,) * 4


def test_domain_sources_are_deduplicated_and_paired_before_m6(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_payload = json.loads(Path("reports/m5/raw/m5_r3_mixture.json").read_text())
    source_manifest = M5R3MixtureManifest.model_validate(source_payload)
    rows = [padded([42, 43], [-100, -100], mode=0)] + [
        thinking_sequence(1000 + index * 4) for index in range(200)
    ]
    with (tmp_path / "sequences.npz").open("wb") as handle:
        np.savez(
            handle,
            input_ids=np.asarray([row.input_ids for row in rows], dtype="<i4"),
            labels=np.asarray([row.labels for row in rows], dtype="<i4"),
            attention_masks=np.asarray([row.attention_mask for row in rows], dtype="u1"),
            modes=np.asarray([row.mode for row in rows], dtype="u1"),
        )
    monkeypatch.setattr(
        correction,
        "open_m5_ablation_mixture",
        lambda _path: SimpleNamespace(root=tmp_path, manifest=source_manifest),
    )

    nonthinking, thinking = correction._domain_source_pairs(source_root=tmp_path)

    assert len(nonthinking) == len(thinking) == 200
    assert all(item.mode == 0 for item in nonthinking)
    assert all(item.mode == 1 for item in thinking)


def test_build_correction_mixture_is_atomic_content_addressed_and_reopenable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_payload = json.loads(Path("reports/m5/raw/m5_r3_mixture.json").read_text())
    source_manifest = M5R3MixtureManifest.model_validate(source_payload)
    (source_root / "manifest.json").write_text(
        json.dumps(source_payload, sort_keys=True),
        encoding="utf-8",
    )
    real_open = open_m5_ablation_mixture

    def fake_open(path: Path) -> object:
        if path == source_root:
            return SimpleNamespace(root=path, manifest=source_manifest)
        return real_open(path)

    parent = SimpleNamespace(
        manifest=SimpleNamespace(
            content_sha256="f82ff32ee98cb852fe6779774d9cce75a71e9430da72a6e5e1f4e3f7c2efd108"
        )
    )
    general = (dense_sequence(mode=0, marker=101),)
    domain_non = tuple(dense_sequence(mode=0, marker=110 + index) for index in range(3))
    domain_think = tuple(dense_sequence(mode=1, marker=120 + index) for index in range(3))
    monkeypatch.setattr(correction, "open_m5_ablation_mixture", fake_open)
    monkeypatch.setattr(correction, "open_registered_dataset", lambda **_kwargs: parent)
    monkeypatch.setattr(correction, "_general_nonthinking_sources", lambda **_kwargs: general)
    monkeypatch.setattr(
        correction,
        "_domain_source_pairs",
        lambda **_kwargs: (domain_non, domain_think),
    )

    manifest = correction.build_m5_dual_mode_correction_mixture(
        artifact_root=tmp_path,
        source_r3_root=source_root,
        output_root=tmp_path / "output",
        build_seed=20260810,
    )
    repeated = correction.build_m5_dual_mode_correction_mixture(
        artifact_root=tmp_path,
        source_r3_root=source_root,
        output_root=tmp_path / "output",
        build_seed=20260810,
    )

    assert isinstance(manifest, M5DualModeCorrectionMixtureManifest)
    assert repeated == manifest
    assert manifest.target_supervised_tokens == 1_000_000
    assert manifest.source_consumed_m6_results is False
    assert manifest.domain_source_pairs == 3
    assert manifest.partially_masked_sequences == 0
