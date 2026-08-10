"""Build a Qwen3-aligned dual-mode correction mixture after the M6 v1 rejection."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from pathlib import Path

import numpy as np

from tinyllm.data.m5_mixture import (
    M5MixtureError,
    M5MixtureSequence,
    open_m5_ablation_mixture,
    select_exact_supervised_tokens,
)
from tinyllm.data.m5_mixture_schema import (
    M5DualModeCorrectionMixtureManifest,
    M5MixtureArtifactFile,
)
from tinyllm.data.m5_r3_mixture_schema import M5R3MixtureManifest
from tinyllm.data.reasoning_schema import content_sha256
from tinyllm.data.registry import open_registered_dataset
from tinyllm.data.tokenization import (
    QWEN3_NONTHINKING_SFT_TEMPLATE_SHA256,
    QWEN3_THINKING_TEMPLATE_SHA256,
)

_SEQUENCE_LENGTH = 1024
_PAD_TOKEN_ID = 151643
_IM_START_TOKEN_ID = 151644
_IM_END_TOKEN_ID = 151645
_THINK_START_TOKEN_ID = 151667
_THINK_END_TOKEN_ID = 151668
_ASSISTANT_TOKEN_ID = 77091
_NEWLINE_TOKEN_ID = 198
_DOUBLE_NEWLINE_TOKEN_ID = 271
_ASSISTANT_HEADER_IDS = (
    _IM_START_TOKEN_ID,
    _ASSISTANT_TOKEN_ID,
    _NEWLINE_TOKEN_ID,
)
_NONTHINKING_CONTEXT_IDS = (
    _THINK_START_TOKEN_ID,
    _DOUBLE_NEWLINE_TOKEN_ID,
    _THINK_END_TOKEN_ID,
    _DOUBLE_NEWLINE_TOKEN_ID,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_content_hash(
    input_ids: np.ndarray,
    labels: np.ndarray,
    attention_masks: np.ndarray,
    modes: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, array in (
        ("input_ids", input_ids),
        ("labels", labels),
        ("attention_masks", attention_masks),
        ("modes", modes),
    ):
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _active_tokens(sequence: M5MixtureSequence) -> tuple[list[int], list[int]]:
    active = sum(sequence.attention_mask)
    if (
        not 1 < active <= _SEQUENCE_LENGTH
        or sequence.attention_mask[:active] != (1,) * active
        or sequence.attention_mask[active:] != (0,) * (_SEQUENCE_LENGTH - active)
        or any(label != -100 for label in sequence.labels[active:])
    ):
        raise M5MixtureError("correction source has invalid padding or Attention Mask")
    return list(sequence.input_ids[:active]), list(sequence.labels[:active])


def _pad(input_ids: list[int], labels: list[int], *, mode: int) -> M5MixtureSequence:
    if len(input_ids) != len(labels) or not 1 < len(input_ids) <= _SEQUENCE_LENGTH:
        raise M5MixtureError("corrected sequence exceeds the frozen sequence length")
    padding = _SEQUENCE_LENGTH - len(input_ids)
    return M5MixtureSequence(
        input_ids=tuple(input_ids) + (_PAD_TOKEN_ID,) * padding,
        labels=tuple(labels) + (-100,) * padding,
        attention_mask=(1,) * len(input_ids) + (0,) * padding,
        mode=mode,
    )


def pack_correction_sequences(
    sequences: tuple[M5MixtureSequence, ...],
    *,
    mode: int,
) -> tuple[M5MixtureSequence, ...]:
    """Greedily repack aligned samples without changing labels or sample order.

    Alignment may add four masked tokens to each legacy sample.  Repacking after
    that insertion avoids running a full 1,024-token forward pass for every
    short source sample while retaining the exact supervised-token budget.
    """

    if mode not in {0, 1} or not sequences:
        raise M5MixtureError("correction packing requires a non-empty valid mode")
    packed: list[M5MixtureSequence] = []
    current_ids: list[int] = []
    current_labels: list[int] = []
    source_supervision = 0
    for sequence in sequences:
        if sequence.mode != mode:
            raise M5MixtureError("correction packing cannot mix modes")
        input_ids, labels = _active_tokens(sequence)
        if len(current_ids) + len(input_ids) > _SEQUENCE_LENGTH:
            packed.append(_pad(current_ids, current_labels, mode=mode))
            current_ids = []
            current_labels = []
        current_ids.extend(input_ids)
        current_labels.extend(labels)
        source_supervision += sequence.supervised_tokens
    if current_ids:
        packed.append(_pad(current_ids, current_labels, mode=mode))
    if sum(sequence.supervised_tokens for sequence in packed) != source_supervision:
        raise M5MixtureError("correction packing changed the supervised-token count")
    return tuple(packed)


def align_legacy_nonthinking_sequence_v2(
    sequence: M5MixtureSequence,
) -> M5MixtureSequence:
    """Insert Qwen3's masked hard-switch context before every Assistant answer."""

    input_ids, labels = _active_tokens(sequence)
    transitions = tuple(
        index
        for index, label in enumerate(labels)
        if label != -100 and (index == 0 or labels[index - 1] == -100)
    )
    if not transitions:
        raise M5MixtureError("legacy Non-thinking sequence has no Assistant supervision")
    for index in transitions:
        header_start = index - len(_ASSISTANT_HEADER_IDS)
        if header_start < 0 or tuple(input_ids[header_start:index]) != _ASSISTANT_HEADER_IDS:
            raise M5MixtureError("legacy Non-thinking supervision lacks an Assistant header")
    corrected_ids: list[int] = []
    corrected_labels: list[int] = []
    transition_set = set(transitions)
    for index, (token_id, label) in enumerate(zip(input_ids, labels, strict=True)):
        if index in transition_set:
            corrected_ids.extend(_NONTHINKING_CONTEXT_IDS)
            corrected_labels.extend((-100,) * len(_NONTHINKING_CONTEXT_IDS))
        corrected_ids.append(token_id)
        corrected_labels.append(label)
    corrected = _pad(corrected_ids, corrected_labels, mode=0)
    if corrected.supervised_tokens != sequence.supervised_tokens:
        raise M5MixtureError("Non-thinking alignment changed the supervised-token count")
    return corrected


def pair_thinking_sequence_as_nonthinking_v2(
    sequence: M5MixtureSequence,
) -> M5MixtureSequence:
    """Reuse a pre-M6 Thinking task as an explicitly conditioned Non-thinking pair."""

    input_ids, labels = _active_tokens(sequence)
    think_starts = [index for index, value in enumerate(labels) if value == _THINK_START_TOKEN_ID]
    think_ends = [index for index, value in enumerate(labels) if value == _THINK_END_TOKEN_ID]
    if len(think_starts) != 1 or len(think_ends) != 1 or think_starts[0] >= think_ends[0]:
        raise M5MixtureError("Thinking correction source must contain one complete Think block")
    start, end = think_starts[0], think_ends[0]
    if (
        start < len(_ASSISTANT_HEADER_IDS)
        or tuple(input_ids[start - len(_ASSISTANT_HEADER_IDS) : start]) != _ASSISTANT_HEADER_IDS
        or end + 1 >= len(input_ids)
        or input_ids[end + 1] != _DOUBLE_NEWLINE_TOKEN_ID
        or labels[end + 1] != _DOUBLE_NEWLINE_TOKEN_ID
    ):
        raise M5MixtureError("Thinking correction source does not match Qwen3 ChatML")
    final_start = end + 2
    final_labels = labels[final_start:]
    if not final_labels or _IM_END_TOKEN_ID not in final_labels:
        raise M5MixtureError("Thinking correction source has a partial final answer")
    corrected_ids = input_ids[:start] + list(_NONTHINKING_CONTEXT_IDS) + input_ids[final_start:]
    corrected_labels = [-100] * (start + len(_NONTHINKING_CONTEXT_IDS)) + final_labels
    corrected = _pad(corrected_ids, corrected_labels, mode=0)
    if corrected.supervised_tokens <= 0:
        raise M5MixtureError("paired Non-thinking correction lost all supervision")
    return corrected


def _general_nonthinking_sources(*, artifact_root: Path) -> tuple[M5MixtureSequence, ...]:
    registered = open_registered_dataset(
        artifact_root=artifact_root,
        dataset_version="m2-sft-v1-f82ff32e",
    )
    sources: list[M5MixtureSequence] = []
    for pack in registered.iter_packs():
        if str(pack.split) != "train":
            continue
        cursor = 0
        for token_count in pack.sample_token_counts:
            end = cursor + token_count
            raw = _pad(
                list(pack.input_ids[cursor:end]),
                list(pack.labels[cursor:end]),
                mode=0,
            )
            cursor = end
            try:
                sources.append(align_legacy_nonthinking_sequence_v2(raw))
            except M5MixtureError as exc:
                if "exceeds the frozen sequence length" not in str(exc):
                    raise
    if not sources:
        raise M5MixtureError("no M2 sources fit the aligned Non-thinking template")
    return tuple(sources)


def general_nonthinking_correction_sources(*, artifact_root: Path) -> tuple[M5MixtureSequence, ...]:
    """Expose verified Qwen3-aligned M2 retention sources to later repairs."""

    return _general_nonthinking_sources(artifact_root=artifact_root)


def _domain_source_pairs(
    *, source_root: Path
) -> tuple[tuple[M5MixtureSequence, ...], tuple[M5MixtureSequence, ...]]:
    opened = open_m5_ablation_mixture(source_root)
    if not isinstance(opened.manifest, M5R3MixtureManifest):
        raise M5MixtureError("dual-mode correction requires a verified R3 source mixture")
    unique: dict[str, M5MixtureSequence] = {}
    with np.load(source_root / opened.manifest.artifact.path, allow_pickle=False) as arrays:
        input_ids = arrays["input_ids"]
        labels = arrays["labels"]
        attention_masks = arrays["attention_masks"]
        modes = arrays["modes"]
        for index, mode_value in enumerate(modes):
            if int(mode_value) == 0:
                continue
            sequence = M5MixtureSequence(
                input_ids=tuple(int(value) for value in input_ids[index]),
                labels=tuple(int(value) for value in labels[index]),
                attention_mask=tuple(int(value) for value in attention_masks[index]),
                mode=1,
            )
            active_ids, active_labels = _active_tokens(sequence)
            supervised = [value for value in active_labels if value != -100]
            if not supervised or supervised[-1] != _IM_END_TOKEN_ID:
                continue
            key = hashlib.sha256(
                np.asarray(active_ids, dtype="<i4").tobytes()
                + np.asarray(active_labels, dtype="<i4").tobytes()
            ).hexdigest()
            unique.setdefault(key, sequence)
    thinking = tuple(unique[key] for key in sorted(unique))
    nonthinking = tuple(pair_thinking_sequence_as_nonthinking_v2(item) for item in thinking)
    if len(thinking) < 200 or len(nonthinking) != len(thinking):
        raise M5MixtureError("dual-mode correction has insufficient paired domain sources")
    return nonthinking, thinking


def build_m5_dual_mode_correction_mixture(
    *,
    artifact_root: Path,
    source_r3_root: Path,
    output_root: Path,
    build_seed: int,
) -> M5DualModeCorrectionMixtureManifest:
    """Atomically build a 1M-token mixture without consuming any M6 result content."""

    source = open_m5_ablation_mixture(source_r3_root)
    if not isinstance(source.manifest, M5R3MixtureManifest):
        raise M5MixtureError("dual-mode correction requires the frozen R3 source")
    if (
        source.manifest.mixture_version != "m5-r3-mixture-v2-b47723e1"
        or source.manifest.consume_m6_frozen_results
    ):
        raise M5MixtureError("R3 source identity or M6 isolation differs")
    source_manifest_bytes = (source_r3_root / "manifest.json").read_bytes()
    source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
    general_sources = _general_nonthinking_sources(artifact_root=artifact_root)
    domain_nonthinking_sources, domain_thinking_sources = _domain_source_pairs(
        source_root=source_r3_root
    )
    general = pack_correction_sequences(general_sources, mode=0)
    domain_nonthinking = pack_correction_sequences(domain_nonthinking_sources, mode=0)
    domain_thinking = pack_correction_sequences(domain_thinking_sources, mode=1)
    selected_general, general_reuse, general_partial = select_exact_supervised_tokens(
        general,
        target=640_000,
        seed=build_seed,
    )
    selected_domain_non, domain_non_reuse, domain_non_partial = select_exact_supervised_tokens(
        domain_nonthinking,
        target=60_000,
        seed=(build_seed + 1) % (2**32),
    )
    selected_domain_think, domain_think_reuse, domain_think_partial = (
        select_exact_supervised_tokens(
            domain_thinking,
            target=300_000,
            seed=(build_seed + 2) % (2**32),
        )
    )
    combined = list(selected_general + selected_domain_non + selected_domain_think)
    random.Random((build_seed + 3) % (2**32)).shuffle(combined)
    input_ids = np.asarray([item.input_ids for item in combined], dtype="<i4")
    labels = np.asarray([item.labels for item in combined], dtype="<i4")
    attention_masks = np.asarray([item.attention_mask for item in combined], dtype="u1")
    modes = np.asarray([item.mode for item in combined], dtype="u1")
    arrays_sha256 = _array_content_hash(input_ids, labels, attention_masks, modes)
    parent = open_registered_dataset(
        artifact_root=artifact_root,
        dataset_version="m2-sft-v1-f82ff32e",
    )
    identity = {
        "arrays_sha256": arrays_sha256,
        "build_seed": build_seed,
        "domain_nonthinking_supervised_tokens": 60_000,
        "domain_thinking_supervised_tokens": 300_000,
        "general_nonthinking_supervised_tokens": 640_000,
        "nonthinking_template_sha256": QWEN3_NONTHINKING_SFT_TEMPLATE_SHA256,
        "parent_content_sha256": parent.manifest.content_sha256,
        "source_r3_manifest_sha256": source_manifest_sha256,
        "thinking_template_sha256": QWEN3_THINKING_TEMPLATE_SHA256,
    }
    identity_sha256 = content_sha256(identity)
    version = f"m5-dual-mode-correction-mixture-v1-{identity_sha256[:8]}"
    destination = output_root / version
    if destination.exists():
        reopened = open_m5_ablation_mixture(destination)
        if not isinstance(reopened.manifest, M5DualModeCorrectionMixtureManifest):
            raise M5MixtureError("existing correction destination has the wrong manifest kind")
        return reopened.manifest
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{version}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        sequence_path = temporary / "sequences.npz"
        with sequence_path.open("wb") as handle:
            np.savez(
                handle,
                input_ids=input_ids,
                labels=labels,
                attention_masks=attention_masks,
                modes=modes,
            )
            handle.flush()
            os.fsync(handle.fileno())
        manifest = M5DualModeCorrectionMixtureManifest(
            mixture_version=version,
            parent_dataset_version="m2-sft-v1-f82ff32e",
            parent_content_sha256=parent.manifest.content_sha256,
            source_r3_mixture_version="m5-r3-mixture-v2-b47723e1",
            source_r3_manifest_sha256=source_manifest_sha256,
            source_consumed_m6_results=False,
            tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
            nonthinking_template_id="qwen3-chatml-nonthinking-sft-v2",
            nonthinking_template_sha256=(
                "fba6724bd16200356794105a2273bbd42e777c8311ef1760059c6f0766171ca2"
            ),
            thinking_template_id="qwen3-chatml-thinking-v1",
            thinking_template_sha256=(
                "4786143dbb7adb72a922d5efdcbe6596f2d65dcdc35d7bbf1b22830b795c2af9"
            ),
            sequence_length=1024,
            pad_token_id=151643,
            target_supervised_tokens=1_000_000,
            thinking_fraction_basis_points=3000,
            nonthinking_supervised_tokens=700_000,
            thinking_supervised_tokens=300_000,
            general_nonthinking_supervised_tokens=640_000,
            domain_nonthinking_supervised_tokens=60_000,
            domain_thinking_supervised_tokens=300_000,
            sequence_count=len(combined),
            nonthinking_sequence_count=len(selected_general) + len(selected_domain_non),
            thinking_sequence_count=len(selected_domain_think),
            general_nonthinking_source_sequences=len(general_sources),
            domain_source_pairs=len(domain_thinking_sources),
            general_nonthinking_reuse_count=general_reuse,
            domain_nonthinking_reuse_count=domain_non_reuse,
            domain_thinking_reuse_count=domain_think_reuse,
            partially_masked_sequences=(
                general_partial + domain_non_partial + domain_think_partial
            ),
            build_seed=build_seed,
            content_sha256=identity_sha256,
            artifact=M5MixtureArtifactFile(
                path="sequences.npz",
                size_bytes=sequence_path.stat().st_size,
                sha256=_sha256_file(sequence_path),
            ),
        )
        manifest_bytes = (
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        (temporary / "COMMITTED").write_text(
            json.dumps(
                {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    reopened = open_m5_ablation_mixture(destination)
    if not isinstance(reopened.manifest, M5DualModeCorrectionMixtureManifest):
        raise M5MixtureError("committed correction destination has the wrong manifest kind")
    return reopened.manifest
