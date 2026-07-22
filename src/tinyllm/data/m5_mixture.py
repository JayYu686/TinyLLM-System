"""Build and verify exact-token M5.2 Non-thinking/Thinking mixtures."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from pydantic import ValidationError
from torch import Tensor
from torch.utils.data import Dataset

from tinyllm.data.m5_mixture_schema import M5MixtureArtifactFile, M5MixtureManifest
from tinyllm.data.reasoning import (
    build_reasoning_dataset,
    generate_reasoning_dev_tasks,
    load_m5_reasoning_data_config,
)
from tinyllm.data.reasoning_schema import (
    M5ReasoningDatasetManifest,
    ReasoningSample,
    ReasoningTask,
    TeacherGenerationRecord,
    content_sha256,
)
from tinyllm.data.registry import open_registered_dataset
from tinyllm.data.schema import ImportedMessage
from tinyllm.data.tokenization import (
    TokenizersBackend,
    load_m2_tokenization_config,
    tokenize_thinking_messages,
)

_SEQUENCE_FILE = "sequences.npz"
_MANIFEST_FILE = "manifest.json"
_COMMIT_FILE = "COMMITTED"


class M5MixtureError(ValueError):
    """Raised when an M5.2 mixture or its lineage fails closed validation."""


@dataclass(frozen=True, slots=True)
class M5MixtureSequence:
    """One fixed-length Assistant-only sequence and its source mode."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    mode: int

    @property
    def supervised_tokens(self) -> int:
        """Count shifted causal-LM labels that contribute to loss."""

        return sum(label != -100 for label in self.labels[1:])


@dataclass(frozen=True, slots=True)
class M5PilotInput:
    """Fully revalidated private Teacher artifact used by the mixture builder."""

    manifest: M5ReasoningDatasetManifest
    samples: tuple[ReasoningSample, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pad_sequence(
    input_ids: tuple[int, ...], labels: tuple[int, ...], *, mode: int
) -> M5MixtureSequence:
    if len(input_ids) != len(labels) or not 1 < len(input_ids) <= 1024:
        raise M5MixtureError("mixture source sequence has an invalid length")
    padding = 1024 - len(input_ids)
    return M5MixtureSequence(
        input_ids=input_ids + (151643,) * padding,
        labels=labels + (-100,) * padding,
        attention_mask=(1,) * len(input_ids) + (0,) * padding,
        mode=mode,
    )


def _trim_supervision(sequence: M5MixtureSequence, keep: int) -> M5MixtureSequence:
    """Keep the first exact number of shifted labels without truncating model input."""

    if not 0 < keep < sequence.supervised_tokens:
        raise M5MixtureError("partial supervision count must be inside the source sequence")
    labels = list(sequence.labels)
    remaining = keep
    for index in range(1, len(labels)):
        if labels[index] == -100:
            continue
        if remaining > 0:
            remaining -= 1
        else:
            labels[index] = -100
    if remaining != 0:
        raise M5MixtureError("could not retain requested partial supervision")
    return M5MixtureSequence(
        input_ids=sequence.input_ids,
        labels=tuple(labels),
        attention_mask=sequence.attention_mask,
        mode=sequence.mode,
    )


def select_exact_supervised_tokens(
    candidates: tuple[M5MixtureSequence, ...],
    *,
    target: int,
    seed: int,
) -> tuple[tuple[M5MixtureSequence, ...], int, int]:
    """Cycle deterministic shuffled epochs until an exact label-token budget is reached."""

    if target < 0 or not candidates or any(item.supervised_tokens <= 0 for item in candidates):
        raise M5MixtureError("exact-token selection requires positive source supervision")
    if target == 0:
        return (), 0, 0
    rng = random.Random(seed)
    selected: list[M5MixtureSequence] = []
    consumed = 0
    reuse_count = 0
    partial = 0
    epoch = 0
    while consumed < target:
        order = list(range(len(candidates)))
        rng.shuffle(order)
        if epoch > 0:
            reuse_count += len(order)
        for index in order:
            sequence = candidates[index]
            remaining = target - consumed
            if sequence.supervised_tokens > remaining:
                sequence = _trim_supervision(sequence, remaining)
                partial += 1
            selected.append(sequence)
            consumed += sequence.supervised_tokens
            if consumed == target:
                return tuple(selected), reuse_count, partial
        epoch += 1
    raise AssertionError("unreachable exact-token selection state")


def load_verified_reasoning_pilot(*, raw_artifact: Path, reasoning_config: Path) -> M5PilotInput:
    """Rebuild a private Teacher artifact before exposing accepted samples to training."""

    try:
        payload = cast(dict[str, object], json.loads(raw_artifact.read_text(encoding="utf-8")))
        tasks = tuple(
            ReasoningTask.model_validate(value) for value in cast(list[object], payload["tasks"])
        )
        generations = tuple(
            TeacherGenerationRecord.model_validate(value)
            for value in cast(list[object], payload["generations"])
        )
        declared = M5ReasoningDatasetManifest.model_validate(payload["manifest"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
        raise M5MixtureError("private reasoning Pilot cannot be parsed") from exc
    config = load_m5_reasoning_data_config(reasoning_config)
    rebuilt = build_reasoning_dataset(
        tasks,
        generations,
        config=config,
        dev_tasks=generate_reasoning_dev_tasks(config),
    )
    if rebuilt.manifest != declared:
        raise M5MixtureError("private reasoning Pilot manifest does not match rebuilt content")
    if not rebuilt.samples:
        raise M5MixtureError("private reasoning Pilot contains no accepted samples")
    return M5PilotInput(manifest=rebuilt.manifest, samples=rebuilt.samples)


def _nonthinking_candidates(*, artifact_root: Path) -> tuple[tuple[M5MixtureSequence, ...], str]:
    registered = open_registered_dataset(
        artifact_root=artifact_root,
        dataset_version="m2-sft-v1-f82ff32e",
    )
    sequences: list[M5MixtureSequence] = []
    for pack in registered.iter_packs():
        if str(pack.split) != "train":
            continue
        cursor = 0
        for token_count in pack.sample_token_counts:
            end = cursor + token_count
            sequence = _pad_sequence(
                pack.input_ids[cursor:end],
                pack.labels[cursor:end],
                mode=0,
            )
            cursor = end
            if sequence.supervised_tokens > 0:
                sequences.append(sequence)
    if not sequences:
        raise M5MixtureError("registered M2 Train split has no supervised sequences")
    return tuple(sequences), registered.manifest.content_sha256


def _thinking_candidates(
    *,
    pilot: M5PilotInput,
    tokenizer_config_path: Path,
    model_dir: Path,
) -> tuple[M5MixtureSequence, ...]:
    tokenization = load_m2_tokenization_config(tokenizer_config_path)
    backend = TokenizersBackend.from_files(
        model_dir / tokenization.tokenizer.tokenizer_file,
        model_dir / tokenization.tokenizer.tokenizer_config_file,
        tokenization.tokenizer,
    )
    sequences: list[M5MixtureSequence] = []
    for sample in pilot.samples:
        encoded = tokenize_thinking_messages(
            (
                ImportedMessage(role="user", content=sample.prompt),
                ImportedMessage(role="assistant", content=sample.final_answer),
            ),
            assistant_reasoning=(sample.reasoning_content,),
            backend=backend,
            tokenizer=tokenization.tokenizer,
        )
        if len(encoded.input_ids) > 1024:
            raise M5MixtureError(
                "accepted Pilot sample exceeds M5 sequence length after tokenization"
            )
        sequence = _pad_sequence(encoded.input_ids, encoded.labels, mode=1)
        if sequence.supervised_tokens <= 0:
            raise M5MixtureError("accepted Pilot sample lost all supervised labels")
        sequences.append(sequence)
    return tuple(sequences)


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


def build_m5_ablation_mixture(
    *,
    artifact_root: Path,
    raw_pilot_artifact: Path,
    reasoning_config_path: Path,
    tokenizer_config_path: Path,
    model_dir: Path,
    output_root: Path,
    thinking_fraction: float,
    build_seed: int,
) -> M5MixtureManifest:
    """Materialize one immutable exact-1M-token mixture and atomically commit it."""

    basis_points = {0.0: 0, 0.3: 3000, 0.5: 5000}.get(thinking_fraction)
    if basis_points is None:
        raise M5MixtureError("Thinking fraction must be 0.0, 0.3, or 0.5")
    pilot = load_verified_reasoning_pilot(
        raw_artifact=raw_pilot_artifact,
        reasoning_config=reasoning_config_path,
    )
    nonthinking, parent_hash = _nonthinking_candidates(artifact_root=artifact_root)
    thinking = _thinking_candidates(
        pilot=pilot,
        tokenizer_config_path=tokenizer_config_path,
        model_dir=model_dir,
    )
    thinking_target = basis_points * 100
    nonthinking_target = 1_000_000 - thinking_target
    selected_non, non_reuse, non_partial = select_exact_supervised_tokens(
        nonthinking,
        target=nonthinking_target,
        seed=build_seed,
    )
    selected_think, think_reuse, think_partial = select_exact_supervised_tokens(
        thinking,
        target=thinking_target,
        seed=(build_seed + 1) % (2**32),
    )
    combined = list(selected_non + selected_think)
    random.Random((build_seed + 2) % (2**32)).shuffle(combined)
    input_ids = np.asarray([item.input_ids for item in combined], dtype="<i4")
    labels = np.asarray([item.labels for item in combined], dtype="<i4")
    attention_masks = np.asarray([item.attention_mask for item in combined], dtype="u1")
    modes = np.asarray([item.mode for item in combined], dtype="u1")
    arrays_hash = _array_content_hash(input_ids, labels, attention_masks, modes)
    identity = {
        "arrays_sha256": arrays_hash,
        "build_seed": build_seed,
        "nonthinking_supervised_tokens": nonthinking_target,
        "parent_content_sha256": parent_hash,
        "pilot_content_sha256": pilot.manifest.content_sha256,
        "thinking_supervised_tokens": thinking_target,
    }
    identity_hash = content_sha256(identity)
    version = f"m5-ablation-mixture-v1-{identity_hash[:8]}"
    destination = output_root / version
    if destination.exists():
        return open_m5_ablation_mixture(destination).manifest
    temporary = output_root / f".{version}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        sequence_path = temporary / _SEQUENCE_FILE
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
        manifest = M5MixtureManifest(
            mixture_version=version,
            parent_dataset_version="m2-sft-v1-f82ff32e",
            parent_content_sha256=parent_hash,
            pilot_dataset_version=pilot.manifest.dataset_version,
            pilot_content_sha256=pilot.manifest.content_sha256,
            tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
            nonthinking_template_id="qwen3-chatml-nonthinking-v1",
            thinking_template_id="qwen3-chatml-thinking-v1",
            sequence_length=1024,
            pad_token_id=151643,
            target_supervised_tokens=1_000_000,
            thinking_fraction_basis_points=cast(Literal[0, 3000, 5000], basis_points),
            nonthinking_supervised_tokens=nonthinking_target,
            thinking_supervised_tokens=thinking_target,
            sequence_count=len(combined),
            nonthinking_sequence_count=len(selected_non),
            thinking_sequence_count=len(selected_think),
            nonthinking_source_sequences=len(nonthinking),
            thinking_source_sequences=len(thinking),
            nonthinking_reuse_count=non_reuse,
            thinking_reuse_count=think_reuse,
            partially_masked_sequences=non_partial + think_partial,
            build_seed=build_seed,
            content_sha256=identity_hash,
            artifact=M5MixtureArtifactFile(
                path="sequences.npz",
                size_bytes=sequence_path.stat().st_size,
                sha256=_sha256_file(sequence_path),
            ),
        )
        (temporary / _MANIFEST_FILE).write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_sha256 = hashlib.sha256((temporary / _MANIFEST_FILE).read_bytes()).hexdigest()
        (temporary / _COMMIT_FILE).write_text(
            json.dumps(
                {"manifest_sha256": manifest_sha256},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return open_m5_ablation_mixture(destination).manifest


@dataclass(frozen=True, slots=True)
class OpenM5Mixture:
    """Verified memory-mapped M5.2 mixture."""

    root: Path
    manifest: M5MixtureManifest


def open_m5_ablation_mixture(root: Path) -> OpenM5Mixture:
    """Validate commit marker, manifest, payload hash, shape, and exact token counts."""

    try:
        manifest_bytes = (root / _MANIFEST_FILE).read_bytes()
        manifest = M5MixtureManifest.model_validate_json(manifest_bytes)
        marker = cast(dict[str, str], json.loads((root / _COMMIT_FILE).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise M5MixtureError("mixture metadata is missing or invalid") from exc
    if marker != {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}:
        raise M5MixtureError("mixture commit marker does not match manifest")
    payload = root / manifest.artifact.path
    if (
        not payload.is_file()
        or payload.is_symlink()
        or payload.stat().st_size != manifest.artifact.size_bytes
        or _sha256_file(payload) != manifest.artifact.sha256
    ):
        raise M5MixtureError("mixture payload failed size or SHA256 validation")
    try:
        with np.load(payload, allow_pickle=False) as arrays:
            input_ids = arrays["input_ids"]
            labels = arrays["labels"]
            attention_masks = arrays["attention_masks"]
            modes = arrays["modes"]
            if set(arrays.files) != {"input_ids", "labels", "attention_masks", "modes"}:
                raise M5MixtureError("mixture payload contains unexpected arrays")
            if input_ids.shape != labels.shape or input_ids.shape != (
                manifest.sequence_count,
                1024,
            ):
                raise M5MixtureError("mixture sequence arrays have invalid shapes")
            if attention_masks.shape != input_ids.shape or not bool(
                np.logical_or(attention_masks == 0, attention_masks == 1).all()
            ):
                raise M5MixtureError("mixture Attention Mask array is invalid")
            if bool(np.logical_and(attention_masks == 0, labels != -100).any()):
                raise M5MixtureError("mixture padding cannot carry supervised labels")
            if modes.shape != (manifest.sequence_count,) or not set(modes.tolist()) <= {0, 1}:
                raise M5MixtureError("mixture mode array is invalid")
            valid = labels[:, 1:] != -100
            non_count = int(valid[modes == 0].sum())
            think_count = int(valid[modes == 1].sum())
            if (
                non_count != manifest.nonthinking_supervised_tokens
                or think_count != manifest.thinking_supervised_tokens
            ):
                raise M5MixtureError("mixture supervised-token counts do not match manifest")
            arrays_hash = _array_content_hash(input_ids, labels, attention_masks, modes)
            expected_content = content_sha256(
                {
                    "arrays_sha256": arrays_hash,
                    "build_seed": manifest.build_seed,
                    "nonthinking_supervised_tokens": manifest.nonthinking_supervised_tokens,
                    "parent_content_sha256": manifest.parent_content_sha256,
                    "pilot_content_sha256": manifest.pilot_content_sha256,
                    "thinking_supervised_tokens": manifest.thinking_supervised_tokens,
                }
            )
            if expected_content != manifest.content_sha256:
                raise M5MixtureError("mixture array content identity differs from manifest")
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, M5MixtureError):
            raise
        raise M5MixtureError("mixture payload cannot be decoded") from exc
    return OpenM5Mixture(root=root, manifest=manifest)


class M5AblationDataset(Dataset[dict[str, Tensor]]):
    """Torch Dataset over one already-verified private M5.2 mixture."""

    def __init__(self, root: Path) -> None:
        opened = open_m5_ablation_mixture(root)
        self.manifest = opened.manifest
        with np.load(opened.root / _SEQUENCE_FILE, allow_pickle=False) as arrays:
            self._input_ids = arrays["input_ids"].copy()
            self._labels = arrays["labels"].copy()
            self._attention_masks = arrays["attention_masks"].copy()

    def __len__(self) -> int:
        return self.manifest.sequence_count

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        input_ids = torch.from_numpy(self._input_ids[index].astype(np.int64, copy=False))
        labels = torch.from_numpy(self._labels[index].astype(np.int64, copy=False))
        attention_mask = torch.from_numpy(self._attention_masks[index].astype(np.int64, copy=False))
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
