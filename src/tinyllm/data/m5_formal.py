"""Build and reopen the immutable M5.3 50M-token dual-mode dataset."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from pydantic import ValidationError
from torch import Tensor
from torch.utils.data import Dataset

from tinyllm.data.m5_formal_schema import (
    M5FormalArtifactFile,
    M5FormalDatasetManifest,
)
from tinyllm.data.m5_mixture import open_m5_ablation_mixture
from tinyllm.data.m5_mixture_schema import M5FormatRepairMixtureManifest
from tinyllm.data.reasoning_schema import content_sha256

_SEQUENCE_FILE = "sequences.npz"
_PLAN_FILE = "epoch_plan.npy"
_MANIFEST_FILE = "manifest.json"
_COMMIT_FILE = "COMMITTED"


class M5FormalDatasetError(ValueError):
    """Raised when formal M5 data identity or content fails closed."""


@dataclass(frozen=True, slots=True)
class OpenM5FormalDataset:
    """Validated formal M5 manifest and private root."""

    root: Path
    manifest: M5FormalDatasetManifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_repeated_epoch_plan(
    source_sequence_count: int,
    *,
    repeated_epochs: int,
    build_seed: int,
) -> np.ndarray:
    """Return deterministic independent permutations for every formal epoch."""

    if source_sequence_count <= 0 or repeated_epochs <= 0:
        raise M5FormalDatasetError("formal epoch plan requires positive dimensions")
    epochs: list[list[int]] = []
    for epoch in range(repeated_epochs):
        order = list(range(source_sequence_count))
        random.Random((build_seed + epoch) % (2**32)).shuffle(order)
        epochs.append(order)
    return np.asarray(epochs, dtype="<i4")


def _plan_content_sha256(plan: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(plan.shape).encode())
    digest.update(str(plan.dtype).encode())
    digest.update(plan.tobytes(order="C"))
    return digest.hexdigest()


def build_m5_formal_dataset(
    *,
    source_root: Path,
    authorization_gate_path: Path,
    output_root: Path,
    build_seed: int,
    git_commit: str,
) -> M5FormalDatasetManifest:
    """Atomically freeze a 50-epoch view over the selected exact-1M R1 mixture."""

    from tinyllm.evaluation.m5_thinking_budget_schema import (
        M5ThinkingBudgetGateResult,
    )

    source = open_m5_ablation_mixture(source_root)
    if not isinstance(source.manifest, M5FormatRepairMixtureManifest):
        raise M5FormalDatasetError("formal M5 source must be the selected R1 mixture")
    source_manifest_bytes = (source_root / _MANIFEST_FILE).read_bytes()
    source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
    try:
        gate_bytes = authorization_gate_path.read_bytes()
        gate = M5ThinkingBudgetGateResult.model_validate_json(gate_bytes)
    except (OSError, ValidationError) as exc:
        raise M5FormalDatasetError("formal M5 authorization gate is invalid") from exc
    if (
        gate.status != "passed"
        or not gate.m5_3_authorized
        or gate.mixture_version != source.manifest.mixture_version
        or gate.mixture_manifest_sha256 != source_manifest_sha256
        or gate.selected_thinking_fraction_basis_points != 3000
    ):
        raise M5FormalDatasetError("formal M5 source is not authorized by protocol v2")
    plan = build_repeated_epoch_plan(
        source.manifest.sequence_count,
        repeated_epochs=50,
        build_seed=build_seed,
    )
    plan_sha256 = _plan_content_sha256(plan)
    identity = {
        "authorization_gate_sha256": hashlib.sha256(gate_bytes).hexdigest(),
        "build_git_commit": git_commit,
        "build_seed": build_seed,
        "plan_sha256": plan_sha256,
        "repeated_epochs": 50,
        "source_content_sha256": source.manifest.content_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "target_supervised_tokens": 50_000_000,
    }
    identity_sha256 = content_sha256(identity)
    version = f"m5-dual-sft-v1-{identity_sha256[:8]}"
    destination = output_root / version
    if destination.exists():
        return open_m5_formal_dataset(destination).manifest
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{version}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        sequence_path = temporary / _SEQUENCE_FILE
        shutil.copyfile(source_root / source.manifest.artifact.path, sequence_path)
        with (temporary / _PLAN_FILE).open("wb") as handle:
            np.save(handle, plan, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        plan_path = temporary / _PLAN_FILE
        manifest = M5FormalDatasetManifest(
            dataset_version=version,
            source_mixture_version=source.manifest.mixture_version,
            source_manifest_sha256=source_manifest_sha256,
            source_content_sha256=source.manifest.content_sha256,
            authorization_gate_sha256=hashlib.sha256(gate_bytes).hexdigest(),
            build_git_commit=git_commit,
            tokenizer_revision=source.manifest.tokenizer_revision,
            nonthinking_template_id=source.manifest.nonthinking_template_id,
            thinking_template_id=source.manifest.thinking_template_id,
            sequence_length=1024,
            pad_token_id=151643,
            thinking_fraction_basis_points=3000,
            source_supervised_tokens=1_000_000,
            source_nonthinking_tokens=700_000,
            source_thinking_tokens=300_000,
            repeated_epochs=50,
            target_supervised_tokens=50_000_000,
            nonthinking_supervised_tokens=35_000_000,
            thinking_supervised_tokens=15_000_000,
            source_sequence_count=source.manifest.sequence_count,
            sequence_count=source.manifest.sequence_count * 50,
            build_seed=build_seed,
            plan_sha256=plan_sha256,
            content_sha256=identity_sha256,
            artifacts=(
                M5FormalArtifactFile(
                    path="sequences.npz",
                    role="source_sequences",
                    size_bytes=sequence_path.stat().st_size,
                    sha256=_sha256_file(sequence_path),
                ),
                M5FormalArtifactFile(
                    path="epoch_plan.npy",
                    role="epoch_plan",
                    size_bytes=plan_path.stat().st_size,
                    sha256=_sha256_file(plan_path),
                ),
            ),
        )
        manifest_bytes = (
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        (temporary / _MANIFEST_FILE).write_bytes(manifest_bytes)
        (temporary / _COMMIT_FILE).write_text(
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
    return open_m5_formal_dataset(destination).manifest


def open_m5_formal_dataset(root: Path) -> OpenM5FormalDataset:
    """Validate commit marker, payload hashes, plan permutations, and exact 50M view."""

    try:
        manifest_bytes = (root / _MANIFEST_FILE).read_bytes()
        manifest = M5FormalDatasetManifest.model_validate_json(manifest_bytes)
        marker = cast(
            dict[str, str],
            json.loads((root / _COMMIT_FILE).read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise M5FormalDatasetError("formal M5 metadata is missing or invalid") from exc
    if marker != {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}:
        raise M5FormalDatasetError("formal M5 commit marker differs from manifest")
    expected_content = content_sha256(
        {
            "authorization_gate_sha256": manifest.authorization_gate_sha256,
            "build_git_commit": manifest.build_git_commit,
            "build_seed": manifest.build_seed,
            "plan_sha256": manifest.plan_sha256,
            "repeated_epochs": manifest.repeated_epochs,
            "source_content_sha256": manifest.source_content_sha256,
            "source_manifest_sha256": manifest.source_manifest_sha256,
            "target_supervised_tokens": manifest.target_supervised_tokens,
        }
    )
    if expected_content != manifest.content_sha256:
        raise M5FormalDatasetError("formal M5 content identity differs from manifest")
    for artifact in manifest.artifacts:
        path = root / artifact.path
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != artifact.size_bytes
            or _sha256_file(path) != artifact.sha256
        ):
            raise M5FormalDatasetError("formal M5 artifact failed size or SHA256 validation")
    try:
        with np.load(root / _SEQUENCE_FILE, allow_pickle=False) as arrays:
            if set(arrays.files) != {"input_ids", "labels", "attention_masks", "modes"}:
                raise M5FormalDatasetError("formal M5 source arrays are incomplete")
            input_ids = arrays["input_ids"]
            labels = arrays["labels"]
            attention_masks = arrays["attention_masks"]
            modes = arrays["modes"]
            if (
                input_ids.shape != (manifest.source_sequence_count, 1024)
                or labels.shape != input_ids.shape
                or attention_masks.shape != input_ids.shape
                or modes.shape != (manifest.source_sequence_count,)
            ):
                raise M5FormalDatasetError("formal M5 source array shapes differ")
            valid = labels[:, 1:] != -100
            if (
                int(valid.sum()) != 1_000_000
                or int(valid[modes == 0].sum()) != 700_000
                or int(valid[modes != 0].sum()) != 300_000
            ):
                raise M5FormalDatasetError("formal M5 source token counts differ")
        plan = np.load(root / _PLAN_FILE, allow_pickle=False)
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, M5FormalDatasetError):
            raise
        raise M5FormalDatasetError("formal M5 payload cannot be decoded") from exc
    if (
        plan.shape != (50, manifest.source_sequence_count)
        or plan.dtype != np.dtype("<i4")
        or _plan_content_sha256(plan) != manifest.plan_sha256
    ):
        raise M5FormalDatasetError("formal M5 epoch plan identity differs")
    expected = np.arange(manifest.source_sequence_count, dtype=np.int32)
    if any(not np.array_equal(np.sort(epoch), expected) for epoch in plan):
        raise M5FormalDatasetError("formal M5 epoch plan is not a set of permutations")
    return OpenM5FormalDataset(root=root, manifest=manifest)


class M5FormalDataset(Dataset[dict[str, Tensor]]):
    """Torch Dataset over the committed 50-epoch dual-mode view."""

    def __init__(self, root: Path) -> None:
        opened = open_m5_formal_dataset(root)
        self.manifest = opened.manifest
        with np.load(root / _SEQUENCE_FILE, allow_pickle=False) as arrays:
            self._input_ids = arrays["input_ids"].copy()
            self._labels = arrays["labels"].copy()
            self._attention_masks = arrays["attention_masks"].copy()
        self._plan = np.load(root / _PLAN_FILE, allow_pickle=False).reshape(-1)

    def __len__(self) -> int:
        return self.manifest.sequence_count

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        source = int(self._plan[index])
        return {
            "input_ids": torch.from_numpy(self._input_ids[source].astype(np.int64, copy=False)),
            "labels": torch.from_numpy(self._labels[source].astype(np.int64, copy=False)),
            "attention_mask": torch.from_numpy(
                self._attention_masks[source].astype(np.int64, copy=False)
            ),
        }
