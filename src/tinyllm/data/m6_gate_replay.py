"""Build an exact-token replay mixture after the M6 R2 forgetting diagnosis."""

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
    M6GateRepairMixtureManifest,
    M6GateReplayMixtureManifest,
)
from tinyllm.data.reasoning_schema import content_sha256


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


def _source_sequences(root: Path) -> tuple[M5MixtureSequence, ...]:
    """Reopen one verified source and materialize its immutable arrays."""

    opened = open_m5_ablation_mixture(root)
    with np.load(root / opened.manifest.artifact.path, allow_pickle=False) as arrays:
        return tuple(
            M5MixtureSequence(
                input_ids=tuple(int(value) for value in arrays["input_ids"][index]),
                labels=tuple(int(value) for value in arrays["labels"][index]),
                attention_mask=tuple(int(value) for value in arrays["attention_masks"][index]),
                mode=int(arrays["modes"][index]),
            )
            for index in range(opened.manifest.sequence_count)
        )


def build_m6_gate_replay_mixture(
    *,
    correction_root: Path,
    repair_root: Path,
    output_root: Path,
    build_seed: int,
) -> M6GateReplayMixtureManifest:
    """Build the preregistered 55% correction / 45% repair replay mixture."""

    correction = open_m5_ablation_mixture(correction_root)
    repair = open_m5_ablation_mixture(repair_root)
    if not isinstance(correction.manifest, M5DualModeCorrectionMixtureManifest):
        raise M5MixtureError("M6 replay correction source has the wrong manifest kind")
    if not isinstance(repair.manifest, M6GateRepairMixtureManifest):
        raise M5MixtureError("M6 replay repair source has the wrong manifest kind")
    correction_manifest_sha256 = hashlib.sha256(
        (correction_root / "manifest.json").read_bytes()
    ).hexdigest()
    repair_manifest_sha256 = hashlib.sha256(
        (repair_root / "manifest.json").read_bytes()
    ).hexdigest()
    if (
        correction.manifest.mixture_version != "m5-dual-mode-correction-mixture-v1-4bc342d4"
        or correction_manifest_sha256
        != "db66ce847fac4bd2966666d125f1bb4e21dd0fd3bb608a1a384806c206f8945c"
        or repair.manifest.mixture_version != "m6-gate-repair-mixture-v1-be2aa7fa"
        or repair_manifest_sha256
        != "13826d120bdbfc3db38ba035f243ddd4e9e85e8f49aec25e8e7ff20f451c7fc1"
    ):
        raise M5MixtureError("M6 replay source identity differs from preregistration")
    if correction.manifest.parent_content_sha256 != repair.manifest.parent_content_sha256:
        raise M5MixtureError("M6 replay sources have different parent datasets")

    correction_sequences = _source_sequences(correction_root)
    repair_sequences = _source_sequences(repair_root)
    correction_non_raw = tuple(item for item in correction_sequences if item.mode == 0)
    correction_think_raw = tuple(item for item in correction_sequences if item.mode == 1)
    repair_non_raw = tuple(item for item in repair_sequences if item.mode == 0)
    repair_think_raw = tuple(item for item in repair_sequences if item.mode == 1)
    strata = (
        select_exact_supervised_tokens(correction_non_raw, target=400_000, seed=build_seed),
        select_exact_supervised_tokens(
            repair_non_raw, target=300_000, seed=(build_seed + 1) % (2**32)
        ),
        select_exact_supervised_tokens(
            correction_think_raw, target=150_000, seed=(build_seed + 2) % (2**32)
        ),
        select_exact_supervised_tokens(
            repair_think_raw, target=150_000, seed=(build_seed + 3) % (2**32)
        ),
    )
    combined = [item for selected, _, _ in strata for item in selected]
    random.Random((build_seed + 4) % (2**32)).shuffle(combined)
    input_ids = np.asarray([item.input_ids for item in combined], dtype="<i4")
    labels = np.asarray([item.labels for item in combined], dtype="<i4")
    attention_masks = np.asarray([item.attention_mask for item in combined], dtype="u1")
    modes = np.asarray([item.mode for item in combined], dtype="u1")
    arrays_sha256 = _array_content_hash(input_ids, labels, attention_masks, modes)
    identity = {
        "arrays_sha256": arrays_sha256,
        "build_seed": build_seed,
        "correction_manifest_sha256": correction_manifest_sha256,
        "correction_nonthinking_supervised_tokens": 400_000,
        "correction_thinking_supervised_tokens": 150_000,
        "diagnostic_protocol_version": "m6-release-v2",
        "parent_content_sha256": correction.manifest.parent_content_sha256,
        "repair_manifest_sha256": repair_manifest_sha256,
        "repair_nonthinking_supervised_tokens": 300_000,
        "repair_thinking_supervised_tokens": 150_000,
    }
    identity_sha256 = content_sha256(identity)
    version = f"m6-gate-replay-mixture-v1-{identity_sha256[:8]}"
    destination = output_root / version
    if destination.exists():
        reopened = open_m5_ablation_mixture(destination)
        if not isinstance(reopened.manifest, M6GateReplayMixtureManifest):
            raise M5MixtureError("existing M6 replay destination has the wrong kind")
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
        non_count = sum(item.mode == 0 for item in combined)
        reuse = tuple(value[1] for value in strata)
        partial = tuple(value[2] for value in strata)
        manifest = M6GateReplayMixtureManifest(
            mixture_version=version,
            parent_dataset_version="m2-sft-v1-f82ff32e",
            parent_content_sha256=correction.manifest.parent_content_sha256,
            diagnostic_protocol_version="m6-release-v2",
            source_consumed_evaluation_content=False,
            evaluation_prompt_overlap_count=0,
            correction_mixture_version="m5-dual-mode-correction-mixture-v1-4bc342d4",
            correction_manifest_sha256=(
                "db66ce847fac4bd2966666d125f1bb4e21dd0fd3bb608a1a384806c206f8945c"
            ),
            repair_mixture_version="m6-gate-repair-mixture-v1-be2aa7fa",
            repair_manifest_sha256=(
                "13826d120bdbfc3db38ba035f243ddd4e9e85e8f49aec25e8e7ff20f451c7fc1"
            ),
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
            correction_nonthinking_supervised_tokens=400_000,
            repair_nonthinking_supervised_tokens=300_000,
            correction_thinking_supervised_tokens=150_000,
            repair_thinking_supervised_tokens=150_000,
            sequence_count=len(combined),
            nonthinking_sequence_count=non_count,
            thinking_sequence_count=len(combined) - non_count,
            correction_nonthinking_source_sequences=len(correction_non_raw),
            repair_nonthinking_source_sequences=len(repair_non_raw),
            correction_thinking_source_sequences=len(correction_think_raw),
            repair_thinking_source_sequences=len(repair_think_raw),
            correction_nonthinking_reuse_count=reuse[0],
            repair_nonthinking_reuse_count=reuse[1],
            correction_thinking_reuse_count=reuse[2],
            repair_thinking_reuse_count=reuse[3],
            partially_masked_sequences=sum(partial),
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
    if not isinstance(reopened.manifest, M6GateReplayMixtureManifest):
        raise M5MixtureError("committed M6 replay destination has the wrong manifest kind")
    return reopened.manifest
