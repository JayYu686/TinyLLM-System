from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tinyllm.data import (
    M5FormalArtifactFile,
    M5FormalDatasetError,
    M5FormalDatasetManifest,
    build_repeated_epoch_plan,
)


def _manifest() -> M5FormalDatasetManifest:
    return M5FormalDatasetManifest(
        dataset_version="m5-dual-sft-v1-aaaaaaaa",
        source_mixture_version="m5-format-repair-mixture-v1-1396b60b",
        source_manifest_sha256=("2467b5dce0d909b865b73219d2f608bdbc9c6fcc1bb09b93c5ebea8a7b60bd0e"),
        source_content_sha256="b" * 64,
        authorization_gate_sha256="c" * 64,
        build_git_commit="d" * 40,
        tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        nonthinking_template_id="qwen3-chatml-nonthinking-v1",
        thinking_template_id="qwen3-chatml-thinking-v1",
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
        source_sequence_count=4,
        sequence_count=200,
        build_seed=20260731,
        plan_sha256="e" * 64,
        content_sha256="a" * 64,
        artifacts=(
            M5FormalArtifactFile(
                path="sequences.npz",
                role="source_sequences",
                size_bytes=1,
                sha256="f" * 64,
            ),
            M5FormalArtifactFile(
                path="epoch_plan.npy",
                role="epoch_plan",
                size_bytes=1,
                sha256="0" * 64,
            ),
        ),
    )


def test_repeated_epoch_plan_is_deterministic_and_each_epoch_is_a_permutation() -> None:
    first = build_repeated_epoch_plan(7, repeated_epochs=3, build_seed=42)
    second = build_repeated_epoch_plan(7, repeated_epochs=3, build_seed=42)

    assert np.array_equal(first, second)
    assert first.shape == (3, 7)
    assert first.dtype == np.dtype("<i4")
    assert all(np.array_equal(np.sort(epoch), np.arange(7)) for epoch in first)
    assert not np.array_equal(first[0], first[1])


def test_repeated_epoch_plan_rejects_empty_dimensions() -> None:
    with pytest.raises(M5FormalDatasetError, match="positive dimensions"):
        build_repeated_epoch_plan(0, repeated_epochs=50, build_seed=42)


def test_formal_manifest_freezes_exact_fifty_million_token_ratio() -> None:
    manifest = _manifest()

    assert manifest.target_supervised_tokens == 50_000_000
    assert manifest.thinking_supervised_tokens == 15_000_000
    assert manifest.sequence_count == manifest.source_sequence_count * 50


def test_formal_manifest_rejects_sequence_and_artifact_drift() -> None:
    manifest = _manifest()
    payload = manifest.model_dump(mode="python")

    with pytest.raises(ValidationError, match="sequence count"):
        M5FormalDatasetManifest.model_validate(payload | {"sequence_count": 199})
    with pytest.raises(ValidationError, match="artifacts must be ordered"):
        M5FormalDatasetManifest.model_validate(
            payload | {"artifacts": tuple(reversed(payload["artifacts"]))}
        )
