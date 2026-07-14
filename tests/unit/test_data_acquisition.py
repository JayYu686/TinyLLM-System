from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from tinyllm.data import (
    COMMITPACKFT_PYTHON_JSONL_ARTIFACT,
    M2_ACQUISITION_MANIFEST,
    OASST1_JSONL_ARTIFACT,
    QWEN3_TOKENIZER_ARTIFACT,
    DataAcquisitionError,
    PinnedDataArtifact,
    acquire_pinned_artifact,
    iter_jsonl_records,
    verify_pinned_artifact,
)


def artifact(payload: bytes) -> PinnedDataArtifact:
    return PinnedDataArtifact(
        name="qwen3-tokenizer",
        url="https://huggingface.co/example/resolve/commit/tokenizer.json",
        cache_path=PurePosixPath("models/example/tokenizer.json"),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        compression="none",
    )


def test_fixed_artifact_identities_are_complete_and_pinned() -> None:
    assert [item.name for item in M2_ACQUISITION_MANIFEST.artifacts] == [
        "commitpackft-python-jsonl",
        "oasst1-jsonl",
        "qwen3-tokenizer",
        "qwen3-tokenizer-config",
    ]
    assert OASST1_JSONL_ARTIFACT.size_bytes == 34_196_309
    assert OASST1_JSONL_ARTIFACT.sha256 == (
        "286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c"
    )
    assert COMMITPACKFT_PYTHON_JSONL_ARTIFACT.size_bytes == 135_858_935
    assert QWEN3_TOKENIZER_ARTIFACT.size_bytes == 11_422_654


def test_acquisition_downloads_to_temporary_then_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified-pinned-content"
    spec = artifact(payload)
    monkeypatch.setattr(
        "tinyllm.data.acquisition.urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )

    path = acquire_pinned_artifact(spec, cache_root=tmp_path, offline=False)

    assert path.read_bytes() == payload
    assert not list(path.parent.glob("*.partial-*"))
    assert acquire_pinned_artifact(spec, cache_root=tmp_path, offline=True) == path


def test_acquisition_refuses_offline_miss_corruption_symlink_and_relative_root(
    tmp_path: Path,
) -> None:
    payload = b"expected"
    spec = artifact(payload)
    with pytest.raises(DataAcquisitionError, match="offline cache miss"):
        acquire_pinned_artifact(spec, cache_root=tmp_path, offline=True)
    with pytest.raises(DataAcquisitionError, match="absolute"):
        acquire_pinned_artifact(spec, cache_root=Path("relative"), offline=True)

    destination = tmp_path.joinpath(*spec.cache_path.parts)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"wrong")
    with pytest.raises(DataAcquisitionError, match="size mismatch"):
        verify_pinned_artifact(destination, spec)
    destination.unlink()
    destination.symlink_to(tmp_path / "missing-target")
    with pytest.raises(DataAcquisitionError, match="missing or not a regular"):
        acquire_pinned_artifact(spec, cache_root=tmp_path, offline=True)


def test_acquisition_removes_failed_partial_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = artifact(b"expected")
    monkeypatch.setattr(
        "tinyllm.data.acquisition.urllib.request.urlopen",
        lambda *_args, **_kwargs: io.BytesIO(b"bad"),
    )

    with pytest.raises(DataAcquisitionError, match="size mismatch"):
        acquire_pinned_artifact(spec, cache_root=tmp_path, offline=False)

    destination = tmp_path.joinpath(*spec.cache_path.parts)
    assert not destination.exists()
    assert not list(destination.parent.glob("*.partial-*"))


def test_jsonl_reader_supports_plain_and_gzip_without_leaking_bad_text(tmp_path: Path) -> None:
    rows = [{"id": 1}, {"id": 2, "nested": {"ok": True}}]
    plain = tmp_path / "data.jsonl"
    plain.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    compressed = tmp_path / "data.jsonl.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    assert list(iter_jsonl_records(plain, compression="none")) == rows
    assert list(iter_jsonl_records(compressed, compression="gzip")) == rows

    private_text = "private-invalid-json"
    plain.write_text(private_text, encoding="utf-8")
    with pytest.raises(DataAcquisitionError, match="line 1") as error:
        list(iter_jsonl_records(plain, compression="none"))
    assert private_text not in str(error.value)

    plain.write_text("[]\n", encoding="utf-8")
    with pytest.raises(DataAcquisitionError, match="must be an object"):
        list(iter_jsonl_records(plain, compression="none"))


def test_artifact_schema_rejects_cache_traversal() -> None:
    raw = artifact(b"value").model_dump()
    raw["cache_path"] = PurePosixPath("../escape")
    with pytest.raises(ValidationError, match="safe and relative"):
        PinnedDataArtifact.model_validate(raw)
