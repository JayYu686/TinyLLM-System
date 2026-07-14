"""Verified acquisition and JSONL readers for pinned M2 artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TextIO, cast

from pydantic import Field, field_validator, model_validator

from tinyllm.schemas.base import StrictSchema

ArtifactName = Literal[
    "oasst1-jsonl",
    "commitpackft-python-jsonl",
    "qwen3-tokenizer",
    "qwen3-tokenizer-config",
]


class PinnedDataArtifact(StrictSchema):
    """Immutable remote identity and safe private-cache location for one M2 input."""

    schema_version: Literal["1.0"] = "1.0"
    name: ArtifactName
    url: str = Field(pattern=r"^https://huggingface\.co/")
    cache_path: PurePosixPath
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compression: Literal["none", "gzip"]

    @field_validator("cache_path")
    @classmethod
    def validate_cache_path(cls, value: PurePosixPath) -> PurePosixPath:
        """Keep every artifact below the caller-provided cache root."""

        if value.is_absolute() or not value.parts or ".." in value.parts:
            raise ValueError("artifact cache path must be safe and relative")
        return value


class M2AcquisitionManifest(StrictSchema):
    """Path-free fixed artifact identities persisted with every registered build."""

    schema_version: Literal["1.0"] = "1.0"
    artifacts: tuple[PinnedDataArtifact, ...]

    @model_validator(mode="after")
    def validate_artifacts(self) -> M2AcquisitionManifest:
        """Require all four fixed inputs once in stable name order."""

        names = tuple(artifact.name for artifact in self.artifacts)
        expected = (
            "commitpackft-python-jsonl",
            "oasst1-jsonl",
            "qwen3-tokenizer",
            "qwen3-tokenizer-config",
        )
        if names != expected:
            raise ValueError("M2 acquisition artifacts must contain all fixed inputs in name order")
        return self


OASST1_JSONL_ARTIFACT = PinnedDataArtifact(
    name="oasst1-jsonl",
    url=(
        "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"
        "fdf72ae0827c1cda404aff25b6603abec9e3399b/"
        "2023-04-12_oasst_ready.messages.jsonl.gz"
    ),
    cache_path=PurePosixPath(
        "datasets/OpenAssistant/oasst1/"
        "fdf72ae0827c1cda404aff25b6603abec9e3399b/"
        "2023-04-12_oasst_ready.messages.jsonl.gz"
    ),
    size_bytes=34_196_309,
    sha256="286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c",
    compression="gzip",
)

COMMITPACKFT_PYTHON_JSONL_ARTIFACT = PinnedDataArtifact(
    name="commitpackft-python-jsonl",
    url=(
        "https://huggingface.co/datasets/bigcode/commitpackft/resolve/"
        "fc56fe33c030c6daa414c2b112c932b8eed085e6/data/python/data.jsonl"
    ),
    cache_path=PurePosixPath(
        "datasets/bigcode/commitpackft/"
        "fc56fe33c030c6daa414c2b112c932b8eed085e6/data/python/data.jsonl"
    ),
    size_bytes=135_858_935,
    sha256="d167da37e1058371c48e057cd8815d03700c867dd8bcf58e61420d4dcd288d73",
    compression="none",
)

QWEN3_TOKENIZER_ARTIFACT = PinnedDataArtifact(
    name="qwen3-tokenizer",
    url=(
        "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/"
        "c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
    ),
    cache_path=PurePosixPath(
        "models/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
    ),
    size_bytes=11_422_654,
    sha256="aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    compression="none",
)

QWEN3_TOKENIZER_CONFIG_ARTIFACT = PinnedDataArtifact(
    name="qwen3-tokenizer-config",
    url=(
        "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/"
        "c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json"
    ),
    cache_path=PurePosixPath(
        "models/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json"
    ),
    size_bytes=9_732,
    sha256="d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    compression="none",
)

M2_PINNED_ARTIFACTS = (
    OASST1_JSONL_ARTIFACT,
    COMMITPACKFT_PYTHON_JSONL_ARTIFACT,
    QWEN3_TOKENIZER_ARTIFACT,
    QWEN3_TOKENIZER_CONFIG_ARTIFACT,
)
M2_ACQUISITION_MANIFEST = M2AcquisitionManifest(
    artifacts=tuple(sorted(M2_PINNED_ARTIFACTS, key=lambda artifact: artifact.name))
)


class DataAcquisitionError(RuntimeError):
    """Raised when a pinned artifact or JSONL stream fails integrity checks."""


@dataclass(frozen=True, slots=True)
class AcquiredM2Artifacts:
    """Verified local paths needed by the fixed M2 pipeline."""

    oasst1_jsonl: Path
    commitpackft_jsonl: Path
    tokenizer_json: Path
    tokenizer_config_json: Path


def sha256_file(path: Path) -> str:
    """Hash one file without constructing an in-memory copy."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pinned_artifact(path: Path, artifact: PinnedDataArtifact) -> None:
    """Refuse missing, non-regular, wrong-size, or wrong-hash cache entries."""

    if not path.is_file() or path.is_symlink():
        raise DataAcquisitionError(
            f"pinned artifact is missing or not a regular file: {artifact.name}"
        )
    if path.stat().st_size != artifact.size_bytes:
        raise DataAcquisitionError(f"pinned artifact size mismatch: {artifact.name}")
    if sha256_file(path) != artifact.sha256:
        raise DataAcquisitionError(f"pinned artifact SHA256 mismatch: {artifact.name}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def acquire_pinned_artifact(
    artifact: PinnedDataArtifact,
    *,
    cache_root: Path,
    offline: bool,
) -> Path:
    """Return an integrity-checked cache entry, downloading atomically when allowed."""

    if not cache_root.is_absolute():
        raise DataAcquisitionError("artifact cache root must be absolute")
    destination = cache_root.joinpath(*artifact.cache_path.parts)
    try:
        resolved_root = cache_root.resolve(strict=False)
        resolved_destination = destination.resolve(strict=False)
    except OSError as exc:
        raise DataAcquisitionError("cannot resolve artifact cache path") from exc
    if not resolved_destination.is_relative_to(resolved_root):
        raise DataAcquisitionError("artifact cache path escapes cache root")
    if destination.exists() or destination.is_symlink():
        verify_pinned_artifact(destination, artifact)
        return destination
    if offline:
        raise DataAcquisitionError(f"offline cache miss: {artifact.name}")

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DataAcquisitionError(f"cannot create cache directory for {artifact.name}") from exc
    temporary = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    request = urllib.request.Request(
        artifact.url,
        headers={"User-Agent": "TinyLLM-System/0.1 pinned-artifact-acquisition"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("xb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        verify_pinned_artifact(temporary, artifact)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        temporary.unlink(missing_ok=True)
        raise DataAcquisitionError(f"cannot acquire pinned artifact: {artifact.name}") from exc
    except DataAcquisitionError:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def acquire_m2_artifacts(*, cache_root: Path, offline: bool = False) -> AcquiredM2Artifacts:
    """Acquire and verify every fixed M2 source and Tokenizer artifact."""

    acquired = {
        artifact.name: acquire_pinned_artifact(
            artifact,
            cache_root=cache_root,
            offline=offline,
        )
        for artifact in M2_PINNED_ARTIFACTS
    }
    return AcquiredM2Artifacts(
        oasst1_jsonl=acquired["oasst1-jsonl"],
        commitpackft_jsonl=acquired["commitpackft-python-jsonl"],
        tokenizer_json=acquired["qwen3-tokenizer"],
        tokenizer_config_json=acquired["qwen3-tokenizer-config"],
    )


@contextmanager
def _open_jsonl_text(path: Path, *, compression: Literal["none", "gzip"]) -> Iterator[TextIO]:
    if compression == "gzip":
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            yield cast(TextIO, handle)
    else:
        with path.open(mode="r", encoding="utf-8") as handle:
            yield handle


def iter_jsonl_records(
    path: Path,
    *,
    compression: Literal["none", "gzip"],
) -> Iterator[Mapping[str, object]]:
    """Yield strict JSON objects without exposing malformed source text in errors."""

    try:
        with _open_jsonl_text(path, compression=compression) as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    decoded: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataAcquisitionError(
                        f"invalid JSONL record at line {line_number}"
                    ) from exc
                if not isinstance(decoded, dict) or any(
                    not isinstance(key, str) for key in decoded
                ):
                    raise DataAcquisitionError(
                        f"JSONL record at line {line_number} must be an object"
                    )
                yield cast(Mapping[str, object], decoded)
    except (OSError, gzip.BadGzipFile, UnicodeError) as exc:
        raise DataAcquisitionError("cannot read pinned JSONL artifact") from exc
