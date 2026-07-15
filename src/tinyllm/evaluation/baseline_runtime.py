"""Verified model acquisition and optional runtime checks for the M2.4c Baseline."""

from __future__ import annotations

import importlib
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from types import ModuleType

from tinyllm.data.acquisition import (
    QWEN3_TOKENIZER_ARTIFACT,
    QWEN3_TOKENIZER_CONFIG_ARTIFACT,
    DataAcquisitionError,
    PinnedDataArtifact,
    acquire_pinned_artifact,
)
from tinyllm.evaluation.baseline_schema import BaselineRunConfig

_MODEL_REPOSITORY = "Qwen/Qwen3-0.6B"
_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
_MODEL_CACHE_PREFIX = f"models/{_MODEL_REPOSITORY}/{_MODEL_REVISION}"

QWEN3_MODEL_CONFIG_ARTIFACT = PinnedDataArtifact(
    name="qwen3-model-config",
    url=f"https://huggingface.co/{_MODEL_REPOSITORY}/resolve/{_MODEL_REVISION}/config.json",
    cache_path=PurePosixPath(f"{_MODEL_CACHE_PREFIX}/config.json"),
    size_bytes=726,
    sha256="660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    compression="none",
)
QWEN3_GENERATION_CONFIG_ARTIFACT = PinnedDataArtifact(
    name="qwen3-generation-config",
    url=(
        f"https://huggingface.co/{_MODEL_REPOSITORY}/resolve/"
        f"{_MODEL_REVISION}/generation_config.json"
    ),
    cache_path=PurePosixPath(f"{_MODEL_CACHE_PREFIX}/generation_config.json"),
    size_bytes=239,
    sha256="2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    compression="none",
)
QWEN3_MODEL_WEIGHTS_ARTIFACT = PinnedDataArtifact(
    name="qwen3-model-weights",
    url=(f"https://huggingface.co/{_MODEL_REPOSITORY}/resolve/{_MODEL_REVISION}/model.safetensors"),
    cache_path=PurePosixPath(f"{_MODEL_CACHE_PREFIX}/model.safetensors"),
    size_bytes=1_503_300_328,
    sha256="f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    compression="none",
)

QWEN3_BASELINE_MODEL_ARTIFACTS = (
    QWEN3_MODEL_CONFIG_ARTIFACT,
    QWEN3_GENERATION_CONFIG_ARTIFACT,
    QWEN3_MODEL_WEIGHTS_ARTIFACT,
    QWEN3_TOKENIZER_ARTIFACT,
    QWEN3_TOKENIZER_CONFIG_ARTIFACT,
)


class BaselineRuntimeError(RuntimeError):
    """Raised when model files or optional runtime dependencies violate the contract."""


class BaselinePreflightError(BaselineRuntimeError):
    """Raised before evaluation when software, artifacts, or the selected GPU are unsafe."""


@dataclass(frozen=True, slots=True)
class BaselineRuntime:
    """Imported modules after exact-version validation."""

    torch: ModuleType
    transformers: ModuleType


@dataclass(frozen=True, slots=True)
class BaselineGpuPreflight:
    """Read-only state of one physical GPU immediately before model loading."""

    physical_index: int
    memory_used_mib: int
    utilization_percent: int
    temperature_c: int


def _validate_artifact_contract(config: BaselineRunConfig) -> None:
    configured = {item.filename: (item.size_bytes, item.sha256) for item in config.model.files}
    pinned = {
        artifact.cache_path.name: (artifact.size_bytes, artifact.sha256)
        for artifact in QWEN3_BASELINE_MODEL_ARTIFACTS
    }
    if configured != pinned:
        raise BaselinePreflightError("Baseline model artifact contract does not match config")


def acquire_baseline_model(
    config: BaselineRunConfig,
    *,
    cache_root: Path,
    offline: bool,
) -> Path:
    """Acquire every pinned model file atomically and return its verified snapshot directory."""

    _validate_artifact_contract(config)
    try:
        paths = tuple(
            acquire_pinned_artifact(artifact, cache_root=cache_root, offline=offline)
            for artifact in QWEN3_BASELINE_MODEL_ARTIFACTS
        )
    except DataAcquisitionError as exc:
        raise BaselinePreflightError(str(exc)) from exc
    parents = {path.parent for path in paths}
    if len(parents) != 1:
        raise BaselinePreflightError("Baseline model artifacts do not share one snapshot directory")
    return parents.pop()


def _installed_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise BaselinePreflightError(
            f"Baseline dependency is not installed: {distribution}"
        ) from exc


def load_baseline_runtime(config: BaselineRunConfig) -> BaselineRuntime:
    """Require the exact reviewed software stack before importing model implementation code."""

    expected = {
        "accelerate": config.software.accelerate,
        "datasets": config.software.datasets,
        "lm_eval": config.software.lm_eval,
        "safetensors": config.software.safetensors,
        "tokenizers": config.software.tokenizers,
        "transformers": config.software.transformers,
    }
    mismatches = [
        f"{name}={actual} (expected {version})"
        for name, version in expected.items()
        if (actual := _installed_version(name)) != version
    ]
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - metadata should fail first
        raise BaselinePreflightError("cannot import a validated Baseline dependency") from exc
    torch_version = str(torch.__version__)
    if torch_version != config.software.torch:
        mismatches.append(f"torch={torch_version} (expected {config.software.torch})")
    if mismatches:
        raise BaselinePreflightError("Baseline dependency mismatch: " + "; ".join(mismatches))
    return BaselineRuntime(torch=torch, transformers=transformers)


def preflight_baseline_gpu(physical_index: int) -> BaselineGpuPreflight:
    """Reject a missing, hot, or materially busy physical GPU before changing visibility."""

    if physical_index < 0:
        raise BaselinePreflightError("physical GPU index must be non-negative")
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BaselinePreflightError("cannot inspect physical GPUs with nvidia-smi") from exc
    selected: BaselineGpuPreflight | None = None
    try:
        for line in completed.stdout.splitlines():
            fields = tuple(int(field.strip()) for field in line.split(","))
            if len(fields) == 4 and fields[0] == physical_index:
                selected = BaselineGpuPreflight(
                    physical_index=fields[0],
                    memory_used_mib=fields[1],
                    utilization_percent=fields[2],
                    temperature_c=fields[3],
                )
                break
    except ValueError as exc:
        raise BaselinePreflightError("nvidia-smi returned an invalid GPU inventory") from exc
    if selected is None:
        raise BaselinePreflightError("selected physical GPU does not exist")
    if selected.temperature_c >= 80:
        raise BaselinePreflightError("selected physical GPU is too hot")
    if selected.memory_used_mib > 1024 or selected.utilization_percent > 10:
        raise BaselinePreflightError("selected physical GPU is busy")
    return selected
