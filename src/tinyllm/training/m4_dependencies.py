"""Versioned evidence for the isolated M4 FSDP2/Qwen dependency gate."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import platform
from typing import Any, Literal

import torch
from pydantic import Field

from tinyllm.schemas.base import StrictSchema

_EXPECTED_BASE_VERSIONS = {
    "torch": "2.7.1",
    "transformers": "4.57.6",
    "accelerate": "1.12.0",
    "safetensors": "0.6.2",
    "tokenizers": "0.22.2",
}


class M4PackageVersions(StrictSchema):
    """Direct M4 package versions plus the Qwen Hub client version."""

    torch: str = Field(min_length=1)
    transformers: str = Field(min_length=1)
    accelerate: str = Field(min_length=1)
    safetensors: str = Field(min_length=1)
    tokenizers: str = Field(min_length=1)
    huggingface_hub: str = Field(min_length=1)


class M4TorchApiEvidence(StrictSchema):
    """PyTorch APIs required before implementing the M4 training path."""

    distributed_available: Literal[True] = True
    fully_shard_importable: Literal[True] = True
    dcp_save_importable: Literal[True] = True
    dcp_load_importable: Literal[True] = True


class M4QwenApiEvidence(StrictSchema):
    """Synthetic local Qwen3 construction and autograd evidence."""

    config_importable: Literal[True] = True
    causal_lm_importable: Literal[True] = True
    parameter_count: int = Field(gt=0)
    forward_finite: Literal[True] = True
    backward_finite: Literal[True] = True


class M4DependencySmokeResult(StrictSchema):
    """Public, privacy-safe result of the bounded M4 dependency Smoke."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["pass"] = "pass"
    profile: Literal["m4-fsdp2-qwen"] = "m4-fsdp2-qwen"
    python_version: str = Field(pattern=r"^3\.11\.")
    packages: M4PackageVersions
    torch_cuda_runtime: str | None
    torch_cuda_available: bool
    torch_nccl_version: tuple[int, ...] | None
    torch_apis: M4TorchApiEvidence
    qwen_apis: M4QwenApiEvidence
    network_accessed: Literal[False] = False
    remote_model_assets_loaded: Literal[False] = False
    fixed_qwen3_8b_revision_verified: Literal[False] = False


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required M4 package is not installed: {distribution}") from exc


def _base_version(version: str) -> str:
    return version.split("+", maxsplit=1)[0]


def _validated_package_versions() -> M4PackageVersions:
    versions = {
        name: _package_version(name) for name in (*_EXPECTED_BASE_VERSIONS, "huggingface-hub")
    }
    mismatches = {
        name: {"actual": _base_version(versions[name]), "expected": expected}
        for name, expected in _EXPECTED_BASE_VERSIONS.items()
        if _base_version(versions[name]) != expected
    }
    if mismatches:
        raise RuntimeError(f"M4 package versions do not match constraints: {mismatches}")
    return M4PackageVersions(
        torch=versions["torch"],
        transformers=versions["transformers"],
        accelerate=versions["accelerate"],
        safetensors=versions["safetensors"],
        tokenizers=versions["tokenizers"],
        huggingface_hub=versions["huggingface-hub"],
    )


def _torch_api_evidence() -> M4TorchApiEvidence:
    fsdp = importlib.import_module("torch.distributed.fsdp")
    checkpoint = importlib.import_module("torch.distributed.checkpoint")
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is unavailable in the M4 profile")
    if not callable(getattr(fsdp, "fully_shard", None)):
        raise RuntimeError("torch.distributed.fsdp.fully_shard is unavailable")
    if not callable(getattr(checkpoint, "save", None)):
        raise RuntimeError("torch.distributed.checkpoint.save is unavailable")
    if not callable(getattr(checkpoint, "load", None)):
        raise RuntimeError("torch.distributed.checkpoint.load is unavailable")
    return M4TorchApiEvidence()


def _qwen_api_evidence() -> M4QwenApiEvidence:
    transformers: Any = importlib.import_module("transformers")
    config_type = getattr(transformers, "Qwen3Config", None)
    model_type = getattr(transformers, "Qwen3ForCausalLM", None)
    if config_type is None or model_type is None:
        raise RuntimeError("Transformers does not expose the required Qwen3 classes")

    torch.manual_seed(42)
    config = config_type(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        use_cache=False,
    )
    model = model_type(config)
    input_ids = torch.arange(32, dtype=torch.long).reshape(2, 16) % config.vocab_size
    output = model(input_ids=input_ids, labels=input_ids)
    loss = getattr(output, "loss", None)
    if not isinstance(loss, torch.Tensor) or not math.isfinite(float(loss.detach())):
        raise RuntimeError("synthetic Qwen3 forward produced a non-finite loss")
    loss.backward()  # type: ignore[no-untyped-call]
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(value).all().item()) for value in gradients):
        raise RuntimeError("synthetic Qwen3 backward produced missing or non-finite gradients")
    return M4QwenApiEvidence(
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def run_m4_dependency_smoke() -> M4DependencySmokeResult:
    """Run a CPU-only, network-free M4 dependency compatibility gate."""

    python_version = platform.python_version()
    if not python_version.startswith("3.11."):
        raise RuntimeError(f"M4 requires Python 3.11, found {python_version}")

    packages = _validated_package_versions()
    torch_apis = _torch_api_evidence()
    qwen_apis = _qwen_api_evidence()
    nccl_version: tuple[int, ...] | None = None
    if torch.version.cuda is not None:
        raw_nccl_version = torch.cuda.nccl.version()  # type: ignore[no-untyped-call]
        if raw_nccl_version is not None:
            nccl_version = tuple(int(part) for part in raw_nccl_version)
    return M4DependencySmokeResult(
        python_version=python_version,
        packages=packages,
        torch_cuda_runtime=torch.version.cuda,
        torch_cuda_available=torch.cuda.is_available(),
        torch_nccl_version=nccl_version,
        torch_apis=torch_apis,
        qwen_apis=qwen_apis,
    )
