from __future__ import annotations

import importlib
import platform
from types import ModuleType

import pytest
import torch
from pydantic import ValidationError

import tinyllm.training.m4_dependencies as subject
from tinyllm.training.m4_dependencies import (
    M4DependencySmokeResult,
    M4PackageVersions,
    M4QwenApiEvidence,
    M4TorchApiEvidence,
)


class FakeQwenConfig:
    """Minimal stand-in for the optional Transformers config class."""

    def __init__(self, **values: object) -> None:
        vocab_size = values["vocab_size"]
        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer")
        self.vocab_size = vocab_size


class FakeQwenOutput:
    """Minimal causal-LM output carrying a differentiable scalar loss."""

    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss


class FakeQwenModel(torch.nn.Module):
    """Small local module used to test the network-free compatibility logic."""

    def __init__(self, config: FakeQwenConfig) -> None:
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> FakeQwenOutput:
        del labels
        loss = self.weight.square() + input_ids.float().sum() * 0.0
        return FakeQwenOutput(loss)


def valid_result() -> M4DependencySmokeResult:
    return M4DependencySmokeResult(
        python_version="3.11.14",
        packages=M4PackageVersions(
            torch="2.7.1+cu118",
            transformers="4.57.6",
            accelerate="1.12.0",
            safetensors="0.6.2",
            tokenizers="0.22.2",
            huggingface_hub="0.36.2",
        ),
        torch_cuda_runtime="11.8",
        torch_cuda_available=True,
        torch_nccl_version=(2, 21, 5),
        torch_apis=M4TorchApiEvidence(),
        qwen_apis=M4QwenApiEvidence(parameter_count=82_304),
    )


def test_m4_dependency_result_keeps_remote_assets_out_of_the_gate() -> None:
    result = valid_result()

    assert result.status == "pass"
    assert result.network_accessed is False
    assert result.remote_model_assets_loaded is False
    assert result.fixed_qwen3_8b_revision_verified is False

    with pytest.raises(ValidationError):
        M4DependencySmokeResult.model_validate(
            {**result.model_dump(mode="python"), "remote_model_assets_loaded": True}
        )


def test_m4_dependency_result_rejects_unknown_fields_and_wrong_python() -> None:
    result = valid_result()

    with pytest.raises(ValidationError):
        M4DependencySmokeResult.model_validate(
            {**result.model_dump(mode="python"), "unreviewed_dependency": "present"}
        )
    with pytest.raises(ValidationError):
        M4DependencySmokeResult.model_validate(
            {**result.model_dump(mode="python"), "python_version": "3.12.0"}
        )


def test_m4_package_versions_are_checked_before_the_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "torch": "2.7.1+cpu",
        "transformers": "4.57.6",
        "accelerate": "1.12.0",
        "safetensors": "0.6.2",
        "tokenizers": "0.22.2",
        "huggingface-hub": "0.36.2",
    }

    def package_version(name: str) -> str:
        return versions[name]

    monkeypatch.setattr(subject, "_package_version", package_version)
    evidence = subject._validated_package_versions()
    assert evidence.torch == "2.7.1+cpu"

    versions["transformers"] = "5.0.0"
    with pytest.raises(RuntimeError, match="do not match constraints"):
        subject._validated_package_versions()


def test_m4_torch_api_gate_requires_fully_shard_and_dcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsdp = ModuleType("torch.distributed.fsdp")
    checkpoint = ModuleType("torch.distributed.checkpoint")

    def api() -> None:
        return None

    fsdp.fully_shard = api  # type: ignore[attr-defined]
    checkpoint.save = api  # type: ignore[attr-defined]
    checkpoint.load = api  # type: ignore[attr-defined]

    def import_module(name: str, package: str | None = None) -> ModuleType:
        del package
        return fsdp if name.endswith("fsdp") else checkpoint

    def distributed_available() -> bool:
        return True

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(torch.distributed, "is_available", distributed_available)
    assert subject._torch_api_evidence().fully_shard_importable is True

    delattr(fsdp, "fully_shard")
    with pytest.raises(RuntimeError, match="fully_shard"):
        subject._torch_api_evidence()


def test_m4_qwen_gate_runs_synthetic_forward_and_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = ModuleType("transformers")
    transformers.Qwen3Config = FakeQwenConfig  # type: ignore[attr-defined]
    transformers.Qwen3ForCausalLM = FakeQwenModel  # type: ignore[attr-defined]

    def import_module(name: str, package: str | None = None) -> ModuleType:
        del package
        assert name == "transformers"
        return transformers

    monkeypatch.setattr(importlib, "import_module", import_module)
    evidence = subject._qwen_api_evidence()
    assert evidence.parameter_count == 1
    assert evidence.backward_finite is True


def test_m4_dependency_smoke_binds_python_and_local_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = valid_result()
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.14")
    monkeypatch.setattr(subject, "_validated_package_versions", lambda: expected.packages)
    monkeypatch.setattr(subject, "_torch_api_evidence", lambda: expected.torch_apis)
    monkeypatch.setattr(subject, "_qwen_api_evidence", lambda: expected.qwen_apis)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = subject.run_m4_dependency_smoke()
    assert result.torch_cuda_runtime is None
    assert result.torch_cuda_available is False
    assert result.remote_model_assets_loaded is False

    monkeypatch.setattr(platform, "python_version", lambda: "3.12.0")
    with pytest.raises(RuntimeError, match="requires Python 3.11"):
        subject.run_m4_dependency_smoke()
