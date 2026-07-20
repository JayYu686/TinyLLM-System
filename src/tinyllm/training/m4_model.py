"""Pinned Qwen3-8B artifact verification, safe loading, and FSDP2 wrapping."""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open  # type: ignore[import-not-found]
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-not-found]
from transformers.models.qwen3.modeling_qwen3 import (  # type: ignore[import-not-found]
    Qwen3DecoderLayer,
)

from tinyllm.training.errors import TrainingError, TrainingErrorCode
from tinyllm.training.m4_model_schema import M4ModelArtifactFile, M4ModelArtifactManifest
from tinyllm.training.m4_qwen_config import M4QwenFSDP2Config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_qwen3_8b_artifact(
    *,
    model_dir: Path,
    config: M4QwenFSDP2Config,
) -> M4ModelArtifactManifest:
    """Fail closed unless every indexed Safetensors shard and metadata file is local."""

    model_dir = model_dir.resolve()
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise ValueError("pinned Qwen model directory is missing or is a symbolic link")
    required_metadata = {
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    try:
        index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("pinned Qwen model metadata cannot be parsed") from exc
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("Qwen Safetensors index has an unsupported schema")
    weight_map = cast(dict[str, str], index["weight_map"])
    shard_names = set(weight_map.values())
    if not shard_names or any(not name.endswith(".safetensors") for name in shard_names):
        raise ValueError("Qwen Safetensors index contains invalid shard names")
    required = required_metadata | shard_names
    paths = tuple(model_dir / name for name in sorted(required))
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("pinned Qwen model snapshot is incomplete or contains a symlink")

    discovered_keys: set[str] = set()
    for shard_name in sorted(shard_names):
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
        expected_keys = {key for key, value in weight_map.items() if value == shard_name}
        if shard_keys != expected_keys:
            raise ValueError(f"Safetensors index differs from shard {shard_name}")
        discovered_keys.update(shard_keys)
    if discovered_keys != set(weight_map):
        raise ValueError("Qwen Safetensors tensor inventory is incomplete")

    hf_config = AutoConfig.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=config.model.trust_remote_code,
    )
    expected_config = {
        "model_type": config.model.model_type,
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 12288,
        "vocab_size": 151936,
        "torch_dtype": "bfloat16",
    }
    actual_config = {key: raw_config.get(key) for key in expected_config}
    if actual_config != expected_config or hf_config.model_type != config.model.model_type:
        raise ValueError("local Qwen config differs from the frozen M4 architecture")
    license_text = (model_dir / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("local Qwen snapshot does not contain the expected Apache-2.0 license")

    files = tuple(
        M4ModelArtifactFile(
            path=path.name,
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )
        for path in paths
    )
    digest = hashlib.sha256()
    digest.update(config.model.repository.encode("utf-8"))
    digest.update(config.model.revision.encode("ascii"))
    for item in files:
        digest.update(item.path.encode("utf-8"))
        digest.update(str(item.size_bytes).encode("ascii"))
        digest.update(item.sha256.encode("ascii"))
    metadata = cast(dict[str, object], index.get("metadata", {}))
    weight_bytes = metadata.get("total_size")
    if type(weight_bytes) is not int or weight_bytes <= 0:
        raise ValueError("Qwen Safetensors index is missing total_size")
    return M4ModelArtifactManifest(
        repository=config.model.repository,
        revision=config.model.revision,
        license=config.model.license,
        model_type=config.model.model_type,
        architecture="Qwen3ForCausalLM",
        hidden_size=4096,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=12288,
        vocab_size=151936,
        weight_bytes=weight_bytes,
        tensor_count=len(weight_map),
        files=files,
        content_sha256=digest.hexdigest(),
    )


def load_qwen3_8b(
    *,
    model_dir: Path,
    config: M4QwenFSDP2Config,
    device: torch.device,
) -> nn.Module:
    """Load only local Safetensors in BF16, disable KV cache, and move to one Rank device."""

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=config.model.trust_remote_code,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.to(device)
    return cast(nn.Module, model)


def apply_qwen_activation_checkpointing(model: nn.Module, *, expected_layers: int) -> int:
    """Apply non-reentrant Activation Checkpointing to every Qwen3 decoder layer."""

    actual = sum(isinstance(module, Qwen3DecoderLayer) for module in model.modules())
    if actual != expected_layers:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
            "Qwen3 decoder-layer count differs from the pinned model manifest",
            context={"actual": actual, "expected": expected_layers},
        )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=lambda module: isinstance(module, Qwen3DecoderLayer),
    )
    wrapped = sum(hasattr(module, "_checkpoint_wrapped_module") for module in model.modules())
    if wrapped != expected_layers:
        raise TrainingError(
            TrainingErrorCode.DISTRIBUTED_STATE_MISMATCH,
            "Activation Checkpointing did not wrap every Qwen3 decoder layer",
            context={"actual": wrapped, "expected": expected_layers},
        )
    return wrapped


def apply_qwen_fully_shard(
    model: nn.Module,
    *,
    config: M4QwenFSDP2Config,
) -> None:
    """FULL_SHARD every decoder layer and then the complete Qwen model."""

    mesh = init_device_mesh("cuda", (config.distributed.world_size,), mesh_dim_names=("fsdp",))
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.float32,
    )
    backbone = getattr(model, "model", None)
    raw_layers = getattr(backbone, "layers", None)
    if not isinstance(raw_layers, nn.ModuleList) or len(raw_layers) != 36:
        raise ValueError("formal Qwen FSDP2 wrapping requires exactly 36 decoder layers")
    layers = tuple(raw_layers)
    if any(
        sum(isinstance(module, Qwen3DecoderLayer) for module in layer.modules()) != 1
        for layer in layers
    ):
        raise ValueError("each Qwen FSDP2 wrap unit must contain one decoder layer")
    for layer in layers:
        fully_shard(
            layer,
            mesh=mesh,
            reshard_after_forward=config.distributed.reshard_after_forward,
            mp_policy=policy,
        )
    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=config.distributed.reshard_after_forward,
        mp_policy=policy,
    )
