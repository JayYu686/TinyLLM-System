"""Collective full-state Safetensors export kept separate from training Checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open  # type: ignore[import-not-found]
from safetensors.torch import save_file  # type: ignore[import-not-found]
from torch import distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_full_safetensors(
    *,
    model: nn.Module,
    export_dir: Path,
    source_model_dir: Path,
    rank: int,
) -> str:
    """Collect full CPU state on Rank zero, atomically publish, and return SHA256."""

    state = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    result: object = None
    if rank == 0:
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            destination = export_dir / "model.safetensors"
            if destination.exists():
                raise ValueError("deployment export already exists")
            tensors: dict[str, torch.Tensor] = {}
            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"full model state contains a non-tensor value: {key}")
                tensors[key] = value.detach().cpu().contiguous()
            temporary = export_dir / f".model.safetensors.tmp-{uuid.uuid4().hex}"
            save_file(tensors, temporary, metadata={"format": "pt", "purpose": "deployment"})
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            for name in (
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ):
                shutil.copyfile(source_model_dir / name, export_dir / name)
            sha256 = _sha256_file(destination)
            manifest = {
                "schema_version": "1.0",
                "purpose": "deployment_export_not_training_checkpoint",
                "filename": destination.name,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256,
                "tensor_count": len(tensors),
            }
            (export_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = {"ok": True, "sha256": sha256}
        except Exception as exc:
            result = {"ok": False, "error_type": type(exc).__name__}
    values = [result if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    result = values[0]
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError("Rank zero failed to publish the Safetensors deployment export")
    return str(result["sha256"])


def validate_safetensors_export(
    *,
    export_dir: Path,
    source_model_dir: Path,
) -> dict[str, Any]:
    """Independently validate hash, tensor inventory, and readable tensor metadata."""

    try:
        manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
        index = json.loads(
            (source_model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Safetensors export metadata cannot be parsed") from exc
    if not isinstance(manifest, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("Safetensors export metadata has an unsupported schema")
    path = export_dir / "model.safetensors"
    if not path.is_file() or path.is_symlink():
        raise ValueError("Safetensors deployment export is missing or unsafe")
    if path.stat().st_size != manifest.get("size_bytes") or _sha256_file(path) != manifest.get(
        "sha256"
    ):
        raise ValueError("Safetensors deployment export failed size or SHA256 validation")
    expected = set(cast(dict[str, str], index["weight_map"]))
    with safe_open(path, framework="pt", device="cpu") as handle:
        actual = set(handle.keys())
        if actual != expected:
            raise ValueError("Safetensors export tensor inventory differs from the pinned model")
        for key in actual:
            shape = handle.get_slice(key).get_shape()
            if not shape or any(dimension <= 0 for dimension in shape):
                raise ValueError(f"Safetensors export contains an invalid tensor shape: {key}")
    if manifest.get("tensor_count") != len(expected):
        raise ValueError("Safetensors export tensor count differs from its Manifest")
    return cast(dict[str, Any], manifest)
