#!/usr/bin/env python3
"""Capture path-free M7 serving software and selected-GPU evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from tinyllm.deployment import (
    M7PackageVersion,
    M7ServingEnvironment,
    M7ServingHardware,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(*command: str) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def _write_new(path: Path, value: object) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("output must be a new absolute path")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o600)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _packages() -> tuple[M7PackageVersion, ...]:
    values = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return tuple(
        M7PackageVersion(name=name, version=version)
        for name, version in sorted(values.items(), key=lambda item: item[0])
    )


def _topology(gpu_index: int) -> tuple[int, str, str]:
    topology = _command("nvidia-smi", "topo", "-m")
    lines = topology.splitlines()
    row = next(
        (line.split() for line in lines if line.split() and line.split()[0] == f"GPU{gpu_index}"),
        None,
    )
    if row is None or len(row) < 13:
        raise ValueError("selected GPU topology row is unavailable")
    cpu_affinity = row[-3]
    numa_affinity = row[-2]
    return int(numa_affinity), cpu_affinity, hashlib.sha256(topology.encode()).hexdigest()


def capture(args: argparse.Namespace) -> tuple[M7ServingEnvironment, M7ServingHardware]:
    commit = _command("git", "rev-parse", "HEAD")
    dirty = bool(_command("git", "status", "--porcelain"))
    environment = M7ServingEnvironment(
        captured_at=datetime.now(UTC),
        python_version=platform.python_version(),
        platform=platform.platform(),
        packages=_packages(),
        serving_constraints_sha256=_sha256(args.constraints),
        vllm_wheel_sha256=_sha256(args.vllm_wheel),
        vllm_wheel_source="vllm-github-release-cu118",
        git_commit=commit,
        git_dirty=dirty,
    )
    query = _command(
        "nvidia-smi",
        f"--id={args.gpu_index}",
        (
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,"
            "temperature.gpu,driver_version"
        ),
        "--format=csv,noheader,nounits",
    ).split(", ")
    if len(query) != 7 or int(query[0]) != args.gpu_index:
        raise ValueError("selected GPU query returned an invalid identity")
    numa_node, cpu_affinity, topology_sha256 = _topology(args.gpu_index)
    import torch

    hardware = M7ServingHardware(
        captured_at=datetime.now(UTC),
        physical_gpu_index=args.gpu_index,
        gpu_name=cast(Any, query[1]),
        memory_total_mib=int(query[2]),
        driver_version=query[6],
        cuda_runtime_version=str(torch.version.cuda),
        bf16_supported=torch.cuda.is_bf16_supported(),
        numa_node=numa_node,
        cpu_affinity=cpu_affinity,
        topology_sha256=topology_sha256,
        memory_used_mib_at_preflight=int(query[3]),
        utilization_percent_at_preflight=int(query[4]),
        temperature_c_at_preflight=int(query[5]),
    )
    _write_new(args.environment_output, environment.to_dict())
    _write_new(args.hardware_output, hardware.to_dict())
    return environment, hardware


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--vllm-wheel", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--hardware-output", type=Path, required=True)
    args = parser.parse_args()
    environment, hardware = capture(args)
    print(
        json.dumps(
            {
                "status": "succeeded",
                "environment_sha256": _sha256(args.environment_output),
                "physical_gpu_index": hardware.physical_gpu_index,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
