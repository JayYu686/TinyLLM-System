"""Environment collectors for ``tinyllm doctor``."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

from tinyllm.doctor.runner import CommandResult, Runner, SubprocessRunner
from tinyllm.doctor.schema import CheckResult, CheckStatus, DoctorReport, aggregate_status

_GIB = 1024**3
_GPU_FIELDS = (
    "index",
    "name",
    "memory_total_mib",
    "memory_used_mib",
    "pci_bus_id",
    "power_limit_w",
    "driver_version",
    "compute_capability",
    "temperature_c",
    "utilization_percent",
)


def _number(value: str, kind: type[int] | type[float]) -> int | float | None:
    cleaned = value.strip().replace(" MiB", "").replace(" W", "").replace(" %", "")
    if cleaned in {"", "N/A", "[Not Supported]"}:
        return None
    try:
        return kind(cleaned)
    except ValueError:
        return None


def parse_gpu_csv(raw: str) -> list[dict[str, object]]:
    """Parse the stable no-header NVIDIA CSV query used by doctor."""

    rows: list[dict[str, object]] = []
    for row in csv.reader(raw.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != len(_GPU_FIELDS):
            raise ValueError(f"expected {len(_GPU_FIELDS)} GPU fields, got {len(row)}")
        values = [value.strip() for value in row]
        rows.append(
            {
                "index": _number(values[0], int),
                "name": values[1],
                "memory_total_mib": _number(values[2], int),
                "memory_used_mib": _number(values[3], int),
                "pci_bus_id": values[4],
                "power_limit_w": _number(values[5], float),
                "driver_version": values[6],
                "compute_capability": values[7],
                "temperature_c": _number(values[8], int),
                "utilization_percent": _number(values[9], int),
            }
        )
    return rows


def _result_evidence(result: CommandResult) -> dict[str, object]:
    return {
        "available": result.available,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "stderr": result.stderr[:300] or None,
    }


def collect_torch(runner: Runner) -> tuple[dict[str, object], CheckResult]:
    """Inspect PyTorch in an isolated process so broken imports do not crash doctor."""

    script = """
import json
import torch
version = getattr(torch, "__version__", None)
if not version:
    raise RuntimeError("torch.__version__ is missing")
cuda_available = bool(torch.cuda.is_available())
payload = {
    "version": version,
    "cuda_runtime": getattr(torch.version, "cuda", None),
    "cuda_available": cuda_available,
    "gpu_count": int(torch.cuda.device_count()),
    "bf16_supported": bool(torch.cuda.is_bf16_supported()) if cuda_available else None,
    "nccl_version": list(torch.cuda.nccl.version()) if cuda_available else None,
}
print(json.dumps(payload, sort_keys=True))
""".strip()
    result = runner.run((sys.executable, "-c", script), timeout=20.0)
    if not result.available or result.returncode != 0:
        evidence = _result_evidence(result)
        return (
            {},
            CheckResult(
                "python.torch_import",
                "fail",
                "PyTorch could not be imported and inspected",
                required=True,
                evidence=evidence,
                remediation="Use a clean project environment with a validated PyTorch/CUDA build.",
            ),
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return (
            {},
            CheckResult(
                "python.torch_import",
                "fail",
                "PyTorch inspection returned invalid JSON",
                required=True,
                evidence={"stdout_prefix": result.stdout[:120]},
                remediation="Reinstall PyTorch in an isolated project environment.",
            ),
        )
    if not isinstance(payload, dict):
        return (
            {},
            CheckResult(
                "python.torch_import",
                "fail",
                "PyTorch inspection returned an unexpected payload",
                required=True,
                remediation="Reinstall PyTorch in an isolated project environment.",
            ),
        )
    cuda_available = payload.get("cuda_available") is True
    return (
        {str(key): value for key, value in payload.items()},
        CheckResult(
            "python.torch_import",
            "pass" if cuda_available else "fail",
            "PyTorch and CUDA are available" if cuda_available else "PyTorch has no CUDA access",
            required=True,
            evidence={"version": payload.get("version"), "cuda_available": cuda_available},
            remediation=(
                None
                if cuda_available
                else "Install a CUDA-enabled PyTorch build and verify the driver."
            ),
        ),
    )


def collect_gpus(runner: Runner) -> tuple[list[dict[str, object]], list[CheckResult]]:
    """Collect GPU inventory and health checks through NVIDIA SMI."""

    query = (
        "index,name,memory.total,memory.used,pci.bus_id,power.limit,driver_version,"
        "compute_cap,temperature.gpu,utilization.gpu"
    )
    result = runner.run(("nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"))
    if not result.available or result.returncode != 0:
        return (
            [],
            [
                CheckResult(
                    "gpu.inventory",
                    "fail",
                    "GPU inventory is unavailable",
                    required=True,
                    evidence=_result_evidence(result),
                    remediation="Install or expose nvidia-smi and verify NVIDIA driver access.",
                )
            ],
        )
    try:
        gpus = parse_gpu_csv(result.stdout)
    except ValueError as exc:
        return (
            [],
            [
                CheckResult(
                    "gpu.inventory",
                    "fail",
                    "GPU inventory output could not be parsed",
                    required=True,
                    evidence={"error": str(exc)},
                    remediation="Record the nvidia-smi version and update the parser fixture.",
                )
            ],
        )
    checks = [
        CheckResult(
            "gpu.inventory",
            "pass" if gpus else "fail",
            f"Discovered {len(gpus)} NVIDIA GPU(s)",
            required=True,
            evidence={"gpu_count": len(gpus)},
            remediation=None if gpus else "Verify NVIDIA driver access.",
        )
    ]
    busy = [
        gpu["index"]
        for gpu in gpus
        if isinstance(utilization := gpu.get("utilization_percent"), int) and utilization >= 10
    ]
    hot = [
        gpu["index"]
        for gpu in gpus
        if isinstance(temperature := gpu.get("temperature_c"), int) and temperature >= 80
    ]
    checks.append(
        CheckResult(
            "gpu.availability",
            "warn" if busy else "pass",
            "Some GPUs are currently busy" if busy else "No busy GPU detected",
            evidence={"busy_gpu_indices": busy},
            remediation="Coordinate GPU allocation before a distributed test." if busy else None,
        )
    )
    checks.append(
        CheckResult(
            "gpu.temperature",
            "warn" if hot else "pass",
            "One or more GPUs are at least 80C" if hot else "GPU temperatures are below 80C",
            evidence={"hot_gpu_indices": hot},
            remediation=(
                "Inspect cooling and sustained clocks before long training." if hot else None
            ),
        )
    )
    return gpus, checks


class DoctorCollector:
    """Collect a complete read-only environment report."""

    def __init__(self, project_root: Path, runner: Runner | None = None) -> None:
        self.project_root = project_root.resolve()
        self.runner = runner or SubprocessRunner()

    def collect(self, *, distributed: bool = False) -> DoctorReport:
        """Collect host, software, GPU, storage, Git, and optional topology facts."""

        checks: list[CheckResult] = []
        inventory: dict[str, object] = {}

        host = self._host_inventory()
        inventory["host"] = host
        checks.append(
            CheckResult(
                "host.linux",
                "pass" if sys.platform.startswith("linux") else "fail",
                "Linux host detected" if sys.platform.startswith("linux") else "Host is not Linux",
                required=True,
                evidence={"platform": sys.platform},
            )
        )

        python_inventory: dict[str, object] = {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        }
        inventory["python"] = python_inventory
        checks.append(
            CheckResult(
                "python.executable",
                "pass",
                "Python executable is available",
                required=True,
                evidence=python_inventory,
            )
        )

        torch_inventory, torch_check = collect_torch(self.runner)
        inventory["torch"] = torch_inventory
        checks.append(torch_check)

        gpus, gpu_checks = collect_gpus(self.runner)
        inventory["gpus"] = gpus
        checks.extend(gpu_checks)

        inventory["cuda"] = self._cuda_inventory(gpus)
        storage, storage_checks = self._storage_inventory()
        inventory["storage"] = storage
        checks.extend(storage_checks)

        git_inventory, git_checks = self._git_inventory()
        inventory["git"] = git_inventory
        checks.extend(git_checks)

        inventory["packages"] = self._package_inventory()

        if distributed:
            topology, topology_checks = self._distributed_inventory()
            inventory["topology"] = topology
            checks.extend(topology_checks)
        else:
            inventory["topology"] = {"collected": False}

        return DoctorReport(
            generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            status=aggregate_status(checks),
            inventory=inventory,
            checks=tuple(checks),
        )

    def _host_inventory(self) -> dict[str, object]:
        os_release: dict[str, str] = {}
        path = Path("/etc/os-release")
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if "=" not in line or line.startswith("#"):
                    continue
                key, value = line.split("=", 1)
                os_release[key.lower()] = value.strip().strip('"')
        return {
            "hostname": socket.gethostname(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "os": os_release,
            "cpu_count": os.cpu_count(),
        }

    def _cuda_inventory(self, gpus: list[dict[str, object]]) -> dict[str, object]:
        summary = self.runner.run(("nvidia-smi",))
        match = re.search(r"CUDA Version:\s*([0-9.]+)", summary.stdout)
        nvcc_path = shutil.which("nvcc")
        fallback = Path("/usr/local/cuda/bin/nvcc")
        selected_nvcc = nvcc_path or (str(fallback) if fallback.is_file() else None)
        nvcc_version: str | None = None
        if selected_nvcc:
            result = self.runner.run((selected_nvcc, "--version"))
            version_match = re.search(r"release\s+([0-9.]+)", result.stdout)
            nvcc_version = version_match.group(1) if version_match else None
        return {
            "driver_version": gpus[0].get("driver_version") if gpus else None,
            "driver_reported_cuda": match.group(1) if match else None,
            "nvcc_on_path": nvcc_path is not None,
            "nvcc_path": selected_nvcc,
            "toolkit_version": nvcc_version,
        }

    def _storage_inventory(self) -> tuple[list[dict[str, object]], list[CheckResult]]:
        entries: list[dict[str, object]] = []
        checks: list[CheckResult] = []
        paths = [self.project_root]
        data_path = Path("/data")
        if data_path.is_dir() and data_path.resolve() != self.project_root:
            paths.append(data_path)
        for path in paths:
            usage = shutil.disk_usage(path)
            free_percent = usage.free / usage.total * 100 if usage.total else 0.0
            entry = {
                "path": str(path),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "free_percent": round(free_percent, 2),
            }
            entries.append(entry)
            low = usage.free < 100 * _GIB or free_percent < 10
            checks.append(
                CheckResult(
                    f"storage.{len(entries) - 1}",
                    "warn" if low else "pass",
                    f"Storage checked for {path}",
                    required=path == self.project_root,
                    evidence={"path": str(path), "free_bytes": usage.free},
                    remediation="Free space or choose a larger artifact root." if low else None,
                )
            )
        return entries, checks

    def _git_inventory(self) -> tuple[dict[str, object], list[CheckResult]]:
        git = ("git", "-C", str(self.project_root))
        inside = self.runner.run((*git, "rev-parse", "--is-inside-work-tree"))
        if inside.returncode != 0:
            return (
                {"is_repository": False},
                [
                    CheckResult(
                        "git.repository",
                        "warn",
                        "Project is not a Git repository",
                        remediation="Initialize Git before creating traceable runs.",
                    )
                ],
            )
        branch = self.runner.run((*git, "symbolic-ref", "--short", "HEAD"))
        commit = self.runner.run((*git, "rev-parse", "HEAD"))
        status = self.runner.run((*git, "status", "--porcelain"))
        has_commit = commit.returncode == 0
        dirty = bool(status.stdout)
        return (
            {
                "is_repository": True,
                "branch": branch.stdout or None,
                "commit": commit.stdout or None if has_commit else None,
                "dirty": dirty,
            },
            [
                CheckResult(
                    "git.lineage",
                    "pass" if has_commit and not dirty else "warn",
                    (
                        "Git lineage is ready"
                        if has_commit and not dirty
                        else "Git lineage is incomplete"
                    ),
                    evidence={"has_commit": has_commit, "dirty": dirty},
                    remediation="Create a reviewed commit before formal runs."
                    if not has_commit or dirty
                    else None,
                )
            ],
        )

    def _package_inventory(self) -> dict[str, object]:
        names = ("torch", "transformers", "trl", "deepspeed", "vllm", "pytest", "ruff", "mypy")
        versions: dict[str, object] = {}
        for name in names:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return versions

    def _distributed_inventory(self) -> tuple[dict[str, object], list[CheckResult]]:
        inventory: dict[str, object] = {"collected": True}
        checks: list[CheckResult] = []
        commands: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("gpu_topology", ("nvidia-smi", "topo", "-m")),
            ("p2p_read", ("nvidia-smi", "topo", "-p2p", "r")),
            ("nvlink", ("nvidia-smi", "nvlink", "--status")),
        )
        for name, command in commands:
            result = self.runner.run(command)
            available = result.available and result.returncode == 0
            combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
            limitation = (
                name == "p2p_read"
                and any(marker in result.stdout for marker in ("CNS", "GNS", "TNS", " NS "))
            ) or (name == "nvlink" and "inactive" in combined_output.lower())
            check_status: CheckStatus = "warn" if available and limitation else "pass"
            if not available:
                check_status = "unavailable"
            inventory[name] = {
                "available": available,
                "output": combined_output if available else None,
                "error": result.stderr[:300] or None,
            }
            checks.append(
                CheckResult(
                    f"distributed.{name}",
                    check_status,
                    (
                        f"{name} reports a topology limitation"
                        if limitation
                        else f"{name} information collected"
                        if available
                        else f"{name} is unavailable"
                    ),
                    evidence=_result_evidence(result),
                    remediation=(
                        "Validate the limitation with a real NCCL collective test."
                        if limitation
                        else f"Provide {command[0]} for distributed diagnostics."
                        if not available
                        else None
                    ),
                )
            )

        numa = self.runner.run(("numactl", "--hardware"))
        numa_source = "numactl"
        if not numa.available or numa.returncode != 0:
            numa = self.runner.run(("lscpu",))
            numa_source = "lscpu"
        numa_available = numa.available and numa.returncode == 0
        inventory["numa"] = {
            "available": numa_available,
            "source": numa_source if numa_available else None,
            "output": numa.stdout if numa_available else None,
            "error": numa.stderr[:300] or None,
        }
        checks.append(
            CheckResult(
                "distributed.numa",
                "pass" if numa_available else "unavailable",
                (
                    f"NUMA information collected with {numa_source}"
                    if numa_available
                    else "NUMA information is unavailable"
                ),
                evidence=_result_evidence(numa),
                remediation="Install numactl or provide lscpu for NUMA diagnostics."
                if not numa_available
                else None,
            )
        )
        nccl_tools: dict[str, object] = {
            name: shutil.which(name)
            for name in ("all_reduce_perf", "all_gather_perf", "reduce_scatter_perf")
        }
        inventory["nccl_tools"] = nccl_tools
        tools_ready = all(path is not None for path in nccl_tools.values())
        checks.append(
            CheckResult(
                "distributed.nccl_tools",
                "pass" if tools_ready else "unavailable",
                (
                    "NCCL test tools are available"
                    if tools_ready
                    else "NCCL test tools are unavailable"
                ),
                evidence=nccl_tools,
                remediation="Build a pinned nccl-tests revision before running the M0 benchmark."
                if not tools_ready
                else None,
            )
        )
        return inventory, checks
