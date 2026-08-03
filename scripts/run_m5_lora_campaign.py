#!/usr/bin/env python3
"""Run the persistent two-segment M5 Qwen3-8B LoRA campaign."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from tinyllm.lineage import read_git_identity
from tinyllm.training.m5_failure import require_child_success
from tinyllm.training.m5_formal import _append_jsonl, _atomic_json, _sha256_file
from tinyllm.training.m5_lora import M5LoRAError
from tinyllm.training.m5_lora_schema import M5LoRACampaignResult, M5LoRARunResult
from tinyllm.training.smoke_preflight import inspect_gpus

_PAUSE_TEMPERATURE_C = 84
_RESUME_TEMPERATURE_C = 74
_POLL_SECONDS = 5


def _run_directories(root: Path) -> set[Path]:
    return {path for path in root.iterdir() if path.is_dir() and (path / "run.json").is_file()}


def _read_result(run: Path) -> tuple[M5LoRARunResult, str]:
    path = run / "result.json"
    try:
        raw = path.read_bytes()
        return M5LoRARunResult.model_validate_json(raw), hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError) as exc:
        raise M5LoRAError("M5 LoRA campaign segment result is invalid") from exc


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGCONT)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def _run_guarded(
    command: list[str],
    *,
    gpu_index: int,
    event_log: Path,
    timeout_seconds: int,
) -> tuple[int, int, int]:
    process = subprocess.Popen(command, start_new_session=True)
    started = time.monotonic()
    paused = False
    pause_count = 0
    max_temperature = 0
    _append_jsonl(
        event_log,
        {
            "event": "segment_started",
            "gpu_index": gpu_index,
            "pid": process.pid,
        },
    )
    try:
        while process.poll() is None:
            if time.monotonic() - started > timeout_seconds:
                raise M5LoRAError("M5 LoRA campaign segment exceeded its wall-clock limit")
            temperature = inspect_gpus((gpu_index,))[0]["temperature_c"]
            max_temperature = max(max_temperature, temperature)
            if not paused and temperature >= _PAUSE_TEMPERATURE_C:
                os.killpg(process.pid, signal.SIGSTOP)
                paused = True
                pause_count += 1
                _append_jsonl(
                    event_log,
                    {
                        "event": "thermal_pause",
                        "gpu_index": gpu_index,
                        "temperature_c": temperature,
                    },
                )
            elif paused and temperature <= _RESUME_TEMPERATURE_C:
                os.killpg(process.pid, signal.SIGCONT)
                paused = False
                _append_jsonl(
                    event_log,
                    {
                        "event": "thermal_resume",
                        "gpu_index": gpu_index,
                        "temperature_c": temperature,
                    },
                )
            time.sleep(_POLL_SECONDS)
        return_code = process.wait()
    except BaseException:
        _terminate_process_group(process)
        raise
    _append_jsonl(
        event_log,
        {
            "event": "segment_finished",
            "gpu_index": gpu_index,
            "return_code": return_code,
            "thermal_pause_count": pause_count,
            "max_observed_temperature_c": max_temperature,
        },
    )
    return return_code, pause_count, max_temperature


def _segment_command(args: argparse.Namespace, *, resume_run: Path | None) -> list[str]:
    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(project_root / "scripts" / "run_m5_lora.py"),
        "--config",
        str(args.config),
        "--dataset-root",
        str(args.dataset_root),
        "--model-dir",
        str(args.model_dir),
        "--output-root",
        str(args.output_root),
        "--m4-config",
        str(args.m4_config),
        "--gpu-index",
        str(args.gpu_index),
        "--timeout-seconds",
        str(args.segment_timeout_seconds),
    ]
    if resume_run is None:
        command.extend(("--stop-after-tokens", str(args.interruption_tokens)))
    else:
        command.extend(("--resume-run", str(resume_run)))
    return command


def run_campaign(args: argparse.Namespace) -> M5LoRACampaignResult:
    """Execute, validate, and summarize the two formal LoRA segments."""

    project_root = Path(__file__).resolve().parents[1]
    git_commit, git_dirty = read_git_identity(project_root)
    if git_dirty:
        raise M5LoRAError("M5 LoRA campaign requires a clean Git worktree")
    args.output_root.mkdir(parents=True, exist_ok=True)
    campaign_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-m5-lora-campaign-gpu{args.gpu_index}"
    )
    campaign_root = args.output_root / "_campaigns" / campaign_id
    campaign_root.mkdir(parents=True, exist_ok=False)
    event_log = campaign_root / "thermal-events.jsonl"
    before = _run_directories(args.output_root)
    first_code, first_pauses, first_maximum = _run_guarded(
        _segment_command(args, resume_run=None),
        gpu_index=args.gpu_index,
        event_log=event_log,
        timeout_seconds=args.segment_timeout_seconds + 600,
    )
    created = _run_directories(args.output_root) - before
    require_child_success(first_code)
    if len(created) != 1:
        raise M5LoRAError("M5 LoRA campaign fresh segment failed")
    run = created.pop()
    interrupted, interrupted_sha256 = _read_result(run)
    if (
        interrupted.status != "interrupted"
        or interrupted.mode != "fresh"
        or not args.interruption_tokens
        <= interrupted.supervised_tokens
        < args.interruption_tokens + 100_000
    ):
        raise M5LoRAError("M5 LoRA campaign interruption boundary is invalid")
    final_code, final_pauses, final_maximum = _run_guarded(
        _segment_command(args, resume_run=run),
        gpu_index=args.gpu_index,
        event_log=event_log,
        timeout_seconds=args.segment_timeout_seconds + 600,
    )
    require_child_success(final_code)
    final, final_sha256 = _read_result(run)
    if (
        final.status != "succeeded"
        or final.mode != "exact_resume"
        or final.resumed_from_tokens != interrupted.supervised_tokens
        or final.supervised_tokens != 10_000_000
        or final.adapter_sha256 is None
    ):
        raise M5LoRAError("M5 LoRA campaign final result is incomplete")
    result = M5LoRACampaignResult(
        status="succeeded",
        campaign_id=campaign_id,
        run_id=final.run_id,
        physical_gpu_index=args.gpu_index,
        segment_count=2,
        interruption_tokens=interrupted.supervised_tokens,
        resumed_from_tokens=final.resumed_from_tokens,
        final_tokens=cast(Literal[10_000_000], final.supervised_tokens),
        adapter_sha256=final.adapter_sha256,
        interrupted_result_sha256=interrupted_sha256,
        final_result_sha256=final_sha256,
        thermal_events_sha256=_sha256_file(event_log),
        thermal_pause_count=first_pauses + final_pauses,
        max_observed_temperature_c=max(first_maximum, final_maximum),
        git_commit=git_commit,
    )
    _atomic_json(campaign_root / "result.json", result.to_dict())
    print(result.model_dump_json(), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft/m5_formal_qwen3_8b_lora.yaml"),
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--m4-config",
        type=Path,
        default=Path("configs/fsdp2/qwen3_8b_four_gpu_formal.yaml"),
    )
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--interruption-tokens", type=int, default=5_000_000)
    parser.add_argument("--segment-timeout-seconds", type=int, default=43_200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_campaign(args)
    except (M5LoRAError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
