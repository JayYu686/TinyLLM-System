#!/usr/bin/env python3
"""Run real M7 API, backend-recovery, and Last Known Good validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from tinyllm.deployment import (
    DeploymentError,
    M7ContractEvidence,
    M7RecoveryEvidence,
    M7RollbackEvidence,
    resolve_model,
)
from tinyllm.schemas import canonical_config_hash
from tinyllm.serving.config import load_gateway_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, value: object) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("evidence output must be a new absolute path")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
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


def _identifier(kind: str, values: object) -> str:
    return f"m7-{kind}-{canonical_config_hash(values)[:8]}"


def _wait_status(client: httpx.Client, url: str, expected: int, timeout: float) -> float:
    started = time.monotonic()
    while time.monotonic() - started <= timeout:
        try:
            if client.get(url).status_code == expected:
                return time.monotonic() - started
        except httpx.HTTPError:
            pass
        time.sleep(0.1 if timeout <= 5 else 0.5)
    raise RuntimeError(f"service did not reach HTTP {expected} within {timeout} seconds")


def _chat(client: httpx.Client, url: str, token: str, *, mode: str, stream: bool) -> Any:
    return client.post(
        f"{url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": "production",
            "messages": [{"role": "user", "content": "Return the word READY."}],
            "mode": mode,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
            "temperature": 0,
            "max_completion_tokens": 64,
        },
    )


def _backend_pid(parent_pid: int) -> int:
    rows = subprocess.run(
        ("ps", "-eo", "pid=,ppid=,args="),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.splitlines()
    children: dict[int, list[int]] = {}
    commands: dict[int, str] = {}
    for row in rows:
        parts = row.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid, parent, command = int(parts[0]), int(parts[1]), parts[2]
        children.setdefault(parent, []).append(pid)
        commands[pid] = command
    pending = list(children.get(parent_pid, ()))
    while pending:
        pid = pending.pop()
        if "tinyllm.serving.vllm_entrypoint" in commands.get(pid, ""):
            return pid
        pending.extend(children.get(pid, ()))
    raise RuntimeError("managed backend process is unavailable")


def _contract(
    client: httpx.Client,
    *,
    gateway_url: str,
    backend_url: str,
    gateway_token: str,
    backend_token: str,
    candidate: str,
    model_sha256: str,
    config_sha256: str,
    environment_sha256: str,
) -> M7ContractEvidence:
    live = client.get(f"{gateway_url}/health/live")
    ready = client.get(f"{gateway_url}/health/ready")
    version = client.get(f"{gateway_url}/version")
    models = client.get(
        f"{gateway_url}/v1/models",
        headers={"Authorization": f"Bearer {gateway_token}"},
    )
    nonthinking = _chat(client, gateway_url, gateway_token, mode="nonthinking", stream=False)
    thinking = _chat(client, gateway_url, gateway_token, mode="thinking", stream=False)
    stream = _chat(client, gateway_url, gateway_token, mode="nonthinking", stream=True)
    stream_text = stream.text
    unauthorized = client.get(f"{gateway_url}/v1/models")
    invalid = client.post(
        f"{gateway_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {gateway_token}"},
        json={"model": "production", "messages": [], "unknown": True},
    )
    wrong_model = client.post(
        f"{gateway_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {gateway_token}"},
        json={"model": "missing", "messages": [{"role": "user", "content": "x"}]},
    )
    backend_unauthorized = client.get(f"{backend_url}/health")
    backend_hidden = client.get(
        f"{backend_url}/metrics", headers={"Authorization": f"Bearer {backend_token}"}
    )
    backend_dynamic = client.post(
        f"{backend_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {backend_token}"},
        json={
            "model": candidate,
            "messages": [{"role": "user", "content": "x"}],
            "chat_template_kwargs": {"enable_thinking": False, "unsafe": True},
        },
    )

    with client.stream(
        "POST",
        f"{gateway_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {gateway_token}"},
        json={
            "model": "production",
            "messages": [{"role": "user", "content": "Write a long numbered list."}],
            "mode": "nonthinking",
            "stream": True,
            "max_completion_tokens": 512,
        },
    ) as cancelled:
        for line in cancelled.iter_lines():
            if line.startswith("data: "):
                break
    time.sleep(1)
    metrics = client.get(
        f"{gateway_url}/metrics",
        headers={"Authorization": f"Bearer {gateway_token}"},
    )
    cancellation = (
        metrics.status_code == 200
        and "tinyllm_gateway_stream_disconnects_total" in metrics.text
        and any(
            line.startswith("tinyllm_gateway_stream_disconnects_total ")
            and float(line.rsplit(" ", 1)[1]) >= 1
            for line in metrics.text.splitlines()
        )
    )
    thinking_payload = thinking.json() if thinking.status_code == 200 else {}
    thinking_serialized = json.dumps(thinking_payload, ensure_ascii=False)
    thinking_hidden = (
        thinking.status_code == 200
        and "reasoning_content" not in thinking_serialized
        and "<think>" not in thinking_serialized
        and "</think>" not in thinking_serialized
    )
    api = all(
        response.status_code == 200
        for response in (live, ready, version, models, nonthinking, thinking)
    )
    streaming = (
        stream.status_code == 200 and "data: [DONE]" in stream_text and '"usage"' in stream_text
    )
    auth = unauthorized.status_code == 401
    errors = invalid.status_code == 422 and wrong_model.status_code == 404
    guard = (
        backend_unauthorized.status_code == 401
        and backend_hidden.status_code == 404
        and backend_dynamic.status_code == 400
    )
    values = {
        "candidate": candidate,
        "environment": environment_sha256,
        "api": api,
        "auth": auth,
        "streaming": streaming,
        "cancellation": cancellation,
        "errors": errors,
        "thinking_hidden": thinking_hidden,
        "guard": guard,
    }
    return M7ContractEvidence(
        evidence_id=_identifier("contract", values),
        evaluated_at=datetime.now(UTC),
        candidate_model_version=candidate,
        model_artifact_sha256=model_sha256,
        serving_config_sha256=config_sha256,
        environment_sha256=environment_sha256,
        api_contract_passed=api,
        auth_contract_passed=auth,
        streaming_contract_passed=streaming,
        cancellation_contract_passed=cancellation,
        error_mapping_contract_passed=errors,
        thinking_content_hidden=thinking_hidden,
        backend_guard_passed=guard,
        passed=all((api, auth, streaming, cancellation, errors, thinking_hidden, guard)),
    )


def validate(args: argparse.Namespace) -> dict[str, object]:
    if not args.output_dir.is_absolute() or args.output_dir.exists():
        raise ValueError("output directory must be new and absolute")
    config = load_gateway_config(args.config)
    config_sha256 = canonical_config_hash(config)
    environment_sha256 = _sha256(args.environment)
    resolved = resolve_model(args.artifact_root, args.candidate)
    gateway_token = secrets.token_urlsafe(48)
    backend_token = secrets.token_urlsafe(48)
    args.output_dir.mkdir(parents=True, mode=0o700)
    log_path = args.output_dir / "service.log"
    command = [
        str(Path(sys.executable).with_name("tinyllm")),
        "serve",
        "--config",
        str(args.config),
        "--artifact-root",
        str(args.artifact_root),
        "--model",
        args.candidate,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "TINYLLM_GATEWAY_BEARER_TOKEN": gateway_token,
            "TINYLLM_VLLM_INTERNAL_TOKEN": backend_token,
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_handle = log_path.open("xb")
    os.fchmod(log_handle.fileno(), 0o600)
    process = subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    gateway_url = f"http://{config.host}:{config.port}"
    backend_url = config.backend_base_url
    try:
        with httpx.Client(follow_redirects=False, trust_env=False, timeout=120) as client:
            _wait_status(client, f"{gateway_url}/health/ready", 200, 180)
            contract = _contract(
                client,
                gateway_url=gateway_url,
                backend_url=backend_url,
                gateway_token=gateway_token,
                backend_token=backend_token,
                candidate=args.candidate,
                model_sha256=resolved.model_artifact_sha256,
                config_sha256=config_sha256,
                environment_sha256=environment_sha256,
            )
            _write_new(args.output_dir / "contract.json", contract.to_dict())

            backend_process = _backend_pid(process.pid)
            before_restarts = backend_process
            os.kill(backend_process, signal.SIGKILL)
            failure_seconds = _wait_status(client, f"{gateway_url}/health/ready", 503, 5)
            recovery_seconds = _wait_status(client, f"{gateway_url}/health/ready", 200, 180)
            post_recovery = _chat(
                client, gateway_url, gateway_token, mode="nonthinking", stream=False
            )
            restarted_backend = _backend_pid(process.pid)
            recovery_values = {
                "candidate": args.candidate,
                "environment": environment_sha256,
                "failure_ms": round(failure_seconds * 1000),
                "recovery_ms": round(recovery_seconds * 1000),
            }
            recovery = M7RecoveryEvidence(
                evidence_id=_identifier("recovery", recovery_values),
                evaluated_at=datetime.now(UTC),
                candidate_model_version=args.candidate,
                model_artifact_sha256=resolved.model_artifact_sha256,
                serving_config_sha256=config_sha256,
                environment_sha256=environment_sha256,
                ready_before_failure=True,
                readiness_failure_milliseconds=round(failure_seconds * 1000),
                backend_recovery_milliseconds=round(recovery_seconds * 1000),
                backend_restart_count=int(restarted_backend != before_restarts),
                post_recovery_request_succeeded=post_recovery.status_code == 200,
                passed=(
                    failure_seconds <= 5
                    and recovery_seconds <= 180
                    and restarted_backend != before_restarts
                    and post_recovery.status_code == 200
                ),
            )
            _write_new(args.output_dir / "recovery.json", recovery.to_dict())

            rollback_started = time.monotonic()
            rejected = False
            try:
                resolve_model(args.artifact_root, "qwen3-0-6b-m6-deadbeef")
            except DeploymentError:
                rejected = True
            version = client.get(f"{gateway_url}/version")
            ready = client.get(f"{gateway_url}/health/ready")
            post_rejection = _chat(
                client, gateway_url, gateway_token, mode="nonthinking", stream=False
            )
            rollback_ms = round((time.monotonic() - rollback_started) * 1000)
            identity_preserved = (
                version.status_code == 200
                and version.json().get("model") == args.candidate
                and version.json().get("model_artifact_sha256") == resolved.model_artifact_sha256
            )
            rollback_values = {
                "candidate": args.candidate,
                "environment": environment_sha256,
                "recovery_ms": rollback_ms,
            }
            rollback = M7RollbackEvidence(
                evidence_id=_identifier("rollback", rollback_values),
                evaluated_at=datetime.now(UTC),
                candidate_model_version=args.candidate,
                model_artifact_sha256=resolved.model_artifact_sha256,
                serving_config_sha256=config_sha256,
                environment_sha256=environment_sha256,
                failure_injected="invalid-model-reference",
                switch_rejected_before_activation=rejected,
                last_known_good_identity_preserved=identity_preserved,
                ready_after_rejection=ready.status_code == 200,
                post_rejection_request_succeeded=post_rejection.status_code == 200,
                rollback_recovery_milliseconds=rollback_ms,
                passed=all(
                    (
                        rejected,
                        identity_preserved,
                        ready.status_code == 200,
                        post_rejection.status_code == 200,
                        rollback_ms <= 180_000,
                    )
                ),
            )
            _write_new(args.output_dir / "rollback.json", rollback.to_dict())
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        log_handle.close()
    return {
        "schema_version": "1.0",
        "status": (
            "succeeded" if contract.passed and recovery.passed and rollback.passed else "failed"
        ),
        "contract_passed": contract.passed,
        "recovery_passed": recovery.passed,
        "rollback_passed": rollback.passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = validate(parser.parse_args())
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "succeeded":
        raise SystemExit(7)


if __name__ == "__main__":
    main()
