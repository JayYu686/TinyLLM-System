#!/usr/bin/env python3
"""Validate the frozen M8 OpenAI Tool Calling matrix against a live Gateway."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from openai import OpenAI

from tinyllm import __version__
from tinyllm.agent import M8ToolCallingCase, M8ToolCallingValidation
from tinyllm.lineage.git import read_git_identity
from tinyllm.schemas import canonical_config_hash

TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_evidence",
        "description": "Search immutable TinyLLM evidence.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def _atomic_new(path: Path, value: object) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("output must be a new absolute path")
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


ToolChoiceMode = Literal["auto", "required", "none", "named"]


def _choice(mode: ToolChoiceMode) -> object:
    if mode == "named":
        return {"type": "function", "function": {"name": "search_evidence"}}
    return mode


def _case(client: OpenAI, model: str, mode: ToolChoiceMode, stream: bool) -> M8ToolCallingCase:
    prompt = (
        "Answer without tools: READY."
        if mode == "none"
        else "Call search_evidence with query M7 Production. Return the tool call now."
    )
    names: list[str] = []
    content = ""
    finish: str | None = None
    error_code: str | None = None
    try:
        create: Any = client.chat.completions.create
        response = create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=[TOOL],
            tool_choice=_choice(mode),
            stream=stream,
            temperature=0,
            max_tokens=256,
            extra_body={"mode": "nonthinking"},
        )
        if stream:
            for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish = choice.finish_reason or finish
                content += choice.delta.content or ""
                for call in choice.delta.tool_calls or ():
                    if call.function and call.function.name:
                        names.append(call.function.name)
        else:
            choice = response.choices[0]
            finish = choice.finish_reason
            content = choice.message.content or ""
            names.extend(
                call.function.name for call in choice.message.tool_calls or () if call.function.name
            )
    except Exception as exc:  # the public evidence records only the exception class
        error_code = type(exc).__name__
    names = list(dict.fromkeys(names))
    expects_tool = mode != "none"
    raw = "<tool_call>" in content or '"name"' in content
    passed = (
        error_code is None
        and not raw
        and (
            (expects_tool and names == ["search_evidence"] and finish == "tool_calls")
            or (not expects_tool and not names and bool(content))
        )
    )
    return M8ToolCallingCase(
        mode=mode,
        stream=stream,
        status="passed" if passed else "failed",
        finish_reason=finish,
        tool_names=tuple(names),
        content_characters=len(content),
        raw_markup_exposed=raw,
        error_code=error_code,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--token-env", default="TINYLLM_GATEWAY_BEARER_TOKEN")
    parser.add_argument("--model", default="production")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("validation endpoint must be loopback")
    token = os.environ.get(args.token_env, "")
    if len(token) < 32:
        raise ValueError("Gateway Bearer Token is missing or too short")
    project_root = args.project_root.resolve(strict=True)
    git_commit, git_dirty = read_git_identity(project_root)
    gpu_name = subprocess.run(
        [
            "nvidia-smi",
            f"--id={args.gpu_index}",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    http = httpx.Client(follow_redirects=False, trust_env=False)
    client = OpenAI(
        api_key=token,
        base_url=args.base_url,
        timeout=120,
        max_retries=0,
        http_client=http,
    )
    try:
        modes: tuple[ToolChoiceMode, ...] = ("auto", "required", "none", "named")
        cases = tuple(
            _case(client, args.model, mode, stream) for mode in modes for stream in (False, True)
        )
    finally:
        client.close()
        http.close()
    identity = {
        "model": args.model,
        "version": __version__,
        "cases": [item.to_dict() for item in cases],
    }
    result = M8ToolCallingValidation(
        validation_id=f"m8-tool-calling-{canonical_config_hash(identity)[:8]}",
        evaluated_at=datetime.now(UTC),
        model=args.model,
        gateway_version=__version__,
        git_commit=git_commit,
        git_dirty=git_dirty,
        physical_gpu_index=args.gpu_index,
        gpu_name=gpu_name,
        cases=cases,
        passed_cases=sum(item.status == "passed" for item in cases),
        passed=all(item.status == "passed" for item in cases) and not git_dirty,
    )
    _atomic_new(args.output, result.to_dict())
    print(result.model_dump_json())
    if not result.passed:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
