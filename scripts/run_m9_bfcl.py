#!/usr/bin/env python3
# mypy: disable-error-code="misc"
"""Run and summarize the pinned TinyLLM BFCL v1.3 Offline Core Profile."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import uuid
from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tinyllm.agent_eval.config import AgentEvalConfigError, load_bfcl_profile_config
from tinyllm.agent_eval.schema import BFCLCategoryResult, BFCLCoreProfileSummary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(roots: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (root.name, path.relative_to(root), path)
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    )
    for root_name, relative, path in files:
        digest.update(root_name.encode())
        digest.update(str(relative).encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
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


def _normalize_bfcl_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project BFCL's response-schema extension onto the OpenAI tool contract."""

    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if set(tool) != {"type", "function"} or tool.get("type") != "function":
            raise RuntimeError("BFCL produced an unsupported tool envelope")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise RuntimeError("BFCL produced an invalid function definition")
        unexpected = set(function).difference({"name", "description", "parameters", "response"})
        if unexpected:
            raise RuntimeError("BFCL produced an unsupported function definition")
        wire_function = dict(function)
        wire_function.pop("response", None)
        normalized.append({"type": "function", "function": wire_function})
    return normalized


def _install_tinyllm_handler(
    *,
    model_name: str,
    served_model: str,
    gateway_base_url: str,
    bearer_token: str,
    mode: str,
    max_completion_tokens: int,
) -> None:
    """Register a process-local BFCL handler without modifying the pinned checkout."""

    model_config_module = importlib.import_module("bfcl_eval.constants.model_config")
    handler_module = importlib.import_module(
        "bfcl_eval.model_handler.api_inference.openai_completion"
    )
    openai_module = importlib.import_module("openai")
    base_handler = handler_module.OpenAICompletionsHandler

    def initialize(self: Any, name: str, temperature: float) -> None:
        previous = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = bearer_token
        try:
            base_handler.__init__(self, name, temperature)
        finally:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous
        self.client = openai_module.OpenAI(
            base_url=f"{gateway_base_url.rstrip('/')}/v1",
            api_key=bearer_token,
            timeout=120.0,
            max_retries=0,
            http_client=openai_module.DefaultHttpxClient(trust_env=False),
        )
        self.is_fc_model = True

    def query_fc(self: Any, inference_data: dict[str, Any]) -> tuple[Any, float]:
        messages = inference_data["message"]
        tools = inference_data["tools"]
        wire_tools = _normalize_bfcl_tools(tools)
        inference_data["inference_input_log"] = {
            "message": repr(messages),
            "tool_count": len(wire_tools),
        }
        started = time.monotonic()
        response = self.client.chat.completions.create(
            model=served_model,
            messages=messages,
            tools=wire_tools or None,
            tool_choice="auto" if wire_tools else None,
            parallel_tool_calls=True if wire_tools else None,
            temperature=0.0,
            max_completion_tokens=max_completion_tokens,
            extra_body={"mode": mode},
        )
        return response, time.monotonic() - started

    tinyllm_handler = type(
        "TinyLLMOpenAIEndpointHandler",
        (base_handler,),
        {"__init__": initialize, "_query_FC": query_fc},
    )
    model_config_module.MODEL_CONFIG_MAPPING[model_name] = model_config_module.ModelConfig(
        model_name=model_name,
        display_name="TinyLLM Qwen3 (FC)",
        url="https://github.com/JayYu686/TinyLLM-System",
        org="TinyLLM-System",
        license="Apache-2.0",
        model_handler=tinyllm_handler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    )


def _run_bfcl(
    *,
    config: Any,
    output: Path,
    bearer_token: str,
) -> tuple[Path, Path]:
    os.environ["BFCL_PROJECT_ROOT"] = str(output)
    generation_module = importlib.import_module("bfcl_eval._llm_response_generation")
    evaluation_module = importlib.import_module("bfcl_eval.eval_checker.eval_runner")
    _install_tinyllm_handler(
        model_name=config.model_name,
        served_model=config.served_model,
        gateway_base_url=config.gateway_base_url,
        bearer_token=bearer_token,
        mode=config.mode,
        max_completion_tokens=config.max_completion_tokens,
    )
    result_root = output / "result"
    score_root = output / "score"
    categories = [item.category for item in config.categories]
    arguments = SimpleNamespace(
        model=[config.model_name],
        test_category=categories,
        temperature=config.temperature,
        include_input_log=False,
        exclude_state_log=False,
        num_gpus=1,
        num_threads=config.num_threads,
        gpu_memory_utilization=0.0,
        backend="vllm",
        skip_server_setup=True,
        local_model_path=None,
        result_dir=result_root,
        allow_overwrite=False,
        run_ids=False,
    )
    generation_module.main(arguments)
    _validate_generation_results(
        config=config,
        result_root=result_root,
    )
    evaluation_module.main([config.model_name], categories, result_root, score_root)
    return result_root, score_root


def _validate_generation_results(*, config: Any, result_root: Path) -> None:
    """Refuse to score incomplete BFCL generations or swallowed endpoint failures."""

    model_directory = config.model_name.replace("/", "_")
    model_root = result_root / model_directory
    total_records = 0
    all_ids: set[str] = set()
    for spec in config.categories:
        path = model_root / f"BFCL_v3_{spec.category}_result.json"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(f"BFCL generation result is missing: {spec.category}") from exc
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"BFCL generation result is invalid: {spec.category} line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"BFCL generation record is not an object: {spec.category} line {line_number}"
                )
            records.append(record)
        if len(records) != spec.item_count:
            raise RuntimeError(
                f"BFCL category {spec.category} has {len(records)} generated items; "
                f"expected {spec.item_count}"
            )
        for record in records:
            item_id = record.get("id")
            if not isinstance(item_id, str) or not item_id:
                raise RuntimeError(f"BFCL category {spec.category} has a missing item id")
            if item_id in all_ids:
                raise RuntimeError(f"BFCL generation has a duplicate item id: {item_id}")
            all_ids.add(item_id)
            if "result" not in record:
                raise RuntimeError(f"BFCL generation result is missing for item: {item_id}")
            result = record["result"]
            if "traceback" in record or (
                isinstance(result, str) and result.lstrip().startswith("Error during inference:")
            ):
                raise RuntimeError(f"BFCL endpoint inference failed for item: {item_id}")
        total_records += len(records)
    expected_total = sum(spec.item_count for spec in config.categories)
    if total_records != expected_total:
        raise RuntimeError(f"BFCL generated {total_records} items; expected {expected_total}")


def _summarize(
    *,
    config: Any,
    result_root: Path,
    score_root: Path,
    model_id: str,
    model_artifact_sha256: str,
) -> BFCLCoreProfileSummary:
    model_directory = config.model_name.replace("/", "_")
    category_results: list[BFCLCategoryResult] = []
    for spec in config.categories:
        path = score_root / model_directory / f"BFCL_v3_{spec.category}_score.json"
        try:
            with path.open(encoding="utf-8") as handle:
                header = json.loads(next(handle))
            if not isinstance(header, dict):
                raise TypeError("BFCL score header is not an object")
            correct = int(header["correct_count"])
            total = int(header["total_count"])
        except (
            OSError,
            StopIteration,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(f"BFCL score is missing or invalid: {spec.category}") from exc
        if total != spec.item_count:
            raise RuntimeError(
                f"BFCL category {spec.category} has {total} items; expected {spec.item_count}"
            )
        category_results.append(
            BFCLCategoryResult(
                category=spec.category,
                item_count=total,
                correct_items=correct,
                accuracy_basis_points=round(correct * 10_000 / total),
                source_score_sha256=_sha256(path),
            )
        )
    correct_items = sum(item.correct_items for item in category_results)
    return BFCLCoreProfileSummary(
        profile_name=config.profile_name,
        bfcl_tag=config.bfcl_tag,
        bfcl_commit=config.bfcl_commit,
        evaluated_at=datetime.now(UTC),
        model_id=model_id,
        model_artifact_sha256=model_artifact_sha256,
        endpoint_handler="tinyllm-openai-chat-completions-v1",
        categories=tuple(category_results),
        total_items=1840,
        correct_items=correct_items,
        overall_accuracy_basis_points=round(correct_items * 10_000 / 1840),
        raw_results_sha256=_tree_sha256((result_root, score_root)),
        completed=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval/m9_bfcl_core.yaml"))
    parser.add_argument("--bfcl-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-artifact-sha256", required=True)
    args: Namespace = parser.parse_args(argv)
    try:
        config = load_bfcl_profile_config(args.config)
    except AgentEvalConfigError as exc:
        parser.error(str(exc))
    if not args.bfcl_checkout.is_absolute() or not args.output.is_absolute():
        parser.error("BFCL checkout and output paths must be absolute")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.bfcl_checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != config.bfcl_commit:
        parser.error("BFCL checkout is not at the frozen v1.3 commit")
    package_root = args.bfcl_checkout / "berkeley-function-call-leaderboard"
    if not package_root.is_dir():
        parser.error("BFCL package directory is missing")
    sys.path.insert(0, str(package_root))
    token = os.environ.get(config.bearer_token_env, "")
    if len(token) < 32:
        parser.error(f"{config.bearer_token_env} must contain a 32-character token")
    if len(args.model_artifact_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.model_artifact_sha256
    ):
        parser.error("model Artifact SHA256 is invalid")
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root, score_root = _run_bfcl(
        config=config,
        output=args.output,
        bearer_token=token,
    )
    summary = _summarize(
        config=config,
        result_root=result_root,
        score_root=score_root,
        model_id=args.model_id,
        model_artifact_sha256=args.model_artifact_sha256,
    )
    _atomic_json(args.output / "tinyllm-bfcl-summary.json", summary.to_dict())
    print(summary.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
