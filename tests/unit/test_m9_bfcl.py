from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.run_m9_bfcl import _install_tinyllm_handler, _summarize
from tinyllm.agent_eval.config import load_bfcl_profile_config


def test_bfcl_profile_is_exactly_1840_offline_tasks() -> None:
    config = load_bfcl_profile_config(Path("configs/eval/m9_bfcl_core.yaml"))

    assert config.profile_name == "TinyLLM BFCL v1.3 Offline Core Profile"
    assert config.bfcl_commit == "ea13468e4423454d0c213704fb87cf7cb3990433"
    assert sum(item.item_count for item in config.categories) == 1840
    assert {item.category for item in config.categories} == {
        "simple",
        "multiple",
        "parallel",
        "parallel_multiple",
        "irrelevance",
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
    }
    assert "live" in config.excluded_categories
    assert "multi_turn_long_context" in config.excluded_categories


def test_bfcl_summary_imports_original_category_counts(tmp_path: Path) -> None:
    config = load_bfcl_profile_config(Path("configs/eval/m9_bfcl_core.yaml"))
    result_root = tmp_path / "result"
    score_root = tmp_path / "score"
    model_root = score_root / "TinyLLM_Qwen3-FC"
    result_root.mkdir()
    model_root.mkdir(parents=True)
    (result_root / "raw.json").write_text("[]\n", encoding="utf-8")
    expected_correct = 0
    for spec in config.categories:
        correct = spec.item_count // 2
        expected_correct += correct
        path = model_root / f"BFCL_v3_{spec.category}_score.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "accuracy": 0.5,
                        "correct_count": correct,
                        "total_count": spec.item_count,
                    }
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    summary = _summarize(
        config=config,
        result_root=result_root,
        score_root=score_root,
        model_id="qwen3-0-6b-m7-fa678d92",
        model_artifact_sha256="a" * 64,
    )

    assert summary.total_items == 1840
    assert summary.correct_items == expected_correct
    assert summary.overall_accuracy_basis_points == 5000
    assert len(summary.categories) == 8
    assert summary.completed is True


def test_bfcl_openai_client_ignores_host_proxy(monkeypatch: Any) -> None:
    mapping: dict[str, Any] = {}
    client_arguments: dict[str, Any] = {}

    class BaseHandler:
        def __init__(self, name: str, temperature: float) -> None:
            self.name = name
            self.temperature = temperature

    class OpenAI:
        def __init__(self, **kwargs: Any) -> None:
            client_arguments.update(kwargs)

    modules = {
        "bfcl_eval.constants.model_config": SimpleNamespace(
            MODEL_CONFIG_MAPPING=mapping,
            ModelConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        "bfcl_eval.model_handler.api_inference.openai_completion": SimpleNamespace(
            OpenAICompletionsHandler=BaseHandler
        ),
        "openai": SimpleNamespace(
            OpenAI=OpenAI,
            DefaultHttpxClient=lambda **kwargs: {"transport": kwargs},
        ),
    }
    monkeypatch.setattr("scripts.run_m9_bfcl.importlib.import_module", lambda name: modules[name])

    _install_tinyllm_handler(
        model_name="TinyLLM/Qwen3-FC",
        served_model="production",
        gateway_base_url="http://127.0.0.1:8000",
        bearer_token="x" * 64,
        mode="nonthinking",
        max_completion_tokens=512,
    )
    mapping["TinyLLM/Qwen3-FC"].model_handler("TinyLLM/Qwen3-FC", 0.0)

    assert client_arguments["http_client"] == {"transport": {"trust_env": False}}
