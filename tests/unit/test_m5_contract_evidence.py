from __future__ import annotations

import json
from pathlib import Path
from typing import cast


def test_m5_contract_report_and_schema_preserve_the_reviewed_boundary() -> None:
    report = Path("reports/m5/m5_dual_mode_contract.md").read_text(encoding="utf-8")
    adr = Path("docs/adr/0005-qwen3-gqa-dual-mode-reasoning.md").read_text(encoding="utf-8")
    schema = json.loads(Path("schemas/m5-sft-config-v1.schema.json").read_text(encoding="utf-8"))
    definitions = cast(dict[str, object], schema["$defs"])
    model = cast(dict[str, object], definitions["M5ModelConfig"])
    reasoning = cast(dict[str, object], definitions["M5ReasoningConfig"])
    model_properties = cast(dict[str, object], model["properties"])
    reasoning_properties = cast(dict[str, object], reasoning["properties"])

    assert "M5 整体仍为 `IN_PROGRESS`" in report
    assert "没有启动 GPU 作业" in report
    assert "不产生 Loss、准确率、吞吐、显存或 Candidate 晋级结论" in report
    assert "442 passed，2 deselected" in report
    assert "保留固定 Qwen3 Revision 的原生 GQA，不实现或转换 MLA" in adr
    assert cast(dict[str, object], schema["properties"])["config_kind"] == {
        "const": "qwen_sft",
        "title": "Config Kind",
        "type": "string",
    }
    assert model_properties["attention_architecture"] == {
        "const": "gqa",
        "title": "Attention Architecture",
        "type": "string",
    }
    assert reasoning_properties["thinking_template_id"] == {
        "const": "qwen3-chatml-thinking-v1",
        "title": "Thinking Template Id",
        "type": "string",
    }
