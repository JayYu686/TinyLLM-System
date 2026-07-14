from __future__ import annotations

import json
from pathlib import Path

from tinyllm.data import load_m2_tokenization_config


def test_committed_qwen_tokenizer_smoke_is_internally_consistent() -> None:
    evidence = json.loads(
        Path("reports/m2/raw/qwen3_tokenizer_smoke.json").read_text(encoding="utf-8")
    )
    config = load_m2_tokenization_config(Path("configs/data/m2_tokenization.yaml"))

    assert evidence["status"] == "pass"
    assert evidence["tokenizer"] == config.tokenizer.to_dict()
    assert evidence["template"] == config.template.to_dict()
    assert evidence["accepted_samples"] == len(evidence["samples"]) == 2
    assert evidence["rejected_samples"] == len(evidence["rejected"]) == 0
    for sample in evidence["samples"]:
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        assert len(input_ids) == len(labels) == sample["token_count"]
        assert sample["supervised_token_count"] == sum(label != -100 for label in labels)
        assert all(
            label == -100 or label == token for token, label in zip(input_ids, labels, strict=True)
        )
        assert labels.count(config.tokenizer.eos_token_id) == sample["supervised_eos_count"] == 1
        assert labels[-1] == -100
