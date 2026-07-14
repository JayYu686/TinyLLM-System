#!/usr/bin/env python3
"""Reproduce the public synthetic M2.4a exact-contamination smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_m2_packing_smoke import _processing_manifest, _source_manifest
from tinyllm.data import (
    M2_ACQUISITION_MANIFEST,
    DatasetLineage,
    ImportedMessage,
    OffsetTokenizer,
    TokenEncoding,
    TokenizationBatch,
    TokenizedSample,
    build_m2_dataset,
    load_m2_packing_config,
    load_m2_tokenization_config,
    open_registered_dataset,
    register_dataset,
    tokenize_messages,
)
from tinyllm.evaluation import (
    AuthoredProvenance,
    CategoryCounts,
    ContaminationPolicy,
    DecodingConfig,
    EvaluationBuildConfig,
    EvaluationItem,
    EvaluationPromptMessage,
    ExactMatchScorer,
    LanguageCounts,
    build_evaluation_manifest,
    scan_exact_contamination,
)


class PinnedCharacterTokenizer(OffsetTokenizer):
    """Small deterministic backend exposing the frozen Qwen special-token identity."""

    def __init__(self) -> None:
        self._special = {
            "<|endoftext|>": 151_643,
            "<|im_start|>": 151_644,
            "<|im_end|>": 151_645,
        }

    @property
    def vocab_size(self) -> int:
        return 151_669

    def token_to_id(self, token: str) -> int | None:
        return self._special.get(token)

    def encode(self, text: str) -> TokenEncoding:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        index = 0
        while index < len(text):
            matched = next(
                (token for token in self._special if text.startswith(token, index)),
                None,
            )
            if matched is not None:
                ids.append(self._special[matched])
                offsets.append((index, index + len(matched)))
                index += len(matched)
                continue
            ids.append(100 + ord(text[index]))
            offsets.append((index, index + 1))
            index += 1
        return TokenEncoding(ids=tuple(ids), offsets=tuple(offsets))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _training_samples(project_root: Path) -> tuple[TokenizedSample, ...]:
    tokenization_config = load_m2_tokenization_config(
        project_root / "configs/data/m2_tokenization.yaml"
    )
    backend = PinnedCharacterTokenizer()
    samples: list[TokenizedSample] = []
    strata = (
        ("oasst1", "zh", 3),
        ("oasst1", "en", 3),
        ("commitpackft", "en", 4),
    )
    for source, language, count in strata:
        for index in range(count):
            global_index = len(samples)
            sample_id = f"{source}:smoke-{language}-{index:02d}"
            messages = (
                ImportedMessage(role="user", content=f"prompt-{global_index:02d}"),
                ImportedMessage(role="assistant", content=f"answer-{global_index:02d}"),
            )
            tokens = tokenize_messages(
                messages,
                backend=backend,
                config=tokenization_config,
            )
            samples.append(
                TokenizedSample.model_validate(
                    {
                        "id": sample_id,
                        "source": source,
                        "split": "train",
                        "component_id": _digest(f"component-{sample_id}"),
                        "group_keys": (f"{source}:group-{global_index:02d}",),
                        "origin_sample_ids": (sample_id,),
                        "origin_record_sha256s": (_digest(f"record-{sample_id}"),),
                        "language": language,
                        "license": "apache-2.0" if source == "oasst1" else "mit",
                        "content_sha256": _digest(f"content-{sample_id}"),
                        "rendered_sha256": tokens.rendered_sha256,
                        "tokenizer_sha256": tokenization_config.tokenizer.tokenizer_sha256,
                        "template_sha256": tokenization_config.template.template_sha256,
                        "max_sequence_length": tokenization_config.max_sequence_length,
                        "input_ids": tokens.input_ids,
                        "labels": tokens.labels,
                        "token_count": len(tokens.input_ids),
                        "supervised_token_count": sum(label != -100 for label in tokens.labels),
                    }
                )
            )
    return tuple(samples)


def _item(suffix: int, *, prompt: str, answer: str) -> EvaluationItem:
    return EvaluationItem(
        id=f"domain-python-{suffix:03d}",
        language="en",
        category="python",
        prompt_messages=(EvaluationPromptMessage(role="user", content=prompt),),
        reference_answer=answer,
        scorer=ExactMatchScorer(
            kind="exact_match",
            accepted_answers=(answer,),
            case_sensitive=True,
            strip_outer_whitespace=True,
        ),
        provenance=AuthoredProvenance(
            origin="tinyllm-authored",
            license="Apache-2.0",
            redistribution_allowed=True,
            source_note="Public synthetic M2.4a smoke fixture.",
        ),
    )


def _evaluation_config(project_root: Path, *, item_count: int) -> EvaluationBuildConfig:
    tokenization = load_m2_tokenization_config(project_root / "configs/data/m2_tokenization.yaml")
    return EvaluationBuildConfig(
        suite_name="tinyllm-smoke",
        version_prefix="tinyllm-smoke-v1",
        expected_items=item_count,
        language_counts=LanguageCounts(en=item_count, zh=0),
        category_counts=CategoryCounts(
            config=0,
            json_items=0,
            linux=0,
            logs=0,
            python=item_count,
            refusal=0,
            short_code=0,
        ),
        tokenizer=tokenization.tokenizer,
        template=tokenization.template,
        max_sequence_length=tokenization.max_sequence_length,
        decoding=DecodingConfig(
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=64,
            seed=42,
        ),
        contamination=ContaminationPolicy(
            split="train",
            full_sequence=True,
            prompt_prefix=True,
            near_dedup=False,
            fingerprint_algorithm="token-sequence-sha256-v1",
        ),
    )


def run_smoke(project_root: Path) -> dict[str, object]:
    """Register synthetic Packs and verify full, Prompt-only, and clean comparisons."""

    samples = _training_samples(project_root)
    tokenization_config = load_m2_tokenization_config(
        project_root / "configs/data/m2_tokenization.yaml"
    )
    packing_config = load_m2_packing_config(project_root / "configs/data/m2_packing.yaml")
    processing_manifest = _processing_manifest(samples)
    source_manifests = (
        _source_manifest("oasst1", samples),
        _source_manifest("commitpackft", samples),
    )
    build = build_m2_dataset(
        TokenizationBatch(samples=samples, rejected=()),
        tokenization_config=tokenization_config,
        packing_config=packing_config,
        processing_manifest=processing_manifest,
        source_manifests=source_manifests,
    )
    lineage = DatasetLineage(
        acquisition_manifest=M2_ACQUISITION_MANIFEST,
        source_manifests=source_manifests,
        processing_manifest=processing_manifest,
        tokenization_config=tokenization_config,
        packing_config=packing_config,
        oasst1_rejected=(),
        commitpackft_rejected=(),
        processing_rejected=(),
    )
    exact = _item(1, prompt="prompt-03", answer="answer-03")
    prompt_only = _item(2, prompt="prompt-03", answer="changed-03")
    clean = _item(3, prompt="clean---03", answer="answer-new")
    contaminated_items = (exact, prompt_only, clean)
    contaminated_manifest = build_evaluation_manifest(
        contaminated_items,
        config=_evaluation_config(project_root, item_count=3),
    )
    clean_manifest = build_evaluation_manifest(
        (clean,),
        config=_evaluation_config(project_root, item_count=1),
    )
    with TemporaryDirectory(prefix="tinyllm-m2-contamination-smoke-") as temporary:
        artifact_root = Path(temporary)
        register_dataset(
            build,
            artifact_root=artifact_root,
            lineage=lineage,
            git_commit="a" * 40,
            git_dirty=False,
            shard_token_limit=1024,
        )
        reopened = open_registered_dataset(
            artifact_root=artifact_root,
            dataset_version=build.manifest.dataset_version,
        )
        contaminated_report = scan_exact_contamination(
            reopened,
            contaminated_items,
            manifest=contaminated_manifest,
            backend=PinnedCharacterTokenizer(),
            tokenization_config=tokenization_config,
        )
        clean_report = scan_exact_contamination(
            reopened,
            (clean,),
            manifest=clean_manifest,
            backend=PinnedCharacterTokenizer(),
            tokenization_config=tokenization_config,
        )
    serialized = contaminated_report.model_dump_json()
    if any(sample.id in serialized for sample in samples):
        raise RuntimeError("contamination report leaked a raw training Sample ID")
    if (
        contaminated_report.full_sequence_matches != 1
        or contaminated_report.prompt_prefix_matches != 2
        or contaminated_report.contaminated_items != 2
        or clean_report.status != "clean"
    ):
        raise RuntimeError("synthetic contamination smoke produced unexpected match counts")
    return {
        "status": "pass",
        "scope": "public-synthetic-evaluation-and-token-arrays",
        "fingerprint_algorithm": contaminated_report.fingerprint_algorithm,
        "near_dedup": contaminated_report.near_dedup,
        "registered_dataset_verified": True,
        "checked_training_samples": contaminated_report.checked_training_samples,
        "checked_evaluation_items": contaminated_report.checked_evaluation_items,
        "full_sequence_matches": contaminated_report.full_sequence_matches,
        "prompt_prefix_matches": contaminated_report.prompt_prefix_matches,
        "contaminated_items": contaminated_report.contaminated_items,
        "raw_training_sample_ids_published": False,
        "clean_control_status": clean_report.status,
        "clean_control_matches": len(clean_report.matches),
    }


def main() -> int:
    """Run the smoke and print stable JSON to standard output."""

    project_root = Path(__file__).resolve().parents[1]
    print(json.dumps(run_smoke(project_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
