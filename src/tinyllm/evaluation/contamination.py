"""Deterministic evaluation identity and exact Train-contamination scanning."""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol, cast

import yaml
from pydantic import ValidationError

from tinyllm.data.acquisition import (
    QWEN3_TOKENIZER_ARTIFACT,
    QWEN3_TOKENIZER_CONFIG_ARTIFACT,
    acquire_pinned_artifact,
)
from tinyllm.data.packing_schema import M2DatasetManifest, PackedSequence
from tinyllm.data.registry import open_registered_dataset
from tinyllm.data.schema import ImportedMessage
from tinyllm.data.tokenization import (
    OffsetTokenizer,
    TokenizersBackend,
    load_m2_tokenization_config,
    tokenize_messages,
)
from tinyllm.data.tokenization_schema import M2TokenizationConfig
from tinyllm.evaluation.schema import (
    CategoryCounts,
    ContaminationMatch,
    ContaminationReport,
    EvaluationBuildConfig,
    EvaluationItem,
    EvaluationSetManifest,
    LanguageCounts,
)


class EvaluationContractError(ValueError):
    """Raised when evaluation identity or contamination inputs violate M2.4."""


class VerifiedPackSource(Protocol):
    """Minimal already-verified Registry interface used by the pure scanner."""

    @property
    def manifest(self) -> M2DatasetManifest:
        """Return the immutable registered Dataset Manifest."""

    def iter_packs(self) -> Iterator[PackedSequence]:
        """Yield validated Packs in deterministic order."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sequence_hash(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = _canonical_json(value)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def fingerprint_token_sequence(token_ids: Iterable[int]) -> str:
    """Hash non-negative Token IDs using the frozen length-delimited v1 encoding."""

    materialized = tuple(token_ids)
    if not materialized:
        raise EvaluationContractError("Token sequence fingerprint input cannot be empty")
    if any(token_id < 0 or token_id > 2**32 - 1 for token_id in materialized):
        raise EvaluationContractError("Token sequence fingerprint input is outside UInt32")
    digest = hashlib.sha256()
    digest.update(len(materialized).to_bytes(8, "big"))
    for token_id in materialized:
        digest.update(struct.pack(">I", token_id))
    return digest.hexdigest()


def load_evaluation_build_config(path: Path) -> EvaluationBuildConfig:
    """Load one strict evaluation-set YAML without accepting unknown fields."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise EvaluationContractError("evaluation config root must be a mapping")
        return EvaluationBuildConfig.model_validate(payload)
    except OSError as exc:
        raise EvaluationContractError("cannot read evaluation config") from exc
    except yaml.YAMLError as exc:
        raise EvaluationContractError("evaluation config is invalid YAML") from exc
    except ValidationError as exc:
        messages = []
        for error in exc.errors(include_url=False, include_context=False):
            location = ".".join(str(part) for part in error["loc"])
            messages.append(f"{location}: {error['msg']}")
        raise EvaluationContractError("invalid evaluation config: " + "; ".join(messages)) from exc


def load_evaluation_items(path: Path) -> tuple[EvaluationItem, ...]:
    """Read strict public JSONL items without echoing rejected content."""

    if not path.is_file() or path.is_symlink():
        raise EvaluationContractError("evaluation JSONL is missing or not a regular file")
    items: list[EvaluationItem] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise EvaluationContractError(
                        f"evaluation JSONL has a blank line at {line_number}"
                    )
                try:
                    payload = json.loads(line)
                    item = EvaluationItem.model_validate(payload)
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise EvaluationContractError(
                        f"evaluation JSONL has an invalid item at line {line_number}"
                    ) from exc
                items.append(item)
    except OSError as exc:
        raise EvaluationContractError("cannot read evaluation JSONL") from exc
    if not items:
        raise EvaluationContractError("evaluation JSONL cannot be empty")
    identifiers = [item.id for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise EvaluationContractError("evaluation item IDs must be unique")
    return tuple(sorted(items, key=lambda item: item.id))


def build_evaluation_manifest(
    items: Iterable[EvaluationItem],
    *,
    config: EvaluationBuildConfig,
) -> EvaluationSetManifest:
    """Build timestamp-free identity after enforcing frozen distribution counts."""

    ordered = tuple(sorted(items, key=lambda item: item.id))
    if not ordered or len({item.id for item in ordered}) != len(ordered):
        raise EvaluationContractError("evaluation items must be non-empty with unique IDs")
    if len(ordered) != config.expected_items:
        raise EvaluationContractError("evaluation item count does not match build config")
    language_counts = Counter(item.language for item in ordered)
    category_counts = Counter(item.category for item in ordered)
    observed_languages = LanguageCounts(
        en=language_counts["en"],
        zh=language_counts["zh"],
    )
    observed_categories = CategoryCounts(
        config=category_counts["config"],
        json_items=category_counts["json"],
        linux=category_counts["linux"],
        logs=category_counts["logs"],
        python=category_counts["python"],
        refusal=category_counts["refusal"],
        short_code=category_counts["short_code"],
    )
    if observed_languages != config.language_counts:
        raise EvaluationContractError("evaluation language counts do not match build config")
    if observed_categories != config.category_counts:
        raise EvaluationContractError("evaluation category counts do not match build config")
    scorer_counts = cast(
        dict[str, int],
        dict(sorted(Counter(item.scorer.kind for item in ordered).items())),
    )
    items_sha256 = _sequence_hash(item.to_dict() for item in ordered)
    config_sha256 = _content_hash(config.to_dict())
    content_sha256 = _content_hash(
        {
            "config_sha256": config_sha256,
            "items_sha256": items_sha256,
        }
    )
    return EvaluationSetManifest(
        suite_name=config.suite_name,
        suite_version=f"{config.version_prefix}-{content_sha256[:8]}",
        content_sha256=content_sha256,
        items_sha256=items_sha256,
        config_sha256=config_sha256,
        item_count=len(ordered),
        language_counts=observed_languages,
        category_counts=observed_categories,
        scorer_counts=scorer_counts,
        tokenizer=config.tokenizer,
        template=config.template,
        max_sequence_length=config.max_sequence_length,
        decoding=config.decoding,
        contamination=config.contamination,
    )


def _prompt_boundary(labels: tuple[int, ...]) -> int:
    try:
        return next(index for index, label in enumerate(labels) if label != -100)
    except StopIteration as exc:
        raise EvaluationContractError("conversation has no supervised Assistant Token") from exc


def _build_train_indexes(
    dataset: VerifiedPackSource,
) -> tuple[dict[str, set[str]], dict[str, set[str]], int]:
    full_index: dict[str, set[str]] = defaultdict(set)
    prompt_index: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    for pack in dataset.iter_packs():
        if pack.split != "train":
            continue
        cursor = 0
        for sample_id, token_count in zip(
            pack.sample_ids,
            pack.sample_token_counts,
            strict=True,
        ):
            end = cursor + token_count
            if sample_id in seen_ids:
                raise EvaluationContractError("Train sample appears more than once in Registry")
            seen_ids.add(sample_id)
            input_ids = pack.input_ids[cursor:end]
            labels = pack.labels[cursor:end]
            boundary = _prompt_boundary(labels)
            sample_id_sha256 = hashlib.sha256(sample_id.encode()).hexdigest()
            full_index[fingerprint_token_sequence(input_ids)].add(sample_id_sha256)
            prompt_index[fingerprint_token_sequence(input_ids[:boundary])].add(sample_id_sha256)
            cursor = end
    return full_index, prompt_index, len(seen_ids)


def _fingerprint_evaluation_item(
    item: EvaluationItem,
    *,
    backend: OffsetTokenizer,
    tokenization_config: M2TokenizationConfig,
) -> tuple[str, str]:
    messages = tuple(
        ImportedMessage(role=message.role, content=message.content)
        for message in item.prompt_messages
    ) + (ImportedMessage(role="assistant", content=item.reference_answer),)
    tokenization = tokenize_messages(
        messages,
        backend=backend,
        config=tokenization_config,
    )
    if len(tokenization.input_ids) > tokenization_config.max_sequence_length:
        raise EvaluationContractError(f"evaluation item exceeds maximum Token length: {item.id}")
    boundary = _prompt_boundary(tokenization.labels)
    return (
        fingerprint_token_sequence(tokenization.input_ids),
        fingerprint_token_sequence(tokenization.input_ids[:boundary]),
    )


def scan_exact_contamination(
    dataset: VerifiedPackSource,
    items: Iterable[EvaluationItem],
    *,
    manifest: EvaluationSetManifest,
    backend: OffsetTokenizer,
    tokenization_config: M2TokenizationConfig,
) -> ContaminationReport:
    """Compare full/reference and prompt-prefix fingerprints against verified Train Packs."""

    ordered = tuple(sorted(items, key=lambda item: item.id))
    if len(ordered) != manifest.item_count or len({item.id for item in ordered}) != len(ordered):
        raise EvaluationContractError("evaluation items do not match manifest item identity")
    observed_items_sha256 = _sequence_hash(item.to_dict() for item in ordered)
    if observed_items_sha256 != manifest.items_sha256:
        raise EvaluationContractError("evaluation item content does not match manifest")
    if manifest.tokenizer != tokenization_config.tokenizer or (
        manifest.template != tokenization_config.template
    ):
        raise EvaluationContractError(
            "evaluation manifest and Tokenization config are incompatible"
        )
    if manifest.max_sequence_length != tokenization_config.max_sequence_length:
        raise EvaluationContractError("evaluation and Tokenization maximum lengths differ")
    if dataset.manifest.tokenizer != manifest.tokenizer or dataset.manifest.template != (
        manifest.template
    ):
        raise EvaluationContractError("Dataset and evaluation Tokenizer/Template identities differ")
    if dataset.manifest.max_sequence_length != manifest.max_sequence_length:
        raise EvaluationContractError("Dataset and evaluation maximum lengths differ")

    full_index, prompt_index, training_samples = _build_train_indexes(dataset)
    matches: list[ContaminationMatch] = []
    for item in ordered:
        full_fingerprint, prompt_fingerprint = _fingerprint_evaluation_item(
            item,
            backend=backend,
            tokenization_config=tokenization_config,
        )
        for sample_id_sha256 in sorted(full_index.get(full_fingerprint, set())):
            matches.append(
                ContaminationMatch(
                    evaluation_item_id=item.id,
                    match_kind="full_sequence",
                    fingerprint_sha256=full_fingerprint,
                    training_sample_id_sha256=sample_id_sha256,
                )
            )
        for sample_id_sha256 in sorted(prompt_index.get(prompt_fingerprint, set())):
            matches.append(
                ContaminationMatch(
                    evaluation_item_id=item.id,
                    match_kind="prompt_prefix",
                    fingerprint_sha256=prompt_fingerprint,
                    training_sample_id_sha256=sample_id_sha256,
                )
            )
    ordered_matches = tuple(
        sorted(
            matches,
            key=lambda item: (
                item.evaluation_item_id,
                item.match_kind,
                item.training_sample_id_sha256,
            ),
        )
    )
    contaminated_items = len({item.evaluation_item_id for item in ordered_matches})
    return ContaminationReport(
        status="contaminated" if ordered_matches else "clean",
        fingerprint_algorithm="token-sequence-sha256-v1",
        near_dedup="not_evaluated",
        evaluation_suite_version=manifest.suite_version,
        evaluation_content_sha256=manifest.content_sha256,
        dataset_version=dataset.manifest.dataset_version,
        dataset_content_sha256=dataset.manifest.content_sha256,
        checked_evaluation_items=len(ordered),
        checked_training_samples=training_samples,
        contaminated_items=contaminated_items,
        full_sequence_matches=sum(item.match_kind == "full_sequence" for item in ordered_matches),
        prompt_prefix_matches=sum(item.match_kind == "prompt_prefix" for item in ordered_matches),
        matches=ordered_matches,
    )


def run_contamination_check(
    *,
    artifact_root: Path,
    dataset_version: str,
    evaluation_set_path: Path,
    evaluation_config_path: Path,
    tokenization_config_path: Path,
) -> ContaminationReport:
    """Load fixed inputs, verify Registry/cache, and return a path-free report."""

    config = load_evaluation_build_config(evaluation_config_path)
    items = load_evaluation_items(evaluation_set_path)
    manifest = build_evaluation_manifest(items, config=config)
    tokenization_config = load_m2_tokenization_config(tokenization_config_path)
    if config.tokenizer != tokenization_config.tokenizer or config.template != (
        tokenization_config.template
    ):
        raise EvaluationContractError("evaluation and Tokenization configs are incompatible")
    if config.max_sequence_length != tokenization_config.max_sequence_length:
        raise EvaluationContractError("evaluation and Tokenization maximum lengths differ")
    dataset = open_registered_dataset(
        artifact_root=artifact_root,
        dataset_version=dataset_version,
    )
    tokenizer_json = acquire_pinned_artifact(
        QWEN3_TOKENIZER_ARTIFACT,
        cache_root=artifact_root / "cache",
        offline=True,
    )
    tokenizer_config_json = acquire_pinned_artifact(
        QWEN3_TOKENIZER_CONFIG_ARTIFACT,
        cache_root=artifact_root / "cache",
        offline=True,
    )
    backend = TokenizersBackend.from_files(
        tokenizer_json,
        tokenizer_config_json,
        tokenization_config.tokenizer,
    )
    return scan_exact_contamination(
        dataset,
        items,
        manifest=manifest,
        backend=cast(OffsetTokenizer, backend),
        tokenization_config=tokenization_config,
    )
