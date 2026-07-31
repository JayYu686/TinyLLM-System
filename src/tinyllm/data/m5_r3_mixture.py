"""Build the corrected, label-aware exact-token M5.2-R3 mixture."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from pydantic import ValidationError

from tinyllm.data.m5_mixture import (
    M5MixtureError,
    M5MixtureSequence,
    _array_content_hash,
    _nonthinking_candidates,
    _thinking_candidates_with_metadata,
    _trim_supervision,
    _with_mode,
    load_verified_reasoning_pilot,
    open_m5_ablation_mixture,
    select_exact_supervised_tokens,
    thinking_candidates_from_samples,
)
from tinyllm.data.m5_mixture_schema import M5MixtureArtifactFile
from tinyllm.data.m5_r3_formal_schema import M5R3FormalSourceResult
from tinyllm.data.m5_r3_mixture_schema import (
    M5R3MixtureConfig,
    M5R3MixtureManifest,
)
from tinyllm.data.m5_r3_p1_schema import M5R3P1CandidateAudit
from tinyllm.data.reasoning_schema import ReasoningSample, content_sha256

_SEQUENCE_FILE = "sequences.npz"
_MANIFEST_FILE = "manifest.json"
_COMMIT_FILE = "COMMITTED"


class M5R3MixtureError(M5MixtureError):
    """Raised when R3 selection, lineage, or exact-token construction differs."""


@dataclass(frozen=True, slots=True)
class M5R3TargetedSource:
    """One accepted formal source paired with content-free selection metrics."""

    sample: ReasoningSample
    label: str
    reasoning_tokens: int
    repeated_8gram_basis_points: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_m5_r3_mixture_config(path: Path) -> M5R3MixtureConfig:
    """Load one strict YAML R3 mixture contract."""

    if path.suffix not in {".yaml", ".yml"}:
        raise M5R3MixtureError("M5 R3 mixture config must use YAML")
    try:
        return M5R3MixtureConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except OSError as exc:
        raise M5R3MixtureError("M5 R3 mixture config cannot be read") from exc
    except (yaml.YAMLError, ValidationError) as exc:
        raise M5R3MixtureError("M5 R3 mixture config violates its schema") from exc


def m5_r3_mixture_config_sha256(config: M5R3MixtureConfig) -> str:
    """Hash the resolved, path-free R3 mixture policy."""

    return content_sha256(config.to_dict())


def load_verified_m5_r3_sources(
    *,
    config: M5R3MixtureConfig,
    formal_result_path: Path,
    formal_raw_artifact: Path,
) -> tuple[M5R3TargetedSource, ...]:
    """Revalidate the public gate and private accepted sources before selection."""

    if _sha256_file(formal_result_path) != config.formal_source.result_sha256:
        raise M5R3MixtureError("M5 R3 formal public result SHA256 differs")
    if _sha256_file(formal_raw_artifact) != config.formal_source.raw_artifact_sha256:
        raise M5R3MixtureError("M5 R3 formal private artifact SHA256 differs")
    try:
        result = M5R3FormalSourceResult.model_validate_json(
            formal_result_path.read_text(encoding="utf-8")
        )
        payload = cast(
            dict[str, object],
            json.loads(formal_raw_artifact.read_text(encoding="utf-8")),
        )
        samples = tuple(
            ReasoningSample.model_validate(item) for item in cast(list[object], payload["samples"])
        )
        audits = tuple(
            M5R3P1CandidateAudit.model_validate(item)
            for item in cast(list[object], payload["audits"])
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
        raise M5R3MixtureError("M5 R3 formal source evidence cannot be parsed") from exc
    if (
        result.status != "pass"
        or not result.formal_source_expansion_complete
        or not result.r3_mixture_authorized
        or result.r3_training_authorized
        or result.raw_artifact_sha256 != config.formal_source.raw_artifact_sha256
        or result.accepted_samples_sha256 != config.formal_source.accepted_samples_sha256
        or content_sha256([sample.to_dict() for sample in samples])
        != config.formal_source.accepted_samples_sha256
        or len(samples) != result.accepted_samples
    ):
        raise M5R3MixtureError("M5 R3 formal source gate or content identity differs")
    accepted_audits = {audit.task_id: audit for audit in audits if audit.status == "accepted"}
    if len(accepted_audits) != len(samples) or len(audits) != result.input_tasks:
        raise M5R3MixtureError("M5 R3 formal audit coverage differs")
    sources: list[M5R3TargetedSource] = []
    for sample in samples:
        audit = accepted_audits.get(sample.task_id)
        try:
            decoded = cast(dict[str, str], json.loads(sample.final_answer))
            label = next(iter(decoded.values()))
        except (json.JSONDecodeError, StopIteration) as exc:
            raise M5R3MixtureError("M5 R3 accepted answer cannot be decoded") from exc
        if (
            audit is None
            or audit.reasoning_tokens is None
            or audit.repeated_8gram_basis_points is None
            or len(decoded) != 1
        ):
            raise M5R3MixtureError("M5 R3 accepted source audit is incomplete")
        sources.append(
            M5R3TargetedSource(
                sample=sample,
                label=label,
                reasoning_tokens=audit.reasoning_tokens,
                repeated_8gram_basis_points=audit.repeated_8gram_basis_points,
            )
        )
    return tuple(sources)


def select_m5_r3_targeted_sources(
    sources: tuple[M5R3TargetedSource, ...],
    *,
    config: M5R3MixtureConfig,
) -> tuple[M5R3TargetedSource, ...]:
    """Apply the frozen family/language/label quotas without consulting Dev results."""

    selected: list[M5R3TargetedSource] = []
    for family, languages in config.selection.quotas.items():
        for language, labels in languages.items():
            for label, required in labels.items():
                eligible = sorted(
                    (
                        source
                        for source in sources
                        if source.sample.task_family == family
                        and source.sample.language == language
                        and source.label == label
                    ),
                    key=lambda source: (
                        source.reasoning_tokens,
                        source.repeated_8gram_basis_points,
                        source.sample.id,
                    ),
                )
                if len(eligible) < required:
                    raise M5R3MixtureError(
                        f"M5 R3 source stratum {family}/{language}/{label} is insufficient"
                    )
                selected.extend(eligible[:required])
    ordered = tuple(sorted(selected, key=lambda source: source.sample.id))
    if len(ordered) != 160 or len({source.sample.id for source in ordered}) != 160:
        raise M5R3MixtureError("M5 R3 targeted selection is incomplete or duplicated")
    return ordered


def select_exact_supervised_tokens_capped(
    candidates: tuple[M5MixtureSequence, ...],
    *,
    target: int,
    seed: int,
    max_source_uses: int,
) -> tuple[tuple[M5MixtureSequence, ...], int, int, tuple[int, ...]]:
    """Select an exact budget while enforcing a per-source total-use ceiling."""

    if (
        target <= 0
        or not candidates
        or max_source_uses <= 0
        or any(item.supervised_tokens <= 0 for item in candidates)
    ):
        raise M5R3MixtureError("capped exact-token selection requires positive inputs")
    if sum(item.supervised_tokens for item in candidates) * max_source_uses < target:
        raise M5R3MixtureError("M5 R3 exact-token target exceeds the source-use cap")
    rng = random.Random(seed)
    selected: list[M5MixtureSequence] = []
    uses = [0] * len(candidates)
    consumed = 0
    partial = 0
    while consumed < target:
        order = [index for index, count in enumerate(uses) if count < max_source_uses]
        if not order:
            raise M5R3MixtureError("M5 R3 source-use cap exhausted before the exact target")
        rng.shuffle(order)
        for index in order:
            sequence = candidates[index]
            remaining = target - consumed
            if sequence.supervised_tokens > remaining:
                sequence = _trim_supervision(sequence, remaining)
                partial += 1
            selected.append(sequence)
            uses[index] += 1
            consumed += sequence.supervised_tokens
            if consumed == target:
                reuse_count = sum(max(count - 1, 0) for count in uses)
                return tuple(selected), reuse_count, partial, tuple(uses)
    raise AssertionError("unreachable capped exact-token selection state")


def build_m5_r3_mixture(
    *,
    artifact_root: Path,
    raw_pilot_artifact: Path,
    formal_result_path: Path,
    formal_raw_artifact: Path,
    config_path: Path,
    reasoning_config_path: Path,
    tokenizer_config_path: Path,
    model_dir: Path,
    output_root: Path,
) -> M5R3MixtureManifest:
    """Materialize the versioned 700K/150K/150K R3 mixture atomically."""

    config = load_m5_r3_mixture_config(config_path)
    if _sha256_file(raw_pilot_artifact) != config.pilot.raw_artifact_sha256:
        raise M5R3MixtureError("M5 R3 Pilot raw artifact SHA256 differs")
    pilot = load_verified_reasoning_pilot(
        raw_artifact=raw_pilot_artifact,
        reasoning_config=reasoning_config_path,
    )
    if (
        pilot.manifest.dataset_version != config.pilot.dataset_version
        or pilot.manifest.content_sha256 != config.pilot.content_sha256
    ):
        raise M5R3MixtureError("M5 R3 Pilot lineage differs")
    formal_sources = load_verified_m5_r3_sources(
        config=config,
        formal_result_path=formal_result_path,
        formal_raw_artifact=formal_raw_artifact,
    )
    selected_sources = select_m5_r3_targeted_sources(formal_sources, config=config)
    nonthinking, parent_hash = _nonthinking_candidates(artifact_root=artifact_root)
    general = tuple(
        item.sequence
        for item in _thinking_candidates_with_metadata(
            pilot=pilot,
            tokenizer_config_path=tokenizer_config_path,
            model_dir=model_dir,
        )
    )
    targeted_candidates = thinking_candidates_from_samples(
        tuple(source.sample for source in selected_sources),
        tokenizer_config_path=tokenizer_config_path,
        model_dir=model_dir,
    )
    targeted = tuple(_with_mode(item.sequence, 2) for item in targeted_candidates)
    if len(general) != 96 or len(targeted) != 160:
        raise M5R3MixtureError("M5 R3 Thinking source counts differ")
    budget = config.token_budget
    selected_non, non_reuse, non_partial = select_exact_supervised_tokens(
        nonthinking,
        target=budget.nonthinking_supervised_tokens,
        seed=config.build_seed,
    )
    selected_general, general_reuse, general_partial = select_exact_supervised_tokens(
        general,
        target=budget.general_thinking_supervised_tokens,
        seed=(config.build_seed + 1) % (2**32),
    )
    selected_targeted, targeted_reuse, targeted_partial, targeted_uses = (
        select_exact_supervised_tokens_capped(
            targeted,
            target=budget.targeted_thinking_supervised_tokens,
            seed=(config.build_seed + 2) % (2**32),
            max_source_uses=config.selection.max_source_uses,
        )
    )
    combined = list(selected_non + selected_general + selected_targeted)
    random.Random((config.build_seed + 3) % (2**32)).shuffle(combined)
    input_ids = np.asarray([item.input_ids for item in combined], dtype="<i4")
    labels = np.asarray([item.labels for item in combined], dtype="<i4")
    attention_masks = np.asarray([item.attention_mask for item in combined], dtype="u1")
    modes = np.asarray([item.mode for item in combined], dtype="u1")
    arrays_hash = _array_content_hash(input_ids, labels, attention_masks, modes)
    selection_sha256 = content_sha256([source.sample.to_dict() for source in selected_sources])
    config_sha256 = m5_r3_mixture_config_sha256(config)
    identity = {
        "arrays_sha256": arrays_hash,
        "build_seed": config.build_seed,
        "config_sha256": config_sha256,
        "formal_accepted_samples_sha256": config.formal_source.accepted_samples_sha256,
        "formal_raw_artifact_sha256": config.formal_source.raw_artifact_sha256,
        "formal_result_sha256": config.formal_source.result_sha256,
        "general_thinking_supervised_tokens": budget.general_thinking_supervised_tokens,
        "nonthinking_supervised_tokens": budget.nonthinking_supervised_tokens,
        "parent_content_sha256": parent_hash,
        "pilot_content_sha256": pilot.manifest.content_sha256,
        "targeted_selection_sha256": selection_sha256,
        "targeted_thinking_supervised_tokens": budget.targeted_thinking_supervised_tokens,
    }
    identity_hash = content_sha256(identity)
    version = f"m5-r3-mixture-v2-{identity_hash[:8]}"
    destination = output_root / version
    if destination.exists():
        opened = open_m5_ablation_mixture(destination)
        if not isinstance(opened.manifest, M5R3MixtureManifest):
            raise M5R3MixtureError("existing R3 destination contains the wrong manifest kind")
        return opened.manifest
    temporary = output_root / f".{version}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        sequence_path = temporary / _SEQUENCE_FILE
        with sequence_path.open("wb") as handle:
            np.savez(
                handle,
                input_ids=input_ids,
                labels=labels,
                attention_masks=attention_masks,
                modes=modes,
            )
            handle.flush()
            os.fsync(handle.fileno())
        family_counts = Counter(source.sample.task_family for source in selected_sources)
        language_counts = Counter(source.sample.language for source in selected_sources)
        label_counts = Counter(source.label for source in selected_sources)
        manifest = M5R3MixtureManifest(
            mixture_version=version,
            config_sha256=config_sha256,
            parent_dataset_version="m2-sft-v1-f82ff32e",
            parent_content_sha256=parent_hash,
            pilot_dataset_version=pilot.manifest.dataset_version,
            pilot_content_sha256=pilot.manifest.content_sha256,
            formal_result_sha256=config.formal_source.result_sha256,
            formal_raw_artifact_sha256=config.formal_source.raw_artifact_sha256,
            formal_accepted_samples_sha256=config.formal_source.accepted_samples_sha256,
            targeted_selection_policy_id=config.selection.policy_id,
            targeted_selection_sha256=selection_sha256,
            tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
            nonthinking_template_id="qwen3-chatml-nonthinking-v1",
            thinking_template_id="qwen3-chatml-thinking-v1",
            sequence_length=1024,
            pad_token_id=151643,
            target_supervised_tokens=budget.total_supervised_tokens,
            thinking_fraction_basis_points=3000,
            nonthinking_supervised_tokens=budget.nonthinking_supervised_tokens,
            thinking_supervised_tokens=300_000,
            general_thinking_supervised_tokens=budget.general_thinking_supervised_tokens,
            targeted_thinking_supervised_tokens=budget.targeted_thinking_supervised_tokens,
            sequence_count=len(combined),
            nonthinking_sequence_count=len(selected_non),
            general_thinking_sequence_count=len(selected_general),
            targeted_thinking_sequence_count=len(selected_targeted),
            nonthinking_source_sequences=len(nonthinking),
            general_thinking_source_sequences=96,
            targeted_thinking_source_sequences=160,
            nonthinking_reuse_count=non_reuse,
            general_thinking_reuse_count=general_reuse,
            targeted_thinking_reuse_count=targeted_reuse,
            targeted_source_supervised_tokens_per_pass=sum(
                item.supervised_tokens for item in targeted
            ),
            targeted_source_use_min=min(targeted_uses),
            targeted_source_use_max=max(targeted_uses),
            partially_masked_sequences=non_partial + general_partial + targeted_partial,
            targeted_source_family_counts={
                "config": family_counts["config"],
                "log_diagnosis": family_counts["log_diagnosis"],
            },
            targeted_source_language_counts={
                "en": language_counts["en"],
                "zh": language_counts["zh"],
            },
            targeted_source_label_counts=dict(sorted(label_counts.items())),
            build_seed=config.build_seed,
            content_sha256=identity_hash,
            artifact=M5MixtureArtifactFile(
                path="sequences.npz",
                size_bytes=sequence_path.stat().st_size,
                sha256=_sha256_file(sequence_path),
            ),
            r3_training_authorized=True,
            consume_m6_frozen_results=False,
        )
        manifest_bytes = (
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        (temporary / _MANIFEST_FILE).write_bytes(manifest_bytes)
        (temporary / _COMMIT_FILE).write_text(
            json.dumps(
                {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    opened = open_m5_ablation_mixture(destination)
    if not isinstance(opened.manifest, M5R3MixtureManifest):
        raise M5R3MixtureError("committed R3 destination contains the wrong manifest kind")
    return opened.manifest
