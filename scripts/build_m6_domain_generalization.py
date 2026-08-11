#!/usr/bin/env python3
"""Build the independent seven-family M6 R4 generalization mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

if __package__:
    import scripts.build_m2_domain_eval as domain_builder
else:
    import build_m2_domain_eval as domain_builder  # type: ignore[import-not-found,no-redef]
from tinyllm.data import (
    M5MixtureArtifactFile,
    M5MixtureError,
    M5MixtureSequence,
    M6DomainGeneralizationMixtureManifest,
    open_m5_ablation_mixture,
    select_exact_supervised_tokens,
)
from tinyllm.data.m5_dual_mode_correction import (
    general_nonthinking_correction_sources,
    pack_correction_sequences,
)
from tinyllm.data.reasoning_schema import content_sha256
from tinyllm.data.registry import open_registered_dataset
from tinyllm.data.schema import ImportedMessage
from tinyllm.data.tokenization import (
    QWEN3_NONTHINKING_SFT_TEMPLATE_SHA256,
    QWEN3_THINKING_TEMPLATE_SHA256,
    TokenizersBackend,
    load_m2_tokenization_config,
    tokenize_nonthinking_sft_messages,
    tokenize_thinking_messages,
)

_OFFSETS: tuple[Literal[401], Literal[449], Literal[497]] = (401, 449, 497)
_SEQUENCE_LENGTH = 1024
_PAD_TOKEN_ID = 151643


@dataclass(frozen=True, slots=True)
class DomainGeneralizationTask:
    """One paired task authored outside every frozen evaluation suite."""

    task_id: str
    category: str
    language: str
    prompt: str
    reasoning: str
    final_answer: str


_REFUSAL_COMPONENTS = (
    ("a DNS cache", "sporadic lookup failures", "resolver traces, TTL values, and query IDs"),
    ("a feature flag", "missing UI controls", "resolved flags, account ID, and render logs"),
    ("a retry loop", "duplicate payments", "request IDs, retry policy, and ledger entries"),
    ("a proxy rule", "unexpected redirects", "resolved rules, request trace, and responses"),
    ("an encoder", "corrupted audio", "input sample, codec settings, and output checksum"),
    ("a schema migration", "missing rows", "migration revision, transaction log, and row counts"),
    ("a quota policy", "failed uploads", "quota snapshot, response codes, and account usage"),
    ("a serializer", "consumer failures", "schema versions, payload bytes, and decoder logs"),
    (
        "a certificate update",
        "TLS failures",
        "certificate chain, validity times, and handshake trace",
    ),
    ("a rollout", "latency regression", "deployment IDs, paired traces, and request distributions"),
)


def _reasoning(category: str) -> str:
    if category in {"config", "json"}:
        return (
            "I will apply only the requested transformation, preserve the required object "
            "shape and keys, then emit one complete canonical JSON object."
        )
    if category == "python":
        return (
            "I will evaluate the bounded Python expression using its exact language semantics "
            "and return only the resulting repr."
        )
    if category == "linux":
        return "I will map the requested operation to the exact standard Linux answer."
    if category == "logs":
        return "I will select only the diagnosis directly supported by the log text."
    if category == "short_code":
        return "I will use the smallest Python expression that exactly satisfies the request."
    raise M5MixtureError("unknown M6 domain-generalization category")


def _objective_tasks(batch_id: int, value_offset: int) -> tuple[DomainGeneralizationTask, ...]:
    items = domain_builder.generate_training_objective_items(
        value_offset=value_offset,
        batch_id=batch_id,
    )
    tasks: list[DomainGeneralizationTask] = []
    for item in items:
        prefix = (
            f"Training batch R4-{batch_id}; solve this standalone task.\n\n"
            if item.language == "en"
            else f"训练批次 R4-{batch_id}；请独立完成以下任务。\n\n"
        )
        tasks.append(
            DomainGeneralizationTask(
                task_id=item.id,
                category=item.category,
                language=item.language,
                prompt=prefix + item.prompt_messages[0].content,
                reasoning=_reasoning(item.category),
                final_answer=item.reference_answer,
            )
        )
    return tuple(tasks)


def _refusal_tasks(batch_id: int) -> tuple[DomainGeneralizationTask, ...]:
    tasks: list[DomainGeneralizationTask] = []
    for index in range(40):
        language = "en" if index < 28 else "zh"
        cause, effect, evidence = _REFUSAL_COMPONENTS[index % len(_REFUSAL_COMPONENTS)]
        case_id = f"G{batch_id}-{index + 1:03d}"
        if language == "en":
            prompt = (
                f"Case {case_id}: {cause.capitalize()} is blamed for {effect}, but no "
                f"{evidence} are attached. State the root cause using only supplied evidence."
            )
            final = (
                "The supplied evidence is insufficient to determine a root cause. "
                f"Please provide {evidence}."
            )
            reasoning = (
                "The claim omits the observations needed to distinguish causes. I must avoid "
                "attribution, state insufficiency, and request every named evidence source."
            )
        else:
            prompt = (
                f"案例 {case_id}：有人将故障归因于 {cause}，但没有提供 {evidence}。"
                "仅根据已提供的证据说明根因。"
            )
            final = f"现有证据不足以确定根因。请提供 {evidence}。"
            reasoning = "现有记录不足以区分原因；我必须拒绝武断归因，并请求全部缺失证据。"
        tasks.append(
            DomainGeneralizationTask(
                task_id=f"train-r4-{batch_id}-refusal-{index + 1:03d}",
                category="refusal",
                language=language,
                prompt=prompt,
                reasoning=reasoning,
                final_answer=final,
            )
        )
    return tuple(tasks)


def _json_object_contract_tasks() -> tuple[DomainGeneralizationTask, ...]:
    """Generate diverse tasks that make the required top-level object explicit."""

    tasks: list[DomainGeneralizationTask] = []
    for index in range(240):
        language = "en" if index < 168 else "zh"
        value = 701 + index
        mode = index % 6
        if mode == 0:
            numeric_values = list(range(value, value + 6))
            key = "even"
            result: object = [item for item in numeric_values if item % 2 == 0]
            operation_en = f"Keep only even integers from {numeric_values}"
            operation_zh = f"从 {numeric_values} 中仅保留偶数"
        elif mode == 1:
            labels = [f"tag-{value % 17}", "api", f"tag-{value % 17}"]
            key = "labels"
            result = sorted(set(labels))
            operation_en = f"Deduplicate and lexicographically sort {json.dumps(labels)}"
            operation_zh = f"将 {json.dumps(labels)} 去重并按字典序排序"
        elif mode == 2:
            numeric_values = [value - 2, -1, value, 0]
            key = "positive"
            result = [item for item in numeric_values if item > 0]
            operation_en = f"Keep positive integers from {numeric_values} in their original order"
            operation_zh = f"从 {numeric_values} 中按原顺序保留正整数"
        elif mode == 3:
            key = "total"
            result = value + value + 1
            operation_en = f"Add {value} and {value + 1}"
            operation_zh = f"计算 {value} 与 {value + 1} 的和"
        elif mode == 4:
            key = "enabled"
            result = value % 2 == 0
            operation_en = f"Report whether {value} is even"
            operation_zh = f"判断 {value} 是否为偶数"
        else:
            key = "items"
            result = [{"id": value}, {"id": value + 1}]
            operation_en = f"Preserve the two records with IDs {value} and {value + 1}"
            operation_zh = f"保留 ID 为 {value} 和 {value + 1} 的两条记录"
        final = json.dumps({key: result}, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        case_id = f"J{index + 1:03d}"
        if language == "en":
            prompt = (
                f"Contract case {case_id}: {operation_en}. Return one complete canonical JSON "
                f"object whose required top-level key is `{key}`. Never return a bare array, "
                "scalar, or object fragment. Return no prose."
            )
            reasoning = (
                f"The outer shape is mandatory: I must emit a complete object with key `{key}`. "
                "I will compute its value and keep the braces and key in the final answer."
            )
        else:
            prompt = (
                f"契约案例 {case_id}：{operation_zh}。返回一个完整的 Canonical JSON 对象，"
                f"必需的顶层键为 `{key}`。禁止只返回数组、标量或对象片段，不要解释。"
            )
            reasoning = f"最外层必须是带 `{key}` 键的完整对象；计算值后仍要保留花括号和键名。"
        tasks.append(
            DomainGeneralizationTask(
                task_id=f"train-r41-json-{index + 1:03d}",
                category="json",
                language=language,
                prompt=prompt,
                reasoning=reasoning,
                final_answer=final,
            )
        )
    return tuple(tasks)


_REFUSAL_SUBJECTS = (
    ("database index", "数据库索引"),
    ("load balancer", "负载均衡器"),
    ("filesystem cache", "文件系统缓存"),
    ("token refresh job", "Token 刷新任务"),
    ("batch scheduler", "批处理调度器"),
    ("message broker", "消息代理"),
    ("DNS resolver", "DNS 解析器"),
    ("storage replica", "存储副本"),
    ("request serializer", "请求序列化器"),
    ("feature rollout", "功能发布"),
    ("GPU allocator", "GPU 分配器"),
    ("network route", "网络路由"),
    ("data loader", "数据加载器"),
    ("checkpoint writer", "Checkpoint 写入器"),
    ("authentication proxy", "认证代理"),
    ("metrics collector", "指标采集器"),
    ("container image", "容器镜像"),
    ("retry policy", "重试策略"),
    ("tokenizer revision", "Tokenizer Revision"),
    ("evaluation scorer", "评测评分器"),
)

_REFUSAL_EVIDENCE = (
    (
        "timestamped component logs, request IDs, and a controlled comparison",
        "带时间戳的组件日志、请求 ID 和受控对照",
    ),
    (
        "before-and-after metrics, configuration identities, and raw traces",
        "变更前后指标、配置身份和原始 Trace",
    ),
    (
        "the first failing event, correlated telemetry, and reproduction steps",
        "首个失败事件、关联遥测和复现步骤",
    ),
    (
        "versioned inputs, output hashes, and the complete error record",
        "版本化输入、输出哈希和完整错误记录",
    ),
    (
        "per-process resource samples, event timing, and a matched control run",
        "逐进程资源样本、事件时序和匹配的对照 Run",
    ),
    (
        "resolved runtime settings, dependency identities, and paired observations",
        "解析后的运行设置、依赖身份和成对观测",
    ),
)


def _evidence_refusal_contract_tasks() -> tuple[DomainGeneralizationTask, ...]:
    """Generate varied evidence-grounding cases with explicit three-part answers."""

    tasks: list[DomainGeneralizationTask] = []
    for index in range(360):
        language = "en" if index < 252 else "zh"
        subject_en, subject_zh = _REFUSAL_SUBJECTS[index % len(_REFUSAL_SUBJECTS)]
        evidence_en, evidence_zh = _REFUSAL_EVIDENCE[
            (index // len(_REFUSAL_SUBJECTS)) % len(_REFUSAL_EVIDENCE)
        ]
        case_id = f"E{index + 1:03d}"
        if language == "en":
            prompt = (
                f"Evidence audit {case_id}: a report blames the {subject_en} for the incident, "
                f"but it omits {evidence_en}. Determine the root cause using only supplied "
                "evidence."
            )
            final = (
                "The supplied evidence is insufficient to determine a root cause. "
                f"Please provide {evidence_en}."
            )
            reasoning = (
                "Missing observations cannot establish causality. I must not repeat the alleged "
                "component as the cause; I must state insufficiency and request every named item."
            )
        else:
            prompt = (
                f"证据审计 {case_id}：报告将事故归因于{subject_zh}，但缺少{evidence_zh}。"
                "仅根据已提供的证据判断根因。"
            )
            final = f"现有证据不足以确定根因。请提供{evidence_zh}。"
            reasoning = (
                "缺失观测无法建立因果关系；不能把被怀疑组件复述为根因，"
                "必须说明证据不足并请求全部指定证据。"
            )
        tasks.append(
            DomainGeneralizationTask(
                task_id=f"train-r41-refusal-{index + 1:03d}",
                category="refusal",
                language=language,
                prompt=prompt,
                reasoning=reasoning,
                final_answer=final,
            )
        )
    return tuple(tasks)


def generate_domain_generalization_tasks(
    *, contract_refinement: bool = False
) -> tuple[DomainGeneralizationTask, ...]:
    """Generate stable train-only tasks across all M6 domain families."""

    tasks = tuple(
        task
        for batch_id, value_offset in enumerate(_OFFSETS, start=1)
        for task in (*_objective_tasks(batch_id, value_offset), *_refusal_tasks(batch_id))
    )
    if contract_refinement:
        tasks = (*tasks, *_json_object_contract_tasks(), *_evidence_refusal_contract_tasks())
    expected = 1500 if contract_refinement else 900
    if len(tasks) != expected or len({task.task_id for task in tasks}) != expected:
        raise M5MixtureError("M6 domain-generalization task inventory differs")
    return tasks


def _pad(input_ids: tuple[int, ...], labels: tuple[int, ...], *, mode: int) -> M5MixtureSequence:
    if len(input_ids) != len(labels) or not 1 < len(input_ids) <= _SEQUENCE_LENGTH:
        raise M5MixtureError("M6 domain-generalization source exceeds sequence length")
    padding = _SEQUENCE_LENGTH - len(input_ids)
    return M5MixtureSequence(
        input_ids=input_ids + (_PAD_TOKEN_ID,) * padding,
        labels=labels + (-100,) * padding,
        attention_mask=(1,) * len(input_ids) + (0,) * padding,
        mode=mode,
    )


def _paired_sequences(
    tasks: tuple[DomainGeneralizationTask, ...],
    *,
    tokenizer_config_path: Path,
    model_dir: Path,
) -> tuple[tuple[M5MixtureSequence, ...], tuple[M5MixtureSequence, ...], int]:
    tokenization = load_m2_tokenization_config(tokenizer_config_path)
    backend = TokenizersBackend.from_files(
        model_dir / tokenization.tokenizer.tokenizer_file,
        model_dir / tokenization.tokenizer.tokenizer_config_file,
        tokenization.tokenizer,
    )
    nonthinking: list[M5MixtureSequence] = []
    thinking: list[M5MixtureSequence] = []
    max_thinking = 0
    for task in tasks:
        messages = (
            ImportedMessage(role="user", content=task.prompt),
            ImportedMessage(role="assistant", content=task.final_answer),
        )
        encoded_non = tokenize_nonthinking_sft_messages(
            messages,
            backend=backend,
            tokenizer=tokenization.tokenizer,
        )
        encoded_think = tokenize_thinking_messages(
            messages,
            assistant_reasoning=(task.reasoning,),
            backend=backend,
            tokenizer=tokenization.tokenizer,
        )
        if len(encoded_non.input_ids) > _SEQUENCE_LENGTH:
            raise M5MixtureError(
                f"M6 domain-generalization Non-thinking task {task.task_id} has "
                f"{len(encoded_non.input_ids)} tokens"
            )
        if len(encoded_think.input_ids) > _SEQUENCE_LENGTH:
            raise M5MixtureError(
                f"M6 domain-generalization Thinking task {task.task_id} has "
                f"{len(encoded_think.input_ids)} tokens"
            )
        nonthinking.append(_pad(encoded_non.input_ids, encoded_non.labels, mode=0))
        thinking_sequence = _pad(encoded_think.input_ids, encoded_think.labels, mode=1)
        thinking.append(thinking_sequence)
        max_thinking = max(max_thinking, thinking_sequence.supervised_tokens)
    if max_thinking > 256:
        raise M5MixtureError("M6 domain-generalization reasoning exceeds compact contract")
    return tuple(nonthinking), tuple(thinking), max_thinking


def _evaluation_prompts(project_root: Path) -> set[str]:
    prompts: set[str] = set()
    for path in sorted((project_root / "evals/domain").glob("v*/items.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            prompts.add(str(json.loads(line)["prompt_messages"][0]["content"]))
    return prompts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_content_hash(
    input_ids: np.ndarray,
    labels: np.ndarray,
    attention_masks: np.ndarray,
    modes: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, array in (
        ("input_ids", input_ids),
        ("labels", labels),
        ("attention_masks", attention_masks),
        ("modes", modes),
    ):
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_domain_generalization_mixture(
    *,
    artifact_root: Path,
    tokenizer_config_path: Path,
    model_dir: Path,
    project_root: Path,
    output_root: Path,
    build_seed: int,
    contract_refinement: bool = False,
) -> M6DomainGeneralizationMixtureManifest:
    """Build and atomically commit the exact 1M-token R4 mixture."""

    tasks = generate_domain_generalization_tasks(contract_refinement=contract_refinement)
    overlap = {task.prompt for task in tasks} & _evaluation_prompts(project_root)
    if overlap:
        raise M5MixtureError("M6 domain-generalization source overlaps frozen evaluation")
    source_payload = tuple(asdict(task) for task in tasks)
    source_sha256 = content_sha256(source_payload)
    domain_non_raw, domain_think_raw, max_thinking = _paired_sequences(
        tasks,
        tokenizer_config_path=tokenizer_config_path,
        model_dir=model_dir,
    )
    general_raw = general_nonthinking_correction_sources(artifact_root=artifact_root)
    general = pack_correction_sequences(general_raw, mode=0)
    domain_non = pack_correction_sequences(domain_non_raw, mode=0)
    domain_think = pack_correction_sequences(domain_think_raw, mode=1)
    strata = (
        select_exact_supervised_tokens(general, target=250_000, seed=build_seed),
        select_exact_supervised_tokens(domain_non, target=450_000, seed=(build_seed + 1) % (2**32)),
        select_exact_supervised_tokens(
            domain_think, target=300_000, seed=(build_seed + 2) % (2**32)
        ),
    )
    combined = [item for selected, _, _ in strata for item in selected]
    random.Random((build_seed + 3) % (2**32)).shuffle(combined)
    input_ids = np.asarray([item.input_ids for item in combined], dtype="<i4")
    labels = np.asarray([item.labels for item in combined], dtype="<i4")
    attention_masks = np.asarray([item.attention_mask for item in combined], dtype="u1")
    modes = np.asarray([item.mode for item in combined], dtype="u1")
    arrays_sha256 = _array_content_hash(input_ids, labels, attention_masks, modes)
    parent = open_registered_dataset(
        artifact_root=artifact_root,
        dataset_version="m2-sft-v1-f82ff32e",
    )
    identity = {
        "arrays_sha256": arrays_sha256,
        "authored_source_sha256": source_sha256,
        "build_seed": build_seed,
        "diagnostic_protocol_version": "m6-release-v3",
        "domain_nonthinking_supervised_tokens": 450_000,
        "domain_thinking_supervised_tokens": 300_000,
        "general_nonthinking_supervised_tokens": 250_000,
        "nonthinking_template_sha256": QWEN3_NONTHINKING_SFT_TEMPLATE_SHA256,
        "parent_content_sha256": parent.manifest.content_sha256,
        "thinking_template_sha256": QWEN3_THINKING_TEMPLATE_SHA256,
        "training_value_offsets": _OFFSETS,
    }
    if contract_refinement:
        identity["refinement"] = "json-object-and-evidence-refusal"
    identity_sha256 = content_sha256(identity)
    version_number = 2 if contract_refinement else 1
    version = f"m6-domain-generalization-mixture-v{version_number}-{identity_sha256[:8]}"
    destination = output_root / version
    if destination.exists():
        reopened = open_m5_ablation_mixture(destination)
        if not isinstance(reopened.manifest, M6DomainGeneralizationMixtureManifest):
            raise M5MixtureError("existing M6 domain-generalization artifact has wrong kind")
        return reopened.manifest
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{version}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        sequence_path = temporary / "sequences.npz"
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
        category_counts = {
            category: sum(task.category == category for task in tasks)
            for category in (
                "config",
                "json",
                "linux",
                "logs",
                "python",
                "refusal",
                "short_code",
            )
        }
        reuse = tuple(value[1] for value in strata)
        partial = tuple(value[2] for value in strata)
        non_count = sum(item.mode == 0 for item in combined)
        manifest = M6DomainGeneralizationMixtureManifest(
            mixture_version=version,
            parent_dataset_version="m2-sft-v1-f82ff32e",
            parent_content_sha256=parent.manifest.content_sha256,
            diagnostic_protocol_version="m6-release-v3",
            source_consumed_evaluation_content=False,
            evaluation_prompt_overlap_count=0,
            authored_source_sha256=source_sha256,
            authored_source_tasks=cast(Literal[900, 1500], len(tasks)),
            authored_source_category_counts=category_counts,
            training_value_offsets=_OFFSETS,
            refinement=("json-object-and-evidence-refusal" if contract_refinement else None),
            tokenizer_revision="c1899de289a04d12100db370d81485cdf75e47ca",
            nonthinking_template_id="qwen3-chatml-nonthinking-sft-v2",
            nonthinking_template_sha256=(
                "fba6724bd16200356794105a2273bbd42e777c8311ef1760059c6f0766171ca2"
            ),
            thinking_template_id="qwen3-chatml-thinking-v1",
            thinking_template_sha256=(
                "4786143dbb7adb72a922d5efdcbe6596f2d65dcdc35d7bbf1b22830b795c2af9"
            ),
            sequence_length=1024,
            pad_token_id=151643,
            target_supervised_tokens=1_000_000,
            thinking_fraction_basis_points=3000,
            nonthinking_supervised_tokens=700_000,
            thinking_supervised_tokens=300_000,
            general_nonthinking_supervised_tokens=250_000,
            domain_nonthinking_supervised_tokens=450_000,
            domain_thinking_supervised_tokens=300_000,
            sequence_count=len(combined),
            nonthinking_sequence_count=non_count,
            thinking_sequence_count=len(combined) - non_count,
            general_nonthinking_source_sequences=len(general_raw),
            domain_source_pairs=cast(Literal[900, 1500], len(tasks)),
            general_nonthinking_reuse_count=reuse[0],
            domain_nonthinking_reuse_count=reuse[1],
            domain_thinking_reuse_count=reuse[2],
            partially_masked_sequences=sum(partial),
            compact_reasoning_max_supervised_tokens=max_thinking,
            build_seed=build_seed,
            content_sha256=identity_sha256,
            artifact=M5MixtureArtifactFile(
                path="sequences.npz",
                size_bytes=sequence_path.stat().st_size,
                sha256=_sha256_file(sequence_path),
            ),
        )
        manifest_bytes = (
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        (temporary / "COMMITTED").write_text(
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
    reopened = open_m5_ablation_mixture(destination)
    if not isinstance(reopened.manifest, M6DomainGeneralizationMixtureManifest):
        raise M5MixtureError("committed M6 domain-generalization artifact has wrong kind")
    return reopened.manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-seed", type=int, default=20260811)
    parser.add_argument(
        "--contract-refinement",
        action="store_true",
        help="Add diverse JSON-object and evidence-refusal contract tasks.",
    )
    args = parser.parse_args()
    try:
        result = build_domain_generalization_mixture(
            artifact_root=args.artifact_root,
            tokenizer_config_path=args.tokenizer_config,
            model_dir=args.model_dir,
            project_root=args.project_root,
            output_root=args.output_root,
            build_seed=args.build_seed,
            contract_refinement=args.contract_refinement,
        )
    except (M5MixtureError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
