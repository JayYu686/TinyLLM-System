"""Build an independent concise dual-mode repair mixture after the M6 v2 diagnosis."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tinyllm.data.m5_dual_mode_correction import (
    general_nonthinking_correction_sources,
    pack_correction_sequences,
)
from tinyllm.data.m5_mixture import (
    M5MixtureError,
    M5MixtureSequence,
    open_m5_ablation_mixture,
    select_exact_supervised_tokens,
)
from tinyllm.data.m5_mixture_schema import (
    M5MixtureArtifactFile,
    M6GateRepairMixtureManifest,
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

_SEQUENCE_LENGTH = 1024
_PAD_TOKEN_ID = 151643


@dataclass(frozen=True, slots=True)
class M6GateRepairTask:
    """One independently authored paired training task."""

    task_id: str
    kind: str
    language: str
    prompt: str
    reasoning: str
    final_answer: str


def _language(index: int) -> str:
    return "en" if index % 10 < 7 else "zh"


def _refusal_tasks() -> tuple[M6GateRepairTask, ...]:
    scenarios = (
        (
            "a database index",
            "slow checkout queries",
            "the query plan and indexed-column statistics",
            "数据库索引",
            "结算查询缓慢",
            "查询计划和索引列统计",
        ),
        (
            "clock skew",
            "duplicate calendar alerts",
            "timestamped host clocks and alert event IDs",
            "时钟偏移",
            "日历提醒重复",
            "带时间戳的主机时钟和提醒事件 ID",
        ),
        (
            "a browser extension",
            "page rendering failures",
            "the extension inventory, console log, and failing URL",
            "浏览器扩展",
            "页面渲染失败",
            "扩展清单、控制台日志和失败 URL",
        ),
        (
            "certificate rotation",
            "client authentication errors",
            "the certificate chain, validity times, and handshake trace",
            "证书轮换",
            "客户端认证错误",
            "证书链、有效期和握手 Trace",
        ),
        (
            "a webhook retry policy",
            "duplicate notifications",
            "request IDs, retry settings, and receiver logs",
            "Webhook 重试策略",
            "通知重复",
            "请求 ID、重试设置和接收端日志",
        ),
        (
            "an audio encoder",
            "distorted recordings",
            "the input sample, codec settings, and encoded output",
            "音频编码器",
            "录音失真",
            "输入样本、Codec 设置和编码输出",
        ),
        (
            "a database migration",
            "missing customer rows",
            "the migration revision, transaction log, and row counts",
            "数据库迁移",
            "客户行缺失",
            "迁移 Revision、事务日志和行计数",
        ),
        (
            "a load balancer rule",
            "intermittent redirects",
            "the resolved rules, request trace, and backend responses",
            "负载均衡规则",
            "间歇性重定向",
            "解析后的规则、请求 Trace 和后端响应",
        ),
        (
            "a message serializer",
            "consumer decode failures",
            "the schema versions, payload bytes, and decoder error",
            "消息序列化器",
            "消费者解码失败",
            "Schema 版本、Payload 字节和解码错误",
        ),
        (
            "a storage quota",
            "failed photo uploads",
            "the quota snapshot, upload response, and account usage",
            "存储配额",
            "照片上传失败",
            "配额快照、上传响应和账户用量",
        ),
    )
    tasks: list[M6GateRepairTask] = []
    for index in range(300):
        language = _language(index)
        en_cause, en_effect, en_missing, zh_cause, zh_effect, zh_missing = scenarios[index % 10]
        case_id = f"R{index + 1:03d}"
        form = (index // 10) % 3
        if language == "en":
            prompts = (
                f"Case {case_id}: An incident note attributes {en_effect} to {en_cause}, "
                f"but it omits {en_missing}. Determine the cause using only the note.",
                f"Case {case_id}: Operators blame {en_cause} for {en_effect}. "
                f"No {en_missing} is attached. Give an evidence-grounded diagnosis.",
                f"Case {case_id}: {en_effect.capitalize()} is reported and {en_cause} is "
                f"proposed as the cause, without {en_missing}. State what can be concluded.",
            )
            final = (
                "The available evidence is insufficient to determine a specific cause. "
                f"Please provide {en_missing}."
            )
            reasoning = (
                "The claim lacks the observations needed to distinguish causes. "
                "I should avoid attribution and request the missing evidence."
            )
        else:
            prompts = (
                f"案例 {case_id}：事件记录将{zh_effect}归因于{zh_cause}，"
                f"但缺少{zh_missing}。仅根据记录判断原因。",
                f"案例 {case_id}：运维人员认为{zh_cause}导致{zh_effect}，"
                f"但未附{zh_missing}。请给出基于证据的诊断。",
                f"案例 {case_id}：报告出现{zh_effect}并提出{zh_cause}是原因，"
                f"但没有{zh_missing}。说明目前能得出的结论。",
            )
            final = f"现有证据不足以确定具体根因。请提供{zh_missing}。"
            reasoning = "现有记录不足以区分不同原因。我应避免武断归因，并请求缺失证据。"
        tasks.append(
            M6GateRepairTask(
                task_id=f"repair-refusal-{index + 1:03d}",
                kind="refusal",
                language=language,
                prompt=prompts[form],
                reasoning=reasoning,
                final_answer=final,
            )
        )
    return tuple(tasks)


def _python_tasks() -> tuple[M6GateRepairTask, ...]:
    tasks: list[M6GateRepairTask] = []
    for index in range(120):
        language = _language(index)
        stop = 10 + index
        divisor = 2 + index % 5
        expression = f"sum(x for x in range({stop}) if x % {divisor} == 0)"
        answer = repr(sum(x for x in range(stop) if x % divisor == 0))
        prompt = (
            f"Evaluate this Python 3 expression and return only its exact repr:\n\n{expression}"
            if language == "en"
            else f"计算以下 Python 3 表达式，仅返回精确 repr：\n\n{expression}"
        )
        reasoning = (
            "I will evaluate the bounded integer expression directly and return only its repr."
            if language == "en"
            else "我将直接计算这个有限整数表达式，并仅返回其 repr。"
        )
        tasks.append(
            M6GateRepairTask(
                f"repair-python-{index + 1:03d}", "python", language, prompt, reasoning, answer
            )
        )
    return tuple(tasks)


def _json_tasks() -> tuple[M6GateRepairTask, ...]:
    tasks: list[M6GateRepairTask] = []
    for index in range(80):
        language = _language(index)
        source = {"job": {"id": index + 101, "retries": index % 4}, "enabled": False}
        expected = {"enabled": True, "job": source["job"]}
        source_text = json.dumps(source, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        answer = json.dumps(expected, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        prompt = (
            "Set `enabled` to true in this object. Return only canonical JSON with "
            f"sorted keys and no spaces: {source_text}"
            if language == "en"
            else f"将对象中的 `enabled` 设为 true。仅返回键排序且无空格的规范 JSON：{source_text}"
        )
        reasoning = (
            "Only one field changes; I should preserve the nested object and serialize canonically."
            if language == "en"
            else "只需修改一个字段；我应保留嵌套对象并按规范序列化。"
        )
        tasks.append(
            M6GateRepairTask(
                f"repair-json-{index + 1:03d}", "json", language, prompt, reasoning, answer
            )
        )
    return tuple(tasks)


def _linux_tasks() -> tuple[M6GateRepairTask, ...]:
    tasks: list[M6GateRepairTask] = []
    for index in range(60):
        language = _language(index)
        path = f"/srv/archive-{index + 41}"
        lines = 5 + index % 20
        modes = (
            (f"du -sh -- {path}", "show the total disk usage", "显示总磁盘占用"),
            (
                f"find {path} -maxdepth 1 -type f | wc -l",
                "count regular files directly inside it",
                "统计其中直接包含的普通文件",
            ),
            (
                f"tail -n {lines} -- {path}/events.log",
                f"show the last {lines} log lines",
                f"显示日志最后 {lines} 行",
            ),
        )
        answer, en_action, zh_action = modes[index % len(modes)]
        prompt = (
            f"Write one shell command to {en_action} for `{path}`. Return only the command."
            if language == "en"
            else f"写一条 Shell 命令，为 `{path}` {zh_action}。仅返回命令。"
        )
        reasoning = (
            "A standard non-interactive command satisfies the requested filesystem operation."
            if language == "en"
            else "使用标准非交互命令即可完成所需文件系统操作。"
        )
        tasks.append(
            M6GateRepairTask(
                f"repair-linux-{index + 1:03d}", "linux", language, prompt, reasoning, answer
            )
        )
    return tuple(tasks)


def _log_tasks() -> tuple[M6GateRepairTask, ...]:
    signatures = (
        (
            "ConnectionRefusedError: [Errno 111] connection refused",
            "The target endpoint refused the connection.",
            "目标端点拒绝连接。",
        ),
        (
            "OSError: [Errno 28] No space left on device",
            "The target filesystem has no free space.",
            "目标文件系统空间不足。",
        ),
        (
            "PermissionError: [Errno 13] Permission denied",
            "The process lacks permission for the operation.",
            "进程缺少执行该操作的权限。",
        ),
        (
            "json.decoder.JSONDecodeError: Expecting value",
            "The input is not valid JSON at the reported position.",
            "输入在报告位置不是合法 JSON。",
        ),
    )
    tasks: list[M6GateRepairTask] = []
    for index in range(60):
        language = _language(index)
        log, en_answer, zh_answer = signatures[index % len(signatures)]
        prompt = (
            "Diagnose only what this log directly supports. Return one sentence.\n\n"
            f"{log} (event {index + 501})"
            if language == "en"
            else f"仅诊断该日志直接支持的事实，返回一句话。\n\n{log}（事件 {index + 501}）"
        )
        answer = en_answer if language == "en" else zh_answer
        reasoning = (
            "The exception text directly identifies the immediate failure; no broader "
            "cause is justified."
            if language == "en"
            else "异常文本直接指出即时故障，不应推断更广泛的原因。"
        )
        tasks.append(
            M6GateRepairTask(
                f"repair-log-{index + 1:03d}", "logs", language, prompt, reasoning, answer
            )
        )
    return tuple(tasks)


def _short_code_tasks() -> tuple[M6GateRepairTask, ...]:
    replacements = (
        (
            "normalized = TODO",
            "text.strip().lower()",
            "trim whitespace and lowercase `text`",
            "去除 `text` 两端空白并转为小写",
        ),
        ("total = TODO", "sum(values)", "sum `values`", "对 `values` 求和"),
        (
            "merged = TODO",
            "{**defaults, **overrides}",
            "merge mappings so `overrides` wins",
            "合并映射并让 `overrides` 覆盖",
        ),
        (
            "absolute = TODO",
            "path.resolve()",
            "resolve `path` to an absolute path",
            "将 `path` 解析为绝对路径",
        ),
    )
    tasks: list[M6GateRepairTask] = []
    for index in range(60):
        language = _language(index)
        statement, answer, en_goal, zh_goal = replacements[index % len(replacements)]
        prompt = (
            f"In `{statement}`, replace TODO with one Python expression that will "
            f"{en_goal}. Return only the expression. Case {index + 701}."
            if language == "en"
            else f"在 `{statement}` 中，用一个能够{zh_goal}的 Python 表达式替换 TODO。"
            f"仅返回表达式。案例 {index + 701}。"
        )
        reasoning = (
            "The requested operation has a direct standard-library expression; I should "
            "return it without explanation."
            if language == "en"
            else "该操作有直接的标准表达式；我应仅返回表达式。"
        )
        tasks.append(
            M6GateRepairTask(
                f"repair-short-code-{index + 1:03d}",
                "short_code",
                language,
                prompt,
                reasoning,
                answer,
            )
        )
    return tuple(tasks)


def generate_m6_gate_repair_tasks() -> tuple[M6GateRepairTask, ...]:
    """Generate stable paired tasks without reading any evaluation response or answer."""

    tasks = (
        _refusal_tasks()
        + _python_tasks()
        + _json_tasks()
        + _linux_tasks()
        + _log_tasks()
        + _short_code_tasks()
    )
    if len(tasks) != 680 or len({task.task_id for task in tasks}) != len(tasks):
        raise M5MixtureError("M6 gate-repair authored task inventory differs")
    return tasks


def _pad(input_ids: tuple[int, ...], labels: tuple[int, ...], *, mode: int) -> M5MixtureSequence:
    if len(input_ids) != len(labels) or not 1 < len(input_ids) <= _SEQUENCE_LENGTH:
        raise M5MixtureError("M6 gate-repair source exceeds the sequence length")
    padding = _SEQUENCE_LENGTH - len(input_ids)
    return M5MixtureSequence(
        input_ids=input_ids + (_PAD_TOKEN_ID,) * padding,
        labels=labels + (-100,) * padding,
        attention_mask=(1,) * len(input_ids) + (0,) * padding,
        mode=mode,
    )


def _paired_sequences(
    tasks: tuple[M6GateRepairTask, ...],
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
    max_thinking_supervised = 0
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
        non_sequence = _pad(encoded_non.input_ids, encoded_non.labels, mode=0)
        think_sequence = _pad(encoded_think.input_ids, encoded_think.labels, mode=1)
        max_thinking_supervised = max(max_thinking_supervised, think_sequence.supervised_tokens)
        nonthinking.append(non_sequence)
        thinking.append(think_sequence)
    if max_thinking_supervised > 256:
        raise M5MixtureError("M6 gate-repair authored reasoning exceeds the compact contract")
    return tuple(nonthinking), tuple(thinking), max_thinking_supervised


def _evaluation_prompts(project_root: Path) -> set[str]:
    prompts: set[str] = set()
    for variant in ("v1", "v2", "v3"):
        path = project_root / f"evals/domain/{variant}/items.jsonl"
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                prompts.add(str(item["prompt_messages"][0]["content"]))
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise M5MixtureError("M6 gate-repair contamination inputs are invalid") from exc
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


def build_m6_gate_repair_mixture(
    *,
    artifact_root: Path,
    tokenizer_config_path: Path,
    model_dir: Path,
    project_root: Path,
    output_root: Path,
    build_seed: int,
) -> M6GateRepairMixtureManifest:
    """Build and atomically commit the exact 1M-token R2 repair mixture."""

    tasks = generate_m6_gate_repair_tasks()
    overlap = {task.prompt for task in tasks} & _evaluation_prompts(project_root)
    if overlap:
        raise M5MixtureError("M6 gate-repair source overlaps a frozen evaluation prompt")
    source_payload = [asdict(task) for task in tasks]
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
    selected_general, general_reuse, general_partial = select_exact_supervised_tokens(
        general,
        target=400_000,
        seed=build_seed,
    )
    selected_domain_non, domain_non_reuse, domain_non_partial = select_exact_supervised_tokens(
        domain_non,
        target=300_000,
        seed=(build_seed + 1) % (2**32),
    )
    selected_domain_think, domain_think_reuse, domain_think_partial = (
        select_exact_supervised_tokens(
            domain_think,
            target=300_000,
            seed=(build_seed + 2) % (2**32),
        )
    )
    combined = list(selected_general + selected_domain_non + selected_domain_think)
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
        "diagnostic_protocol_version": "m6-release-v2",
        "domain_nonthinking_supervised_tokens": 300_000,
        "domain_thinking_supervised_tokens": 300_000,
        "general_nonthinking_supervised_tokens": 400_000,
        "nonthinking_template_sha256": QWEN3_NONTHINKING_SFT_TEMPLATE_SHA256,
        "parent_content_sha256": parent.manifest.content_sha256,
        "thinking_template_sha256": QWEN3_THINKING_TEMPLATE_SHA256,
    }
    identity_sha256 = content_sha256(identity)
    version = f"m6-gate-repair-mixture-v1-{identity_sha256[:8]}"
    destination = output_root / version
    if destination.exists():
        reopened = open_m5_ablation_mixture(destination)
        if not isinstance(reopened.manifest, M6GateRepairMixtureManifest):
            raise M5MixtureError("existing M6 gate-repair destination has the wrong kind")
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
        manifest = M6GateRepairMixtureManifest(
            mixture_version=version,
            parent_dataset_version="m2-sft-v1-f82ff32e",
            parent_content_sha256=parent.manifest.content_sha256,
            diagnostic_protocol_version="m6-release-v2",
            source_consumed_evaluation_content=False,
            evaluation_prompt_overlap_count=0,
            authored_source_sha256=source_sha256,
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
            general_nonthinking_supervised_tokens=400_000,
            domain_nonthinking_supervised_tokens=300_000,
            domain_thinking_supervised_tokens=300_000,
            sequence_count=len(combined),
            nonthinking_sequence_count=len(selected_general) + len(selected_domain_non),
            thinking_sequence_count=len(selected_domain_think),
            general_nonthinking_source_sequences=len(general_raw),
            authored_domain_source_pairs=len(tasks),
            authored_refusal_source_pairs=sum(task.kind == "refusal" for task in tasks),
            general_nonthinking_reuse_count=general_reuse,
            domain_nonthinking_reuse_count=domain_non_reuse,
            domain_thinking_reuse_count=domain_think_reuse,
            partially_masked_sequences=(
                general_partial + domain_non_partial + domain_think_partial
            ),
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
    if not isinstance(reopened.manifest, M6GateRepairMixtureManifest):
        raise M5MixtureError("committed M6 gate-repair destination has the wrong kind")
    return reopened.manifest
