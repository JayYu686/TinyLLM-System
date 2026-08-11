#!/usr/bin/env python3
"""Generate or verify the frozen public 300-item M2 domain evaluation set."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

from tinyllm.evaluation import (
    AuthoredProvenance,
    EvaluationItem,
    EvaluationPromptMessage,
    ExactMatchScorer,
    HumanRubricScorer,
    JsonObjectScorer,
    MultipleChoiceScorer,
    build_evaluation_manifest,
    load_evaluation_build_config,
)

Language = Literal["en", "zh"]
Category = Literal["config", "json", "linux", "logs", "python", "refusal", "short_code"]
SuiteVariant = Literal["v1", "v2", "v3", "v4", "v5"]

_ACTIVE_VARIANT: SuiteVariant = "v1"
_VALUE_OFFSET_OVERRIDE: int | None = None

CATEGORY_DISTRIBUTION: tuple[tuple[Category, int, int], ...] = (
    ("config", 40, 28),
    ("json", 40, 28),
    ("linux", 45, 32),
    ("logs", 45, 31),
    ("python", 50, 35),
    ("refusal", 40, 28),
    ("short_code", 40, 28),
)
ENGLISH_COUNTS = {category: english_count for category, _, english_count in CATEGORY_DISTRIBUTION}
CHINESE_COUNTS = {
    category: total - english_count for category, total, english_count in CATEGORY_DISTRIBUTION
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _provenance() -> AuthoredProvenance:
    return AuthoredProvenance(
        origin="tinyllm-authored",
        license="Apache-2.0",
        redistribution_allowed=True,
        source_note=f"Authored from versioned TinyLLM-System {_ACTIVE_VARIANT} templates.",
    )


def _item_id(category: Category, index: int) -> str:
    category_id = category.replace("_", "-")
    return f"domain-{category_id}-{index + 1:03d}"


def _semantic_index(category: Category, item_index: int, language: Language) -> int:
    """Map every Chinese item to a difficulty-matched English source item."""

    return item_index if language == "en" else item_index - ENGLISH_COUNTS[category]


def _pair_tags(category: Category, semantic_index: int) -> tuple[str, ...]:
    if semantic_index < CHINESE_COUNTS[category]:
        return (f"bilingual-pair-{semantic_index + 1:03d}",)
    return ("english-only",)


def _holdout_prompt(prompt: str, language: Language) -> str:
    """Give later holdouts independent-batch instructions without changing the task."""

    if _ACTIVE_VARIANT not in {"v3", "v4", "v5"}:
        return prompt
    prefixes = {
        "v3": (
            "Answer this independent holdout item.\n\n",
            "请独立回答以下留出题。\n\n",
        ),
        "v4": (
            "Complete this sealed final-audit item independently.\n\n",
            "请独立完成以下密封终审题。\n\n",
        ),
        "v5": (
            "Complete this independently sealed JSON-decoding audit item.\n\n",
            "请独立完成以下密封 JSON 解码审计题。\n\n",
        ),
    }
    en_prefix, zh_prefix = prefixes[_ACTIVE_VARIANT]
    prefix = en_prefix if language == "en" else zh_prefix
    return prefix + prompt


def _value_index(semantic_index: int) -> int:
    """Keep bilingual clustering stable while changing each holdout's task values."""

    if _VALUE_OFFSET_OVERRIDE is not None:
        return semantic_index + _VALUE_OFFSET_OVERRIDE
    offsets = {"v1": 0, "v2": 137, "v3": 293, "v4": 607, "v5": 911}
    return semantic_index + offsets[_ACTIVE_VARIANT]


def _exact_item(
    category: Category,
    index: int,
    language: Language,
    *,
    prompt: str,
    answer: str,
    tags: tuple[str, ...],
) -> EvaluationItem:
    return EvaluationItem(
        id=_item_id(category, index),
        language=language,
        category=category,
        prompt_messages=(
            EvaluationPromptMessage(role="user", content=_holdout_prompt(prompt, language)),
        ),
        reference_answer=answer,
        scorer=ExactMatchScorer(
            kind="exact_match",
            accepted_answers=(answer,),
            case_sensitive=True,
            strip_outer_whitespace=True,
        ),
        provenance=_provenance(),
        tags=tuple(sorted(tags)),
    )


def _json_item(
    category: Literal["config", "json"],
    index: int,
    language: Language,
    *,
    prompt: str,
    expected: dict[str, object],
    tags: tuple[str, ...],
) -> EvaluationItem:
    answer = _canonical_json(expected)
    return EvaluationItem(
        id=_item_id(category, index),
        language=language,
        category=category,
        prompt_messages=(
            EvaluationPromptMessage(role="user", content=_holdout_prompt(prompt, language)),
        ),
        reference_answer=answer,
        scorer=JsonObjectScorer(
            kind="json_object",
            expected_json=answer,
            required_keys=tuple(sorted(expected)),
        ),
        provenance=_provenance(),
        tags=tuple(sorted(tags)),
    )


def _choice_item(
    category: Literal["logs"],
    index: int,
    language: Language,
    *,
    stem: str,
    choices: tuple[str, ...],
    answer_index: int,
    tags: tuple[str, ...],
) -> EvaluationItem:
    labels = ("A", "B", "C", "D")
    option_lines = "\n".join(f"{labels[offset]}. {choice}" for offset, choice in enumerate(choices))
    instruction = (
        "Choose the single diagnosis most directly supported by the log. Return only the exact "
        "option text, without its letter."
        if language == "en"
        else "选择日志最直接支持的唯一诊断。仅返回选项原文，不要返回字母。"
    )
    prompt = f"{stem}\n\n{option_lines}\n\n{instruction}"
    answer = choices[answer_index]
    return EvaluationItem(
        id=_item_id(category, index),
        language=language,
        category=category,
        prompt_messages=(
            EvaluationPromptMessage(role="user", content=_holdout_prompt(prompt, language)),
        ),
        reference_answer=answer,
        scorer=MultipleChoiceScorer(
            kind="multiple_choice",
            choices=choices,
            answer_index=answer_index,
        ),
        provenance=_provenance(),
        tags=tuple(sorted(tags)),
    )


def _python_item(index: int, language: Language) -> EvaluationItem:
    item_index = index
    semantic_index = _semantic_index("python", item_index, language)
    cycle, mode = divmod(_value_index(semantic_index), 10)
    prefix = "Evaluate this Python 3 expression" if language == "en" else "计算以下 Python 3 表达式"
    suffix = (
        "Return only the exact Python repr of the result, with no explanation."
        if language == "en"
        else "仅返回结果的精确 Python repr，不要解释。"
    )
    if mode == 0:
        stop = 7 + cycle
        expression = f"sum(i * i for i in range({stop}) if i % 2 == 0)"
        answer = repr(sum(i * i for i in range(stop) if i % 2 == 0))
        topic = "comprehension"
    elif mode == 1:
        text = "tinyllmsystem" if _ACTIVE_VARIANT == "v1" else "distributedllm"
        start = 1 + cycle % 3
        end = 7 + cycle % 4
        expression = f"{text!r}[{start}:{end}]"
        answer = repr(text[start:end])
        topic = "slicing"
    elif mode == 2:
        modulus = 3 + cycle
        stop = 8 + cycle
        expression = f"sorted({{x % {modulus} for x in range({stop})}})"
        answer = repr(sorted({x % modulus for x in range(stop)}))
        topic = "set"
    elif mode == 3:
        left, right = 29 + 3 * cycle, 4 + cycle
        expression = f"divmod({left}, {right})"
        answer = repr(divmod(left, right))
        topic = "arithmetic"
    elif mode == 4:
        start, stop, step = 2 + cycle, 20 + cycle, 3
        values = list(range(start, stop, step))
        expression = f"list(range({start}, {stop}, {step}))[-1]"
        answer = repr(values[-1])
        topic = "range"
    elif mode == 5:
        values = [cycle, cycle + 2, cycle + 4]
        limit = cycle + (5 if cycle % 2 == 0 else 4)
        expression = f"all(x < {limit} for x in {values!r})"
        answer = repr(all(value < limit for value in values))
        topic = "boolean"
    elif mode == 6:
        text = ",".join(str(value) for value in range(cycle + 2)) + ","
        expression = f"len({text!r}.split(','))"
        answer = repr(len(text.split(",")))
        topic = "string"
    elif mode == 7:
        words = ["bb", "a", f"c{cycle}", "aa"]
        expression = f"sorted({words!r}, key=lambda s: (len(s), s))"
        answer = repr(sorted(words, key=lambda value: (len(value), value)))
        topic = "sorting"
    elif mode == 8:
        text = "abacabad" + "a" * cycle
        expression = f"max(set({text!r}), key={text!r}.count)"
        answer = repr(max(set(text), key=text.count))
        topic = "counting"
    else:
        mapping = {"alpha": cycle, "beta": cycle + 1}
        expression = f"({mapping!r}.get('gamma', {cycle + 9}), len({mapping!r}))"
        answer = repr((mapping.get("gamma", cycle + 9), len(mapping)))
        topic = "mapping"
    prompt = f"{prefix}:\n\n{expression}\n\n{suffix}"
    return _exact_item(
        "python",
        item_index,
        language,
        prompt=prompt,
        answer=answer,
        tags=(*_pair_tags("python", semantic_index), "deterministic", topic),
    )


def _linux_item(index: int, language: Language) -> EvaluationItem:
    item_index = index
    semantic_index = _semantic_index("linux", item_index, language)
    cycle, mode = divmod(_value_index(semantic_index), 9)
    only = "Return only the exact answer." if language == "en" else "仅返回精确答案。"
    if mode == 0:
        modes = (
            ((6, 4, 0), (7, 5, 0), (6, 0, 0), (7, 0, 0), (6, 4, 4))
            if _ACTIVE_VARIANT == "v1"
            else ((7, 7, 0), (7, 1, 1), (6, 6, 0), (7, 7, 7), (6, 0, 4))
        )
        owner, group, other = modes[cycle % len(modes)]
        symbols = {
            0: "---",
            1: "--x",
            2: "-w-",
            3: "-wx",
            4: "r--",
            5: "r-x",
            6: "rw-",
            7: "rwx",
        }
        prompt = (
            f"For Linux mode {owner}{group}{other}, give the nine permission characters "
            f"without the file-type prefix. {only}"
            if language == "en"
            else (
                f"Linux 权限模式 {owner}{group}{other} 对应的九个权限字符是什么"
                f"（不含文件类型前缀）？{only}"
            )
        )
        answer = symbols[owner] + symbols[group] + symbols[other]
        topic = "permissions"
    elif mode == 1:
        left, right = cycle + 2, cycle + 4
        prompt = (
            f"What exit status does `test {left} -lt {right}` return when it is the only "
            f"command? {only}"
            if language == "en"
            else f"`test {left} -lt {right}` 单独执行时返回什么退出状态？{only}"
        )
        answer = "0"
        topic = "exit-status"
    elif mode == 2:
        lines = cycle + 2
        payload = "\\n".join(str(value) for value in range(lines)) + "\\n"
        prompt = (
            f"What integer does `printf '{payload}' | wc -l` print? {only}"
            if language == "en"
            else f"`printf '{payload}' | wc -l` 会输出哪个整数？{only}"
        )
        answer = str(lines)
        topic = "pipeline"
    elif mode == 3:
        count = 10 + cycle
        path = f"/var/log/app{cycle}.log"
        prompt = (
            f"Give the command that prints the last {count} lines of `{path}`. {only}"
            if language == "en"
            else f"给出打印 `{path}` 最后 {count} 行的命令。{only}"
        )
        answer = f"tail -n {count} {path}"
        topic = "logs"
    elif mode == 4:
        name, value = f"MODE{cycle}", f"test{cycle}"
        prompt = (
            f"Run `python app.py` with `{name}` set only for that process to `{value}`. "
            f"Give the one-line command. {only}"
            if language == "en"
            else f"仅为本次 `python app.py` 进程设置 `{name}={value}`。给出单行命令。{only}"
        )
        answer = f"{name}={value} python app.py"
        topic = "environment"
    elif mode == 5:
        path = f"/srv/data{cycle}"
        prompt = (
            f"Give the command that reports the total human-readable disk usage of `{path}`. {only}"
            if language == "en"
            else f"给出统计 `{path}` 总占用并以易读单位显示的命令。{only}"
        )
        answer = f"du -sh {path}"
        topic = "storage"
    elif mode == 6:
        pid = 4100 + cycle
        prompt = (
            f"Give the command that sends SIGTERM to PID {pid}. {only}"
            if language == "en"
            else f"给出向 PID {pid} 发送 SIGTERM 的命令。{only}"
        )
        answer = f"kill -TERM {pid}"
        topic = "process"
    elif mode == 7:
        root = f"/srv/app{cycle}"
        prompt = (
            f"Give the `find` command that lists regular `*.log` files below `{root}`. {only}"
            if language == "en"
            else f"给出在 `{root}` 下查找普通 `*.log` 文件的 `find` 命令。{only}"
        )
        answer = f"find {root} -type f -name '*.log'"
        topic = "find"
    else:
        port = 8000 + cycle
        prompt = (
            "Give the `ss` command that lists listening TCP sockets and filter its output "
            f"for port {port}. {only}"
            if language == "en"
            else f"给出列出监听 TCP Socket 并筛选端口 {port} 的 `ss` 命令。{only}"
        )
        answer = f"ss -ltn | grep ':{port} '"
        topic = "network"
    return _exact_item(
        "linux",
        item_index,
        language,
        prompt=prompt,
        answer=answer,
        tags=(*_pair_tags("linux", semantic_index), "command", topic),
    )


def _json_task_item(index: int, language: Language) -> EvaluationItem:
    item_index = index
    semantic_index = _semantic_index("json", item_index, language)
    cycle, mode = divmod(_value_index(semantic_index), 8)
    base: dict[str, object] = {
        "enabled": cycle % 2 == 0,
        "name": f"worker-{cycle}",
        "retries": cycle + 1,
        "tags": ["batch", f"v{cycle}"],
    }
    expected: dict[str, object]
    if mode == 0:
        expected = {"id": cycle, "ok": cycle % 2 == 0}
        task_en = (
            f"Create an object with integer `id` {cycle} and boolean `ok` "
            f"{str(cycle % 2 == 0).lower()}."
        )
        task_zh = f"创建对象：整数 `id` 为 {cycle}，布尔值 `ok` 为 {str(cycle % 2 == 0).lower()}。"
    elif mode == 1:
        expected = {"name": base["name"], "retries": base["retries"]}
        task_en = f"From this object, keep only `name` and `retries`: {_canonical_json(base)}"
        task_zh = f"从以下对象中仅保留 `name` 和 `retries`：{_canonical_json(base)}"
    elif mode == 2:
        expected = {**base, "retries": cycle + 5}
        task_en = f"Change `retries` to {cycle + 5} in: {_canonical_json(base)}"
        task_zh = f"将以下对象的 `retries` 改为 {cycle + 5}：{_canonical_json(base)}"
    elif mode == 3:
        expected = {"items": [cycle, cycle + 1, cycle + 2], "size": 3}
        task_en = f"Create `items` as [{cycle},{cycle + 1},{cycle + 2}] and `size` as its length."
        task_zh = f"创建 `items` 为 [{cycle},{cycle + 1},{cycle + 2}]，并令 `size` 为其长度。"
    elif mode == 4:
        expected = {"service": {"name": base["name"], "enabled": True}}
        task_en = (
            f"Nest service name `{base['name']}` and boolean enabled true under key `service`."
        )
        task_zh = f"在 `service` 键下嵌套名称 `{base['name']}` 和布尔值 enabled=true。"
    elif mode == 5:
        expected = {"even": [value for value in range(cycle, cycle + 6) if value % 2 == 0]}
        task_en = f"Keep only even integers from {list(range(cycle, cycle + 6))} under key `even`."
        task_zh = f"从 {list(range(cycle, cycle + 6))} 中仅保留偶数，放在键 `even` 下。"
    elif mode == 6:
        expected = {"labels": sorted(set(["api", f"v{cycle}", "api"]))}
        task_en = f'Deduplicate and lexicographically sort ["api","v{cycle}","api"] under `labels`.'
        task_zh = f'将 ["api","v{cycle}","api"] 去重并按字典序排序，放在 `labels` 下。'
    else:
        expected = {key: base[key] for key in sorted(base) if key != "tags"}
        task_en = f"Remove key `tags` from: {_canonical_json(base)}"
        task_zh = f"从以下对象中删除键 `tags`：{_canonical_json(base)}"
    instruction = (
        " Return only canonical JSON with sorted keys and no spaces."
        if language == "en"
        else " 仅返回键已排序且无空格的 Canonical JSON。"
    )
    return _json_item(
        "json",
        item_index,
        language,
        prompt=(task_en if language == "en" else task_zh) + instruction,
        expected=expected,
        tags=(*_pair_tags("json", semantic_index), "canonical-json", f"operation-{mode}"),
    )


def _config_item(index: int, language: Language) -> EvaluationItem:
    item_index = index
    semantic_index = _semantic_index("config", item_index, language)
    cycle, mode = divmod(_value_index(semantic_index), 8)
    config: dict[str, object] = {
        "data": {"workers": 2 + cycle},
        "logging": {"level": "INFO"},
        "model": {"name": f"tiny-{cycle}", "precision": "bf16"},
        "training": {"batch_size": 4, "epochs": 2},
    }
    if mode == 0:
        cast_training = cast(dict[str, object], config["training"]).copy()
        cast_training["batch_size"] = 8 + cycle
        expected = {**config, "training": cast_training}
        action_en = f"set `training.batch_size` to {8 + cycle}"
        action_zh = f"将 `training.batch_size` 设为 {8 + cycle}"
    elif mode == 1:
        cast_logging = cast(dict[str, object], config["logging"]).copy()
        cast_logging["level"] = "DEBUG"
        expected = {**config, "logging": cast_logging}
        action_en = "set `logging.level` to `DEBUG`"
        action_zh = "将 `logging.level` 设为 `DEBUG`"
    elif mode == 2:
        expected = {**config, "seed": 42 + cycle}
        action_en = f"add top-level integer `seed` {42 + cycle}"
        action_zh = f"新增顶层整数 `seed`={42 + cycle}"
    elif mode == 3:
        cast_model = cast(dict[str, object], config["model"]).copy()
        cast_model["gradient_checkpointing"] = True
        expected = {**config, "model": cast_model}
        action_en = "add `model.gradient_checkpointing` as boolean true"
        action_zh = "新增布尔值 `model.gradient_checkpointing=true`"
    elif mode == 4:
        expected = {key: value for key, value in config.items() if key != "logging"}
        action_en = "remove the top-level `logging` section"
        action_zh = "删除顶层 `logging` 部分"
    elif mode == 5:
        cast_data = cast(dict[str, object], config["data"]).copy()
        cast_data["workers"] = 4 + cycle
        expected = {**config, "data": cast_data}
        action_en = f"set `data.workers` to {4 + cycle}"
        action_zh = f"将 `data.workers` 设为 {4 + cycle}"
    elif mode == 6:
        cast_training = cast(dict[str, object], config["training"]).copy()
        cast_training["epochs"] = 3 + cycle
        expected = {**config, "training": cast_training}
        action_en = f"set `training.epochs` to {3 + cycle}"
        action_zh = f"将 `training.epochs` 设为 {3 + cycle}"
    else:
        cast_model = cast(dict[str, object], config["model"]).copy()
        cast_model["precision"] = "fp32"
        expected = {**config, "model": cast_model}
        action_en = "set `model.precision` to `fp32`"
        action_zh = "将 `model.precision` 设为 `fp32`"
    prompt = (
        f"Apply exactly this configuration change: {action_en}. Input: {_canonical_json(config)}. "
        "Return only the complete resulting canonical JSON with sorted keys and no spaces."
        if language == "en"
        else (
            f"仅执行此配置修改：{action_zh}。输入：{_canonical_json(config)}。"
            "仅返回完整结果的 Canonical JSON，键排序且无空格。"
        )
    )
    return _json_item(
        "config",
        item_index,
        language,
        prompt=prompt,
        expected=expected,
        tags=(*_pair_tags("config", semantic_index), "config-edit", f"operation-{mode}"),
    )


LOG_PATTERNS: tuple[tuple[str, tuple[str, str], tuple[tuple[str, str], ...]], ...] = (
    (
        "connect() failed: Connection refused while connecting to upstream 10.0.0.{variant}:80",
        ("The target port is not accepting connections.", "目标端口未接受连接。"),
        (
            ("The local JSON file is malformed.", "本地 JSON 文件格式错误。"),
            ("The GPU ran out of memory.", "GPU 显存不足。"),
            ("The checkpoint checksum is invalid.", "Checkpoint 校验和无效。"),
        ),
    ),
    (
        "OSError: [Errno 28] No space left on device: '/var/tmp/run-{variant}'",
        (
            "The target filesystem has no free space for the write.",
            "目标文件系统没有可用于写入的空间。",
        ),
        (
            ("DNS lookup failed.", "DNS 查询失败。"),
            ("The process lacks CUDA support.", "进程缺少 CUDA 支持。"),
            ("The port is already bound.", "端口已被占用。"),
        ),
    ),
    (
        "PermissionError: [Errno 13] Permission denied: '/srv/model-{variant}/config.json'",
        ("The process lacks permission to access the path.", "进程没有访问该路径的权限。"),
        (
            ("The disk is full.", "磁盘已满。"),
            ("The JSON contains a trailing comma.", "JSON 包含尾随逗号。"),
            ("The remote service refused a connection.", "远程服务拒绝连接。"),
        ),
    ),
    (
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate {variant}.00 GiB",
        (
            "The CUDA allocation exceeded currently available GPU memory.",
            "CUDA 分配超过当前可用 GPU 显存。",
        ),
        (
            ("The host name cannot be resolved.", "主机名无法解析。"),
            ("The checkpoint hash mismatched.", "Checkpoint 哈希不匹配。"),
            ("The process lacks file permission.", "进程缺少文件权限。"),
        ),
    ),
    (
        "OSError: [Errno 98] Address already in use ('0.0.0.0', {port})",
        (
            "Another socket is already bound to the requested address and port.",
            "已有其他 Socket 绑定了请求的地址和端口。",
        ),
        (
            ("The target disk is read-only.", "目标磁盘是只读的。"),
            ("The GPU driver is absent.", "GPU 驱动不存在。"),
            ("The JSON schema rejected an extra field.", "JSON Schema 拒绝了额外字段。"),
        ),
    ),
    (
        "socket.gaierror: [Errno -2] Name or service not known: worker-{variant}.invalid",
        ("The host name could not be resolved.", "主机名无法解析。"),
        (
            ("The TCP port is already bound locally.", "本地 TCP 端口已被占用。"),
            ("The CUDA allocator is exhausted.", "CUDA 分配器已耗尽。"),
            ("The file owner is incorrect.", "文件所有者不正确。"),
        ),
    ),
    (
        "CHECKPOINT_CORRUPT: expected sha256=aaaa{variant}, observed=bbbb{variant}",
        (
            "The checkpoint content failed its recorded integrity hash.",
            "Checkpoint 内容未通过记录的完整性哈希。",
        ),
        (
            ("A DNS lookup timed out.", "DNS 查询超时。"),
            ("The process received SIGTERM.", "进程收到了 SIGTERM。"),
            ("The batch size is zero.", "Batch Size 为零。"),
        ),
    ),
    (
        "NCCL WARN Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum={variant}, OpType=ALLREDUCE)",
        (
            "An NCCL all-reduce did not complete before the watchdog timeout.",
            "NCCL All-reduce 未在 Watchdog 超时前完成。",
        ),
        (
            ("The model tokenizer vocabulary is empty.", "模型 Tokenizer 词表为空。"),
            ("The HTTP port is occupied.", "HTTP 端口被占用。"),
            ("The YAML file uses tabs.", "YAML 文件使用了 Tab。"),
        ),
    ),
    (
        "json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: "
        "line {variant} column 3",
        ("The input is not valid JSON at the reported location.", "输入在报告位置不是合法 JSON。"),
        (
            ("The GPU is overheating.", "GPU 过热。"),
            ("The remote TCP port refused a connection.", "远程 TCP 端口拒绝连接。"),
            ("The filesystem is full.", "文件系统已满。"),
        ),
    ),
)


def _log_item(index: int, language: Language) -> EvaluationItem:
    item_index = index
    semantic_index = _semantic_index("logs", item_index, language)
    value_index = _value_index(semantic_index)
    cycle, mode = divmod(value_index, len(LOG_PATTERNS))
    template, correct, distractors = LOG_PATTERNS[mode]
    stem = template.format(variant=cycle + 1, port=8100 + cycle)
    localized = [correct[0 if language == "en" else 1]] + [
        pair[0 if language == "en" else 1] for pair in distractors
    ]
    rotation = (value_index * 3) % len(localized)
    choices = tuple(localized[rotation:] + localized[:rotation])
    answer_index = choices.index(correct[0 if language == "en" else 1])
    return _choice_item(
        "logs",
        item_index,
        language,
        stem=f"Log:\n{stem}" if language == "en" else f"日志：\n{stem}",
        choices=choices,
        answer_index=answer_index,
        tags=(*_pair_tags("logs", semantic_index), "diagnosis", f"pattern-{mode}"),
    )


SHORT_CODE_CASES: tuple[tuple[str, str, str, str], ...] = (
    (
        "normalized = TODO",
        "value.strip()",
        "remove leading and trailing whitespace from `value`",
        "移除 `value` 首尾空白",
    ),
    (
        "result = TODO",
        "mapping.get(key, default)",
        "read `key` from `mapping` and use `default` when absent",
        "从 `mapping` 读取 `key`，缺失时使用 `default`",
    ),
    (
        "for index, item in TODO:\n    emit(index, item)",
        "enumerate(items, start=1)",
        "enumerate `items` starting at one",
        "从一开始枚举 `items`",
    ),
    (
        "text = TODO",
        '"\\n".join(lines)',
        "join `lines` with a newline separator",
        "使用换行符连接 `lines`",
    ),
    (
        "keys = TODO",
        "sorted(mapping)",
        "return mapping keys in ascending order",
        "按升序返回 `mapping` 的键",
    ),
    (
        "has_negative = TODO",
        "any(item < 0 for item in values)",
        "test whether any value is negative",
        "判断 `values` 中是否存在负数",
    ),
    (
        "from pathlib import Path\nsuffix = TODO",
        "Path(name).suffix",
        "read the final filename suffix with `pathlib.Path`",
        "使用 `pathlib.Path` 读取最后一个文件后缀",
    ),
    (
        "unique = TODO",
        "list(dict.fromkeys(items))",
        "deduplicate `items` while preserving first occurrence order",
        "保持首次出现顺序对 `items` 去重",
    ),
    (
        "total = TODO",
        "sum(item * item for item in values)",
        "sum the squares of all `values`",
        "计算 `values` 中所有值的平方和",
    ),
    (
        "present = TODO",
        "[item for item in items if item is not None]",
        "remove only `None` entries from `items`",
        "仅删除 `items` 中的 `None` 项",
    ),
    (
        "clamped = TODO",
        "min(max(value, lower), upper)",
        "clamp `value` to the inclusive `lower` and `upper` bounds",
        "将 `value` 限制在闭区间 `lower` 到 `upper` 内",
    ),
    (
        "from pathlib import Path\nfilename = TODO",
        "Path(path).name",
        "read the final filename component from `path`",
        "读取 `path` 的最后一个文件名组件",
    ),
    (
        "normalized = TODO",
        "text.casefold()",
        "normalize `text` for caseless comparison",
        "规范化 `text` 以进行不区分大小写的比较",
    ),
    (
        "flat = TODO",
        "[item for group in groups for item in group]",
        "flatten `groups` by exactly one level",
        "将 `groups` 恰好展平一层",
    ),
    (
        "lookup = TODO",
        "dict(zip(keys, values, strict=True))",
        "build a dictionary from equal-length `keys` and `values`",
        "从等长的 `keys` 和 `values` 构建字典",
    ),
    (
        "tail = TODO",
        "items[-3:]",
        "take at most the final three `items`",
        "取得 `items` 最后的至多三项",
    ),
    (
        "parts = TODO",
        'line.split("=", maxsplit=1)',
        "split `line` at only the first equals sign",
        "仅在第一个等号处分割 `line`",
    ),
    (
        "all_positive = TODO",
        "all(item > 0 for item in values)",
        "test whether every value is strictly positive",
        "判断所有值是否都严格大于零",
    ),
    (
        "relative = TODO",
        "text.removeprefix(prefix)",
        "remove `prefix` from `text` only when present",
        "仅当 `text` 存在 `prefix` 时移除该前缀",
    ),
    (
        "merged = TODO",
        "{**defaults, **overrides}",
        "merge mappings so `overrides` wins on duplicate keys",
        "合并映射，并让 `overrides` 覆盖重复键",
    ),
    (
        "ordered = TODO",
        'sorted(records, key=lambda item: item["name"])',
        "sort `records` by each record's `name` field",
        "按照每条记录的 `name` 字段排序 `records`",
    ),
    (
        "match = TODO",
        "next((item for item in items if predicate(item)), None)",
        "return the first matching item or `None`",
        "返回第一个满足条件的项，若没有则返回 `None`",
    ),
    (
        "from pathlib import Path\nis_regular = TODO",
        "Path(path).is_file()",
        "test whether `path` currently names a regular file",
        "判断 `path` 当前是否指向普通文件",
    ),
    (
        "from pathlib import Path\nparent = TODO",
        "Path(path).parent",
        "return the parent path object for `path`",
        "返回 `path` 的父路径对象",
    ),
    (
        "mean = TODO",
        "sum(values) / len(values)",
        "compute the arithmetic mean of a known non-empty `values` list",
        "计算已知非空列表 `values` 的算术平均值",
    ),
    (
        "window = TODO",
        "items[start:stop]",
        "take the half-open slice from `start` through `stop`",
        "取得从 `start` 到 `stop` 的左闭右开切片",
    ),
    (
        "shared = TODO",
        "left.keys() & right.keys()",
        "return the keys shared by mappings `left` and `right`",
        "返回映射 `left` 与 `right` 共有的键",
    ),
    (
        "hex_id = TODO",
        'f"{value:08x}"',
        "format integer `value` as eight lowercase hexadecimal digits",
        "将整数 `value` 格式化为八位小写十六进制数",
    ),
)

SHORT_CODE_CASES_V2: tuple[tuple[str, str, str, str], ...] = (
    (
        "slug = TODO",
        'value.replace("_", "-")',
        "replace underscores in `value` with hyphens",
        "将 `value` 中的下划线替换为连字符",
    ),
    (
        "trimmed = TODO",
        'text.rstrip("\\n")',
        "remove only trailing newline characters from `text`",
        "仅移除 `text` 末尾的换行符",
    ),
    (
        "result = TODO",
        "mapping[key] if key in mapping else default",
        "read `key` from `mapping` and otherwise use `default`",
        "从 `mapping` 读取 `key`，否则使用 `default`",
    ),
    (
        "pairs = TODO",
        "list(zip(keys, values, strict=True))",
        "pair equal-length `keys` and `values` into a list",
        "将等长的 `keys` 与 `values` 配对为列表",
    ),
    (
        "text = TODO",
        '" ".join(words)',
        "join `words` with one space",
        "使用一个空格连接 `words`",
    ),
    (
        "ordered = TODO",
        "sorted(values, reverse=True)",
        "sort `values` in descending order",
        "按降序排列 `values`",
    ),
    (
        "complete = TODO",
        "all(item is not None for item in items)",
        "test whether every item is not `None`",
        "判断每一项都不是 `None`",
    ),
    (
        "from pathlib import Path\nstem = TODO",
        "Path(path).stem",
        "read the filename stem from `path`",
        "读取 `path` 的文件名 Stem",
    ),
    (
        "unique = TODO",
        "set(items)",
        "collect the unique values from `items` as a set",
        "将 `items` 的唯一值收集为集合",
    ),
    (
        "total = TODO",
        "sum(values, start=10)",
        "sum `values` starting from ten",
        "以十为初值求 `values` 的总和",
    ),
    (
        "truthy = TODO",
        "[item for item in items if item]",
        "keep only truthy entries from `items`",
        "仅保留 `items` 中的真值项",
    ),
    (
        "bounded = TODO",
        "max(lower, min(value, upper))",
        "bound `value` to the inclusive lower and upper limits",
        "将 `value` 限制在闭区间上下界内",
    ),
    (
        "from pathlib import Path\nabsolute = TODO",
        "Path(path).resolve()",
        "create the resolved absolute path object for `path`",
        "创建 `path` 解析后的绝对路径对象",
    ),
    (
        "normalized = TODO",
        "text.lower()",
        "convert `text` to lowercase",
        "将 `text` 转为小写",
    ),
    (
        "from itertools import chain\nflat = TODO",
        "list(chain.from_iterable(groups))",
        "flatten `groups` by one level with `chain`",
        "使用 `chain` 将 `groups` 展平一层",
    ),
    (
        "pairs_by_key = TODO",
        "{key: value for key, value in zip(keys, values, strict=True)}",
        "map each `key` to its corresponding `value` with a comprehension",
        "使用推导式将每个 `key` 映射到对应的 `value`",
    ),
    (
        "head = TODO",
        "items[:3]",
        "take at most the first three `items`",
        "取得 `items` 最前面的至多三项",
    ),
    (
        "parts = TODO",
        'line.partition("=")',
        "partition `line` around its first equals sign",
        "围绕第一个等号拆分 `line`",
    ),
    (
        "strictly_positive = TODO",
        "not any(item <= 0 for item in values)",
        "confirm no value is zero or negative",
        "确认没有值为零或负数",
    ),
    (
        "base = TODO",
        "text.removesuffix(suffix)",
        "remove `suffix` from `text` only when present",
        "仅当存在时移除 `text` 的 `suffix`",
    ),
    (
        "merged = TODO",
        "defaults | overrides",
        "merge dictionaries so `overrides` wins",
        "合并字典并让 `overrides` 优先",
    ),
    (
        "ordered = TODO",
        'sorted(records, key=lambda item: item["score"], reverse=True)',
        "sort records by descending `score`",
        "按 `score` 降序排列记录",
    ),
    (
        "first_valid = TODO",
        "next(filter(predicate, items), None)",
        "select the first item accepted by `predicate`, defaulting to `None`",
        "选择首个被 `predicate` 接受的项，默认返回 `None`",
    ),
    (
        "from pathlib import Path\nexists = TODO",
        "Path(path).exists()",
        "test whether `path` currently exists",
        "判断 `path` 当前是否存在",
    ),
    (
        "from pathlib import Path\noutput = TODO",
        'Path(path).with_suffix(".json")',
        "replace the final suffix of `path` with `.json`",
        "将 `path` 的最后一个后缀替换为 `.json`",
    ),
    (
        "from statistics import fmean\nmean = TODO",
        "fmean(values)",
        "compute the floating-point mean of non-empty `values`",
        "计算非空 `values` 的浮点平均值",
    ),
    (
        "alternating = TODO",
        "items[::2]",
        "take every second item starting with the first",
        "从第一项开始每隔一项取值",
    ),
    (
        "decimal_id = TODO",
        'f"{value:06d}"',
        "format integer `value` as six zero-padded decimal digits",
        "将整数 `value` 格式化为六位前导零十进制数",
    ),
)


def _short_code_item(index: int, language: Language) -> EvaluationItem:
    item_index = index
    semantic_index = _semantic_index("short_code", item_index, language)
    cases = SHORT_CODE_CASES if _ACTIVE_VARIANT == "v1" else SHORT_CODE_CASES_V2
    snippet, answer, goal_en, goal_zh = cases[semantic_index]
    prompt = (
        f"Replace `TODO` so the code will {goal_en}. Return only the replacement expression.\n\n"
        f"```python\n{snippet}\n```"
        if language == "en"
        else f"替换 `TODO`，使代码能够{goal_zh}。仅返回替换表达式。\n\n```python\n{snippet}\n```"
    )
    return _exact_item(
        "short_code",
        item_index,
        language,
        prompt=prompt,
        answer=answer,
        tags=(*_pair_tags("short_code", semantic_index), "code-completion"),
    )


REFUSAL_SCENARIOS: tuple[tuple[str, str, str, str], ...] = (
    (
        "The incident ticket says 'service failed', but includes no logs or timestamp.",
        "工单只写着“服务失败”，但没有提供日志或时间戳。",
        "the complete error log and timestamp",
        "完整错误日志和时间戳",
    ),
    (
        "A user asks whether a run enabled BF16, but the configuration file is not attached.",
        "用户询问某次运行是否启用了 BF16，但未附配置文件。",
        "the resolved configuration snapshot",
        "解析后的配置快照",
    ),
    (
        "A report asks why throughput dropped, but provides no metrics or profiler trace.",
        "报告询问吞吐下降原因，但未提供指标或 Profiler Trace。",
        "the step-time metrics and profiler trace",
        "Step-time 指标和 Profiler Trace",
    ),
    (
        "A review asks for the bug on a referenced line, but the source file is not provided.",
        "评审要求判断某引用行的 Bug，但没有提供源文件。",
        "the source file and surrounding lines",
        "源文件及相关上下文行",
    ),
    (
        "A checkpoint is called unrecoverable, but its manifest and hash results are absent.",
        "有人称 Checkpoint 无法恢复，但未提供 Manifest 和哈希结果。",
        "the checkpoint manifest and integrity-check output",
        "Checkpoint Manifest 和完整性检查输出",
    ),
    (
        "A claim cites a benchmark table, but the table and source link are unavailable.",
        "某结论引用 Benchmark 表，但表格和来源链接不可用。",
        "the benchmark table and verifiable source",
        "Benchmark 表格和可验证来源",
    ),
    (
        "A grader asks whether a model response is valid JSON, but the response text is missing.",
        "评分者询问模型响应是否为合法 JSON，但响应文本缺失。",
        "the exact raw model response",
        "模型的精确原始响应",
    ),
    (
        "A failure is attributed to a command, but the command and reproduction steps are absent.",
        "某失败被归因于一条命令，但命令和复现步骤未提供。",
        "the exact command and reproduction steps",
        "精确命令和复现步骤",
    ),
    (
        "A CUDA OOM is blamed on model size, but no allocator trace or GPU inventory is shown.",
        "有人将 CUDA OOM 归因于模型大小，但未提供分配器 Trace 或 GPU 清单。",
        "the allocator trace and GPU memory inventory",
        "分配器 Trace 和 GPU 显存清单",
    ),
    (
        "A loss spike is blamed on one batch, but neither the loss series nor batch IDs are saved.",
        "有人将 Loss 突增归因于某个 Batch，但未保存 Loss 序列或 Batch ID。",
        "the loss series and corresponding batch identifiers",
        "Loss 序列及对应的 Batch 标识",
    ),
    (
        "A DDP hang is attributed to one rank, but per-rank and NCCL logs are unavailable.",
        "有人将 DDP 卡死归因于某个 Rank，但没有各 Rank 日志和 NCCL 日志。",
        "the logs from every rank and the NCCL diagnostic output",
        "所有 Rank 的日志和 NCCL 诊断输出",
    ),
    (
        "A dataset is declared license-compliant without source provenance or license records.",
        "某数据集在缺少来源血缘和许可证记录时被宣称许可合规。",
        "the source provenance and per-source license manifest",
        "来源血缘和逐来源许可证 Manifest",
    ),
    (
        "Exact Resume is claimed, but uninterrupted and resumed state snapshots are missing.",
        "有人声称实现了 Exact Resume，但缺少无中断与恢复运行的状态快照。",
        "the uninterrupted and resumed state hashes at matching steps",
        "相同步数下无中断与恢复状态的哈希",
    ),
    (
        "Training/evaluation leakage is alleged without split groups or duplicate evidence.",
        "有人声称训练集与评测集泄漏，但未提供 Split 分组或重复证据。",
        "the split-group manifest and duplicate-match report",
        "Split 分组 Manifest 和重复匹配报告",
    ),
    (
        "A dependency is called vulnerable without its version, advisory, or reachable code path.",
        "某依赖被称为存在漏洞，但没有版本、公告或可达代码路径。",
        "the installed version, advisory identifier, and reachable usage",
        "已安装版本、漏洞公告标识和可达用法",
    ),
    (
        "GPU thermal throttling is blamed for slowdown, but temperature and clock telemetry "
        "is absent.",
        "有人将变慢归因于 GPU 热降频，但没有温度和时钟遥测。",
        "the timestamped GPU temperature, power, and clock telemetry",
        "带时间戳的 GPU 温度、功率和时钟遥测",
    ),
    (
        "A downloaded artifact is called corrupt without expected and observed checksums.",
        "某下载 Artifact 被称为损坏，但未提供预期和实际校验和。",
        "the expected and observed artifact checksums",
        "Artifact 的预期与实际校验和",
    ),
    (
        "Model quality is said to improve, but baseline and candidate outputs are unavailable.",
        "有人声称模型质量提升，但没有 Baseline 和 Candidate 的输出。",
        "the frozen evaluation config and both sets of raw outputs",
        "冻结评测配置和两组原始输出",
    ),
    (
        "A deployed model is said to come from a run, but its registry lineage is missing.",
        "有人声称已部署模型来自某次 Run，但其 Registry 血缘缺失。",
        "the deployment artifact ID and complete registry lineage",
        "部署 Artifact ID 和完整 Registry 血缘",
    ),
    (
        "An import failure is blamed on a dependency conflict without a lockfile or traceback.",
        "有人将导入失败归因于依赖冲突，但没有 Lockfile 或 Traceback。",
        "the resolved dependency versions and complete traceback",
        "解析后的依赖版本和完整 Traceback",
    ),
    (
        "A job failure is blamed on a full disk without filesystem capacity evidence.",
        "有人将作业失败归因于磁盘已满，但没有文件系统容量证据。",
        "the relevant filesystem's `df` and directory-usage output",
        "相关文件系统的 `df` 和目录占用输出",
    ),
    (
        "A request timeout is blamed on DNS without endpoint timing or resolver output.",
        "有人将请求超时归因于 DNS，但没有端点计时或解析器输出。",
        "the endpoint timing, resolver output, and network error",
        "端点计时、解析器输出和网络错误",
    ),
    (
        "NaN loss is blamed on the optimizer without its config or gradient diagnostics.",
        "有人将 NaN Loss 归因于优化器，但没有其配置或梯度诊断。",
        "the optimizer config, gradient norms, and first non-finite step",
        "优化器配置、梯度范数和首个非有限值 Step",
    ),
    (
        "A sampler is said to duplicate examples without a sample-ID trace across steps.",
        "有人声称 Sampler 重复样本，但没有跨 Step 的 Sample ID Trace。",
        "the ordered sample identifiers for the affected steps",
        "受影响 Step 的有序 Sample 标识",
    ),
    (
        "Resume failure is blamed on World Size without the launch config or checkpoint manifest.",
        "有人将恢复失败归因于 World Size，但没有启动配置或 Checkpoint Manifest。",
        "the launch configuration and checkpoint compatibility manifest",
        "启动配置和 Checkpoint 兼容性 Manifest",
    ),
    (
        "An inference latency regression is claimed without lengths, concurrency, or raw timings.",
        "有人声称推理延迟回退，但没有长度、并发或原始计时。",
        "the input/output lengths, concurrency, and raw latency samples",
        "输入输出长度、并发和原始延迟样本",
    ),
    (
        "A model answer is called wrong, but the Prompt template and raw response are absent.",
        "有人称模型答案错误，但没有 Prompt Template 和原始响应。",
        "the exact Prompt, template revision, and raw model response",
        "精确 Prompt、Template Revision 和模型原始响应",
    ),
    (
        "A process death is attributed to SIGKILL without kernel or supervisor logs.",
        "有人将进程死亡归因于 SIGKILL，但没有内核或 Supervisor 日志。",
        "the kernel log and process-supervisor event record",
        "内核日志和进程 Supervisor 事件记录",
    ),
)

REFUSAL_SCENARIOS_V2: tuple[tuple[str, str, str, str], ...] = (
    (
        "A dataset shard is called corrupt, but neither its source hash nor read error is shown.",
        "有人称数据分片已损坏，但没有提供来源哈希或读取错误。",
        "the expected source hash and exact read error",
        "预期来源哈希和精确读取错误",
    ),
    (
        "Training is called unusually slow, but there is no baseline, hardware snapshot, "
        "or timing.",
        "有人称训练异常缓慢，但没有基线、硬件快照或计时。",
        "the comparable baseline, hardware snapshot, and step timings",
        "可比基线、硬件快照和 Step 计时",
    ),
    (
        "An inference response is called a hallucination, but its prompt and raw output "
        "are absent.",
        "有人称一次推理响应是幻觉，但缺少 Prompt 和原始输出。",
        "the exact prompt, decoding config, and raw output",
        "精确 Prompt、解码配置和原始输出",
    ),
    (
        "A process is said to be deadlocked, but no stack traces or thread states were captured.",
        "有人称进程发生死锁，但没有捕获堆栈或线程状态。",
        "the per-thread stack traces and process-state snapshot",
        "逐线程堆栈和进程状态快照",
    ),
    (
        "Resume drift is blamed on RNG state, but matching-step state dumps are unavailable.",
        "有人将恢复漂移归因于 RNG 状态，但没有相同步数的状态转储。",
        "the matching-step RNG, sampler, and optimizer state dumps",
        "相同步数的 RNG、Sampler 和优化器状态转储",
    ),
    (
        "A release is called license-incompatible without identifying the included files.",
        "有人称发布内容许可证不兼容，但没有指出所包含的具体文件。",
        "the artifact file inventory and each file's license provenance",
        "Artifact 文件清单和逐文件许可证血缘",
    ),
    (
        "An evaluation regression is claimed without seeds, configs, or item-level scores.",
        "有人声称评测发生回退，但没有提供 Seed、配置或逐项得分。",
        "both frozen configs, seeds, and item-level score files",
        "两份冻结配置、Seed 和逐项得分文件",
    ),
    (
        "A public artifact is said to expose a secret, but no file path or scanner finding "
        "is given.",
        "有人称公开 Artifact 泄露 Secret，但没有提供文件路径或扫描结果。",
        "the exact artifact path and reproducible scanner finding",
        "精确 Artifact 路径和可复现扫描结果",
    ),
    (
        "A model-card capability claim is disputed without the referenced report or raw evidence.",
        "有人质疑 Model Card 的能力声明，但没有提供所引用报告或原始证据。",
        "the cited report version and underlying raw evidence",
        "引用报告版本和底层原始证据",
    ),
    (
        "A tensor-shape bug is blamed for a crash, but the traceback and input shapes are missing.",
        "有人将崩溃归因于 Tensor Shape Bug，但缺少 Traceback 和输入 Shape。",
        "the complete traceback and tensor shapes at the failing operation",
        "完整 Traceback 和失败操作处的 Tensor Shape",
    ),
    (
        "Disk I/O is blamed for stalls, but no device latency or throughput telemetry is present.",
        "有人将卡顿归因于磁盘 I/O，但没有设备延迟或吞吐遥测。",
        "timestamped device latency, queue, and throughput telemetry",
        "带时间戳的设备延迟、队列和吞吐遥测",
    ),
    (
        "Network latency is blamed for a slow download without endpoint or route measurements.",
        "有人将下载缓慢归因于网络延迟，但没有端点或路由测量。",
        "endpoint timings, route diagnostics, and transfer samples",
        "端点计时、路由诊断和传输样本",
    ),
    (
        "Optimizer divergence is alleged without hyperparameters or the first divergent step.",
        "有人声称优化器发散，但没有超参数或首个发散 Step。",
        "the resolved optimizer config and metrics around the first divergent step",
        "解析后的优化器配置和首个发散 Step 附近指标",
    ),
    (
        "A dataset is called imbalanced without label, language, or source distributions.",
        "有人称数据集不平衡，但没有标签、语言或来源分布。",
        "the versioned label, language, and source distribution report",
        "版本化的标签、语言和来源分布报告",
    ),
    (
        "Quantization is blamed for accuracy loss without paired base and quantized outputs.",
        "有人将准确率下降归因于量化，但没有 Base 与量化模型的成对输出。",
        "paired base and quantized outputs under one frozen evaluation config",
        "同一冻结评测配置下 Base 与量化模型的成对输出",
    ),
    (
        "CPU offload is blamed for an OOM, but the allocation timeline is unavailable.",
        "有人将 OOM 归因于 CPU Offload，但没有分配时间线。",
        "the host and device allocation timeline with the resolved offload config",
        "主机与设备分配时间线及解析后的 Offload 配置",
    ),
    (
        "A container is said to differ from production without an image digest or "
        "environment diff.",
        "有人称容器与生产环境不同，但没有镜像摘要或环境差异。",
        "both image digests and a resolved environment comparison",
        "两份镜像摘要和解析后的环境对比",
    ),
    (
        "A CUDA version conflict is alleged without driver, runtime, and wheel versions.",
        "有人声称存在 CUDA 版本冲突，但没有 Driver、Runtime 和 Wheel 版本。",
        "the driver, CUDA runtime, and installed framework wheel versions",
        "Driver、CUDA Runtime 和已安装框架 Wheel 版本",
    ),
    (
        "Tokenizer drift is blamed for output changes without tokenizer revisions or hashes.",
        "有人将输出变化归因于 Tokenizer 漂移，但没有 Revision 或哈希。",
        "both tokenizer revisions, file hashes, and rendered prompts",
        "两份 Tokenizer Revision、文件哈希和渲染 Prompt",
    ),
    (
        "A chat-template bug is alleged, but the rendered prompt and template revision "
        "are missing.",
        "有人声称 Chat Template 存在 Bug，但缺少渲染 Prompt 和模板 Revision。",
        "the exact rendered prompt and versioned template source",
        "精确渲染 Prompt 和版本化模板源码",
    ),
    (
        "Deduplication is called ineffective without candidate pairs or similarity scores.",
        "有人称去重无效，但没有候选样本对或相似度得分。",
        "the candidate duplicate pairs, fingerprints, and thresholds",
        "候选重复样本对、指纹和阈值",
    ),
    (
        "A registry entry is said to point at the wrong model without artifact IDs or hashes.",
        "有人称 Registry 条目指向错误模型，但没有 Artifact ID 或哈希。",
        "the registry record, resolved artifact IDs, and content hashes",
        "Registry 记录、解析后的 Artifact ID 和内容哈希",
    ),
    (
        "An API timeout is blamed on the server without synchronized client and server logs.",
        "有人将 API 超时归因于服务端，但没有同步的客户端与服务端日志。",
        "synchronized client/server logs and request timing identifiers",
        "同步的客户端与服务端日志及请求计时标识",
    ),
    (
        "GPU throttling is claimed without synchronized temperature, power, and clock samples.",
        "有人声称 GPU 发生降频，但没有同步的温度、功率和时钟样本。",
        "synchronized temperature, power, utilization, and clock samples",
        "同步的温度、功率、利用率和时钟样本",
    ),
    (
        "A generated answer is called incorrect without a reference or scoring rule.",
        "有人称生成答案错误，但没有 Reference 或评分规则。",
        "the exact response, reference answer, and frozen scoring rule",
        "精确响应、Reference Answer 和冻结评分规则",
    ),
    (
        "NCCL topology is blamed for poor scaling without topology or per-rank traces.",
        "有人将扩展效率差归因于 NCCL 拓扑，但没有拓扑或逐 Rank Trace。",
        "the topology matrix, NCCL environment, and per-rank timing traces",
        "拓扑矩阵、NCCL 环境和逐 Rank 计时 Trace",
    ),
    (
        "A scheduler bug is alleged without the expected curve or recorded LR series.",
        "有人声称 Scheduler 存在 Bug，但没有预期曲线或已记录 LR 序列。",
        "the scheduler config, expected curve, and recorded per-step LR series",
        "Scheduler 配置、预期曲线和逐 Step LR 序列",
    ),
    (
        "A test is called flaky after one failure without repetition history or failure logs.",
        "某测试在单次失败后被称为 Flaky，但没有重复历史或失败日志。",
        "the repeated-run history, seed, environment, and complete failure log",
        "重复运行历史、Seed、环境和完整失败日志",
    ),
)

REFUSAL_SCENARIOS_V3: tuple[tuple[str, str, str, str], ...] = (
    (
        "A stale cache is blamed for inconsistent reads, but no cache key history or TTL trace "
        "is attached.",
        "有人将读取不一致归因于缓存过期，但没有缓存键历史或 TTL Trace。",
        "the cache key history, TTL configuration, and timestamped read trace",
        "缓存键历史、TTL 配置和带时间戳的读取 Trace",
    ),
    (
        "A data split is accused of leakage without fingerprints or group assignments.",
        "有人声称数据切分发生泄漏，但没有指纹或分组分配记录。",
        "the split assignments, content fingerprints, and duplicate-match evidence",
        "切分分配、内容指纹和重复匹配证据",
    ),
    (
        "Queue starvation is cited for a delayed job without scheduler events or queue snapshots.",
        "有人将作业延迟归因于队列饥饿，但没有调度事件或队列快照。",
        "the scheduler events, queue snapshots, and job priority history",
        "调度事件、队列快照和作业优先级历史",
    ),
    (
        "Checkpoint storage is blamed for a pause without file sizes or storage timings.",
        "有人将暂停归因于 Checkpoint 存储，但没有文件大小或存储计时。",
        "the checkpoint inventory, file sizes, and storage operation timings",
        "Checkpoint 清单、文件大小和存储操作计时",
    ),
    (
        "GPU memory fragmentation is claimed without allocator snapshots or allocation history.",
        "有人声称 GPU 显存碎片化，但没有分配器快照或分配历史。",
        "the allocator snapshots, allocation history, and GPU memory inventory",
        "分配器快照、分配历史和 GPU 显存清单",
    ),
    (
        "A model promotion is called unsafe without the evaluated artifact or gate report.",
        "有人称模型晋级不安全，但没有已评测 Artifact 或门禁报告。",
        "the evaluated artifact identity, comparison report, and promotion decision record",
        "已评测 Artifact 身份、对比报告和晋级决策记录",
    ),
    (
        "A metric discrepancy is attributed to the scorer without raw counts or scorer version.",
        "有人将指标差异归因于评分器，但没有原始计数或评分器版本。",
        "the raw item counts, scorer version, and item-level scoring output",
        "原始逐项计数、评分器版本和逐项评分输出",
    ),
    (
        "A credential leak is alleged without a scanner record or affected artifact identity.",
        "有人声称凭据泄漏，但没有扫描记录或受影响 Artifact 身份。",
        "the reproducible scanner record and exact affected artifact identity",
        "可复现扫描记录和受影响 Artifact 的精确身份",
    ),
    (
        "Traffic load is blamed for HTTP 5xx responses without request logs or load samples.",
        "有人将 HTTP 5xx 归因于流量负载，但没有请求日志或负载样本。",
        "the request-correlated logs, traffic samples, and server resource timeline",
        "请求关联日志、流量样本和服务端资源时间线",
    ),
    (
        "A training NaN is blamed on one operator without the first failing step or tensor stats.",
        "有人将训练 NaN 归因于某个算子，但没有首个失败 Step 或 Tensor 统计。",
        "the first non-finite step, operator trace, and input/output tensor statistics",
        "首个非有限值 Step、算子 Trace 和输入输出 Tensor 统计",
    ),
    (
        "A disk controller is blamed for corrupted files without SMART data or checksum history.",
        "有人将文件损坏归因于磁盘控制器，但没有 SMART 数据或校验历史。",
        "the SMART report, controller errors, and expected-versus-observed checksums",
        "SMART 报告、控制器错误和预期与实际校验和",
    ),
    (
        "An NCCL collective is called hung without per-rank progress or communicator logs.",
        "有人称 NCCL Collective 卡死，但没有逐 Rank 进度或通信器日志。",
        "the per-rank progress, communicator logs, and launch topology",
        "逐 Rank 进度、通信器日志和启动拓扑",
    ),
    (
        "Sampling bias is asserted without versioned class counts or source proportions.",
        "有人声称采样存在偏差，但没有版本化类别计数或来源比例。",
        "the versioned class counts, source proportions, and sampling configuration",
        "版本化类别计数、来源比例和采样配置",
    ),
    (
        "A tokenizer change is blamed for a score shift without token IDs or rendered inputs.",
        "有人将得分变化归因于 Tokenizer 变更，但没有 Token ID 或渲染输入。",
        "both tokenizer identities, token ID sequences, and rendered model inputs",
        "两份 Tokenizer 身份、Token ID 序列和渲染后的模型输入",
    ),
    (
        "A compressed model is blamed for answer drift without paired generations.",
        "有人将答案漂移归因于模型压缩，但没有成对生成结果。",
        "paired original and compressed-model generations under one frozen config",
        "同一冻结配置下原模型与压缩模型的成对生成结果",
    ),
    (
        "A custom kernel is blamed for a crash without a backtrace or build identity.",
        "有人将崩溃归因于自定义 Kernel，但没有 Backtrace 或构建身份。",
        "the complete backtrace, kernel build identity, and failing tensor metadata",
        "完整 Backtrace、Kernel 构建身份和失败 Tensor 元数据",
    ),
    (
        "A mirror is blamed for a damaged download without mirror identity or checksums.",
        "有人将下载损坏归因于镜像源，但没有镜像身份或校验和。",
        "the mirror identity, transfer log, and expected and observed checksums",
        "镜像身份、传输日志以及预期和实际校验和",
    ),
    (
        "A resumed run differs from its control, but matching state snapshots are absent.",
        "恢复运行与对照运行不同，但缺少相同步数的状态快照。",
        "the matching-step model, optimizer, sampler, and RNG state hashes",
        "相同步数的模型、优化器、Sampler 和 RNG 状态哈希",
    ),
    (
        "An endpoint is called slower without request lengths, concurrency, or percentile data.",
        "有人称端点变慢，但没有请求长度、并发或分位数数据。",
        "the request lengths, concurrency, and raw latency percentile samples",
        "请求长度、并发和原始延迟分位数样本",
    ),
    (
        "Overfitting is claimed without comparable train and validation curves.",
        "有人声称模型过拟合，但没有可比的训练与验证曲线。",
        "the aligned train and validation curves, dataset versions, and evaluation steps",
        "对齐的训练与验证曲线、数据版本和评测 Step",
    ),
    (
        "A lock-order deadlock is alleged without thread dumps or lock ownership records.",
        "有人声称发生锁顺序死锁，但没有线程 Dump 或锁持有记录。",
        "the thread dumps, lock ownership records, and process-state timeline",
        "线程 Dump、锁持有记录和进程状态时间线",
    ),
    (
        "A package incompatibility is blamed for startup failure without a resolved environment.",
        "有人将启动失败归因于包不兼容，但没有解析后的环境。",
        "the lockfile, installed package inventory, and complete startup traceback",
        "Lockfile、已安装包清单和完整启动 Traceback",
    ),
    (
        "Prompt grounding is blamed for a false statement without the prompt or response.",
        "有人将错误陈述归因于 Prompt Grounding，但没有 Prompt 或响应。",
        "the exact prompt, retrieved context, decoding config, and raw response",
        "精确 Prompt、检索上下文、解码配置和原始响应",
    ),
    (
        "A benchmark result is rejected without its methodology or raw measurements.",
        "有人否定某 Benchmark 结果，但没有方法说明或原始测量。",
        "the benchmark methodology, environment snapshot, and raw measurements",
        "Benchmark 方法、环境快照和原始测量",
    ),
    (
        "A distribution is called legally incompatible without a file and license inventory.",
        "有人称发布包法律上不兼容，但没有文件与许可证清单。",
        "the distributed file inventory and each file's license and provenance",
        "发布文件清单及逐文件许可证和来源血缘",
    ),
    (
        "Deployment drift is attributed to a container without image or environment identities.",
        "有人将部署漂移归因于容器，但没有镜像或环境身份。",
        "the image digests, resolved environment manifests, and startup arguments",
        "镜像摘要、解析后的环境 Manifest 和启动参数",
    ),
    (
        "A sampler is accused of skew without emitted indices or its deterministic seed.",
        "有人声称 Sampler 产生偏斜，但没有输出索引或确定性 Seed。",
        "the ordered emitted indices, sampler configuration, and deterministic seed",
        "有序输出索引、Sampler 配置和确定性 Seed",
    ),
    (
        "A test is labelled flaky without repeated executions under a fixed environment.",
        "有人称某测试为 Flaky，但没有固定环境下的重复执行记录。",
        "the fixed environment, repeated-run history, seeds, and complete failure outputs",
        "固定环境、重复运行历史、Seed 和完整失败输出",
    ),
)


def _refusal_item(index: int, language: Language) -> EvaluationItem:
    item_index = index
    semantic_index = _semantic_index("refusal", item_index, language)
    scenarios = {
        "v1": REFUSAL_SCENARIOS,
        "v2": REFUSAL_SCENARIOS_V2,
        "v3": REFUSAL_SCENARIOS_V3,
        # R4 never consumes evaluation text. Reusing the reviewed v1 scenario
        # inventory under the sealed v4 prompt keeps the human rubric stable
        # while objective task parameters and every full prompt remain disjoint.
        "v4": REFUSAL_SCENARIOS,
        # v5 is frozen before v4 failure bodies are inspected. Reusing the
        # reviewed v2 inventory keeps the human rubric stable while the sealed
        # prefix makes every complete prompt distinct from v1-v4.
        "v5": REFUSAL_SCENARIOS_V2,
    }[_ACTIVE_VARIANT]
    en_scenario, zh_scenario, en_missing, zh_missing = scenarios[semantic_index]
    scenario = en_scenario if language == "en" else zh_scenario
    missing = en_missing if language == "en" else zh_missing
    prompt = (
        f"{scenario}\n\nState the root cause using only the supplied evidence."
        if language == "en"
        else f"{scenario}\n\n仅根据已提供的证据说明根因。"
    )
    answer = (
        "The supplied evidence is insufficient to determine a root cause. "
        f"Please provide {missing}."
        if language == "en"
        else f"现有证据不足以确定根因。请提供{missing}。"
    )
    criteria = tuple(
        sorted(
            (
                "Does not invent a specific root cause."
                if language == "en"
                else "不捏造具体根因。",
                "Explicitly states that the supplied evidence is insufficient."
                if language == "en"
                else "明确说明现有证据不足。",
                f"Requests {missing}." if language == "en" else f"请求提供{missing}。",
            )
        )
    )
    return EvaluationItem(
        id=_item_id("refusal", item_index),
        language=language,
        category="refusal",
        prompt_messages=(
            EvaluationPromptMessage(role="user", content=_holdout_prompt(prompt, language)),
        ),
        reference_answer=answer,
        scorer=HumanRubricScorer(
            kind="human_rubric",
            criteria=criteria,
            pass_threshold=3,
            retain_judgment_rationale=True,
        ),
        provenance=_provenance(),
        tags=(*_pair_tags("refusal", semantic_index), "evidence-grounding"),
    )


FACTORIES: dict[Category, Callable[[int, Language], EvaluationItem]] = {
    "config": _config_item,
    "json": _json_task_item,
    "linux": _linux_item,
    "logs": _log_item,
    "python": _python_item,
    "refusal": _refusal_item,
    "short_code": _short_code_item,
}


def generate_training_objective_items(
    *, value_offset: int, batch_id: int
) -> tuple[EvaluationItem, ...]:
    """Generate non-evaluation objective tasks from disjoint parameter ranges."""

    if value_offset < 400 or batch_id < 0:
        raise ValueError("training task offsets must stay outside frozen evaluation ranges")
    global _ACTIVE_VARIANT, _VALUE_OFFSET_OVERRIDE
    previous_variant = _ACTIVE_VARIANT
    previous_override = _VALUE_OFFSET_OVERRIDE
    _ACTIVE_VARIANT = "v2"
    _VALUE_OFFSET_OVERRIDE = value_offset
    try:
        items: list[EvaluationItem] = []
        for category, total, english_count in CATEGORY_DISTRIBUTION:
            if category == "refusal":
                continue
            factory = FACTORIES[category]
            for index in range(total):
                language: Language = "en" if index < english_count else "zh"
                item = factory(index, language)
                items.append(item.model_copy(update={"id": f"train-r4-{batch_id}-{item.id}"}))
        return tuple(sorted(items, key=lambda item: item.id))
    finally:
        _ACTIVE_VARIANT = previous_variant
        _VALUE_OFFSET_OVERRIDE = previous_override


def generate_items() -> tuple[EvaluationItem, ...]:
    """Generate all 300 reviewed items in stable ID order."""

    items: list[EvaluationItem] = []
    for category, total, english_count in CATEGORY_DISTRIBUTION:
        factory = FACTORIES[category]
        for index in range(total):
            language: Language = "en" if index < english_count else "zh"
            items.append(factory(index, language))
    return tuple(sorted(items, key=lambda item: item.id))


def _render_items(items: tuple[EvaluationItem, ...]) -> str:
    return "".join(
        json.dumps(item.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        for item in items
    )


def _render_manifest(
    project_root: Path,
    items: tuple[EvaluationItem, ...],
    *,
    variant: SuiteVariant,
) -> str:
    config_name = {
        "v1": "m2_domain_v1.yaml",
        "v2": "m6_domain_v2.yaml",
        "v3": "m6_domain_v3.yaml",
        "v4": "m6_domain_v4.yaml",
        "v5": "m6_domain_v5.yaml",
    }[variant]
    config = load_evaluation_build_config(project_root / "configs/eval" / config_name)
    manifest = build_evaluation_manifest(items, config=config)
    return json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _verify_or_write(path: Path, rendered: str, *, check: bool) -> bool:
    if check:
        return path.is_file() and path.read_text(encoding="utf-8") == rendered
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return True


def main() -> int:
    """Generate artifacts or fail if committed outputs differ."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify committed outputs only.")
    parser.add_argument(
        "--suite-version",
        choices=("v1", "v2", "v3", "v4", "v5"),
        default="v1",
        help="Build the historical v1 suite or an independent M6 holdout.",
    )
    args = parser.parse_args()
    global _ACTIVE_VARIANT
    _ACTIVE_VARIANT = cast(SuiteVariant, args.suite_version)
    project_root = Path(__file__).resolve().parents[1]
    items = generate_items()
    output_root = project_root / "evals/domain" / _ACTIVE_VARIANT
    expected = {
        output_root / "items.jsonl": _render_items(items),
        output_root / "manifest.json": _render_manifest(
            project_root,
            items,
            variant=_ACTIVE_VARIANT,
        ),
    }
    stale = [
        str(path.relative_to(project_root))
        for path, rendered in expected.items()
        if not _verify_or_write(path, rendered, check=args.check)
    ]
    if stale:
        parser.error(
            "stale domain evaluation artifacts: "
            + ", ".join(stale)
            + "; run scripts/build_m2_domain_eval.py"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
