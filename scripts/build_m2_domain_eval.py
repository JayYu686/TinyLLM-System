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
        source_note="Authored from versioned TinyLLM-System v1 templates.",
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
        prompt_messages=(EvaluationPromptMessage(role="user", content=prompt),),
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
        prompt_messages=(EvaluationPromptMessage(role="user", content=prompt),),
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
        prompt_messages=(EvaluationPromptMessage(role="user", content=prompt),),
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
    cycle, mode = divmod(semantic_index, 10)
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
        text = "tinyllmsystem"
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
    cycle, mode = divmod(semantic_index, 9)
    only = "Return only the exact answer." if language == "en" else "仅返回精确答案。"
    if mode == 0:
        modes = ((6, 4, 0), (7, 5, 0), (6, 0, 0), (7, 0, 0), (6, 4, 4))
        owner, group, other = modes[cycle]
        symbols = {0: "---", 4: "r--", 5: "r-x", 6: "rw-", 7: "rwx"}
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
    cycle, mode = divmod(semantic_index, 8)
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
    cycle, mode = divmod(semantic_index, 8)
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
    cycle, mode = divmod(semantic_index, len(LOG_PATTERNS))
    template, correct, distractors = LOG_PATTERNS[mode]
    stem = template.format(variant=cycle + 1, port=8100 + cycle)
    localized = [correct[0 if language == "en" else 1]] + [
        pair[0 if language == "en" else 1] for pair in distractors
    ]
    rotation = (semantic_index * 3) % len(localized)
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


def _short_code_item(index: int, language: Language) -> EvaluationItem:
    item_index = index
    semantic_index = _semantic_index("short_code", item_index, language)
    snippet, answer, goal_en, goal_zh = SHORT_CODE_CASES[semantic_index]
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


def _refusal_item(index: int, language: Language) -> EvaluationItem:
    item_index = index
    semantic_index = _semantic_index("refusal", item_index, language)
    en_scenario, zh_scenario, en_missing, zh_missing = REFUSAL_SCENARIOS[semantic_index]
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
        prompt_messages=(EvaluationPromptMessage(role="user", content=prompt),),
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


def _render_manifest(project_root: Path, items: tuple[EvaluationItem, ...]) -> str:
    config = load_evaluation_build_config(project_root / "configs/eval/m2_domain_v1.yaml")
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
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    items = generate_items()
    output_root = project_root / "evals/domain/v1"
    expected = {
        output_root / "items.jsonl": _render_items(items),
        output_root / "manifest.json": _render_manifest(project_root, items),
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
