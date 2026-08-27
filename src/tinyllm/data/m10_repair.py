# ruff: noqa: E501
"""Deterministic M10.5 repair trajectories focused on end-to-end Agent success."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from tinyllm.agent import AgentToolDefinition
from tinyllm.agent_eval.schema import AgentEvalCategory, AgentEvalLanguage
from tinyllm.agent_eval.suite import tool_catalog
from tinyllm.data.m10_devops import (
    M10_DEVOPS_REPAIR_SEED,
    M10_DEVOPS_REPAIR_V3_SEED,
    REPAIR_CATEGORY_COUNTS,
    REPAIR_LANGUAGE_COUNTS,
    M10DevOpsDataError,
    _context,
    _language,
    _message,
    _missing_argument,
    _no_tool,
    _parallel,
    _security,
    _sequential,
    _single_tool,
    _system,
    _tool_call,
    _tool_result,
    _validate_arguments,
    _wrong_tool,
)
from tinyllm.data.m10_devops_schema import (
    M10DevOpsToolCall,
    M10DevOpsTrainingMessage,
    M10DevOpsTrainingSample,
    canonical_json_sha256,
)

_NO_CALL_CATEGORIES: Final = {
    "no_tool",
    "wrong_tool_irrelevance",
    "missing_argument_clarification",
}
_BANNED_GENERIC_ANSWERS: Final = (
    "the read-only check is complete",
    "已完成只读核查",
    "ties the recorded",
    "与评测事件一致",
    "matches the evaluation fixture",
)
_FAMILY_COUNTS: Final[dict[AgentEvalCategory, int]] = {
    "single_tool": 12,
    "no_tool": 6,
    "wrong_tool_irrelevance": 6,
    "missing_argument_clarification": 6,
    "sequential_multi_step": 12,
    "parallel_independent_tools": 1,
    "tool_failure_recovery": 2,
    "grounding_approval_security": 3,
}


def _calls(messages: Sequence[M10DevOpsTrainingMessage]) -> tuple[M10DevOpsToolCall, ...]:
    return tuple(call for message in messages for call in message.tool_calls)


def _replace_final(
    messages: tuple[M10DevOpsTrainingMessage, ...], final: str
) -> tuple[M10DevOpsTrainingMessage, ...]:
    if not messages or messages[-1].role != "assistant" or not messages[-1].content:
        raise M10DevOpsDataError("repair trajectory must end with a final assistant answer")
    return (*messages[:-1], _message("assistant", content=final))


def _citation(call: M10DevOpsToolCall) -> str:
    return f"[evidence:{call.id}]"


def _single_final(
    language: AgentEvalLanguage,
    ordinal: int,
    context: Mapping[str, str],
    messages: tuple[M10DevOpsTrainingMessage, ...],
) -> str:
    call = _calls(messages)[0]
    citation = _citation(call)
    variant = ordinal % 6
    if language == "zh":
        facts = (
            f"恢复文档要求 {context['service']} 仅在哈希校验通过后重启",
            f"运行 {context['run_id']} 的状态是 failed，失败码为 {context['failure']}",
            f"日志 18–37 行记录的首个故障是 {context['failure']}",
            f"最近 24 条观测的 step_time_ms P95 为 {842 + ordinal % 40}，最大显存字节数为 {21300000000 + ordinal * 1000}",
            f"配置 {context['config']} 使用 bf16，修订号为 cfg-{ordinal:04d}",
            f"失败运行共 1 个：{context['run_id']}",
        )[variant]
        leads = ("核查结果", "记录显示", "根据工具返回", "可确认", "实际值", "结论")
        return (
            f"{leads[ordinal % len(leads)]}：{facts}。工具记录 {context['doc']}；证据：{citation}。"
        )
    facts = (
        f"the recovery document requires restarting {context['service']} only after hash verification",
        f"Run {context['run_id']} is failed with failure code {context['failure']}",
        f"the first failure in log lines 18–37 is {context['failure']}",
        f"the latest 24 observations have step_time_ms P95 {842 + ordinal % 40} and maximum GPU memory {21300000000 + ordinal * 1000} bytes",
        f"configuration {context['config']} uses bf16 at revision cfg-{ordinal:04d}",
        f"the failed-Run query returned exactly 1 Run: {context['run_id']}",
    )[variant]
    leads = (
        "Result",
        "The record shows",
        "From the tool output",
        "Verified fact",
        "Observed value",
        "Conclusion",
    )
    return (
        f"{leads[ordinal % len(leads)]}: {facts}. "
        f"Tool record {context['doc']}; evidence: {citation}."
    )


def _no_tool_final(
    language: AgentEvalLanguage,
    ordinal: int,
    messages: tuple[M10DevOpsTrainingMessage, ...],
) -> str:
    core = messages[-1].content
    assert core is not None
    if language == "zh":
        variants = (
            "直接回答：{core}",
            "核心区别是：{core}",
            "这里不需要读取运行数据。{core}",
            "从系统语义看，{core}",
            "简要结论：{core}",
            "概念层面的答案是：{core}",
            "无需调用工具即可判断：{core}",
            "结论如下。{core}",
        )
    else:
        variants = (
            "Direct answer: {core}",
            "The core distinction is: {core}",
            "No Run lookup is needed here. {core}",
            "At the system level, {core}",
            "In short: {core}",
            "The conceptual answer is: {core}",
            "This can be answered without a tool call. {core}",
            "The conclusion is straightforward. {core}",
        )
    return variants[ordinal % len(variants)].format(core=core)


def _wrong_tool_final(language: AgentEvalLanguage, ordinal: int, context: Mapping[str, str]) -> str:
    action = ordinal % 6
    if language == "zh":
        denied = (
            f"删除运行 {context['run_id']} 的 Checkpoint",
            f"通过 SSH 登录 {context['service']}",
            f"发送事件 {context['incident']} 的邮件",
            f"重启承载 {context['service']} 的主机",
            "下载并执行未注册的 MCP Server",
            f"修改 {context['service']} 的生产数据库",
        )[action]
        alternatives = (
            "我只能使用注册的只读证据工具",
            "可以改为检查现有运行与日志证据",
            "当前工具策略没有授予该权限",
            "不会用相似工具绕过权限边界",
        )
        return f"请求 {context['incident']} 无法执行“{denied}”；{alternatives[ordinal % 4]}。"
    denied = (
        f"delete checkpoints for Run {context['run_id']}",
        f"open SSH access to {context['service']}",
        f"send email for incident {context['incident']}",
        f"restart the host running {context['service']}",
        "download and execute an unregistered MCP server",
        f"modify the production database for {context['service']}",
    )[action]
    alternatives = (
        "I can only use the registered read-only evidence tools",
        "I can instead inspect existing Run and log evidence",
        "the current tool policy grants no such authority",
        "I will not bypass the boundary with a similar tool",
    )
    return f"I cannot {denied} for request {context['incident']}; {alternatives[ordinal % 4]}."


def _missing_final(language: AgentEvalLanguage, ordinal: int, context: Mapping[str, str]) -> str:
    questions_en = (
        "Which allowlisted relative configuration path should I inspect?",
        "What exact Run ID should I look up?",
        "Which allowlisted relative log path and line range should I read?",
        "Which metrics artifact path and metric names should I query?",
        "Which service or failure identifier should the evidence query contain?",
        "Which source-relative configuration path and exact key updates should I propose?",
    )
    questions_zh = (
        "请提供需要检查的白名单内相对配置路径，可以吗？",
        "请提供需要查询的精确 Run ID，可以吗？",
        "请提供白名单内相对日志路径和行号范围，可以吗？",
        "请提供指标产物路径和需要查询的指标名称，可以吗？",
        "请提供检索词应包含的服务名或故障标识，可以吗？",
        "请提供源配置相对路径和需要修改的精确键值，可以吗？",
    )
    question = (questions_zh if language == "zh" else questions_en)[ordinal % 6]
    if language == "zh":
        return f"请求 {context['incident']} 缺少执行所需参数。{question}"
    return f"Request {context['incident']} is missing a required argument. {question}"


def _sequential_final(
    language: AgentEvalLanguage,
    ordinal: int,
    context: Mapping[str, str],
    messages: tuple[M10DevOpsTrainingMessage, ...],
) -> str:
    first, second = _calls(messages)
    evidence = f"{_citation(first)} {_citation(second)}"
    variant = ordinal % 3
    if language == "zh":
        facts = (
            f"失败运行列表返回 {context['run_id']}，其失败码是 {context['failure']}",
            f"证据定位到日志 42–58 行，其中记录的故障是 {context['failure']}",
            f"运行 {context['run_id']} 的最后 loss 为 {round(0.7 + (ordinal % 30) / 100, 2)}，step_time_ms P95 为 {810 + ordinal % 60}",
        )[variant]
        return f"顺序执行两个只读步骤后可确认：{facts}。证据：{evidence}。"
    facts = (
        f"the failed-Run list returned {context['run_id']}, whose failure code is {context['failure']}",
        f"the evidence resolves to log lines 42–58, which record {context['failure']}",
        f"Run {context['run_id']} has final loss {round(0.7 + (ordinal % 30) / 100, 2)} and step_time_ms P95 {810 + ordinal % 60}",
    )[variant]
    return f"After the two ordered read steps, {facts}. Evidence: {evidence}."


def _parallel_final(
    language: AgentEvalLanguage,
    ordinal: int,
    context: Mapping[str, str],
    messages: tuple[M10DevOpsTrainingMessage, ...],
) -> str:
    metrics, config = _calls(messages)
    memory = 21900000000 + ordinal * 100
    throughput = 1420 + ordinal % 50
    evidence = f"{_citation(metrics)} {_citation(config)}"
    if language == "zh":
        return f"并行结果显示最大显存为 {memory} 字节、吞吐中位数为 {throughput} tokens/s；配置使用 bf16 且启用了 gradient checkpointing。证据：{evidence}。"
    return f"The parallel results show maximum GPU memory {memory} bytes and median throughput {throughput} tokens/s; the configuration uses bf16 with gradient checkpointing enabled. Evidence: {evidence}."


def _recovery(
    language: AgentEvalLanguage, ordinal: int, context: Mapping[str, str]
) -> tuple[str, tuple[M10DevOpsTrainingMessage, ...]]:
    call = _tool_call(
        f"call_recovery_{ordinal:04d}",
        "read_log_excerpt",
        {"relative_path": context["log"], "start_line": 70, "end_line": 92},
    )
    error = "TRANSIENT_ARTIFACT_BUSY" if ordinal % 2 == 0 else "TOOL_EXECUTION_TIMEOUT"
    prompt = (
        f"读取 {context['incident']} 的有界日志；只读工具首次遇到可重试故障时由运行时重试一次，并报告最终事实。"
        if language == "zh"
        else f"Read the bounded log for {context['incident']}; let the runtime retry one transient read failure, then report the final fact."
    )
    result = {
        "status": "succeeded",
        "attempts": 2,
        "first_error": error,
        "failure_code": context["failure"],
        "evidence_id": context["doc"],
    }
    final = (
        f"运行时在 {error} 后重试成功；日志中的实际故障是 {context['failure']}。证据：{_citation(call)}。"
        if language == "zh"
        else f"The runtime recovered after {error}; the log records {context['failure']}. Evidence: {_citation(call)}."
    )
    return prompt, (
        _message("assistant", tool_calls=(call,)),
        _tool_result(call, result),
        _message("assistant", content=final),
    )


def _security_final(
    language: AgentEvalLanguage,
    ordinal: int,
    context: Mapping[str, str],
    messages: tuple[M10DevOpsTrainingMessage, ...],
) -> str:
    call = _calls(messages)[0]
    citation = _citation(call)
    variant = ordinal % 3
    if language == "zh":
        facts = (
            f"证据中的操作指令不可信；可采信事实是 {context['service']} 恢复前必须校验产物哈希",
            f"获批补丁只写入 agent-sandboxes/m10-{ordinal:04d}/stage.yaml，源配置保持不变",
            f"读取保持在白名单日志 9–21 行内，{context['failure']} 发生在 Checkpoint 提交前",
        )[variant]
        return f"安全核查结论：{facts}。工具记录 {context['doc']}；证据：{citation}。"
    facts = (
        f"the operational instruction is untrusted; the supported fact is that {context['service']} requires artifact hash verification before recovery",
        f"the approved patch writes only agent-sandboxes/m10-{ordinal:04d}/stage.yaml and leaves the source configuration unchanged",
        f"the read remains within allowlisted log lines 9–21, where {context['failure']} occurs before checkpoint commit",
    )[variant]
    return f"Security conclusion: {facts}. Tool record {context['doc']}; evidence: {citation}."


def _repair_tail(
    category: AgentEvalCategory,
    language: AgentEvalLanguage,
    ordinal: int,
    context: Mapping[str, str],
) -> tuple[str, tuple[M10DevOpsTrainingMessage, ...]]:
    if category == "tool_failure_recovery":
        return _recovery(language, ordinal, context)
    builders = {
        "single_tool": _single_tool,
        "no_tool": _no_tool,
        "wrong_tool_irrelevance": _wrong_tool,
        "missing_argument_clarification": _missing_argument,
        "sequential_multi_step": _sequential,
        "parallel_independent_tools": _parallel,
        "grounding_approval_security": _security,
    }
    prompt, messages = builders[category](language, ordinal, context)
    finals = {
        "single_tool": lambda: _single_final(language, ordinal, context, messages),
        "no_tool": lambda: _no_tool_final(language, ordinal, messages),
        "wrong_tool_irrelevance": lambda: _wrong_tool_final(language, ordinal, context),
        "missing_argument_clarification": lambda: _missing_final(language, ordinal, context),
        "sequential_multi_step": lambda: _sequential_final(language, ordinal, context, messages),
        "parallel_independent_tools": lambda: _parallel_final(language, ordinal, context, messages),
        "grounding_approval_security": lambda: _security_final(
            language, ordinal, context, messages
        ),
    }
    return prompt, _replace_final(messages, finals[category]())


def _sequential_v3(
    language: AgentEvalLanguage,
    ordinal: int,
    context: Mapping[str, str],
) -> tuple[str, tuple[M10DevOpsTrainingMessage, ...]]:
    """Cover four generic two-step plans while preserving the requested entity."""

    service = context["service"]
    variant = ordinal % 4
    first_result: dict[str, Any]
    second_result: dict[str, Any]
    if variant == 0:
        first = _tool_call(f"call_seq3_{ordinal:04d}_1", "get_run", {"run_id": context["run_id"]})
        first_result = {
            "status": "succeeded",
            "run_id": context["run_id"],
            "service": service,
            "config_path": context["config"],
            "evidence_id": f"{context['doc']}-run",
        }
        second = _tool_call(
            f"call_seq3_{ordinal:04d}_2",
            "inspect_config",
            {"relative_path": context["config"]},
        )
        second_result = {
            "status": "succeeded",
            "precision": "bf16",
            "gradient_checkpointing": True,
            "evidence_id": f"{context['doc']}-config",
        }
        task_en = f"resolve Run {context['run_id']}, inspect its returned configuration, and state the precision used by {service}"
        task_zh = f"先解析运行 {context['run_id']}，再检查返回的配置，并说明 {service} 使用的精度"
        fact_en = f"{service} uses bf16 with gradient checkpointing enabled"
        fact_zh = f"{service} 使用 bf16，并启用了 gradient checkpointing"
    elif variant == 1:
        first = _tool_call(
            f"call_seq3_{ordinal:04d}_1",
            "search_evidence",
            {"query": f"{service} {context['incident']}", "top_k": 4},
        )
        first_result = {
            "status": "succeeded",
            "documents": [
                {
                    "document_id": context["doc"],
                    "relative_path": context["log"],
                    "line_start": 42,
                    "line_end": 58,
                }
            ],
        }
        second = _tool_call(
            f"call_seq3_{ordinal:04d}_2",
            "read_log_excerpt",
            {"relative_path": context["log"], "start_line": 42, "end_line": 58},
        )
        second_result = {
            "status": "succeeded",
            "service": service,
            "failure_code": context["failure"],
            "evidence_id": context["doc"],
        }
        task_en = f"locate evidence for incident {context['incident']}, read only the returned range, and report the failure affecting {service}"
        task_zh = f"先定位事件 {context['incident']} 的证据，再仅读取返回范围，并报告影响 {service} 的故障"
        fact_en = f"{service} recorded {context['failure']}"
        fact_zh = f"{service} 记录的故障是 {context['failure']}"
    elif variant == 2:
        first = _tool_call(f"call_seq3_{ordinal:04d}_1", "get_run", {"run_id": context["run_id"]})
        first_result = {
            "status": "succeeded",
            "run_id": context["run_id"],
            "service": service,
            "metrics_path": context["metrics"],
            "evidence_id": f"{context['doc']}-run",
        }
        second = _tool_call(
            f"call_seq3_{ordinal:04d}_2",
            "query_metrics",
            {
                "relative_path": context["metrics"],
                "metric_names": ["loss", "step_time_ms"],
                "limit": 32,
            },
        )
        loss = round(0.7 + (ordinal % 30) / 100, 2)
        latency = 810 + ordinal % 60
        second_result = {
            "status": "succeeded",
            "loss_last": loss,
            "step_time_ms_p95": latency,
            "evidence_id": f"{context['doc']}-metrics",
        }
        task_en = f"resolve Run {context['run_id']}, query only its returned metrics path, and summarize the measurements for {service}"
        task_zh = (
            f"先解析运行 {context['run_id']}，再仅查询返回的指标路径，并汇总 {service} 的观测值"
        )
        fact_en = f"{service} has final loss {loss} and step_time_ms P95 {latency}"
        fact_zh = f"{service} 的最后 loss 为 {loss}，step_time_ms P95 为 {latency}"
    else:
        first = _tool_call(
            f"call_seq3_{ordinal:04d}_1", "list_runs", {"limit": 20, "status": "failed"}
        )
        first_result = {
            "status": "succeeded",
            "runs": [context["run_id"]],
            "count": 1,
            "evidence_id": f"{context['doc']}-list",
        }
        second = _tool_call(f"call_seq3_{ordinal:04d}_2", "get_run", {"run_id": context["run_id"]})
        second_result = {
            "status": "succeeded",
            "run_id": context["run_id"],
            "service": service,
            "failure_code": context["failure"],
            "evidence_id": f"{context['doc']}-run",
        }
        task_en = f"list failed Runs, inspect the returned Run, and identify the failure associated with {service}"
        task_zh = f"先列出失败运行，再检查返回的 Run，并识别与 {service} 相关的故障"
        fact_en = f"{service} Run {context['run_id']} failed with {context['failure']}"
        fact_zh = f"{service} 的运行 {context['run_id']} 因 {context['failure']} 失败"
    prompt = (
        f"{task_zh}；最终答案必须保留服务名并引用两个工具结果。"
        if language == "zh"
        else f"{task_en}; preserve the service name in the final answer and cite both tool results."
    )
    evidence = f"{_citation(first)} {_citation(second)}"
    final = (
        f"顺序诊断结论：{fact_zh}。证据：{evidence}。"
        if language == "zh"
        else f"Ordered diagnostic result: {fact_en}. Evidence: {evidence}."
    )
    return prompt, (
        _message("assistant", tool_calls=(first,)),
        _tool_result(first, first_result),
        _message("assistant", tool_calls=(second,)),
        _tool_result(second, second_result),
        _message("assistant", content=final),
    )


def _parallel_v3(
    language: AgentEvalLanguage,
    ordinal: int,
    context: Mapping[str, str],
) -> tuple[str, tuple[M10DevOpsTrainingMessage, ...]]:
    """Teach one decision containing two independent, non-duplicated calls."""

    service = context["service"]
    first_result: dict[str, Any]
    second_result: dict[str, Any]
    if ordinal % 2:
        first = _tool_call(
            f"call_parallel3_{ordinal:04d}_log",
            "read_log_excerpt",
            {"relative_path": context["log"], "start_line": 24, "end_line": 46},
        )
        first_result = {
            "status": "succeeded",
            "service": service,
            "failure_code": context["failure"],
            "evidence_id": f"{context['doc']}-log",
        }
        second = _tool_call(
            f"call_parallel3_{ordinal:04d}_metrics",
            "query_metrics",
            {
                "relative_path": context["metrics"],
                "metric_names": ["gpu_memory_bytes", "step_time_ms"],
                "limit": 24,
            },
        )
        memory = 21_900_000_000 + ordinal * 100
        latency = 820 + ordinal % 50
        second_result = {
            "status": "succeeded",
            "gpu_memory_bytes_max": memory,
            "step_time_ms_p95": latency,
            "evidence_id": f"{context['doc']}-metrics",
        }
        task_en = (
            f"read the bounded log and query the independent metrics for {service} in parallel"
        )
        task_zh = f"并行读取 {service} 的有界日志和独立指标"
        fact_en = f"{service} recorded {context['failure']}, peak memory {memory} bytes, and step_time_ms P95 {latency}"
        fact_zh = f"{service} 记录了 {context['failure']}，最大显存 {memory} 字节，step_time_ms P95 为 {latency}"
    else:
        first = _tool_call(
            f"call_parallel3_{ordinal:04d}_metrics",
            "query_metrics",
            {
                "relative_path": context["metrics"],
                "metric_names": ["gpu_memory_bytes", "tokens_per_second"],
                "limit": 16,
            },
        )
        memory = 21_900_000_000 + ordinal * 100
        throughput = 1420 + ordinal % 50
        first_result = {
            "status": "succeeded",
            "gpu_memory_bytes_max": memory,
            "tokens_per_second_median": throughput,
            "evidence_id": f"{context['doc']}-metrics",
        }
        second = _tool_call(
            f"call_parallel3_{ordinal:04d}_config",
            "inspect_config",
            {"relative_path": context["config"]},
        )
        second_result = {
            "status": "succeeded",
            "precision": "bf16",
            "gradient_checkpointing": True,
            "evidence_id": f"{context['doc']}-config",
        }
        task_en = f"inspect the independent metrics and configuration for {service} in parallel"
        task_zh = f"并行检查 {service} 的独立指标与配置"
        fact_en = f"{service} reached {throughput} tokens/s at {memory} bytes and uses bf16 with gradient checkpointing"
        fact_zh = f"{service} 达到 {throughput} tokens/s、显存 {memory} 字节，并使用 bf16 和 gradient checkpointing"
    prompt = (
        f"{task_zh}；在同一个决策中发出两个工具调用，最终保留服务名和两个证据。"
        if language == "zh"
        else f"{task_en}; issue both tool calls in one decision and preserve the service name and both citations."
    )
    final = (
        f"并行诊断结论：{fact_zh}。证据：{_citation(first)} {_citation(second)}。"
        if language == "zh"
        else f"Parallel diagnostic result: {fact_en}. Evidence: {_citation(first)} {_citation(second)}."
    )
    return prompt, (
        _message("assistant", tool_calls=(first, second)),
        _tool_result(first, first_result),
        _tool_result(second, second_result),
        _message("assistant", content=final),
    )


def _recovery_v3(
    language: AgentEvalLanguage,
    ordinal: int,
    context: Mapping[str, str],
) -> tuple[str, tuple[M10DevOpsTrainingMessage, ...]]:
    """Represent a transparent runtime retry as one logical model tool call."""

    service = context["service"]
    fact_key: str
    fact_value: str | int
    if ordinal % 2:
        call = _tool_call(
            f"call_recovery3_{ordinal:04d}",
            "read_log_excerpt",
            {"relative_path": context["log"], "start_line": 70, "end_line": 92},
        )
        fact_key, fact_value = "failure_code", context["failure"]
        task_en = f"read the bounded log for {service}"
        task_zh = f"读取 {service} 的有界日志"
        fact_en = f"{service} recorded {context['failure']}"
        fact_zh = f"{service} 记录的故障是 {context['failure']}"
    else:
        call = _tool_call(
            f"call_recovery3_{ordinal:04d}",
            "query_metrics",
            {
                "relative_path": context["metrics"],
                "metric_names": ["step_time_ms"],
                "limit": 20,
            },
        )
        fact_key, fact_value = "step_time_ms_p95", 830 + ordinal % 40
        task_en = f"query the bounded latency metrics for {service}"
        task_zh = f"查询 {service} 的有界延迟指标"
        fact_en = f"{service} has step_time_ms P95 {fact_value}"
        fact_zh = f"{service} 的 step_time_ms P95 为 {fact_value}"
    error = "TRANSIENT_ARTIFACT_BUSY" if ordinal % 4 < 2 else "TOOL_EXECUTION_TIMEOUT"
    result: dict[str, Any] = {
        "status": "succeeded",
        "attempts": 2,
        "first_error": error,
        fact_key: fact_value,
        "evidence_id": context["doc"],
    }
    prompt = (
        f"{task_zh}；若首次出现可重试故障，由运行时透明重试一次。模型只发出一个逻辑调用，最终保留服务名。"
        if language == "zh"
        else f"{task_en}; let the runtime transparently retry one transient failure. The model must issue one logical call and preserve the service name."
    )
    final = (
        f"运行时从 {error} 恢复后确认：{fact_zh}。证据：{_citation(call)}。"
        if language == "zh"
        else f"After runtime recovery from {error}, {fact_en}. Evidence: {_citation(call)}."
    )
    return prompt, (
        _message("assistant", tool_calls=(call,)),
        _tool_result(call, result),
        _message("assistant", content=final),
    )


def _repair_v3_tail(
    category: AgentEvalCategory,
    language: AgentEvalLanguage,
    ordinal: int,
    context: Mapping[str, str],
) -> tuple[str, tuple[M10DevOpsTrainingMessage, ...]]:
    if category == "sequential_multi_step":
        return _sequential_v3(language, ordinal, context)
    if category == "parallel_independent_tools":
        return _parallel_v3(language, ordinal, context)
    if category == "tool_failure_recovery":
        return _recovery_v3(language, ordinal, context)
    return _repair_tail(category, language, ordinal, context)


def _build_repair_sample(
    category: AgentEvalCategory,
    category_ordinal: int,
    global_ordinal: int,
    tools: tuple[AgentToolDefinition, ...],
    *,
    source_revision: Literal["m10-devops-training-v2", "m10-devops-training-v3"] = (
        "m10-devops-training-v2"
    ),
) -> M10DevOpsTrainingSample:
    language = _language(category_ordinal)
    context = _context(global_ordinal)
    prompt, tail = (
        _repair_v3_tail(category, language, global_ordinal, context)
        if source_revision == "m10-devops-training-v3"
        else _repair_tail(category, language, global_ordinal, context)
    )
    messages = (
        _message(
            "system",
            content=_system(language, global_ordinal).replace(
                "m10-policy",
                "m10.5-v3-policy"
                if source_revision == "m10-devops-training-v3"
                else "m10.5-policy",
            ),
        ),
        _message("user", content=prompt),
        *tail,
    )
    tool_map = {item.tool_name: item for item in tools}
    for message in messages:
        for call in message.tool_calls:
            _validate_arguments(call, tool_map)
    family = global_ordinal % _FAMILY_COUNTS[category] + 1
    slug = category.replace("_", "-")
    source_record = {
        "generator_contract_version": (
            "m10-devops-generator-v3"
            if source_revision == "m10-devops-training-v3"
            else "m10-devops-generator-v2"
        ),
        "seed": (
            M10_DEVOPS_REPAIR_V3_SEED
            if source_revision == "m10-devops-training-v3"
            else M10_DEVOPS_REPAIR_SEED
        ),
        "category": category,
        "category_ordinal": category_ordinal,
        "global_ordinal": global_ordinal,
        "context": context,
        "family": family,
    }
    prompt_payload = [
        {"role": item.role, "content": item.content} for item in messages if item.role == "user"
    ]
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "sample_id": f"m10-devops-{language}-{slug}-{category_ordinal:04d}",
        "source_id": "tinyllm_devops",
        "source_revision": source_revision,
        "license": "Apache-2.0",
        "language": language,
        "category": category,
        "template_family": f"family-{slug}-{family:02d}",
        "group_id": f"group-{slug}-{family:02d}",
        "mode": "nonthinking",
        "available_tools": [item.to_dict() for item in tools],
        "messages": [item.to_dict() for item in messages],
        "source_record_sha256": canonical_json_sha256(source_record),
        "prompt_sha256": canonical_json_sha256(prompt_payload),
        "tool_schema_sha256": canonical_json_sha256([item.to_dict() for item in tools]),
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return M10DevOpsTrainingSample.model_validate(payload)


def build_repair_samples() -> tuple[M10DevOpsTrainingSample, ...]:
    """Build the 2,400-record repair source without evaluation-set content."""

    tools = tool_catalog()
    samples: list[M10DevOpsTrainingSample] = []
    global_ordinal = 0
    for category, count in REPAIR_CATEGORY_COUNTS.items():
        for category_ordinal in range(1, count + 1):
            global_ordinal += 1
            samples.append(_build_repair_sample(category, category_ordinal, global_ordinal, tools))
    if Counter(item.category for item in samples) != Counter(REPAIR_CATEGORY_COUNTS):
        raise M10DevOpsDataError("repair category distribution is inconsistent")
    if Counter(item.language for item in samples) != Counter(REPAIR_LANGUAGE_COUNTS):
        raise M10DevOpsDataError("repair language distribution is inconsistent")
    validate_repair_samples(samples)
    return tuple(samples)


def build_repair_v3_samples() -> tuple[M10DevOpsTrainingSample, ...]:
    """Build the 2,400-record v3 source without copying evaluation prompts."""

    tools = tool_catalog()
    samples: list[M10DevOpsTrainingSample] = []
    global_ordinal = 0
    for category, count in REPAIR_CATEGORY_COUNTS.items():
        for category_ordinal in range(1, count + 1):
            global_ordinal += 1
            samples.append(
                _build_repair_sample(
                    category,
                    category_ordinal,
                    global_ordinal,
                    tools,
                    source_revision="m10-devops-training-v3",
                )
            )
    if Counter(item.category for item in samples) != Counter(REPAIR_CATEGORY_COUNTS):
        raise M10DevOpsDataError("v3 repair category distribution is inconsistent")
    if Counter(item.language for item in samples) != Counter(REPAIR_LANGUAGE_COUNTS):
        raise M10DevOpsDataError("v3 repair language distribution is inconsistent")
    validate_repair_v3_samples(samples)
    return tuple(samples)


def _tool_result_scalars(messages: Sequence[M10DevOpsTrainingMessage]) -> tuple[str, ...]:
    values: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            rendered = str(value)
            if rendered.casefold() not in {"succeeded", "failed", "approved"}:
                values.append(rendered)

    for message in messages:
        if message.role == "tool" and message.content:
            visit(json.loads(message.content))
    return tuple(values)


def validate_repair_samples(
    samples: Sequence[M10DevOpsTrainingSample],
) -> dict[str, int | str]:
    """Fail closed on the exact defects observed in the rejected v1 campaign."""

    if len(samples) != 2400 or any(
        item.source_revision != "m10-devops-training-v2" for item in samples
    ):
        raise M10DevOpsDataError("repair quality gate requires one complete v2 source")
    grounded = 0
    recovery_single_call = 0
    clarification_questions = 0
    banned = 0
    finals: list[str] = []
    for sample in samples:
        final = sample.messages[-1].content or ""
        finals.append(final)
        folded = final.casefold()
        banned += sum(term in folded for term in _BANNED_GENERIC_ANSWERS)
        calls = _calls(sample.messages)
        if sample.category not in _NO_CALL_CATEGORIES:
            citations_ok = all(_citation(call) in final for call in calls)
            facts = _tool_result_scalars(sample.messages)
            facts_ok = any(value.casefold() in folded for value in facts)
            if not citations_ok or not facts_ok:
                raise M10DevOpsDataError(
                    f"repair final answer is not grounded in Tool Result: {sample.sample_id}"
                )
            grounded += 1
        if sample.category == "tool_failure_recovery":
            tool_results = [item for item in sample.messages if item.role == "tool"]
            if (
                len(calls) != 1
                or len(tool_results) != 1
                or '"attempts":2' not in (tool_results[0].content or "")
            ):
                raise M10DevOpsDataError("repair recovery must use one runtime-retried call")
            recovery_single_call += 1
        if sample.category == "missing_argument_clarification":
            if "?" not in final and "？" not in final:
                raise M10DevOpsDataError("repair clarification answer must contain a question")
            clarification_questions += 1
    frequencies = Counter(finals)
    unique = len(frequencies)
    maximum = max(frequencies.values())
    if banned or unique < 2200 or maximum > 8:
        raise M10DevOpsDataError("repair answer diversity gate failed")
    return {
        "schema_version": "1.0",
        "status": "pass",
        "item_count": len(samples),
        "tool_grounded_samples": grounded,
        "recovery_single_call_samples": recovery_single_call,
        "clarification_question_samples": clarification_questions,
        "banned_generic_answer_matches": banned,
        "unique_final_answers": unique,
        "maximum_exact_final_answer_frequency": maximum,
    }


def validate_repair_v3_samples(
    samples: Sequence[M10DevOpsTrainingSample],
) -> dict[str, int | str]:
    """Fail closed on v3 planning, entity preservation, and retry semantics."""

    if len(samples) != 2400 or any(
        item.source_revision != "m10-devops-training-v3" for item in samples
    ):
        raise M10DevOpsDataError("v3 repair quality gate requires one complete v3 source")
    grounded = 0
    recovery_single_call = 0
    clarification_questions = 0
    sequential_two_step = 0
    parallel_two_call = 0
    entity_preserved = 0
    banned = 0
    finals: list[str] = []
    entity_categories = {
        "sequential_multi_step",
        "parallel_independent_tools",
        "tool_failure_recovery",
    }
    for global_ordinal, sample in enumerate(samples, start=1):
        final = sample.messages[-1].content or ""
        finals.append(final)
        folded = final.casefold()
        banned += sum(term in folded for term in _BANNED_GENERIC_ANSWERS)
        calls = _calls(sample.messages)
        if sample.category not in _NO_CALL_CATEGORIES:
            citations_ok = all(_citation(call) in final for call in calls)
            facts = _tool_result_scalars(sample.messages)
            facts_ok = any(value.casefold() in folded for value in facts)
            if not citations_ok or not facts_ok:
                raise M10DevOpsDataError(
                    f"v3 repair final answer is not grounded in Tool Result: {sample.sample_id}"
                )
            grounded += 1
        if sample.category == "tool_failure_recovery":
            tool_results = [item for item in sample.messages if item.role == "tool"]
            if (
                len(calls) != 1
                or len(tool_results) != 1
                or '"attempts":2' not in (tool_results[0].content or "")
            ):
                raise M10DevOpsDataError("v3 recovery must use one runtime-retried logical call")
            recovery_single_call += 1
        if sample.category == "missing_argument_clarification":
            if "?" not in final and "？" not in final:
                raise M10DevOpsDataError("v3 clarification answer must contain a question")
            clarification_questions += 1
        if sample.category == "sequential_multi_step":
            call_messages = [message for message in sample.messages if message.tool_calls]
            if len(calls) != 2 or [len(message.tool_calls) for message in call_messages] != [1, 1]:
                raise M10DevOpsDataError("v3 sequential trajectory must contain two ordered calls")
            sequential_two_step += 1
        if sample.category == "parallel_independent_tools":
            if len(calls) != 2 or not any(
                len(message.tool_calls) == 2 for message in sample.messages
            ):
                raise M10DevOpsDataError("v3 parallel trajectory must issue two calls together")
            parallel_two_call += 1
        if sample.category in entity_categories:
            service = _context(global_ordinal)["service"]
            if service.casefold() not in folded:
                raise M10DevOpsDataError(
                    f"v3 final answer dropped the requested entity: {sample.sample_id}"
                )
            entity_preserved += 1
    frequencies = Counter(finals)
    unique = len(frequencies)
    maximum = max(frequencies.values())
    if banned or unique < 2200 or maximum > 8:
        raise M10DevOpsDataError("v3 repair answer diversity gate failed")
    return {
        "schema_version": "1.0",
        "status": "pass",
        "item_count": len(samples),
        "tool_grounded_samples": grounded,
        "recovery_single_call_samples": recovery_single_call,
        "clarification_question_samples": clarification_questions,
        "sequential_two_step_samples": sequential_two_step,
        "parallel_two_call_samples": parallel_two_call,
        "entity_preserved_samples": entity_preserved,
        "banned_generic_answer_matches": banned,
        "unique_final_answers": unique,
        "maximum_exact_final_answer_frequency": maximum,
    }
