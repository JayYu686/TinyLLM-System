from __future__ import annotations

from tinyllm.agent_eval import AgentEvalObservedCall, aggregate_results, score_task
from tinyllm.agent_eval.suite import build_tasks


def _observed(task_index: int = 0) -> tuple[AgentEvalObservedCall, ...]:
    task = build_tasks("dev")[task_index]
    calls = task.allowed_trajectories[0].calls
    return tuple(
        AgentEvalObservedCall(
            sequence=index,
            tool_name=call.tool_name,
            arguments=call.arguments,
            schema_valid=True,
            result_status=call.result_status,
        )
        for index, call in enumerate(calls, start=1)
    )


def test_exact_single_tool_trace_scores_success() -> None:
    task = build_tasks("dev")[0]
    result = score_task(
        task,
        run_id="agent-test-1",
        status="succeeded",
        calls=_observed(),
        final_answer=("Run 20260820T011500Z-serving-smoke-b2c3d4e5-0002 is recorded."),
        evidence_citations=("call_test",),
        duration_milliseconds=10,
        input_tokens=20,
        output_tokens=10,
    )

    assert result.tool_selection_correct is True
    assert result.argument_correct is True
    assert result.grounding_correct is True
    assert result.task_success is True


def test_wrong_tool_is_hallucination_and_fails() -> None:
    task = build_tasks("dev")[0]
    result = score_task(
        task,
        run_id="agent-test-2",
        status="succeeded",
        calls=(
            AgentEvalObservedCall(
                sequence=1,
                tool_name="search_evidence",
                arguments={"query": "wrong"},
                schema_valid=True,
                result_status="succeeded",
            ),
        ),
        final_answer="unsupported",
    )

    assert result.tool_selection_correct is False
    assert result.tool_hallucination is True
    assert result.task_success is False


def test_parallel_calls_accept_either_order() -> None:
    task = next(
        item for item in build_tasks("dev") if item.category == "parallel_independent_tools"
    )
    expected = task.allowed_trajectories[0].calls
    calls = tuple(
        AgentEvalObservedCall(
            sequence=index,
            tool_name=call.tool_name,
            arguments=call.arguments,
            schema_valid=True,
            result_status="succeeded",
        )
        for index, call in enumerate(reversed(expected), start=1)
    )
    result = score_task(
        task,
        run_id="agent-test-3",
        status="succeeded",
        calls=calls,
        final_answer="registry diagnosis",
        evidence_citations=("call_a", "call_b"),
    )

    assert result.argument_correct is True
    assert result.task_success is True


def test_no_tool_and_clarification_metrics_use_relevant_denominators() -> None:
    no_tool = next(item for item in build_tasks("dev") if item.category == "no_tool")
    missing = next(
        item for item in build_tasks("dev") if item.category == "missing_argument_clarification"
    )
    first = score_task(
        no_tool,
        run_id="agent-test-4",
        status="succeeded",
        calls=(),
        final_answer="DDP replicates state; FSDP2 shards it.",
        duration_milliseconds=10,
    )
    required = missing.final_assertions.required_terms[0]
    second = score_task(
        missing,
        run_id="agent-test-5",
        status="succeeded",
        calls=(),
        final_answer=f"Which {required}?",
        duration_milliseconds=20,
    )
    summary = aggregate_results((first, second))

    assert first.task_success is True
    assert second.task_success is True
    assert summary.no_tool_accuracy_basis_points == 10_000
    assert summary.tool_hallucination_rate_basis_points == 0
    assert summary.p95_end_to_end_milliseconds == 20


def test_failed_recovery_trace_does_not_pass() -> None:
    task = next(item for item in build_tasks("dev") if item.category == "tool_failure_recovery")
    expected = task.allowed_trajectories[0].calls[0]
    result = score_task(
        task,
        run_id="agent-test-6",
        status="failed",
        calls=(
            AgentEvalObservedCall(
                sequence=1,
                tool_name=expected.tool_name,
                arguments=expected.arguments,
                schema_valid=True,
                result_status="failed",
            ),
        ),
        final_answer="",
    )

    assert result.error_recovery_success is False
    assert result.task_success is False


def test_transparent_read_retry_counts_as_recovery() -> None:
    task = next(item for item in build_tasks("dev") if item.category == "tool_failure_recovery")
    expected = task.allowed_trajectories[0].calls[0]
    result = score_task(
        task,
        run_id="agent-test-7",
        status="succeeded",
        calls=(
            AgentEvalObservedCall(
                sequence=1,
                tool_name=expected.tool_name,
                arguments=expected.arguments,
                schema_valid=True,
                result_status="succeeded",
                attempts=2,
            ),
        ),
        final_answer="trainer evidence",
        evidence_citations=("call_retry",),
    )

    assert result.error_recovery_success is True
    assert result.task_success is True
