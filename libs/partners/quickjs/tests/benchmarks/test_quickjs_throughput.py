"""Wall-time throughput benchmarks for QuickJS REPL middleware.

Run locally:  `make benchmark`
Run with CodSpeed:  `uv run --group test pytest ./tests -m benchmark --codspeed`

These tests measure throughput for many single-thread eval iterations where the
workload combines PTC tool calls with ``console.log`` output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage

from langchain_quickjs import REPLMiddleware
from tests.benchmarks._common import (
    PTC_AND_CONSOLE_CODE,
    THROUGHPUT_ITERATIONS,
    FakeChatModel,
    assert_eval_succeeded,
    echo_payload,
    invoke_payload,
    make_agent,
    tool_call_message,
)

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.benchmark
class TestQuickJSThroughputBenchmarks:
    """Benchmarks that track eval throughput for hot single-thread loops."""

    def _make_multi_turn_agent(
        self,
        *,
        middleware: REPLMiddleware,
        codes: list[str],
    ) -> Any:
        messages: list[AIMessage] = []
        for index, code in enumerate(codes):
            messages.append(tool_call_message(code, call_id=f"call_{index}"))
            messages.append(AIMessage(content="done"))
        return create_deep_agent(
            model=FakeChatModel(messages=iter(messages)),
            middleware=[middleware],
        )

    def _record_turn_metrics(
        self,
        *,
        benchmark: BenchmarkFixture,
        turns_per_round: int,
    ) -> None:
        benchmark.extra_info["turns_per_round"] = turns_per_round
        stats = getattr(getattr(benchmark, "stats", None), "stats", None)
        mean_seconds = getattr(stats, "mean", None)
        if isinstance(mean_seconds, (int, float)) and mean_seconds > 0:
            benchmark.extra_info["turns_per_second"] = round(
                turns_per_round / mean_seconds,
                3,
            )

    def test_single_thread_many_iterations_ptc_and_console_log(
        self,
        benchmark: BenchmarkFixture,
    ) -> None:
        """Measure throughput for many eval calls in one thread and process."""
        middleware = REPLMiddleware(capture_console=True, ptc=[echo_payload])

        @benchmark
        def _() -> None:
            agent = make_agent(
                code=PTC_AND_CONSOLE_CODE,
                middleware=middleware,
                repeats=THROUGHPUT_ITERATIONS,
            )
            for _ in range(THROUGHPUT_ITERATIONS):
                result = agent.invoke(
                    invoke_payload(),
                    config={"configurable": {"thread_id": "throughput-bench-thread"}},
                )
                assert_eval_succeeded(result)

        benchmark.extra_info["thread_count"] = 1
        benchmark.extra_info["iterations_per_round"] = THROUGHPUT_ITERATIONS
        benchmark.extra_info["workload"] = "ptc_tools_plus_console_log"
        self._record_turn_metrics(
            benchmark=benchmark,
            turns_per_round=THROUGHPUT_ITERATIONS,
        )
