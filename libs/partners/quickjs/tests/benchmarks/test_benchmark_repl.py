"""Wall-time benchmarks for QuickJS middleware and standalone REPL evaluation.

Run locally: `uv run --group test pytest ./tests/benchmarks -m benchmark`

These tests cover two layers:
- a per-call public-API path that constructs a fresh agent with `REPLMiddleware`,
  runs one trivial eval through the `eval` tool, and verifies the result;
- the underlying `_ThreadREPL` directly, to measure raw interpreter throughput
  without agent construction or tool wiring overhead.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field
from pytest_benchmark.fixture import BenchmarkFixture

from typing import cast

from langchain_quickjs import REPLMiddleware
from langchain_quickjs._repl import _Registry

_EXPRESSION = "1 + 1"
_PARALLEL_WORKERS = 20


class _FakeChatModel(GenericFakeChatModel):
    """Generic fake chat model whose `bind_tools` returns self."""

    messages: Iterator[AIMessage | str] = Field(exclude=True)

    def bind_tools(self, tools: list[Any], **_: Any) -> _FakeChatModel:
        del tools
        return self


def _script(code: str, *, final_message: str = "Done.") -> Iterator[AIMessage]:
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "eval",
                        "args": {"code": code},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content=final_message),
        ]
    )


def _make_agent(code: str, middleware: REPLMiddleware | None = None) -> Any:
    return create_deep_agent(
        model=_FakeChatModel(messages=_script(code, final_message="The answer is 2.")),
        middleware=[middleware or REPLMiddleware()],
    )


def _eval_tool_message(result: dict[str, Any]) -> ToolMessage:
    messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.name == "eval"
    ]
    assert messages, "expected at least one eval ToolMessage"
    return messages[-1]


def _invoke_once(middleware: REPLMiddleware | None = None) -> None:
    result = _make_agent(_EXPRESSION, middleware).invoke(
        {"messages": [HumanMessage(content="Use the eval tool to calculate 1 + 1")]}
    )
    tool_message = _eval_tool_message(result)
    assert "<error" not in tool_message.content, tool_message.content
    assert "2" in tool_message.content, tool_message.content
    assert result["messages"][-1].content == "The answer is 2."


def _echo_program(*, line_count: int = 1000) -> str:
    return "\n".join(['echo("hello");' for _ in range(line_count)])


def _install_echo_host_function(registry: _Registry, *, thread_id: str) -> None:
    repl = registry.get(thread_id)

    async def _ainstall() -> None:
        ctx = cast(Any, repl._ctx)
        ctx.register("echo", lambda value: value)

    repl._worker.run_sync(_ainstall())


def _eval_in_fresh_repl(code: str, *, expected_result: str) -> None:
    registry = _Registry(memory_limit=64 * 1024 * 1024, timeout=5.0, capture_console=True)
    thread_id = "benchmark"
    repl = registry.get(thread_id)
    try:
        _install_echo_host_function(registry, thread_id=thread_id)
        outcome = repl.eval_sync(code)
        assert outcome.error_type is None
        assert outcome.result == expected_result
    finally:
        registry.close()


@pytest.mark.benchmark
class TestQuickJSMiddlewareBenchmark:
    """Wall-time benchmarks for public QuickJS middleware execution."""

    def test_trivial_expression_with_fresh_agent(
        self, benchmark: BenchmarkFixture
    ) -> None:
        """Create a fresh agent and run one trivial expression through `eval`."""
        _invoke_once()
        benchmark(_invoke_once)

    def test_trivial_expression_with_parallel_fresh_agents(
        self, benchmark: BenchmarkFixture
    ) -> None:
        """Run the same trivial expression on 20 fresh agents in parallel."""
        shared_middleware = REPLMiddleware()

        def _run_once() -> None:
            with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as executor:
                list(
                    executor.map(
                        lambda _: _invoke_once(shared_middleware),
                        range(_PARALLEL_WORKERS),
                    )
                )

        _run_once()
        benchmark(_run_once)


@pytest.mark.benchmark
class TestQuickJSStandaloneReplBenchmark:
    """Wall-time benchmarks for direct `_ThreadREPL` evaluation."""

    def test_trivial_expression_in_fresh_repl(
        self, benchmark: BenchmarkFixture
    ) -> None:
        _eval_in_fresh_repl(_EXPRESSION, expected_result="2")
        benchmark(lambda: _eval_in_fresh_repl(_EXPRESSION, expected_result="2"))

    def test_thousand_line_program_in_fresh_repl(
        self, benchmark: BenchmarkFixture
    ) -> None:
        program = _echo_program()
        _eval_in_fresh_repl(program, expected_result="hello")
        benchmark(lambda: _eval_in_fresh_repl(program, expected_result="hello"))
