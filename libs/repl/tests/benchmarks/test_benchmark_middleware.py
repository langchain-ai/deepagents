"""Wall-time benchmarks for end-to-end REPL middleware evaluation.

Run locally: `uv run --group test pytest ./tests/benchmarks -m benchmark`
Run with CodSpeed: `uv run --group test pytest ./tests/benchmarks -m benchmark --codspeed`

These tests measure a per-call public-API path that constructs a fresh agent with
`ReplMiddleware`, runs one trivial eval through the `repl` tool, and verifies the
result. This keeps the number focused on setup overhead plus a minimal end-to-end
execution.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from deepagents.graph import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage
from pytest_benchmark.fixture import BenchmarkFixture

from langchain_repl import ReplMiddleware
from tests.benchmarks._fake_chat_model import GenericFakeChatModel

_EXPRESSION = "print(1 + 1)"
_PARALLEL_WORKERS = 20


def _script(code: str, *, final_message: str = "Done.") -> Iterator[AIMessage]:
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "repl",
                        "args": {"code": code},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=final_message),
        ]
    )


def _make_agent(code: str, middleware: ReplMiddleware | None = None) -> Any:
    return create_deep_agent(
        model=GenericFakeChatModel(
            messages=_script(code, final_message="The answer is 2.")
        ),
        middleware=[middleware or ReplMiddleware()],
    )


def _invoke_once(middleware: ReplMiddleware | None = None) -> None:
    result = _make_agent(_EXPRESSION, middleware).invoke(
        {"messages": [HumanMessage(content="Use the repl to calculate 1 + 1")]}
    )
    assert "messages" in result
    tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
    assert [msg.content for msg in tool_messages] == ["2"]
    assert result["messages"][-1].content == "The answer is 2."


@pytest.mark.benchmark
class TestReplMiddlewareBenchmark:
    """Wall-time benchmarks for public REPL middleware execution."""

    def test_trivial_expression_with_fresh_agent(
        self, benchmark: BenchmarkFixture
    ) -> None:
        """Create a fresh agent and run one trivial expression through `repl`."""
        _invoke_once()
        benchmark(_invoke_once)

    def test_trivial_expression_with_parallel_fresh_agents(
        self, benchmark: BenchmarkFixture
    ) -> None:
        """Run the same trivial expression on 20 fresh agents in parallel."""
        shared_middleware = ReplMiddleware()

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
