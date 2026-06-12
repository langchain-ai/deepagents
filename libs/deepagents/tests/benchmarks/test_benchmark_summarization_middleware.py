"""Wall-time benchmarks for `SummarizationMiddleware` per-model-call overhead.

Run locally:  `make -C libs/deepagents benchmark`
Run with CodSpeed:  `make -C libs/deepagents bench`

These tests measure the wall time of `wrap_model_call` / `awrap_model_call`
on the *common* path -- the one taken on every model invocation where nothing
is truncated and nothing is summarized. That path runs the token counter, and
counting with `tools=` is expensive because every tool schema is converted on
each invocation.

The benchmarks pin a deterministic token counter that reproduces that cost
(it converts each tool to an OpenAI schema per call) and scale the tool count,
so the per-call counting overhead dominates the measurement. They exist to
guard the optimization in PR #3877, which collapsed two full token counts per
model call down to one on this path; with a duplicate count the numbers here
roughly double.

Regression detection is handled by CodSpeed in CI. Local runs produce
pytest-benchmark tables (min/max/mean/stddev) for human inspection.
"""

import asyncio
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pytest_benchmark.fixture import BenchmarkFixture

from deepagents.middleware.summarization import SummarizationMiddleware
from tests.unit_tests.middleware.test_summarization_middleware import (
    MockBackend,
    make_conversation_messages,
    make_mock_model,
    make_mock_runtime,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState, ModelResponse

# A high threshold both triggers never reach: the conversation stays small, so
# neither truncation nor summarization fires and the no-op path is exercised.
_NEVER = 1_000_000_000

# Tool counts to scale the per-call schema-conversion cost. 23 matches the
# production deep-agent trace cited in PR #3877.
_TOOL_COUNTS = [1, 5, 23]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(idx: int) -> BaseTool:
    """Create a named tool with a non-trivial schema for conversion cost."""

    @tool(description=f"Tool number {idx} used for benchmarking schema conversion")
    def dynamic_tool(query: str, count: int = 1, *, verbose: bool = False) -> str:
        """Echo the query a number of times."""
        return f"tool_{idx}({query}, {count}, {verbose})"

    dynamic_tool.name = f"tool_{idx}"
    return dynamic_tool


def _converting_token_counter(messages: list[BaseMessage], **kwargs: Any) -> int:
    """Token counter that pays the real per-call tool-conversion cost.

    Production token counters convert every tool schema on each call (see
    langchain-ai/langchain#38073). This reproduces that cost deterministically
    so the benchmark measures what the optimization actually saves, rather than
    a trivial counter whose double-invocation is invisible.
    """
    tools: list[BaseTool | dict[str, Any]] | None = kwargs.get("tools")
    total = 0
    if tools is not None:
        for t in tools:
            schema = convert_to_openai_tool(t)
            total += len(str(schema)) // 4
    for message in messages:
        total += len(str(message.content)) // 4
    return total


def _make_middleware() -> SummarizationMiddleware:
    """Build a middleware whose triggers never fire (common no-op path)."""
    return SummarizationMiddleware(
        model=make_mock_model(),
        backend=MockBackend(),
        trigger=("tokens", _NEVER),
        token_counter=_converting_token_counter,
        truncate_args_settings={
            "trigger": ("tokens", _NEVER),
            "keep": ("messages", 2),
        },
    )


def _make_request(tools: list[BaseTool]) -> ModelRequest:
    """Build a `ModelRequest` carrying the tools and a realistic conversation."""
    state = cast("AgentState[Any]", {"messages": make_conversation_messages()})
    return ModelRequest(
        model=make_mock_model(),
        messages=state["messages"],
        system_message=None,
        tools=tools,
        runtime=make_mock_runtime(),
        state=state,
    )


def _handler(_request: ModelRequest) -> "ModelResponse":
    """No-op handler standing in for the wrapped model call."""
    return AIMessage(content="ok")


async def _ahandler(_request: ModelRequest) -> "ModelResponse":
    """Async no-op handler standing in for the wrapped model call."""
    return AIMessage(content="ok")


@pytest.fixture
def bench_event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """A dedicated event loop, closed on teardown to avoid leaking loops.

    Named distinctly from pytest-asyncio's reserved `event_loop` fixture, which
    is deprecated to override; these benchmarks are sync and drive the loop
    manually via `run_until_complete`.
    """
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestSummarizationWrapModelCall:
    """Per-call overhead of the no-summarization, no-truncation path."""

    @pytest.mark.parametrize("tool_count", _TOOL_COUNTS, ids=lambda n: f"{n}_tools")
    def test_wrap_model_call(self, benchmark: BenchmarkFixture, tool_count: int) -> None:
        """Sync path: token counting must run once, not once per check."""
        middleware = _make_middleware()
        tools = [_make_tool(i) for i in range(tool_count)]

        @benchmark  # type: ignore[misc]
        def _() -> None:
            middleware.wrap_model_call(_make_request(tools), _handler)

    @pytest.mark.parametrize("tool_count", _TOOL_COUNTS, ids=lambda n: f"{n}_tools")
    def test_awrap_model_call(
        self,
        benchmark: BenchmarkFixture,
        tool_count: int,
        bench_event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Async path: same single-count guarantee as the sync path."""
        middleware = _make_middleware()
        tools = [_make_tool(i) for i in range(tool_count)]

        @benchmark  # type: ignore[misc]
        def _() -> None:
            bench_event_loop.run_until_complete(middleware.awrap_model_call(_make_request(tools), _ahandler))
