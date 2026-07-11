"""Unit tests for `ToolSelectionMiddleware`."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest
from langchain.tools import ToolRuntime
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as tool_decorator

from deepagents.middleware.tool_selection import DISCOVER_TOOLS_NAME, ToolSelectionMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def _tool(name: str, description: str) -> dict[str, Any]:
    return {"name": name, "description": description}


def _request(
    tools: list[dict[str, Any]],
    *,
    query: str = "hello",
    pinned: list[str] | None = None,
) -> ModelRequest:
    state: dict[str, Any] = {"messages": []}
    if pinned is not None:
        state["tool_selection_pinned"] = pinned
    return ModelRequest(
        model=GenericFakeChatModel(messages=iter([])),
        messages=[HumanMessage(content=query)],
        tools=tools,
        state=state,
    )


def _tool_runtime(
    tools: list[dict[str, Any]],
    *,
    pinned: list[str] | None = None,
    tool_call_id: str = "call-1",
) -> ToolRuntime:
    return ToolRuntime(
        state={"tool_selection_pinned": pinned or []},
        context=None,
        config={},
        stream_writer=None,
        tool_call_id=tool_call_id,
        store=None,
        tools=tools,
    )


def test_no_op_below_top_k() -> None:
    """When total tool count is already <= top_k, filtering is skipped entirely."""
    middleware = ToolSelectionMiddleware(top_k=5)
    tools = [_tool(f"tool_{i}", f"description {i}") for i in range(3)]
    request = _request(tools)

    filtered = middleware._filter_request(request)

    assert filtered is request
    assert filtered.tools == tools


def test_top_k_limits_tool_count() -> None:
    """Non always-include tools are limited to top_k after filtering."""
    middleware = ToolSelectionMiddleware(top_k=2, always_include=frozenset())
    tools = [_tool(f"tool_{i}", "search the web for info") for i in range(10)]
    request = _request(tools, query="search the web for info")

    filtered = middleware._filter_request(request)

    assert len(filtered.tools) == 2


def test_always_include_never_filtered() -> None:
    """Tools named in `always_include` survive filtering regardless of score."""
    middleware = ToolSelectionMiddleware(top_k=1, always_include=frozenset({"read_file"}))
    tools = [
        _tool("read_file", "completely unrelated to the query"),
        *[_tool(f"other_{i}", "irrelevant filler tool") for i in range(5)],
        _tool("matching_tool", "banana apple orange fruit query"),
    ]
    request = _request(tools, query="banana apple orange fruit query")

    filtered = middleware._filter_request(request)

    names = {t["name"] for t in filtered.tools}
    assert "read_file" in names
    assert "matching_tool" in names  # highest lexical score, fills the single top_k slot


def test_lexical_scoring_ranks_relevant_tools_higher() -> None:
    """A tool whose description overlaps the query keywords ranks above an unrelated one."""
    middleware = ToolSelectionMiddleware(top_k=1, always_include=frozenset())
    relevant = _tool("weather_lookup", "get the current weather forecast for a city")
    unrelated = _tool("stock_price", "look up the stock price for a ticker symbol")
    request = _request([relevant, unrelated], query="what is the weather forecast today")

    filtered = middleware._filter_request(request)

    names = {t["name"] for t in filtered.tools}
    assert "weather_lookup" in names
    assert "stock_price" not in names


def test_discover_tools_pins_match_for_next_turn() -> None:
    """Calling `discover_tools` with a matching query pins the tool for future turns."""
    middleware = ToolSelectionMiddleware(top_k=1, always_include=frozenset())
    hidden_tool = _tool("weather_lookup", "get the current weather forecast for a city")
    unrelated = _tool("stock_price", "look up the stock price for a ticker symbol")
    all_tools = [hidden_tool, unrelated]

    discover = middleware.tools[0]
    runtime = _tool_runtime(all_tools)
    result = discover.func(query="weather forecast", runtime=runtime)

    assert result.update["tool_selection_pinned"] == ["weather_lookup"]

    # Next turn: the pinned tool survives filtering even though top_k=1 would
    # otherwise favor `unrelated` for this query.
    request = _request(all_tools, query="stock price lookup", pinned=result.update["tool_selection_pinned"])
    filtered = middleware._filter_request(request)

    names = {t["name"] for t in filtered.tools}
    assert "weather_lookup" in names


def test_discover_tools_reports_no_match() -> None:
    """`discover_tools` reports no match found and pins nothing when nothing scores above zero."""
    middleware = ToolSelectionMiddleware(top_k=1, always_include=frozenset())
    tools = [_tool("stock_price", "look up the stock price for a ticker symbol")]

    discover = middleware.tools[0]
    runtime = _tool_runtime(tools)
    result = discover.func(query="completely unrelated gibberish xyzzy", runtime=runtime)

    assert result.update["tool_selection_pinned"] == []
    assert "No tool found" in result.update["messages"][0].content


class _FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings: exact text lookup into a hand-picked vector table."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectors[text]


def test_embeddings_scorer_used_when_provided() -> None:
    """When `embeddings` is provided, scoring goes through cosine similarity, not lexical overlap."""
    relevant = _tool("weather_lookup", "forecast")
    unrelated = _tool("stock_price", "ticker")
    vectors = {
        "weather forecast query": [1.0, 0.0],
        "weather_lookup: forecast": [1.0, 0.0],
        "stock_price: ticker": [0.0, 1.0],
    }
    middleware = ToolSelectionMiddleware(top_k=1, always_include=frozenset(), embeddings=_FakeEmbeddings(vectors))
    request = _request([relevant, unrelated], query="weather forecast query")

    filtered = middleware._filter_request(request)

    names = {t["name"] for t in filtered.tools}
    assert "weather_lookup" in names
    assert "stock_price" not in names


def test_wrap_model_call_filters_tools() -> None:
    """`wrap_model_call` applies filtering before invoking the handler."""
    middleware = ToolSelectionMiddleware(top_k=1, always_include=frozenset())
    relevant = _tool("weather_lookup", "get the current weather forecast for a city")
    unrelated = _tool("stock_price", "look up the stock price for a ticker symbol")
    request = _request([relevant, unrelated], query="what is the weather forecast today")

    captured: dict[str, Any] = {}

    def handler(req: ModelRequest) -> str:
        captured["tools"] = req.tools
        return "ok"

    middleware.wrap_model_call(request, handler)

    names = {t["name"] for t in captured["tools"]}
    assert "weather_lookup" in names
    assert "stock_price" not in names


async def test_awrap_model_call_filters_tools() -> None:
    """`awrap_model_call` applies the same filtering as the sync path."""
    middleware = ToolSelectionMiddleware(top_k=1, always_include=frozenset())
    relevant = _tool("weather_lookup", "get the current weather forecast for a city")
    unrelated = _tool("stock_price", "look up the stock price for a ticker symbol")
    request = _request([relevant, unrelated], query="what is the weather forecast today")

    captured: dict[str, Any] = {}

    async def handler(req: ModelRequest) -> str:
        captured["tools"] = req.tools
        return "ok"

    await middleware.awrap_model_call(request, handler)

    names = {t["name"] for t in captured["tools"]}
    assert "weather_lookup" in names
    assert "stock_price" not in names


def test_discover_tools_name_always_kept_when_present() -> None:
    """The `discover_tools` tool itself is never filtered out of the request."""
    middleware = ToolSelectionMiddleware(top_k=1, always_include=frozenset())
    discover_dict = _tool(DISCOVER_TOOLS_NAME, "search the full tool registry")
    filler = [_tool(f"other_{i}", "irrelevant filler tool") for i in range(5)]
    request = _request([discover_dict, *filler], query="totally unrelated query text")

    filtered = middleware._filter_request(request)

    names = {t["name"] for t in filtered.tools}
    assert DISCOVER_TOOLS_NAME in names


def test_discover_tools_runs_through_real_tool_node() -> None:
    """`discover_tools` must work when invoked by the framework's own ToolNode, not just by calling `.func()` directly.

    Regression test: `ToolRuntime` injection is detected by inspecting the tool's raw
    function signature (`inspect.signature(fn).parameters[...].annotation`). If that
    annotation isn't a live type at definition time (e.g. `runtime: ToolRuntime` stored as
    a postponed string because the module used `from __future__ import annotations`), the
    tool node fails to recognize `runtime` as injectable and raises `TypeError: missing 1
    required positional argument: 'runtime'` at call time -- a failure calling `.func()`
    directly never exercises, since that bypasses the framework's injection path entirely.
    """
    middleware = ToolSelectionMiddleware(top_k=1, always_include=frozenset())

    @tool_decorator
    def calculator(expression: str) -> str:
        """Evaluate a basic arithmetic expression."""
        return expression

    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": DISCOVER_TOOLS_NAME, "args": {"query": "arithmetic expression"}, "id": "call_1"}],
                ),
                AIMessage(content="Found it."),
            ]
        )
    )
    agent = create_agent(model=fake_model, tools=[calculator], middleware=[middleware])

    result = agent.invoke({"messages": [HumanMessage(content="do some math")]})

    tool_message = next(m for m in result["messages"] if getattr(m, "tool_call_id", None) == "call_1")
    assert "Found tool `calculator`" in tool_message.content
