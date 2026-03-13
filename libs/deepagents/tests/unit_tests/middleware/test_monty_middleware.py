from __future__ import annotations

from langchain.tools import ToolRuntime

from deepagents.backends.state import StateBackend
from deepagents.middleware.repl import MontyMiddleware


class _DummyRuntimeState(dict):
    pass


def test_monty_middleware_repl_tool_returns_result_string() -> None:
    mw = MontyMiddleware(backend=StateBackend)
    tool = next(t for t in mw.tools if t.name == "repl")

    state = _DummyRuntimeState()
    runtime = ToolRuntime(state=state, context=None, config={}, stream_writer=None, tool_call_id="x", store=None)  # type: ignore[arg-type]

    result = tool.func(code="1 + 1", runtime=runtime, timeout=5)

    assert result == "2"


def test_monty_middleware_does_not_set_repl_state() -> None:
    mw = MontyMiddleware(backend=StateBackend)
    tool = next(t for t in mw.tools if t.name == "repl")

    state = _DummyRuntimeState()
    runtime = ToolRuntime(state=state, context=None, config={}, stream_writer=None, tool_call_id="x", store=None)  # type: ignore[arg-type]

    result = tool.func(code="1 + 1", runtime=runtime)

    assert result == "2"
    assert "repl_state" not in state
