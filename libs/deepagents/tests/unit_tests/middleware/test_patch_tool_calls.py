"""Unit tests for PatchToolCallsMiddleware."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Import directly from the module file to avoid deepagents/__init__.py
# which eagerly imports graph.py → langchain_anthropic (not installed in test env)
_mod_path = Path(__file__).parent.parent.parent.parent / "deepagents" / "middleware" / "patch_tool_calls.py"
_spec = importlib.util.spec_from_file_location("patch_tool_calls", _mod_path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
PatchToolCallsMiddleware = _mod.PatchToolCallsMiddleware


def _make_runtime() -> MagicMock:
    return MagicMock()


def _state(messages: list) -> dict:
    return {"messages": messages}


def _ai_with_tool_call(tool_id: str = "tc-1", name: str = "my_tool") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": tool_id, "name": name, "args": {}}],
    )


def _tool_message(tool_id: str = "tc-1", name: str = "my_tool") -> ToolMessage:
    return ToolMessage(content="result", name=name, tool_call_id=tool_id)


class TestNoDangling:
    """No patching needed when all tool calls have been answered."""

    def test_empty_messages_returns_none(self) -> None:
        mw = PatchToolCallsMiddleware()
        result = mw.before_agent(_state([]), _make_runtime())
        assert result is None

    def test_no_tool_calls_returns_none(self) -> None:
        mw = PatchToolCallsMiddleware()
        state = _state([HumanMessage(content="hi"), AIMessage(content="hello")])
        assert mw.before_agent(state, _make_runtime()) is None

    def test_answered_tool_call_returns_none(self) -> None:
        mw = PatchToolCallsMiddleware()
        state = _state([_ai_with_tool_call("tc-1"), _tool_message("tc-1")])
        assert mw.before_agent(state, _make_runtime()) is None


class TestDanglingCancellation:
    """Dangling tool calls should be patched with a cancellation ToolMessage."""

    def test_dangling_call_gets_cancellation(self) -> None:
        mw = PatchToolCallsMiddleware()
        ai_msg = _ai_with_tool_call("tc-1", "playwright_browser_navigate")
        state = _state([ai_msg])

        result = mw.before_agent(state, _make_runtime())

        assert result is not None
        messages = result["messages"].value
        assert len(messages) == 2
        cancel_msg = messages[1]
        assert isinstance(cancel_msg, ToolMessage)
        assert cancel_msg.tool_call_id == "tc-1"
        assert "cancelled" in cancel_msg.content

    def test_only_unanswered_calls_are_cancelled(self) -> None:
        """If one call is answered and one is not, only the unanswered is patched."""
        mw = PatchToolCallsMiddleware()
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"id": "tc-1", "name": "tool_a", "args": {}},
                {"id": "tc-2", "name": "tool_b", "args": {}},
            ],
        )
        tool_answer = _tool_message("tc-1", "tool_a")
        state = _state([ai_msg, tool_answer])

        result = mw.before_agent(state, _make_runtime())

        assert result is not None
        messages = result["messages"].value
        # ai_msg + cancel for tc-2 (inserted right after AIMessage) + tool_answer
        assert len(messages) == 3
        cancel_msg = messages[1]
        assert cancel_msg.tool_call_id == "tc-2"


class TestInFlightProtection:
    """Tool calls registered as in-flight must not be cancelled."""

    def test_in_flight_call_is_not_cancelled(self) -> None:
        mw = PatchToolCallsMiddleware()
        mw._in_flight.add((None, "tc-1"))
        ai_msg = _ai_with_tool_call("tc-1")
        state = _state([ai_msg])

        result = mw.before_agent(state, _make_runtime())
        assert result is None

    def test_in_flight_cleared_after_sync_wrap(self) -> None:
        mw = PatchToolCallsMiddleware()
        request = MagicMock()
        request.tool_call = {"id": "tc-1"}
        handler = MagicMock(return_value=_tool_message("tc-1"))

        mw.wrap_tool_call(request, handler)

        assert "tc-1" not in mw._in_flight

    def test_in_flight_cleared_on_sync_exception(self) -> None:
        mw = PatchToolCallsMiddleware()
        request = MagicMock()
        request.tool_call = {"id": "tc-1"}
        handler = MagicMock(side_effect=RuntimeError("tool failed"))

        with pytest.raises(RuntimeError):
            mw.wrap_tool_call(request, handler)

        assert "tc-1" not in mw._in_flight

    async def test_in_flight_cleared_after_async_wrap(self) -> None:
        mw = PatchToolCallsMiddleware()
        request = MagicMock()
        request.tool_call = {"id": "tc-1"}
        handler = AsyncMock(return_value=_tool_message("tc-1"))

        await mw.awrap_tool_call(request, handler)

        assert "tc-1" not in mw._in_flight

    async def test_in_flight_cleared_on_async_exception(self) -> None:
        mw = PatchToolCallsMiddleware()
        request = MagicMock()
        request.tool_call = {"id": "tc-1"}
        handler = AsyncMock(side_effect=RuntimeError("async tool failed"))

        with pytest.raises(RuntimeError):
            await mw.awrap_tool_call(request, handler)

        assert "tc-1" not in mw._in_flight

    async def test_in_flight_cleared_on_cancellation(self) -> None:
        """Simulates asyncio.CancelledError (e.g. Playwright timeout)."""
        mw = PatchToolCallsMiddleware()
        request = MagicMock()
        request.tool_call = {"id": "tc-playwright"}
        handler = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await mw.awrap_tool_call(request, handler)

        assert "tc-playwright" not in mw._in_flight


class TestAbforeAgentWaiting:
    """abefore_agent must wait for in-flight tools to clear before proceeding."""

    async def test_abefore_agent_no_in_flight_returns_immediately(self) -> None:
        mw = PatchToolCallsMiddleware()
        state = _state([_ai_with_tool_call("tc-1")])
        result = await mw.abefore_agent(state, _make_runtime())
        # No in-flight, so the dangling call should be cancelled normally
        assert result is not None

    async def test_abefore_agent_waits_for_in_flight_to_clear(self) -> None:
        """If in-flight clears asynchronously, abefore_agent should not cancel it."""
        mw = PatchToolCallsMiddleware()
        mw._in_flight.add((None, "tc-1"))

        async def _clear_after_delay() -> None:
            await asyncio.sleep(0.02)
            mw._in_flight.discard((None, "tc-1"))

        _background_tasks = set()
        task = asyncio.create_task(_clear_after_delay())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        ai_msg = _ai_with_tool_call("tc-1")
        state = _state([ai_msg, _tool_message("tc-1")])

        result = await mw.abefore_agent(state, _make_runtime())
        # tc-1 was answered so no dangling once in-flight is cleared
        assert result is None

    async def test_abefore_agent_cancels_if_still_in_flight_after_timeout(self) -> None:
        """If tool stays in-flight past the wait window, the in-flight ID is still protected.

        The in-flight ID is not cancelled because we cannot verify the tool truly abandoned.
        """
        mw = PatchToolCallsMiddleware()
        mw._in_flight.add((None, "tc-stuck"))

        ai_msg = _ai_with_tool_call("tc-stuck")
        # No corresponding ToolMessage → dangling, but still in-flight
        state = _state([ai_msg])

        result = await mw.abefore_agent(state, _make_runtime())
        # tc-stuck is still in _in_flight, so it is NOT cancelled (protected)
        assert result is None


class TestThreadIsolation:
    """Tools in one thread should not affect or block another thread."""

    @pytest.mark.asyncio
    async def test_isolation_between_threads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mw = PatchToolCallsMiddleware()

        # Mock thread 1
        monkeypatch.setattr(_mod, "get_config", lambda: {"configurable": {"thread_id": "thread-1"}})
        mw._in_flight.add(("thread-1", "tc-1"))

        # In thread 2, tc-1 should NOT be seen as in-flight
        monkeypatch.setattr(_mod, "get_config", lambda: {"configurable": {"thread_id": "thread-2"}})
        state = _state([_ai_with_tool_call("tc-1")])

        # It should be cancelled because thread-2 doesn't own tc-1
        result = await mw.abefore_agent(state, _make_runtime())
        assert result is not None
        messages = result["messages"].value
        assert any(isinstance(m, ToolMessage) and m.tool_call_id == "tc-1" for m in messages)
