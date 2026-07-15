"""Unit tests for textual_adapter functions."""

import asyncio
import sys
from asyncio import Future
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from datetime import datetime
from pathlib import Path
from time import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import ValidationError
from rich.console import Console

from deepagents_code import config as config_module
from deepagents_code._ask_user_types import AskUserWidgetResult, Question
from deepagents_code._tool_stream import (
    TOOL_OUTPUT_TRUNCATION_MARKER,
    UNRENDERABLE_TOOL_OUTPUT,
)
from deepagents_code.approval_mode import APPROVAL_MODE_NAMESPACE, approval_mode_key
from deepagents_code.client.non_interactive import (
    StreamState,
    _process_ai_message,
    _process_message_chunk,
)
from deepagents_code.config import ASCII_GLYPHS, UNICODE_GLYPHS, build_stream_config
from deepagents_code.tui.textual_adapter import (
    SessionStats,
    TextualUIAdapter,
    _build_interrupted_ai_message,
    _format_rubric_details,
    _format_rubric_event,
    _handle_interrupt_cleanup,
    _interrupt_owned_tool_rows,
    _is_summarization_chunk,
    _read_mentioned_file,
    execute_task_textual,
)
from deepagents_code.tui.widgets.messages import (
    AppMessage,
    RubricResultMessage,
    SummarizationMessage,
    ToolCallMessage,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


async def _mock_mount(widget: object) -> None:
    """Mock mount function for tests."""


def _mock_approval() -> Future[object]:
    """Mock approval function for tests."""
    future: Future[object] = Future()
    return future


def _noop_status(_: str) -> None:
    """No-op status callback for tests."""


class TestTextualUIAdapterInit:
    """Tests for `TextualUIAdapter` initialization."""

    def test_set_spinner_callback_stored(self) -> None:
        """Verify `set_spinner` callback is properly stored."""

        async def mock_spinner(status: str | None) -> None:
            pass

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=mock_spinner,
        )
        assert adapter._set_spinner is mock_spinner

    def test_set_spinner_defaults_to_none(self) -> None:
        """Verify `set_spinner` is optional and defaults to `None`."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        assert adapter._set_spinner is None

    def test_current_tool_messages_initialized_empty(self) -> None:
        """Verify `_current_tool_messages` is initialized as empty dict."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        assert adapter._current_tool_messages == {}

    def test_token_callbacks_initialized_none(self) -> None:
        """Verify token callbacks are initialized as `None`."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        assert adapter._on_tokens_update is None
        assert adapter._on_tokens_pending is None
        assert adapter._on_tokens_show is None

    def test_on_tool_complete_defaults_to_none_and_accepts_callback(self) -> None:
        """Verify `on_tool_complete` is optional and can be assigned via init."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        assert adapter._on_tool_complete is None

        callback = MagicMock()
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            on_tool_complete=callback,
        )
        assert adapter._on_tool_complete is callback

    def test_on_user_visible_output_started_defaults_to_none_and_accepts_callback(
        self,
    ) -> None:
        """Verify the user-visible-output callback can be assigned."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        assert adapter._on_user_visible_output_started is None

        callback = MagicMock()
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            on_user_visible_output_started=callback,
        )
        assert adapter._on_user_visible_output_started is callback

    def test_set_token_callbacks(self) -> None:
        """Verify token callbacks can be assigned."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        def update_cb(count: int, *, approximate: bool = False) -> None:
            pass

        def pending_cb() -> None:
            pass

        def show_cb(*, approximate: bool = False) -> None:
            pass

        adapter._on_tokens_update = update_cb
        adapter._on_tokens_pending = pending_cb
        adapter._on_tokens_show = show_cb
        assert adapter._on_tokens_update is update_cb
        assert adapter._on_tokens_pending is pending_cb
        assert adapter._on_tokens_show is show_cb

    def test_finalize_pending_tools_with_error_marks_and_clears(self) -> None:
        """Pending tool widgets should be marked error and then cleared."""
        set_active = MagicMock()
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_active_message=set_active,
        )

        tool_1 = MagicMock()
        tool_2 = MagicMock()
        adapter._current_tool_messages = {"a": tool_1, "b": tool_2}

        adapter.finalize_pending_tools_with_error("Agent error: boom")

        tool_1.set_error.assert_called_once_with("Agent error: boom")
        tool_2.set_error.assert_called_once_with("Agent error: boom")
        assert adapter._current_tool_messages == {}
        set_active.assert_called_once_with(None)

    def test_finalize_pending_tools_dispatches_terminal_hooks(self) -> None:
        """Aborting a stream mid-tool closes each pending tool.use.

        Each pending widget already had its `tool.use` dispatched at mount, so
        the safety-net path must emit a terminal `tool.error`/`tool.result` for
        it — otherwise the aborted tool leaves an unterminated `tool.use`.
        """
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_active_message=MagicMock(),
        )
        tool_widget = MagicMock()
        tool_widget.tool_name = "read_file"
        tool_widget.args = {"path": "notes.txt"}
        adapter._current_tool_messages = {"call-1": tool_widget}

        with patch(
            "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
        ) as mock_dispatch:
            adapter.finalize_pending_tools_with_error("Agent error: boom")

        events = [(c[0][0], c[0][1]) for c in mock_dispatch.call_args_list]
        assert ("tool.error", {"tool_names": ["read_file"]}) in events
        result_payloads = [p for e, p in events if e == "tool.result"]
        assert result_payloads == [
            {
                "tool_name": "read_file",
                "tool_id": "call-1",
                "tool_args": {"path": "notes.txt"},
                "tool_status": "error",
                "tool_output": "Agent error: boom",
            }
        ]
        assert adapter._current_tool_messages == {}


class TestInterruptCleanup:
    """Tests for interrupt cleanup token handling."""

    async def test_tool_only_interrupt_marks_tokens_approximate(self) -> None:
        """Tool-only interrupted turns should keep the stale-token marker."""
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            mounted.append(widget)
            await asyncio.sleep(0)

        set_spinner = AsyncMock()
        set_active = MagicMock()
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=set_spinner,
            set_active_message=set_active,
        )

        tool_widget = MagicMock()
        tool_widget._tool_name = "read_file"
        tool_widget._args = {"path": "notes.txt"}
        adapter._current_tool_messages = {"call-1": tool_widget}

        show_calls: list[bool] = []

        def show_cb(*, approximate: bool = False) -> None:
            show_calls.append(approximate)

        adapter._on_tokens_show = show_cb

        agent = SimpleNamespace(aupdate_state=AsyncMock())
        turn_stats = SessionStats()
        config = {"configurable": {"thread_id": "t-1"}}

        with patch(
            "deepagents_code.tui.textual_adapter.time.monotonic", return_value=101.0
        ):
            await _handle_interrupt_cleanup(
                adapter=adapter,
                agent=agent,
                config=config,  # ty: ignore
                pending_text_by_namespace={},
                captured_input_tokens=0,
                captured_output_tokens=0,
                turn_stats=turn_stats,
                start_time=100.0,
            )

        assert mounted
        assert show_calls == [True]
        assert turn_stats.wall_time_seconds == pytest.approx(1.0)
        set_active.assert_called_once_with(None)
        set_spinner.assert_awaited_once_with(None)
        tool_widget.set_rejected.assert_called_once_with()
        assert adapter._current_tool_messages == {}

        interrupted_payload = agent.aupdate_state.await_args_list[0].args[1]
        interrupted_msg = interrupted_payload["messages"][0]
        assert interrupted_msg.tool_calls[0]["id"] == "call-1"
        assert interrupted_msg.tool_calls[0]["name"] == "read_file"

    async def test_interrupt_cleanup_dispatches_terminal_hooks(self) -> None:
        """A cancelled turn closes each pending tool.use with terminal hooks.

        A tool whose `tool.use` fired but whose `ToolMessage` never arrived
        (because Ctrl+C aborted the turn) must still get a terminal
        `tool.error`/`tool.result`, mirroring the HITL-reject branches.
        """
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_active_message=MagicMock(),
        )
        tool_widget = MagicMock()
        # Public accessors feed the terminal-hook payload; the private backing
        # fields feed the interrupted-AIMessage rebuild — set both consistently,
        # mirroring the real `ToolCallMessage` where the accessors read these.
        tool_widget.tool_name = "execute"
        tool_widget.args = {"command": "sleep 100"}
        tool_widget._tool_name = "execute"
        tool_widget._args = {"command": "sleep 100"}
        adapter._current_tool_messages = {"call-1": tool_widget}
        agent = SimpleNamespace(aupdate_state=AsyncMock())

        with patch(
            "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
        ) as mock_dispatch:
            await _handle_interrupt_cleanup(
                adapter=adapter,
                agent=agent,
                config={"configurable": {"thread_id": "t-1"}},  # ty: ignore
                pending_text_by_namespace={},
                captured_input_tokens=0,
                captured_output_tokens=0,
                turn_stats=SessionStats(),
                start_time=0.0,
            )

        events = [(c[0][0], c[0][1]) for c in mock_dispatch.call_args_list]
        assert ("tool.error", {"tool_names": ["execute"]}) in events
        result_payloads = [p for e, p in events if e == "tool.result"]
        assert result_payloads == [
            {
                "tool_name": "execute",
                "tool_id": "call-1",
                "tool_args": {"command": "sleep 100"},
                "tool_status": "error",
                "tool_output": "Turn cancelled",
            }
        ]
        tool_widget.set_rejected.assert_called_once_with()
        assert adapter._current_tool_messages == {}

    async def test_terminal_hooks_dispatched_before_state_writes(self) -> None:
        """Terminal hooks are scheduled before the (possibly slow) state writes.

        On an interactive quit the graceful-exit drain in `app.py` snapshots the
        in-flight hook tasks right after cancelling the worker. If the terminal
        hooks fired only *after* `aupdate_state`'s awaited remote writes, a slow
        checkpointer could push them past that snapshot and they would be
        cancelled at loop teardown — a silent audit gap. Pin that every hook
        dispatch precedes the first state write.
        """
        order: list[str] = []

        async def _record_update(*_: Any, **__: Any) -> None:
            # A real remote state write awaits; the yield also mirrors how a slow
            # checkpointer would interleave with the scheduled hook tasks.
            await asyncio.sleep(0)
            order.append("aupdate_state")

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_active_message=MagicMock(),
        )
        tool_widget = MagicMock()
        tool_widget.tool_name = "execute"
        tool_widget.args = {"command": "sleep 100"}
        tool_widget._tool_name = "execute"
        tool_widget._args = {"command": "sleep 100"}
        adapter._current_tool_messages = {"call-1": tool_widget}
        agent = SimpleNamespace(aupdate_state=_record_update)

        def _record_dispatch(event: str, _payload: dict[str, Any]) -> None:
            order.append(f"dispatch:{event}")

        with patch(
            "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget",
            side_effect=_record_dispatch,
        ):
            await _handle_interrupt_cleanup(
                adapter=adapter,
                agent=agent,
                config={"configurable": {"thread_id": "t-1"}},  # ty: ignore
                pending_text_by_namespace={},
                captured_input_tokens=0,
                captured_output_tokens=0,
                turn_stats=SessionStats(),
                start_time=0.0,
            )

        assert "dispatch:tool.result" in order
        assert "aupdate_state" in order
        first_state_write = order.index("aupdate_state")
        assert all(
            order.index(item) < first_state_write
            for item in order
            if item.startswith("dispatch:")
        )

    async def test_interrupt_stops_active_assistant_streams(self) -> None:
        """Interrupted streaming messages should not leave flush timers running."""
        sync_message_content = MagicMock()
        assistant_msg = SimpleNamespace(
            id="asst-1",
            _content="partial response",
            stop_stream=AsyncMock(),
        )
        assistant_messages = {(): assistant_msg}

        adapter = TextualUIAdapter(
            mount_message=AsyncMock(),
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
            sync_message_content=sync_message_content,
        )
        agent = SimpleNamespace(aupdate_state=AsyncMock())

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config={"configurable": {"thread_id": "t-1"}},
            pending_text_by_namespace={(): "partial response"},
            assistant_message_by_namespace=assistant_messages,
            captured_input_tokens=0,
            captured_output_tokens=0,
            turn_stats=SessionStats(),
            start_time=0.0,
        )

        assistant_msg.stop_stream.assert_awaited_once_with()
        sync_message_content.assert_called_once_with("asst-1", "partial response")
        assert assistant_messages == {}

    async def test_interrupt_cancels_active_remote_runs_before_state_writes(
        self,
    ) -> None:
        """Remote runs should be interrupted before recovery state is persisted."""
        calls: list[str] = []

        # Sync side effects are fine: the AsyncMock wrapping them is awaitable,
        # and recording into `calls` is enough to assert relative ordering.
        def cancel_runs(_config: object) -> None:
            calls.append("cancel")

        def update_state(_config: object, _values: dict[str, Any]) -> None:
            calls.append("update")

        adapter = TextualUIAdapter(
            mount_message=AsyncMock(),
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )
        agent = SimpleNamespace(
            acancel_active_runs=AsyncMock(side_effect=cancel_runs),
            aupdate_state=AsyncMock(side_effect=update_state),
        )
        config: RunnableConfig = {"configurable": {"thread_id": "t-1"}}

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config=config,
            pending_text_by_namespace={},
            captured_input_tokens=0,
            captured_output_tokens=0,
            turn_stats=SessionStats(),
            start_time=0.0,
        )

        agent.acancel_active_runs.assert_awaited_once_with(config)
        assert calls == ["cancel", "update"]

    async def test_criteria_cancel_skips_conversation_interruption_recovery(
        self,
    ) -> None:
        """Criteria cancellation cleans runtime UI without adding chat messages."""
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            mounted.append(widget)
            await asyncio.sleep(0)

        tool_widget = MagicMock()
        tool_widget.tool_name = "docs_search"
        tool_widget.args = {"query": "login"}
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )
        adapter._current_tool_messages = {"call-1": tool_widget}
        agent = SimpleNamespace(
            acancel_active_runs=AsyncMock(),
            aupdate_state=AsyncMock(),
        )

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config={"configurable": {"thread_id": "t-1"}},
            pending_text_by_namespace={(): "partial criteria output"},
            captured_input_tokens=10,
            captured_output_tokens=5,
            turn_stats=SessionStats(),
            start_time=0.0,
            recover_interrupted_turn=False,
        )

        agent.acancel_active_runs.assert_awaited_once()
        agent.aupdate_state.assert_not_awaited()
        tool_widget.set_rejected.assert_called_once_with()
        assert adapter._current_tool_messages == {}
        assert not any(
            isinstance(widget, AppMessage)
            and "Interrupted by user" in str(widget._content)
            for widget in mounted
        )

    async def test_chat_cancel_retains_conversation_interruption_recovery(
        self,
    ) -> None:
        """Ordinary chat cancellation still records and displays interruption."""
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            mounted.append(widget)
            await asyncio.sleep(0)

        agent = SimpleNamespace(aupdate_state=AsyncMock())
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config={"configurable": {"thread_id": "t-1"}},
            pending_text_by_namespace={(): "partial answer"},
            captured_input_tokens=10,
            captured_output_tokens=5,
            turn_stats=SessionStats(),
            start_time=0.0,
        )

        assert any(
            isinstance(widget, AppMessage)
            and str(widget._content) == "Interrupted by user"
            for widget in mounted
        )
        assert len(agent.aupdate_state.await_args_list) == 2
        persisted = [
            value
            for call in agent.aupdate_state.await_args_list
            for value in call.args[1]["messages"]
        ]
        assert any("partial answer" in str(message.content) for message in persisted)
        assert any(
            "Task interrupted by user" in str(message.content) for message in persisted
        )

    async def test_remote_run_cancel_failure_does_not_skip_state_writes(self) -> None:
        """Interrupt cleanup remains best-effort when remote cancel fails."""
        agent = SimpleNamespace(
            acancel_active_runs=AsyncMock(side_effect=RuntimeError("down")),
            aupdate_state=AsyncMock(),
        )
        adapter = TextualUIAdapter(
            mount_message=AsyncMock(),
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config={"configurable": {"thread_id": "t-1"}},
            pending_text_by_namespace={},
            captured_input_tokens=0,
            captured_output_tokens=0,
            turn_stats=SessionStats(),
            start_time=0.0,
        )

        agent.acancel_active_runs.assert_awaited_once()
        agent.aupdate_state.assert_awaited_once()

    async def test_remote_run_cancel_value_error_propagates(self) -> None:
        """A `ValueError` (missing `thread_id`) propagates instead of warning.

        It is a contract bug rather than a transient remote failure, so it must
        surface and the recovery-state write must be skipped.
        """
        agent = SimpleNamespace(
            acancel_active_runs=AsyncMock(side_effect=ValueError("missing thread_id")),
            aupdate_state=AsyncMock(),
        )
        adapter = TextualUIAdapter(
            mount_message=AsyncMock(),
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )

        with pytest.raises(ValueError, match="missing thread_id"):
            await _handle_interrupt_cleanup(
                adapter=adapter,
                agent=agent,
                config={"configurable": {"thread_id": "t-1"}},
                pending_text_by_namespace={},
                captured_input_tokens=0,
                captured_output_tokens=0,
                turn_stats=SessionStats(),
                start_time=0.0,
            )

        agent.acancel_active_runs.assert_awaited_once()
        # The re-raise short-circuits before the recovery-state write, which
        # is what distinguishes it from the swallowed-transient-failure path.
        agent.aupdate_state.assert_not_awaited()

    async def test_local_agent_without_cancel_method_still_writes_state(self) -> None:
        """Local agents lack `acancel_active_runs`; cleanup must skip it cleanly."""
        agent = SimpleNamespace(aupdate_state=AsyncMock())
        assert not hasattr(agent, "acancel_active_runs")
        adapter = TextualUIAdapter(
            mount_message=AsyncMock(),
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config={"configurable": {"thread_id": "t-1"}},
            pending_text_by_namespace={},
            captured_input_tokens=0,
            captured_output_tokens=0,
            turn_stats=SessionStats(),
            start_time=0.0,
        )

        agent.aupdate_state.assert_awaited_once()

    async def test_disables_tracing_during_state_save(self) -> None:
        """Interrupt-cleanup `aupdate_state` calls must run with tracing disabled.

        Interrupt state writes (partial AI message + cancellation notice) are
        internal recovery mechanics. Surfacing them as standalone `UpdateState`
        runs in LangSmith would add noise unrelated to user-visible agent activity.
        """
        from langsmith import get_tracing_context

        captured: list[object] = []

        async def _capture(*_args: object, **_kwargs: object) -> None:  # noqa: RUF029
            captured.append(get_tracing_context().get("enabled"))

        agent = SimpleNamespace(aupdate_state=AsyncMock(side_effect=_capture))
        adapter = TextualUIAdapter(
            mount_message=AsyncMock(),
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config={"configurable": {"thread_id": "t-1"}},
            pending_text_by_namespace={},
            captured_input_tokens=0,
            captured_output_tokens=0,
            turn_stats=SessionStats(),
            start_time=0.0,
        )

        assert captured, "aupdate_state was never called"
        assert all(v is False for v in captured), (
            f"tracing was not disabled: {captured}"
        )

    async def test_disables_tracing_when_interrupted_msg_present(self) -> None:
        """Both `aupdate_state` calls disable tracing when interrupted_msg is set.

        When there is a partial AI message to save, both writes (interrupted AI
        message and cancellation notice) must be suppressed from LangSmith traces.
        """
        from langsmith import get_tracing_context

        captured: list[object] = []

        async def _capture(*_args: object, **_kwargs: object) -> None:  # noqa: RUF029
            captured.append(get_tracing_context().get("enabled"))

        tool_widget = MagicMock()
        tool_widget._tool_name = "read_file"
        tool_widget._args = {"path": "notes.txt"}

        agent = SimpleNamespace(aupdate_state=AsyncMock(side_effect=_capture))
        adapter = TextualUIAdapter(
            mount_message=AsyncMock(),
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )
        adapter._current_tool_messages = {"call-1": tool_widget}

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config={"configurable": {"thread_id": "t-1"}},
            pending_text_by_namespace={},
            captured_input_tokens=0,
            captured_output_tokens=0,
            turn_stats=SessionStats(),
            start_time=0.0,
        )

        assert len(captured) == 2, (
            f"expected 2 aupdate_state calls, got {len(captured)}"
        )
        assert all(v is False for v in captured), (
            f"tracing was not disabled: {captured}"
        )


class TestInterruptCleanupTokenPersist:
    """`_context_tokens` rides on the cancellation `aupdate_state` write."""

    async def test_includes_context_tokens_in_cancellation_update(self) -> None:
        """The cancellation HumanMessage write carries the latest token count."""
        captured: list[dict[str, Any]] = []

        async def _capture(_config: object, values: dict[str, Any]) -> None:  # noqa: RUF029
            captured.append(values)

        agent = SimpleNamespace(aupdate_state=AsyncMock(side_effect=_capture))
        adapter = TextualUIAdapter(
            mount_message=AsyncMock(),
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config={"configurable": {"thread_id": "t-1"}},
            pending_text_by_namespace={},
            captured_input_tokens=4321,
            captured_output_tokens=0,
            turn_stats=SessionStats(),
            start_time=0.0,
        )

        # Only the cancellation write happens (no partial AI message in this test);
        # it carries both `messages` and `_context_tokens`.
        assert len(captured) == 1
        assert captured[0]["_context_tokens"] == 4321
        assert "messages" in captured[0]

    async def test_omits_context_tokens_when_no_usage_captured(self) -> None:
        """Zero tokens means we never saw `usage_metadata`; preserve the prior value."""
        captured: list[dict[str, Any]] = []

        async def _capture(_config: object, values: dict[str, Any]) -> None:  # noqa: RUF029
            captured.append(values)

        agent = SimpleNamespace(aupdate_state=AsyncMock(side_effect=_capture))
        adapter = TextualUIAdapter(
            mount_message=AsyncMock(),
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config={"configurable": {"thread_id": "t-1"}},
            pending_text_by_namespace={},
            captured_input_tokens=0,
            captured_output_tokens=0,
            turn_stats=SessionStats(),
            start_time=0.0,
        )

        assert len(captured) == 1
        assert "_context_tokens" not in captured[0]

    async def test_includes_context_tokens_for_output_only_turn(self) -> None:
        """Output-only AI turns (no input usage) still persist a count."""
        captured: list[dict[str, Any]] = []

        async def _capture(_config: object, values: dict[str, Any]) -> None:  # noqa: RUF029
            captured.append(values)

        agent = SimpleNamespace(aupdate_state=AsyncMock(side_effect=_capture))
        adapter = TextualUIAdapter(
            mount_message=AsyncMock(),
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config={"configurable": {"thread_id": "t-1"}},
            pending_text_by_namespace={},
            captured_input_tokens=0,
            captured_output_tokens=500,
            turn_stats=SessionStats(),
            start_time=0.0,
        )

        assert len(captured) == 1
        assert captured[0]["_context_tokens"] == 500

    async def test_remote_agent_interrupt_write_carries_context_tokens(self) -> None:
        """Remote agents are not skipped on the interrupt-cleanup write.

        Locks in the deletion of the old `_persist_context_tokens` `RemoteAgent`
        short-circuit so a future refactor cannot silently re-introduce it.
        """
        from deepagents_code.client.remote_client import RemoteAgent

        captured: list[dict[str, Any]] = []

        async def _capture(_config: object, values: dict[str, Any]) -> None:  # noqa: RUF029
            captured.append(values)

        agent = MagicMock(spec=RemoteAgent)
        agent.aupdate_state = AsyncMock(side_effect=_capture)
        adapter = TextualUIAdapter(
            mount_message=AsyncMock(),
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config={"configurable": {"thread_id": "t-1"}},
            pending_text_by_namespace={},
            captured_input_tokens=1234,
            captured_output_tokens=88,
            turn_stats=SessionStats(),
            start_time=0.0,
        )

        assert isinstance(agent, RemoteAgent)
        assert len(captured) == 1
        assert captured[0]["_context_tokens"] == 1322

    async def test_partial_ai_message_write_does_not_carry_tokens(self) -> None:
        """Only the cancellation write carries `_context_tokens`."""
        captured: list[dict[str, Any]] = []

        async def _capture(_config: object, values: dict[str, Any]) -> None:  # noqa: RUF029
            captured.append(values)

        tool_widget = MagicMock()
        tool_widget._tool_name = "read_file"
        tool_widget._args = {"path": "notes.txt"}

        agent = SimpleNamespace(aupdate_state=AsyncMock(side_effect=_capture))
        adapter = TextualUIAdapter(
            mount_message=AsyncMock(),
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=AsyncMock(),
            set_active_message=MagicMock(),
        )
        adapter._current_tool_messages = {"call-1": tool_widget}

        await _handle_interrupt_cleanup(
            adapter=adapter,
            agent=agent,
            config={"configurable": {"thread_id": "t-1"}},
            pending_text_by_namespace={},
            captured_input_tokens=7777,
            captured_output_tokens=0,
            turn_stats=SessionStats(),
            start_time=0.0,
        )

        assert len(captured) == 2
        # First write is the interrupted AI message; should not be polluted.
        assert "_context_tokens" not in captured[0]
        # Second write is the cancellation HumanMessage; carries the token count.
        assert captured[1]["_context_tokens"] == 7777


class TestBuildStreamConfig:
    """Tests for `build_stream_config` metadata construction."""

    def setup_method(self) -> None:
        """Clear the git lookup caches between tests."""
        config_module._git_branch_cache.clear()
        config_module._repo_metadata_cache.clear()

    @pytest.fixture(autouse=True)
    def _hermetic_git(self) -> Generator[None, None, None]:
        """Stub git/repo lookups so tests don't read the host repo's real `.git`.

        These tests assert on the identity/turn keys, not on git attribution, so
        pinning the repo/commit lookups keeps them deterministic in exported
        checkouts (e.g. a CI tarball with no `.git`).
        """
        with (
            patch.object(config_module, "_get_repository_metadata", return_value=None),
            patch.object(config_module, "_get_git_commit_sha", return_value=None),
        ):
            yield

    def test_coding_agent_identity_block_present(self) -> None:
        """The coding-agent-v1 identity block is stamped on every config."""
        from deepagents_code._version import __version__

        metadata = build_stream_config("t-id", assistant_id=None)["metadata"]
        assert metadata["ls_agent_purpose"] == "coding"
        assert metadata["ls_integration"] == "deepagents-code"
        assert metadata["ls_agent_runtime"] == "Deep Agents Code"
        assert metadata["ls_trace_schema_version"] == "coding-agent-v1"
        assert metadata["ls_integration_version"] == __version__
        assert metadata["ls_agent_runtime_version"] == __version__

    def test_thread_id_set_as_top_level_metadata(self) -> None:
        """thread_id is mirrored to top-level metadata for contract grouping."""
        config = build_stream_config("t-group", assistant_id=None)
        assert config["metadata"]["thread_id"] == "t-group"
        assert config["configurable"]["thread_id"] == "t-group"

    def test_turn_markers_passed_through(self) -> None:
        """turn_id / turn_number reach metadata when provided."""
        metadata = build_stream_config(
            "t-turn", assistant_id=None, turn_id="turn-9", turn_number=4
        )["metadata"]
        assert metadata["turn_id"] == "turn-9"
        assert metadata["turn_number"] == 4

    def test_turn_markers_absent_when_unset(self) -> None:
        """turn_id / turn_number are omitted when not provided."""
        metadata = build_stream_config("t-noturn", assistant_id=None)["metadata"]
        assert "turn_id" not in metadata
        assert "turn_number" not in metadata

    def test_scope_restricted_keys_not_emitted(self) -> None:
        """approval_policy / ls_subagent_* are never stamped trace-wide."""
        metadata = build_stream_config(
            "t-scope", assistant_id="agent", turn_id="t", turn_number=1
        )["metadata"]
        assert "approval_policy" not in metadata
        assert "ls_subagent_id" not in metadata
        assert "ls_subagent_type" not in metadata

    def test_dcode_agent_fields_present(self) -> None:
        """Selected dcode agent metadata should be present."""
        config = build_stream_config("t-456", assistant_id="my-agent")
        assert "assistant_id" not in config["metadata"]
        assert config["metadata"]["dcode_agent_name"] == "my-agent"
        assert config["metadata"]["agent_name"] == "my-agent"
        assert "updated_at" in config["metadata"]
        assert "cwd" in config["metadata"]

    def test_updated_at_is_valid_iso_timestamp(self) -> None:
        """`updated_at` should be a valid timezone-aware ISO 8601 timestamp."""
        config = build_stream_config("t-456", assistant_id="my-agent")
        raw = config["metadata"]["updated_at"]
        assert isinstance(raw, str)
        parsed = datetime.fromisoformat(raw)
        assert parsed.tzinfo is not None

    def test_no_dcode_agent_fields_when_none(self) -> None:
        """Selected dcode agent fields should be absent when unset."""
        config = build_stream_config("t-789", assistant_id=None)
        metadata = config["metadata"]
        assert "assistant_id" not in metadata
        assert "dcode_agent_name" not in metadata
        assert "agent_name" not in metadata
        assert "updated_at" not in metadata
        assert "cwd" in metadata

    def test_no_dcode_agent_fields_when_empty_string(self) -> None:
        """Empty-string `assistant_id` should be treated as absent."""
        config = build_stream_config("t-000", assistant_id="")
        metadata = config["metadata"]
        assert "assistant_id" not in metadata
        assert "dcode_agent_name" not in metadata
        assert "agent_name" not in metadata
        assert "updated_at" not in metadata
        assert "cwd" in metadata

    def test_git_branch_included_when_available(self) -> None:
        """Git branch should be included in metadata when in a git repo."""
        with patch(
            "deepagents_code.config._get_git_branch",
            return_value="feature-branch",
        ):
            config = build_stream_config("t-git", assistant_id="agent")
        assert config["metadata"]["git_branch"] == "feature-branch"

    def test_git_branch_absent_when_not_in_repo(self) -> None:
        """Git branch should be absent when not in a git repo."""
        with patch(
            "deepagents_code.config._get_git_branch",
            return_value=None,
        ):
            config = build_stream_config("t-nogit", assistant_id="agent")
        assert "git_branch" not in config["metadata"]

    def test_configurable_thread_id(self) -> None:
        """`configurable.thread_id` should match the provided thread ID."""
        config = build_stream_config("t-abc", assistant_id=None)
        assert config["configurable"]["thread_id"] == "t-abc"

    def test_sandbox_type_included_when_set(self) -> None:
        """Sandbox type should appear in metadata when provided."""
        config = build_stream_config("t-sb", assistant_id=None, sandbox_type="daytona")
        assert config["metadata"]["sandbox_type"] == "daytona"

    def test_sandbox_type_absent_when_none(self) -> None:
        """Sandbox type should be absent from metadata when not provided."""
        config = build_stream_config("t-nosb", assistant_id=None)
        assert "sandbox_type" not in config["metadata"]

    def test_sandbox_type_none_string_excluded(self) -> None:
        """The argparse sentinel `"none"` should not leak into metadata."""
        config = build_stream_config("t-none", assistant_id=None, sandbox_type="none")
        assert "sandbox_type" not in config["metadata"]

    def test_no_model_keys_in_configurable(self) -> None:
        """Model/model_params should not be in configurable."""
        config = build_stream_config("t-no-model", assistant_id=None)
        assert "model" not in config["configurable"]
        assert "model_params" not in config["configurable"]

    def test_versions_contains_cli_version(self) -> None:
        """CLI version should always be present in metadata.lc_versions."""
        from deepagents_code._version import __version__

        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch("deepagents_code.config._get_deepagents_version", return_value=None),
        ):
            config = build_stream_config("t-ver", assistant_id=None)
        assert config["metadata"]["lc_versions"] == {"deepagents-code": __version__}

    def test_versions_marks_editable_cli_version(self) -> None:
        """Editable dcode installs should be visible in metadata.lc_versions."""
        from deepagents_code._version import __version__

        with (
            patch("deepagents_code.config._is_editable_install", return_value=True),
            patch("deepagents_code.config._get_deepagents_version", return_value=None),
        ):
            config = build_stream_config("t-editable", assistant_id=None)
        assert config["metadata"]["lc_versions"] == {
            "deepagents-code": f"{__version__} (editable)"
        }

    def test_dcode_client_deepagents_version_is_diagnostic_metadata(self) -> None:
        """Client-side SDK version should not be reported as graph instrumentation."""
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.extras_info.resolve_sdk_version",
                return_value=("1.2.3", "resolved"),
            ),
        ):
            config = build_stream_config("t-sdk", assistant_id=None)
        assert config["metadata"]["dcode_client_deepagents_version"] == "1.2.3"
        assert "deepagents" not in config["metadata"]["lc_versions"]

    def test_dcode_client_deepagents_version_absent_when_metadata_missing(
        self,
    ) -> None:
        """Missing SDK metadata should not prevent stream config construction."""
        with (
            patch("deepagents_code.config._is_editable_install", return_value=False),
            patch(
                "deepagents_code.extras_info.resolve_sdk_version",
                return_value=(None, "not_installed"),
            ),
        ):
            config = build_stream_config("t-missing-sdk", assistant_id=None)

        assert "dcode_client_deepagents_version" not in config["metadata"]

    def test_dcode_client_deepagents_version_does_not_import_sdk(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SDK version metadata lookup should not import the SDK package."""
        for module in list(sys.modules):
            if module == "deepagents" or module.startswith("deepagents."):
                monkeypatch.delitem(sys.modules, module, raising=False)

        with patch("deepagents_code.config._is_editable_install", return_value=False):
            build_stream_config("t-no-sdk-import", assistant_id=None)

        assert not any(
            module == "deepagents" or module.startswith("deepagents.")
            for module in sys.modules
        )

    def test_get_deepagents_version_maps_status_to_value(self) -> None:
        """Only a `resolved` status yields a version; other statuses map to None.

        The guard keys on `status`, not on the version string, so a non-resolved
        status must drop even a non-`None` version the resolver flagged as
        untrustworthy.
        """
        from deepagents_code.config import _get_deepagents_version

        with patch(
            "deepagents_code.extras_info.resolve_sdk_version",
            return_value=("1.2.3", "error"),
        ):
            assert _get_deepagents_version() is None

        with patch(
            "deepagents_code.extras_info.resolve_sdk_version",
            return_value=("1.2.3", "resolved"),
        ):
            assert _get_deepagents_version() == "1.2.3"

    def test_versions_editable_with_resolved_sdk_version(self) -> None:
        """Editable suffix and SDK diagnostic version are populated independently."""
        from deepagents_code._version import __version__

        with (
            patch("deepagents_code.config._is_editable_install", return_value=True),
            patch(
                "deepagents_code.extras_info.resolve_sdk_version",
                return_value=("1.2.3", "resolved"),
            ),
        ):
            config = build_stream_config("t-editable-sdk", assistant_id=None)
        assert config["metadata"]["lc_versions"] == {
            "deepagents-code": f"{__version__} (editable)"
        }
        assert config["metadata"]["dcode_client_deepagents_version"] == "1.2.3"

    def test_user_id_included_when_set(self) -> None:
        """DEEPAGENTS_CODE_USER_ID should appear in metadata when set."""
        with patch.dict("os.environ", {"DEEPAGENTS_CODE_USER_ID": "mason"}):
            config = build_stream_config("t-uid", assistant_id=None)
        assert config["metadata"]["user_id"] == "mason"

    def test_user_id_absent_when_unset(self) -> None:
        """user_id should be absent from metadata when env var is not set."""
        with patch.dict("os.environ", {"DEEPAGENTS_CODE_USER_ID": ""}):
            config = build_stream_config("t-nouid", assistant_id=None)
        assert "user_id" not in config["metadata"]

    def test_experimental_included_when_enabled(self) -> None:
        """Experimental runs should be identifiable in trace metadata."""
        with patch.dict("os.environ", {"DEEPAGENTS_CODE_EXPERIMENTAL": "true"}):
            config = build_stream_config("t-experimental", assistant_id=None)
        assert config["metadata"]["dcode_experimental"] is True

    def test_experimental_absent_when_disabled(self) -> None:
        """Default runs should not be labeled experimental."""
        with patch.dict("os.environ", {"DEEPAGENTS_CODE_EXPERIMENTAL": "false"}):
            config = build_stream_config("t-stable", assistant_id=None)
        assert "dcode_experimental" not in config["metadata"]

    def test_experimental_absent_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default path (env var unset) is not labeled experimental."""
        monkeypatch.delenv("DEEPAGENTS_CODE_EXPERIMENTAL", raising=False)
        config = build_stream_config("t-default", assistant_id=None)
        assert "dcode_experimental" not in config["metadata"]


class TestGetGitBranch:
    """Tests for `_get_git_branch` caching."""

    def setup_method(self) -> None:
        """Clear the git-branch cache between tests."""
        config_module._git_branch_cache.clear()

    def test_reuses_cached_branch_for_same_working_directory(self) -> None:
        """Repeated lookups in one repo should only resolve the branch once."""
        with (
            patch(
                "deepagents_code.config.Path.cwd",
                return_value=Path("/tmp/repo"),
            ),
            patch(
                "deepagents_code.config.resolve_git_branch",
                return_value="feature-branch",
            ) as mock_resolve,
        ):
            assert config_module._get_git_branch() == "feature-branch"
            assert config_module._get_git_branch() == "feature-branch"

        mock_resolve.assert_called_once_with("/tmp/repo")


class TestGetGitCommitSha:
    """Tests for `_get_git_commit_sha` freshness."""

    def test_resolves_commit_fresh_on_each_call(self) -> None:
        """HEAD moves mid-session, so the SHA must be re-resolved every call."""
        with (
            patch(
                "deepagents_code.config.Path.cwd",
                return_value=Path("/tmp/repo"),
            ),
            patch(
                "deepagents_code._git.resolve_git_commit_sha",
                side_effect=["sha-before", "sha-after"],
            ) as mock_resolve,
        ):
            assert config_module._get_git_commit_sha() == "sha-before"
            assert config_module._get_git_commit_sha() == "sha-after"

        assert mock_resolve.call_count == 2


class TestGetGitBranchOSError:
    """Tests for _get_git_branch when Path.cwd() raises OSError."""

    def setup_method(self) -> None:
        """Clear the git-branch cache between tests."""
        config_module._git_branch_cache.clear()

    def test_returns_none_on_cwd_oserror(self) -> None:
        """_get_git_branch should return None when cwd is inaccessible."""
        with patch(
            "deepagents_code.config.Path.cwd",
            side_effect=OSError("deleted"),
        ):
            assert config_module._get_git_branch() is None


class TestBuildStreamConfigOSError:
    """Tests for build_stream_config when Path.cwd() raises OSError."""

    def setup_method(self) -> None:
        """Clear the git lookup caches between tests."""
        config_module._git_branch_cache.clear()
        config_module._repo_metadata_cache.clear()

    def test_cwd_absent_on_oserror(self) -> None:
        """Cwd should be absent from metadata when Path.cwd() raises."""
        with patch(
            "deepagents_code.config.Path.cwd",
            side_effect=OSError("deleted"),
        ):
            config = build_stream_config("t-err", assistant_id="agent")
        assert "cwd" not in config["metadata"]


class TestIsSummarizationChunk:
    """Tests for `_is_summarization_chunk` detection."""

    def test_returns_true_for_summarization_source(self) -> None:
        """Should return `True` when `lc_source` is `'summarization'`."""
        metadata = {"lc_source": "summarization"}
        assert _is_summarization_chunk(metadata) is True

    def test_returns_false_for_none_metadata(self) -> None:
        """Should return `False` when `metadata` is `None`."""
        assert _is_summarization_chunk(None) is False
        assert _is_summarization_chunk({}) is False

    def test_returns_false_for_none_lc_source(self) -> None:
        """Should return `False` when `lc_source` is not `'summarization'`."""
        metadata_none = {"lc_source": None}
        assert _is_summarization_chunk(metadata_none) is False

        metadata_other = {"lc_source": "other"}
        assert _is_summarization_chunk(metadata_other) is False

        metadata_missing = {"other_key": "value"}
        assert _is_summarization_chunk(metadata_missing) is False

    def test_returns_false_for_unrelated_metadata(self) -> None:
        """Should return `False` when only unrelated keys are present."""
        assert _is_summarization_chunk({"langgraph_node": "model"}) is False
        assert _is_summarization_chunk({"langgraph_node": None}) is False


class TestFormatRubricEvent:
    """Tests for rubric custom-stream event formatting."""

    @pytest.fixture(autouse=True)
    def _pin_unicode_glyphs(self) -> Generator[None, None, None]:
        """Pin Unicode glyphs so literal assertions hold on any terminal.

        `_format_rubric_event` resolves glyphs via `get_glyphs()`, which depends
        on charset detection. Pinning keeps these assertions deterministic in
        CI; `test_ascii_mode_degrades_to_ascii_glyphs` covers the ASCII path.
        """
        with patch(
            "deepagents_code.tui.textual_adapter.get_glyphs",
            return_value=UNICODE_GLYPHS,
        ):
            yield

    def test_start_event_omits_iteration_by_default(self) -> None:
        """Start events should avoid noisy iteration numbers by default."""
        assert (
            _format_rubric_event(
                {"type": "rubric_evaluation_start", "iteration": 1},
            )
            == "⏳ Checking acceptance criteria…"
        )

    def test_start_event_mentions_explicit_iteration(self) -> None:
        """Explicit iteration display should surface the 1-based count."""
        assert (
            _format_rubric_event(
                {
                    "type": "rubric_evaluation_start",
                    "iteration": 1,
                    "show_iteration": True,
                },
            )
            == "⏳ Checking acceptance criteria (iteration 2)…"
        )

    def test_needs_revision_uses_work_status_copy(self) -> None:
        """An unmet rubric should describe the work, not call the rubric deficient."""
        assert (
            _format_rubric_event(
                {
                    "type": "rubric_evaluation_end",
                    "result": "needs_revision",
                    "explanation": "missing coverage",
                    "criteria": [
                        {"name": "tests pass", "passed": False, "gap": "not run"},
                        {"name": "docs", "passed": True},
                    ],
                },
            )
            == "↻ Acceptance criteria not yet satisfied"
        )

    def test_satisfied_event(self) -> None:
        """Satisfied events should render compact success text."""
        assert (
            _format_rubric_event(
                {"type": "rubric_evaluation_end", "result": "satisfied"},
            )
            == "✓ Acceptance criteria satisfied"
        )

    def test_start_event_without_int_iteration_omits_number(self) -> None:
        """A non-integer iteration should fall back to the unnumbered label."""
        assert (
            _format_rubric_event(
                {"type": "rubric_evaluation_start", "iteration": None},
            )
            == "⏳ Checking acceptance criteria…"
        )

    def test_max_iterations_reached_event(self) -> None:
        """The summary stays concise while details preserve goal recovery guidance."""
        event = {
            "type": "rubric_evaluation_end",
            "result": "max_iterations_reached",
            "explanation": "coverage is still missing",
            "criteria": [
                {
                    "name": "tests pass",
                    "passed": False,
                    "gap": "integration test failed",
                },
                {"name": "docs updated", "passed": True},
            ],
        }

        assert _format_rubric_event(event) == (
            "⚠ Acceptance criteria not yet satisfied (iteration limit reached)"
        )
        details = _format_rubric_details(event, goal_active=True)
        assert "coverage is still missing" in details
        assert "tests pass" in details
        assert "integration test failed" in details
        assert "The goal remains active" in details
        assert "`/goal <objective>`" in details
        assert "`/goal clear`" in details

    def test_invalid_rubric_and_grader_failure_have_distinct_copy(self) -> None:
        """Rubric validity and grader infrastructure failures must stay distinct."""
        assert (
            _format_rubric_event(
                {
                    "type": "rubric_evaluation_end",
                    "result": "failed",
                    "explanation": "contradictory criteria",
                },
            )
            == "⚠ Rubric is invalid or cannot be evaluated"
        )
        assert (
            _format_rubric_event(
                {"type": "rubric_evaluation_end", "result": "grader_error"},
            )
            == "⚠ Acceptance criteria check failed"
        )

    def test_unknown_terminal_result_renders_fallback(self) -> None:
        """An unrecognized terminal result must not be silently dropped."""
        assert (
            _format_rubric_event(
                {"type": "rubric_evaluation_end", "result": "something_new"},
            )
            == "⚠ Acceptance criteria check ended"
        )

    def test_complete_details_preserve_explanation_gaps_and_next_step(self) -> None:
        """Details should retain every structured field without repr truncation."""
        explanation = "First paragraph.\n\nSecond paragraph.\n" + "detail " * 1000
        details = _format_rubric_details(
            {
                "type": "rubric_evaluation_end",
                "result": "needs_revision",
                "explanation": explanation,
                "criteria": [
                    {
                        "name": "Exact copy remains intact",
                        "passed": False,
                        "gap": "Expected 'Not ready'; found 'Pending'.",
                    },
                    {"name": "Passing criterion", "passed": True},
                ],
            }
        )

        assert explanation.strip() in details
        assert "Exact copy remains intact" in details
        assert "Expected 'Not ready'; found 'Pending'." in details
        assert "Passing criterion" not in details
        assert "Address every unmet criterion, then retry the check." in details
        assert "{'name':" not in details
        assert "truncated" not in details

    def test_invalid_rubric_and_grader_failure_details_recommend_next_steps(
        self,
    ) -> None:
        """Each terminal failure class should provide the relevant recovery action."""
        invalid = _format_rubric_details(
            {
                "result": "failed",
                "explanation": "The criteria contradict each other.",
            }
        )
        grader_error = _format_rubric_details(
            {"result": "grader_error", "explanation": "Provider timeout."}
        )

        assert "Review or replace the rubric" in invalid
        assert "Retry the check, or choose a different grader model." in grader_error

    def test_no_detail_results_return_empty_string(self) -> None:
        """`None` and satisfied verdicts have no failure detail to expand."""
        assert _format_rubric_details({"result": None}) == ""
        assert _format_rubric_details({"result": "satisfied"}) == ""

    def test_failure_without_explanation_returns_next_step_only(self) -> None:
        """A failure with no explanation still yields an actionable next step."""
        details = _format_rubric_details({"result": "failed"})

        assert "Explanation" not in details
        assert "Unmet criteria" not in details
        assert details == (
            "Next step\nReview or replace the rubric before grading again."
        )

    def test_unknown_result_uses_generic_next_step(self) -> None:
        """An unrecognized terminal verdict falls back to a generic recovery step."""
        details = _format_rubric_details({"result": "something_new"})

        assert "Next step\nReview the grader details before continuing." in details

    def test_unnamed_failing_criterion_without_gap(self) -> None:
        """A failing criterion missing name/gap renders a bare default bullet."""
        details = _format_rubric_details(
            {
                "result": "needs_revision",
                "criteria": [{"passed": False}],
            }
        )

        assert "- Unnamed criterion" in details
        # No gap line follows a criterion with no gap.
        assert "- Unnamed criterion\n  " not in details

    def test_active_goal_max_iterations_details_offer_goal_commands(self) -> None:
        """An unfinished goal that hit the limit should point at goal recovery."""
        details = _format_rubric_details(
            {"result": "max_iterations_reached"},
            goal_active=True,
        )

        assert "The goal remains active" in details
        assert "`/goal clear`" in details

    def test_end_event_without_result_returns_none(self) -> None:
        """Partial end events should not render a spurious warning."""
        assert _format_rubric_event({"type": "rubric_evaluation_end"}) is None

    def test_unrelated_event_returns_none(self) -> None:
        """Only rubric events should render rubric messages."""
        assert _format_rubric_event({"type": "subagent_start"}) is None

    def test_ascii_mode_degrades_to_ascii_glyphs(self) -> None:
        """In ASCII mode the transcript glyphs must degrade, not stay Unicode."""
        with patch(
            "deepagents_code.tui.textual_adapter.get_glyphs",
            return_value=ASCII_GLYPHS,
        ):
            start = _format_rubric_event(
                {"type": "rubric_evaluation_start", "iteration": 0},
            )
            revision = _format_rubric_event(
                {
                    "type": "rubric_evaluation_end",
                    "result": "needs_revision",
                    "criteria": [
                        {"name": "tests pass", "passed": False, "gap": "not run"},
                    ],
                },
            )
            satisfied = _format_rubric_event(
                {"type": "rubric_evaluation_end", "result": "satisfied"},
            )
            failed = _format_rubric_event(
                {"type": "rubric_evaluation_end", "result": "failed"},
            )
        assert start == f"{ASCII_GLYPHS.hourglass} Checking acceptance criteria..."
        assert revision == (
            f"{ASCII_GLYPHS.retry} Acceptance criteria not yet satisfied"
        )
        assert satisfied == f"{ASCII_GLYPHS.checkmark} Acceptance criteria satisfied"
        assert failed == (
            f"{ASCII_GLYPHS.warning} Rubric is invalid or cannot be evaluated"
        )


class _FakeAgent:
    """Minimal async stream agent used for adapter execution tests."""

    def __init__(self, chunks: list[tuple]) -> None:
        self._chunks = chunks

    async def astream(self, *_: Any, **__: Any) -> AsyncIterator[tuple[Any, ...]]:
        """Yield preconfigured stream chunks."""
        for chunk in self._chunks:
            yield chunk


class _RaisingAgent:
    """Async stream agent that yields chunks then raises mid-stream.

    Models a provider/transport failure (or a cancellation) partway through a
    turn, exercising the non-clean-exit paths of `execute_task_textual` where the
    `else`-branch (clean end) code never runs.
    """

    def __init__(self, chunks: list[tuple], error: BaseException) -> None:
        self._chunks = chunks
        self._error = error

    async def astream(self, *_: Any, **__: Any) -> AsyncIterator[tuple[Any, ...]]:
        """Yield the preconfigured chunks, then raise the configured error."""
        for chunk in self._chunks:
            yield chunk
        raise self._error


class _SequencedAgent:
    """Agent test double that returns a different stream per call."""

    def __init__(self, streams_by_call: list[list[tuple[Any, ...]]]) -> None:
        self._streams_by_call = streams_by_call
        self.stream_inputs: list[dict | Command] = []
        self.contexts: list[Any] = []
        self.configs: list[Any] = []
        self.store_items: list[tuple[tuple[str, ...], str, dict[str, Any]]] = []

    async def aput_store_item(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> None:
        """Record store writes requested by `execute_task_textual`."""
        self.store_items.append((namespace, key, value))

    async def astream(
        self,
        stream_input: dict | Command,
        *_: Any,
        context: object = None,
        config: object = None,
        **__: Any,
    ) -> AsyncIterator[tuple[Any, ...]]:
        """Yield chunks for this invocation and record stream inputs/context.

        `execute_task_textual` mutates a single `context` dict in place across
        stream iterations (production reads the value at each call), so snapshot
        a copy here to capture the per-iteration state rather than aliasing the
        final mutation.
        """
        self.stream_inputs.append(stream_input)
        self.contexts.append(dict(context) if isinstance(context, dict) else context)
        self.configs.append(config)
        chunks = self._streams_by_call.pop(0) if self._streams_by_call else []
        for chunk in chunks:
            yield chunk


class _FailingApprovalStoreAgent(_SequencedAgent):
    """Agent test double whose approval-mode store writes fail."""

    async def aput_store_item(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> None:
        """Raise while preserving the production store-writer signature."""
        _ = (namespace, key, value)
        msg = "approval-mode store unavailable"
        raise RuntimeError(msg)


class TestExecuteTaskTextualTurnMarkers:
    """End-to-end: turn markers advance and reach the stream config metadata."""

    async def test_turn_markers_flow_into_stream_config_and_advance(self) -> None:
        """A real session state advances turn markers into each turn's config.

        Guards the full wiring (`advance_turn` -> `build_stream_config` ->
        `astream` config) that the per-piece unit tests don't exercise together:
        a dropped `advance_turn()` call or mis-passed turn tuple would still pass
        those, but not this.
        """
        from deepagents_code.app import TextualSessionState

        session_state = TextualSessionState(thread_id="thread-1", auto_approve=True)
        agent = _SequencedAgent([[], []])
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        # Stub git lookups so the captured metadata is deterministic.
        with (
            patch.object(config_module, "_get_repository_metadata", return_value=None),
            patch.object(config_module, "_get_git_commit_sha", return_value=None),
        ):
            await execute_task_textual(
                user_input="first",
                agent=agent,
                assistant_id="assistant",
                session_state=session_state,
                adapter=adapter,
            )
            await execute_task_textual(
                user_input="second",
                agent=agent,
                assistant_id="assistant",
                session_state=session_state,
                adapter=adapter,
            )

        first_meta = agent.configs[0]["metadata"]
        second_meta = agent.configs[1]["metadata"]
        assert first_meta["turn_number"] == 1
        assert second_meta["turn_number"] == 2
        assert first_meta["turn_id"]
        assert second_meta["turn_id"]
        assert first_meta["turn_id"] != second_meta["turn_id"]
        # The session state itself reflects the latest turn.
        assert session_state.turn_number == 2


class TestExecuteTaskTextualAutoApproveInput:
    """Auto-approve must ride on run context, never a first-turn `Command`."""

    async def test_pre_enabled_auto_approve_uses_plain_dict_and_context(self) -> None:
        """A fresh turn sends a plain dict input; auto-approve rides on context.

        A first-turn `Command(update=...)` is rebuilt with `goto=None` by the
        LangGraph API server's `map_cmd`, crashing `_control_branch` on a fresh
        thread. The flag must travel via run context instead.
        """
        agent = _SequencedAgent([[]])
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        await execute_task_textual(
            user_input="hi",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        stream_input = agent.stream_inputs[0]
        assert not isinstance(stream_input, Command)
        assert stream_input == {
            "messages": [{"role": "user", "content": "hi"}],
            "goal_criteria_request": None,
        }
        assert agent.contexts[0]["auto_approve"] is True
        assert agent.contexts[0]["thread_id"] == "thread-1"
        key = approval_mode_key("thread-1")
        assert agent.contexts[0]["approval_mode_key"] == key
        assert agent.store_items == [
            (APPROVAL_MODE_NAMESPACE, key, {"auto_approve": True})
        ]

    async def test_rubric_is_sent_as_graph_state(self) -> None:
        """Rubrics should travel beside messages, not inside user content."""
        agent = _SequencedAgent([[]])
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        await execute_task_textual(
            user_input="hi",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
            rubric="tests pass",
        )

        stream_input = agent.stream_inputs[0]
        assert not isinstance(stream_input, Command)
        assert stream_input == {
            "messages": [{"role": "user", "content": "hi"}],
            "rubric": "tests pass",
            "goal_criteria_request": None,
        }

    async def test_blocked_goal_retry_context_is_not_user_input(
        self,
        tmp_path: Path,
    ) -> None:
        """Retry context should not be parsed for file mentions or checkpointed."""
        secret = tmp_path / "secret.txt"
        secret.write_text("do not attach me")
        agent = _SequencedAgent([[]])
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        await execute_task_textual(
            user_input="continue now",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
            blocked_goal_retry_context=f"blocked on @{secret}",
        )

        stream_input = agent.stream_inputs[0]
        assert not isinstance(stream_input, Command)
        assert stream_input == {
            "messages": [{"role": "user", "content": "continue now"}],
            "goal_criteria_request": None,
        }
        assert (
            agent.contexts[0]["blocked_goal_retry_context"] == f"blocked on @{secret}"
        )

    async def test_stale_blocked_goal_retry_context_is_cleared(self) -> None:
        """A reused context must not leak a prior turn's retry context.

        `CLIContext` is reused across turns, so a turn with no blocked goal
        (`blocked_goal_retry_context=None`) must actively pop any stale value
        left by an earlier turn rather than silently carrying it forward.
        """
        agent = _SequencedAgent([[]])
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        # Simulate a context carried over from an earlier blocked-goal turn.
        stale_context: dict[str, Any] = {
            "blocked_goal_retry_context": "stale blocker from a prior turn"
        }

        await execute_task_textual(
            user_input="continue now",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
            context=cast("Any", stale_context),
            blocked_goal_retry_context=None,
        )

        assert "blocked_goal_retry_context" not in agent.contexts[0]

    async def test_live_approval_write_failure_fails_closed_context(self) -> None:
        """A failed live-mode write must not reuse a stale approval key."""
        agent = _FailingApprovalStoreAgent([[]])
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        session_state = SimpleNamespace(
            thread_id="thread-1",
            auto_approve=True,
            approval_mode_key="stale",
        )

        await execute_task_textual(
            user_input="hi",
            agent=agent,
            assistant_id="assistant",
            session_state=session_state,
            adapter=adapter,
        )

        stream_input = agent.stream_inputs[0]
        assert not isinstance(stream_input, Command)
        assert agent.contexts[0]["auto_approve"] is False
        assert "approval_mode_key" not in agent.contexts[0]
        assert agent.store_items == []
        # The stale key must be cleared so later turns don't reuse it.
        assert session_state.approval_mode_key is None

    @pytest.mark.parametrize("use_async_callback", [True, False])
    async def test_mid_turn_auto_approve_all_propagates_to_resume_context(
        self,
        use_async_callback: bool,
    ) -> None:
        """Choosing "auto-approve all" mid-turn flips the resuming stream's context.

        The PR's headline behavior: iteration 1 interrupts for approval, the
        user picks `auto_approve_all`, and the per-iteration context refresh
        re-reads `session_state.auto_approve` so iteration 2 (the resume)
        carries `auto_approve=True`. Guards against hoisting the refresh out of
        the stream loop (which would leave the first-iteration value frozen and
        keep interrupting the rest of the turn).

        Parametrized over an async and a sync `on_auto_approve_enabled` callback
        to cover the `Awaitable[None] | None` union the adapter awaits only when
        the result is non-`None`.
        """
        action_requests = [{"name": "execute", "args": {"command": "echo hi"}}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    (
                        (),
                        "messages",
                        (
                            _tool_call_message(
                                "execute", {"command": "echo hi"}, "tool-1"
                            ),
                            {},
                        ),
                    ),
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": action_requests,
                            "review_configs": [
                                {
                                    "action_name": "execute",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        }
                    ),
                ],
                [],
            ]
        )

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            future: asyncio.Future[object] = asyncio.Future()
            future.set_result({"type": "auto_approve_all"})
            return future

        callback_seen: list[bool] = []

        on_auto_approve_enabled: Callable[[], Awaitable[None] | None]
        if use_async_callback:

            async def _async_callback() -> None:
                await asyncio.sleep(0)
                callback_seen.append(True)

            on_auto_approve_enabled = _async_callback
        else:

            def _sync_callback() -> None:
                callback_seen.append(True)

            on_auto_approve_enabled = _sync_callback

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=request_approval,
            on_auto_approve_enabled=on_auto_approve_enabled,
        )
        session_state = SimpleNamespace(thread_id="thread-1", auto_approve=False)

        await execute_task_textual(
            user_input="hi",
            agent=agent,
            assistant_id="assistant",
            session_state=session_state,
            adapter=adapter,
        )

        # Two stream iterations: the initial turn and the resume after the
        # decision. The flag must flip between them, not stay frozen.
        assert len(agent.contexts) == 2
        assert agent.contexts[0]["auto_approve"] is False
        assert agent.contexts[1]["auto_approve"] is True
        assert agent.contexts[0]["thread_id"] == "thread-1"
        assert agent.contexts[1]["thread_id"] == "thread-1"
        key = approval_mode_key("thread-1")
        assert agent.contexts[0]["approval_mode_key"] == key
        assert agent.contexts[1]["approval_mode_key"] == key
        assert agent.store_items == [
            (APPROVAL_MODE_NAMESPACE, key, {"auto_approve": False}),
            (APPROVAL_MODE_NAMESPACE, key, {"auto_approve": True}),
        ]
        assert callback_seen == [True]
        assert session_state.auto_approve is True


def _ask_user_interrupt_chunk(payload: dict[str, Any]) -> tuple[Any, ...]:
    """Build an updates-stream chunk containing one ask_user interrupt."""
    interrupt = SimpleNamespace(id="interrupt-1", value=payload)
    return ((), "updates", {"__interrupt__": [interrupt]})


def _hitl_interrupt_chunk(payload: dict[str, Any]) -> tuple[Any, ...]:
    """Build an updates-stream chunk containing one HITL interrupt."""
    interrupt = SimpleNamespace(id="interrupt-1", value=payload)
    return ((), "updates", {"__interrupt__": [interrupt]})


def _tool_chunk(
    *,
    name: str | None,
    args: str,
    chunk_id: str | None,
    index: int = 0,
) -> tuple[Any, ...]:
    """Build a `messages`-stream chunk carrying one streamed tool-call fragment."""
    from langchain_core.messages import AIMessageChunk

    message = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": name,
                "args": args,
                "id": chunk_id,
                "index": index,
                "type": "tool_call_chunk",
            }
        ],
    )
    return ((), "messages", (message, {}))


def _usage_chunk(*, input_tokens: int, output_tokens: int) -> tuple[Any, ...]:
    """Build a `messages`-stream chunk carrying only `usage_metadata`."""
    from langchain_core.messages import AIMessageChunk

    message = AIMessageChunk(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return ((), "messages", (message, {}))


class TestExecuteTaskTextualUsageStats:
    """`execute_task_textual` forwards the active provider into usage stats.

    The per-model recording API is unit-tested directly elsewhere; this guards
    the call site actually reading `settings.model_provider` and threading it
    through `record_request`.
    """

    async def test_records_provider_from_settings(self) -> None:
        """A usage chunk records the configured provider on `turn_stats`."""

        async def mount_message(_: object) -> None:
            await asyncio.sleep(0)

        turn_stats = SessionStats()
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with patch("deepagents_code.config.settings") as mock_settings:
            mock_settings.model_name = "gpt-5.5"
            mock_settings.model_provider = "openai"
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent([_usage_chunk(input_tokens=100, output_tokens=50)]),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
                turn_stats=turn_stats,
            )

        assert turn_stats.per_model["openai", "gpt-5.5"].input_tokens == 100
        assert turn_stats.per_model["openai", "gpt-5.5"].output_tokens == 50


class TestExecuteTaskTextualToolCallStreaming:
    """Tests for incremental tool-call argument accumulation."""

    async def test_fragmented_args_mount_once_when_json_completes(self) -> None:
        """Args streamed across many fragments parse once the JSON is whole.

        The tool row mounts a single time with fully accumulated args, even
        though the JSON arrives split across several chunks.
        """
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        # Split a JSON object across fragments; only the last one closes it.
        chunks = [
            _tool_chunk(name="edit_file", args='{"path": ', chunk_id="t1"),
            _tool_chunk(name=None, args='"a.py", ', chunk_id=None),
            _tool_chunk(name=None, args='"content": "x"}', chunk_id=None),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        tool_msgs = [m for m in mounted if isinstance(m, ToolCallMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]._tool_name == "edit_file"
        assert tool_msgs[0]._args == {"path": "a.py", "content": "x"}

    async def test_incomplete_args_do_not_mount(self) -> None:
        """A tool row stays unmounted while its JSON args are still partial."""
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        # JSON never closes — the row must not mount with partial args.
        chunks = [
            _tool_chunk(name="edit_file", args='{"path": ', chunk_id="t1"),
            _tool_chunk(name=None, args='"a.py"', chunk_id=None),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert not [m for m in mounted if isinstance(m, ToolCallMessage)]

    async def test_scalar_args_mount_eagerly_when_complete(self) -> None:
        """Non-object JSON args parse as soon as the scalar is whole.

        Scalars never close with `}`/`]`, so the bracket heuristic that defers
        large objects never fires for them; they must still mount (wrapped as
        `{"value": ...}`) once the accumulated fragment is valid JSON.
        """
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        # A JSON string split so the first fragment is not yet valid JSON.
        chunks = [
            _tool_chunk(name="echo", args='"hel', chunk_id="t1"),
            _tool_chunk(name=None, args='lo"', chunk_id=None),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        tool_msgs = [m for m in mounted if isinstance(m, ToolCallMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]._args == {"value": "hello"}

    async def test_dict_args_resolve_without_reparsing(self) -> None:
        """A complete `tool_call` block mounts with its dict args verbatim."""
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        chunks = [
            (
                (),
                "messages",
                (_tool_call_message("read_file", {"path": "a.py"}, "t1"), {}),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        tool_msgs = [m for m in mounted if isinstance(m, ToolCallMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]._args == {"path": "a.py"}

    async def test_interleaved_fragments_accumulate_per_tool(self) -> None:
        """Fragments for two concurrent tool calls accumulate independently.

        Each tool call carries a distinct stream index, so interleaved argument
        fragments must not bleed across buffers.
        """
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        # Two tools (index 0 and 1) with interleaved argument fragments.
        chunks = [
            _tool_chunk(name="read_file", args='{"path": ', chunk_id="t0", index=0),
            _tool_chunk(name="grep", args='{"pattern": ', chunk_id="t1", index=1),
            _tool_chunk(name=None, args='"a.py"}', chunk_id=None, index=0),
            _tool_chunk(name=None, args='"x"}', chunk_id=None, index=1),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        tool_msgs = [m for m in mounted if isinstance(m, ToolCallMessage)]
        by_name = {m._tool_name: m._args for m in tool_msgs}
        assert by_name == {
            "read_file": {"path": "a.py"},
            "grep": {"pattern": "x"},
        }


class TestExecuteTaskTextualSummarizationFeedback:
    """Tests for summarization spinner and notification feedback."""

    async def test_spinner_transitions_for_summarization_stream(self) -> None:
        """Spinner should move Thinking -> Offloading -> Thinking."""
        statuses: list[str | None] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        async def mount_message(_widget: object) -> None:
            await asyncio.sleep(0)

        chunks = [
            (
                (),
                "messages",
                (AIMessage(content="summary chunk"), {"lc_source": "summarization"}),
            ),
            ((), "messages", (HumanMessage(content="regular chunk"), {})),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert statuses[0] == "Thinking"
        assert "Offloading" in statuses
        assert statuses[-1] == "Thinking"

    async def test_mounts_summarization_notification_on_regular_chunk(self) -> None:
        """Notification should render when regular chunks resume after summarization."""
        statuses: list[str | None] = []
        mounted_widgets: list[object] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted_widgets.append(widget)

        chunks = [
            (
                (),
                "messages",
                (AIMessage(content="summary chunk"), {"lc_source": "summarization"}),
            ),
            # Regular chunk from the actual model — signals summarization ended.
            ((), "messages", (HumanMessage(content="regular"), {})),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert any(
            isinstance(widget, SummarizationMessage) for widget in mounted_widgets
        )

    async def test_mounts_notification_when_stream_ends_mid_summarization(self) -> None:
        """Notification should still render if stream exhausts during summarization."""
        mounted_widgets: list[object] = []

        async def record_spinner(_status: str | None) -> None:
            await asyncio.sleep(0)

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted_widgets.append(widget)

        # Only summarization chunks, no regular chunks follow.
        chunks = [
            (
                (),
                "messages",
                (AIMessage(content="summary chunk"), {"lc_source": "summarization"}),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert any(
            isinstance(widget, SummarizationMessage) for widget in mounted_widgets
        )


def _tool_call_message(
    name: str, args: dict[str, Any], tool_id: str
) -> SimpleNamespace:
    """Build a message-like object with content_blocks containing one tool call."""
    return SimpleNamespace(
        content_blocks=[
            {"type": "tool_call", "name": name, "args": args, "id": tool_id}
        ]
    )


def _text_message(text: str) -> SimpleNamespace:
    """Build a message-like object with content_blocks containing one text block."""
    return SimpleNamespace(content_blocks=[{"type": "text", "text": text}])


class TestExecuteTaskTextualUserVisibleOutputStarted:
    """The callback fires once on the first output rendered for the user."""

    async def test_fires_once_on_first_streamed_text(self) -> None:
        """Streaming text triggers `on_user_visible_output_started` a single time."""
        user_visible_output_started = MagicMock()
        chunks = [
            ((), "messages", (_text_message("hello"), {})),
            ((), "messages", (_text_message(" world"), {})),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            on_user_visible_output_started=user_visible_output_started,
        )

        await execute_task_textual(
            user_input="hi",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        user_visible_output_started.assert_called_once_with()

    async def test_fires_once_across_text_then_tool_call(self) -> None:
        """Text followed by a tool call in one turn still fires exactly once."""
        user_visible_output_started = MagicMock()
        chunks = [
            ((), "messages", (_text_message("thinking"), {})),
            ((), "messages", (_tool_call_message("task", {"task": "a"}, "t-a"), {})),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            on_user_visible_output_started=user_visible_output_started,
        )

        await execute_task_textual(
            user_input="hi",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        user_visible_output_started.assert_called_once_with()

    async def test_fires_on_first_tool_call_without_text(self) -> None:
        """A turn that opens with a tool call still reports output started."""
        user_visible_output_started = MagicMock()
        chunks = [
            ((), "messages", (_tool_call_message("task", {"task": "a"}, "t-a"), {})),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            on_user_visible_output_started=user_visible_output_started,
        )

        await execute_task_textual(
            user_input="hi",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        user_visible_output_started.assert_called_once_with()

    async def test_fires_on_synthesized_ask_user_tool_call(self) -> None:
        """An updates-only `ask_user` row reports visible output after mounting."""
        user_visible_output_started = MagicMock()
        future: asyncio.Future[AskUserWidgetResult] = asyncio.Future()
        future.set_result({"type": "answered", "answers": ["Alice"]})

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return future

        questions: list[Question] = [{"question": "Name?", "type": "text"}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": questions,
                            "tool_call_id": "ask-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
            on_user_visible_output_started=user_visible_output_started,
        )

        await execute_task_textual(
            user_input="hi",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        user_visible_output_started.assert_called_once_with()

    async def test_not_fired_when_no_output_is_produced(self) -> None:
        """A turn that streams no text or tool call never reports output."""
        user_visible_output_started = MagicMock()

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            on_user_visible_output_started=user_visible_output_started,
        )

        await execute_task_textual(
            user_input="hi",
            agent=_FakeAgent([]),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        user_visible_output_started.assert_not_called()

    async def test_not_fired_for_subagent_output(self) -> None:
        """Text and tool calls hidden in a subagent namespace do not count."""
        user_visible_output_started = MagicMock()
        chunks = [
            (("subagent",), "messages", (_text_message("hidden"), {})),
            (
                ("subagent",),
                "messages",
                (_tool_call_message("read_file", {"path": "x"}, "t-a"), {}),
            ),
        ]
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            on_user_visible_output_started=user_visible_output_started,
        )

        await execute_task_textual(
            user_input="hi",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        user_visible_output_started.assert_not_called()

    async def test_not_fired_for_hidden_summarization_output(self) -> None:
        """Hidden main-namespace summarization text does not count."""
        user_visible_output_started = MagicMock()
        chunks = [
            (
                (),
                "messages",
                (_text_message("hidden summary"), {"lc_source": "summarization"}),
            ),
        ]
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            on_user_visible_output_started=user_visible_output_started,
        )

        await execute_task_textual(
            user_input="hi",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        user_visible_output_started.assert_not_called()

    async def test_not_fired_when_tool_widget_does_not_mount(self) -> None:
        """A tool call that never reaches the transcript does not count."""
        user_visible_output_started = MagicMock()

        async def fail_mount(_widget: object) -> None:
            await asyncio.sleep(0)
            msg = "mount failed"
            raise RuntimeError(msg)

        adapter = TextualUIAdapter(
            mount_message=fail_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            on_user_visible_output_started=user_visible_output_started,
        )

        await execute_task_textual(
            user_input="hi",
            agent=_FakeAgent(
                [
                    (
                        (),
                        "messages",
                        (_tool_call_message("read_file", {"path": "x"}, "t-a"), {}),
                    )
                ]
            ),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        user_visible_output_started.assert_not_called()


class TestExecuteTaskTextualParallelToolSpinner:
    """Regression tests for #1796: premature spinner with parallel tools."""

    async def test_spinner_stays_up_across_parallel_tools(self) -> None:
        """With two parallel tools, the spinner stays "Thinking" and is never hidden."""
        statuses: list[str | None] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        async def mount_message(_widget: object) -> None:
            await asyncio.sleep(0)

        chunks = [
            (
                (),
                "messages",
                (
                    _tool_call_message("task", {"task": "a"}, "tool-a"),
                    {},
                ),
            ),
            (
                (),
                "messages",
                (
                    _tool_call_message("task", {"task": "b"}, "tool-b"),
                    {},
                ),
            ),
            (
                (),
                "messages",
                (
                    ToolMessage(content="result a", tool_call_id="tool-a"),
                    {},
                ),
            ),
            (
                (),
                "messages",
                (
                    ToolMessage(content="result b", tool_call_id="tool-b"),
                    {},
                ),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        assert statuses[0] == "Thinking"
        assert statuses[-1] == "Thinking"
        # Stable turn-level indicator: never hidden while parallel tools run.
        assert None not in statuses

    async def test_on_tool_complete_fires_per_tool_message(self) -> None:
        """`on_tool_complete` should fire once per `ToolMessage`, even in parallel."""
        tool_complete = MagicMock()
        tc = _tool_call_message
        chunks = [
            ((), "messages", (tc("task", {"task": "a"}, "tool-a"), {})),
            ((), "messages", (tc("task", {"task": "b"}, "tool-b"), {})),
            ((), "messages", (ToolMessage(content="a", tool_call_id="tool-a"), {})),
            ((), "messages", (ToolMessage(content="b", tool_call_id="tool-b"), {})),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            on_tool_complete=tool_complete,
        )

        await execute_task_textual(
            user_input="hi",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        assert tool_complete.call_count == 2

    async def test_on_tool_complete_exception_is_swallowed(self) -> None:
        """A raising `on_tool_complete` must not break agent streaming."""
        tc = _tool_call_message
        chunks = [
            ((), "messages", (tc("task", {"task": "a"}, "tool-a"), {})),
            ((), "messages", (ToolMessage(content="a", tool_call_id="tool-a"), {})),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            on_tool_complete=MagicMock(side_effect=RuntimeError("boom")),
        )

        await execute_task_textual(
            user_input="hi",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

    async def test_spinner_shown_after_single_tool_completes(self) -> None:
        """Spinner should show Thinking after the only tool completes."""
        statuses: list[str | None] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        chunks = [
            (
                (),
                "messages",
                (
                    _tool_call_message("ls", {"path": "."}, "tool-1"),
                    {},
                ),
            ),
            (
                (),
                "messages",
                (
                    ToolMessage(content="file1.py", tool_call_id="tool-1"),
                    {},
                ),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        await execute_task_textual(
            user_input="list files",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        assert statuses[-1] == "Thinking"

    async def test_edit_file_tool_keeps_thinking_spinner_while_pending(self) -> None:
        """`edit_file` should not leave a visual gap before approval/execution."""
        statuses: list[str | None] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        chunks = [
            (
                (),
                "messages",
                (
                    _tool_call_message(
                        "edit_file",
                        {
                            "file_path": "example.py",
                            "old_string": "old",
                            "new_string": "new",
                        },
                        "tool-1",
                    ),
                    {},
                ),
            ),
            (
                (),
                "messages",
                (
                    ToolMessage(content="edited", tool_call_id="tool-1"),
                    {},
                ),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        await execute_task_textual(
            user_input="edit the file",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        assert statuses[:2] == ["Thinking", "Thinking"]
        assert None not in statuses

    async def test_auto_executed_tool_shows_running_at_mount(self) -> None:
        """Auto-executed tools (no approval) spin immediately when mounted.

        Regression guard: read-only tools such as `grep`/`glob` previously sat
        visually idle from mount until their result arrived. The stream here
        ends right after the tool call (no result), so the row is observed in
        its mount-time state.
        """
        chunks = [
            (
                (),
                "messages",
                (_tool_call_message("grep", {"pattern": "foo"}, "tool-1"), {}),
            ),
        ]

        # Capture the widget at mount: the clean-completion orphan close clears
        # `_current_tool_messages` (hooks-only, so the widget keeps its state),
        # so read the mounted row directly rather than the post-turn tracking.
        mounted_tools: list[ToolCallMessage] = []

        async def capture_mount(widget: object) -> None:
            await asyncio.sleep(0)
            if isinstance(widget, ToolCallMessage):
                mounted_tools.append(widget)

        adapter = TextualUIAdapter(
            mount_message=capture_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        await execute_task_textual(
            user_input="search",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        assert len(mounted_tools) == 1
        assert mounted_tools[0]._status == "running"

    async def test_edit_file_marks_running_at_mount(self) -> None:
        """All tool rows are marked running at mount, including `edit_file`.

        The row is hidden inside its collapsed group, so "running" drives the
        group's live progress state rather than showing a duplicate spinner
        alongside the global "Thinking" indicator.
        """
        chunks = [
            (
                (),
                "messages",
                (
                    _tool_call_message(
                        "edit_file",
                        {
                            "file_path": "example.py",
                            "old_string": "old",
                            "new_string": "new",
                        },
                        "tool-1",
                    ),
                    {},
                ),
            ),
        ]

        # Capture the widget at mount: the clean-completion orphan close clears
        # `_current_tool_messages` (hooks-only, so the widget keeps its state),
        # so read the mounted row directly rather than the post-turn tracking.
        mounted_tools: list[ToolCallMessage] = []

        async def capture_mount(widget: object) -> None:
            await asyncio.sleep(0)
            if isinstance(widget, ToolCallMessage):
                mounted_tools.append(widget)

        adapter = TextualUIAdapter(
            mount_message=capture_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        await execute_task_textual(
            user_input="edit",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        assert len(mounted_tools) == 1
        assert mounted_tools[0]._status == "running"

    async def test_spinner_with_three_parallel_tools_out_of_order(self) -> None:
        """Three parallel tools complete out of order; spinner stays up throughout."""
        statuses: list[str | None] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        tc = _tool_call_message
        chunks = [
            ((), "messages", (tc("task", {"task": "a"}, "tool-a"), {})),
            ((), "messages", (tc("task", {"task": "b"}, "tool-b"), {})),
            ((), "messages", (tc("task", {"task": "c"}, "tool-c"), {})),
            # Complete out of dispatch order: B, A, C
            (
                (),
                "messages",
                (
                    ToolMessage(
                        content="result b",
                        tool_call_id="tool-b",
                    ),
                    {},
                ),
            ),
            (
                (),
                "messages",
                (
                    ToolMessage(
                        content="result a",
                        tool_call_id="tool-a",
                    ),
                    {},
                ),
            ),
            (
                (),
                "messages",
                (
                    ToolMessage(
                        content="result c",
                        tool_call_id="tool-c",
                    ),
                    {},
                ),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        assert statuses[-1] == "Thinking"
        # Stable turn-level indicator: never hidden while parallel tools run.
        assert None not in statuses

    async def test_spinner_recovers_with_untracked_tool_id(self) -> None:
        """Spinner still shows Thinking with an untracked tool_call_id."""
        statuses: list[str | None] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        tc = _tool_call_message
        chunks = [
            ((), "messages", (tc("task", {"task": "a"}, "tool-a"), {})),
            # Result with a tool_call_id that was never dispatched
            (
                (),
                "messages",
                (
                    ToolMessage(
                        content="result a",
                        tool_call_id="tool-a",
                    ),
                    {},
                ),
            ),
            (
                (),
                "messages",
                (
                    ToolMessage(
                        content="unknown",
                        tool_call_id="tool-unknown",
                    ),
                    {},
                ),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
            adapter=adapter,
        )

        # After the tracked tool completes, dict is empty so spinner should show.
        # The untracked ToolMessage should not break spinner recovery.
        thinking_calls = [i for i, s in enumerate(statuses) if s == "Thinking"]
        assert len(thinking_calls) >= 2, (
            f"Expected at least 2 Thinking calls; got {len(thinking_calls)}: {statuses}"
        )


class TestExecuteTaskTextualTextThenToolSpinner:
    """Regression tests: spinner must stay visible between text and tool call.

    When the assistant streams explanatory text and then emits a tool call,
    the model often pauses between finishing the text and producing the tool
    call. The spinner should remain visible during that pause rather than
    disappearing as soon as the first text chunk arrives.
    """

    async def test_spinner_not_hidden_when_text_chunk_arrives(self) -> None:
        """Streaming a text block must not hide the Thinking spinner."""
        statuses: list[str | None] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        chunks = [
            ((), "messages", (_text_message("Now I'll call a tool..."), {})),
            ((), "messages", (_tool_call_message("ls", {"path": "."}, "tool-1"), {})),
            ((), "messages", (ToolMessage(content="ok", tool_call_id="tool-1"), {})),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        # Patch AssistantMessage so it doesn't require a real Textual DOM.
        fake_msg = AsyncMock()
        fake_msg.id = "asst-test"
        with patch(
            "deepagents_code.tui.textual_adapter.AssistantMessage",
            return_value=fake_msg,
        ):
            await execute_task_textual(
                user_input="hi",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
                adapter=adapter,
            )

        # The spinner is a stable turn-level indicator: it shows "Thinking"
        # before the stream, stays up while text streams and while the tool
        # runs (the tool's own progress shows in its collapsed group row), and
        # is never hidden mid-turn — so it no longer flickers off for each tool.
        assert statuses[0] == "Thinking"
        assert statuses[-1] == "Thinking"
        assert None not in statuses, f"Spinner was hidden mid-turn: {statuses}"
        assert all(s == "Thinking" for s in statuses)

    async def test_spinner_reanchors_for_text_after_tool_cycle(self) -> None:
        """Text -> tool_call -> tool_result -> text must re-anchor the spinner.

        After a tool cycle completes, the tool_call handler pops the previous
        AssistantMessage from `assistant_message_by_namespace`, so the next
        text chunk mounts a fresh widget. The re-anchor call must fire for that
        second text burst so the spinner stays visible between it and any
        follow-up tool call.
        """
        statuses: list[str | None] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        chunks = [
            ((), "messages", (_text_message("First, I'll inspect..."), {})),
            ((), "messages", (_tool_call_message("ls", {"path": "."}, "tool-1"), {})),
            ((), "messages", (ToolMessage(content="ok", tool_call_id="tool-1"), {})),
            ((), "messages", (_text_message("Now the second step..."), {})),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        fake_msg = AsyncMock()
        fake_msg.id = "asst-test"
        with patch(
            "deepagents_code.tui.textual_adapter.AssistantMessage",
            return_value=fake_msg,
        ):
            await execute_task_textual(
                user_input="hi",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
                adapter=adapter,
            )

        # Expected Thinking calls:
        #   1. Before astream
        #   2. After first AssistantMessage mount (re-anchor)
        #   3. After tool result
        #   4. After second AssistantMessage mount (re-anchor again)
        thinking_count = sum(1 for s in statuses if s == "Thinking")
        assert thinking_count >= 4, (
            f"Expected at least 4 Thinking calls including re-anchors after "
            f"each text mount; got {thinking_count}: {statuses}"
        )

    async def test_spinner_stays_up_when_text_arrives_mid_tool(self) -> None:
        """A text chunk arriving while a tool is in flight keeps the spinner up.

        Contrived sequence: a tool call mounts (populating
        `_current_tool_messages`), then a text chunk arrives before the tool
        result. The spinner stays "Thinking" throughout and is never hidden —
        the text re-anchor stays gated on `not _current_tool_messages`.
        """
        statuses: list[str | None] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        chunks = [
            ((), "messages", (_tool_call_message("ls", {"path": "."}, "tool-1"), {})),
            ((), "messages", (_text_message("Meanwhile..."), {})),
            ((), "messages", (ToolMessage(content="ok", tool_call_id="tool-1"), {})),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        fake_msg = AsyncMock()
        fake_msg.id = "asst-test"
        with patch(
            "deepagents_code.tui.textual_adapter.AssistantMessage",
            return_value=fake_msg,
        ):
            await execute_task_textual(
                user_input="hi",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
                adapter=adapter,
            )

        # The spinner stays "Thinking" and is never hidden while the tool is in
        # flight; the text re-anchor stays gated on no pending tools.
        assert statuses[0] == "Thinking"
        assert statuses[-1] == "Thinking"
        assert None not in statuses


class TestExecuteTaskTextualRubricRevisionStreaming:
    """Regression coverage for rubric-driven assistant reattempts."""

    async def test_rubric_feedback_starts_new_assistant_message(self) -> None:
        """A rubric-injected human turn must separate assistant attempts."""
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        class FakeAssistantMessage:
            def __init__(self, content: str = "", **kwargs: str | None) -> None:
                self.id = kwargs.get("id")
                self._content = content

            async def append_content(self, text: str) -> None:
                self._content += text

            async def stop_stream(self) -> None:
                pass

            async def write_initial_content(self) -> None:
                pass

        chunks = [
            ((), "messages", (_text_message("Hi Mason."), {})),
            (
                (),
                "messages",
                (
                    HumanMessage(
                        content="Please revise.",
                        name="rubric_grader",
                        additional_kwargs={"lc_source": "rubric_grader"},
                    ),
                    {},
                ),
            ),
            (
                (),
                "custom",
                {"type": "rubric_evaluation_start", "iteration": 0},
            ),
            (
                (),
                "custom",
                {
                    "type": "rubric_evaluation_end",
                    "result": "needs_revision",
                    "explanation": "say yellow",
                    "criteria": [],
                },
            ),
            ((), "messages", (_text_message("yellow yellow"), {})),
        ]
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.AssistantMessage",
                side_effect=FakeAssistantMessage,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.get_glyphs",
                return_value=UNICODE_GLYPHS,
            ),
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
                adapter=adapter,
            )

        assistant_messages = [
            widget for widget in mounted if isinstance(widget, FakeAssistantMessage)
        ]
        assert [msg._content for msg in assistant_messages] == [
            "Hi Mason.",
            "yellow yellow",
        ]

        app_messages = [widget for widget in mounted if isinstance(widget, AppMessage)]
        assert [str(widget._content) for widget in app_messages] == [
            (
                f"{UNICODE_GLYPHS.hourglass} Checking acceptance criteria"
                f"{UNICODE_GLYPHS.ellipsis}"
            )
        ]
        rubric_messages = [
            widget for widget in mounted if isinstance(widget, RubricResultMessage)
        ]
        assert len(rubric_messages) == 1
        assert rubric_messages[0]._summary == (
            f"{UNICODE_GLYPHS.retry} Acceptance criteria not yet satisfied"
        )
        assert rubric_messages[0]._details == (
            "Explanation\nsay yellow\n\n"
            "Next step\nAddress every unmet criterion, then retry the check."
        )


class TestExecuteTaskTextualHITLShellSuppression:
    """Tests for shell-tool widget suppression during HITL approval."""

    async def _run_with_decision(
        self,
        *,
        tool_call_name: str,
        tool_call_id: str,
        approval_decision: dict[str, Any],
        extra_tool_calls: list[tuple[str, dict[str, Any], str]] | None = None,
    ) -> tuple[
        TextualUIAdapter,
        list[object],
        dict[str, tuple[bool, bool, str]],
    ]:
        """Drive a HITL flow and snapshot widget visibility during the await.

        Returns the adapter, the mounted widgets, and a mapping of
        `tool_call_id -> (display, _awaiting_approval, _status)` captured while
        the approval future is pending. The status entry locks in the pause
        behavior: tools start their spinner at mount but are reverted to
        `pending` while blocked on the approval decision.
        """
        mounted: list[object] = []
        snapshots: dict[str, tuple[bool, bool, str]] = {}

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        future: asyncio.Future[object] = asyncio.Future()

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            for tid, tool_msg in adapter._current_tool_messages.items():
                snapshots[tid] = (
                    bool(tool_msg.display),
                    tool_msg._awaiting_approval,
                    tool_msg._status,
                )
            future.set_result(approval_decision)
            return future

        message_chunks: list[tuple[Any, ...]] = [
            (
                (),
                "messages",
                (
                    _tool_call_message(
                        tool_call_name, {"command": "echo hi"}, tool_call_id
                    ),
                    {},
                ),
            )
        ]
        for name, args, tid in extra_tool_calls or []:
            message_chunks.append(
                ((), "messages", (_tool_call_message(name, args, tid), {}))
            )

        action_requests = [{"name": tool_call_name, "args": {"command": "echo hi"}}]
        for name, args, _tid in extra_tool_calls or []:
            action_requests.append({"name": name, "args": args})

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    *message_chunks,
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": action_requests,
                            "review_configs": [
                                {
                                    "action_name": req["name"],
                                    "allowed_decisions": ["approve", "reject"],
                                }
                                for req in action_requests
                            ],
                        }
                    ),
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )
        return adapter, mounted, snapshots

    async def test_shell_tool_widget_suppressed_during_approval(self) -> None:
        """`execute` widget should be hidden during the await and restored after."""
        _adapter, mounted, snapshots = await self._run_with_decision(
            tool_call_name="execute",
            tool_call_id="tool-shell",
            approval_decision={"type": "approve"},
        )
        tool_rows = [w for w in mounted if isinstance(w, ToolCallMessage)]
        assert len(tool_rows) == 1
        # While the future was pending, the widget was hidden and its spinner
        # paused (reverted from the mount-time "running" to "pending").
        assert snapshots["tool-shell"] == (False, True, "pending")
        # After the finally block, it was restored and the spinner resumed
        # (the resumed stream is empty, so the row never reaches a result).
        assert tool_rows[0].display is True
        assert tool_rows[0]._awaiting_approval is False
        assert tool_rows[0]._status == "running"

    async def test_non_shell_tool_widget_not_suppressed(self) -> None:
        """`read_file` widget should stay visible — only shell tools are hidden."""
        _adapter, mounted, snapshots = await self._run_with_decision(
            tool_call_name="read_file",
            tool_call_id="tool-read",
            approval_decision={"type": "approve"},
        )
        tool_rows = [w for w in mounted if isinstance(w, ToolCallMessage)]
        assert len(tool_rows) == 1
        # Visible the whole time, never marked as awaiting approval, but the
        # spinner is paused to "pending" while the decision is outstanding.
        assert snapshots["tool-read"] == (True, False, "pending")
        assert tool_rows[0].display is True
        assert tool_rows[0]._awaiting_approval is False
        # Resumed to "running" after approval (resumed stream yields no result).
        assert tool_rows[0]._status == "running"

    async def test_batch_approval_keeps_all_widgets_visible(self) -> None:
        """Batched approvals (>1 request) must not hide any tool widget.

        The approval dialog only renders a per-tool command preview for
        single-tool approvals. For batches it shows just a count header,
        so suppressing the streamed rows would leave the user with no
        preview of what's being approved.
        """
        _adapter, _mounted, snapshots = await self._run_with_decision(
            tool_call_name="execute",
            tool_call_id="tool-shell",
            approval_decision={"type": "approve"},
            extra_tool_calls=[("read_file", {"path": "notes.txt"}, "tool-read")],
        )
        assert snapshots["tool-shell"] == (True, False, "pending")
        assert snapshots["tool-read"] == (True, False, "pending")

    async def test_batch_of_shell_tools_keeps_all_widgets_visible(self) -> None:
        """Multiple parallel `execute` calls: all rows stay visible.

        Regression guard: the batch approval dialog does not render
        per-tool commands, so hiding every `execute` row left users with
        only a generic "N Tool Calls Require Approval" header.
        """
        _adapter, _mounted, snapshots = await self._run_with_decision(
            tool_call_name="execute",
            tool_call_id="tool-shell-1",
            approval_decision={"type": "approve"},
            extra_tool_calls=[
                ("execute", {"command": "echo bye"}, "tool-shell-2"),
            ],
        )
        assert snapshots["tool-shell-1"] == (True, False, "pending")
        assert snapshots["tool-shell-2"] == (True, False, "pending")

    async def test_shell_widget_restored_when_approval_raises(self) -> None:
        """`finally` must restore the widget even if approval raises."""
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            msg = "boom"
            raise RuntimeError(msg)

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    (
                        (),
                        "messages",
                        (
                            _tool_call_message(
                                "execute", {"command": "echo hi"}, "tool-shell"
                            ),
                            {},
                        ),
                    ),
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": [
                                {"name": "execute", "args": {"command": "echo hi"}}
                            ],
                            "review_configs": [
                                {
                                    "action_name": "execute",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        }
                    ),
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        with pytest.raises(RuntimeError, match="boom"):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        tool_rows = [w for w in mounted if isinstance(w, ToolCallMessage)]
        assert len(tool_rows) == 1
        assert tool_rows[0].display is True
        assert tool_rows[0]._awaiting_approval is False


class TestInterruptOwnedToolRows:
    """`_interrupt_owned_tool_rows` scopes pause/resume to the right rows."""

    def test_matches_row_by_name_and_args(self) -> None:
        """An action request owns the tracked row with the same name and args."""
        execute_row = ToolCallMessage("execute", {"command": "echo hi"})
        task_row = ToolCallMessage("task", {"description": "research"})
        current = {"exec-1": execute_row, "task-1": task_row}

        owned = _interrupt_owned_tool_rows(
            [{"name": "execute", "args": {"command": "echo hi"}}], current
        )

        assert owned == [execute_row]

    def test_nested_child_request_owns_no_outer_task_row(self) -> None:
        """A subagent child tool (untracked) matches no outer `task` row.

        The child `fetch_url`/`execute` calls that interrupt live inside the
        subagent and are never tracked in `_current_tool_messages`; only the
        outer `task` rows are. So the child's action request must own nothing,
        leaving both task timers untouched.
        """
        task_one = ToolCallMessage("task", {"description": "a"})
        task_two = ToolCallMessage("task", {"description": "b"})
        current = {"task-1": task_one, "task-2": task_two}

        owned = _interrupt_owned_tool_rows(
            [{"name": "fetch_url", "args": {"url": "http://example.com"}}], current
        )

        assert owned == []

    def test_duplicate_calls_map_to_distinct_rows(self) -> None:
        """Two identical calls claim two distinct rows, not the same one twice."""
        first = ToolCallMessage("execute", {"command": "echo hi"})
        second = ToolCallMessage("execute", {"command": "echo hi"})
        current = {"exec-1": first, "exec-2": second}

        owned = _interrupt_owned_tool_rows(
            [
                {"name": "execute", "args": {"command": "echo hi"}},
                {"name": "execute", "args": {"command": "echo hi"}},
            ],
            current,
        )

        assert len(owned) == 2
        assert {id(row) for row in owned} == {id(first), id(second)}

    def test_mismatched_args_are_not_owned(self) -> None:
        """A same-named call with different args does not own the row."""
        execute_row = ToolCallMessage("execute", {"command": "echo hi"})
        current = {"exec-1": execute_row}

        owned = _interrupt_owned_tool_rows(
            [{"name": "execute", "args": {"command": "echo bye"}}], current
        )

        assert owned == []


class TestExecuteTaskTextualTaskTimerAcrossInterrupts:
    """A running `task` timer stays monotonic across nested subagent HITL.

    Regression guard for the bug where any interrupt paused *every* tracked
    tool row and each approval reset *every* row's start time — so an unrelated
    nested child approval restarted the outer `task` elapsed timer from zero.
    """

    @staticmethod
    def _task_chunk(tool_id: str, description: str) -> tuple[Any, ...]:
        """A main-agent `task` tool call that mounts and starts running."""
        return (
            (),
            "messages",
            (
                _tool_call_message(
                    "task",
                    {"description": description, "subagent_type": "general-purpose"},
                    tool_id,
                ),
                {},
            ),
        )

    @staticmethod
    def _child_interrupt_chunk(
        namespace: tuple[str, ...], tool_name: str, args: dict[str, Any]
    ) -> tuple[Any, ...]:
        """A HITL interrupt raised by a nested subagent's child tool call."""
        interrupt = SimpleNamespace(
            id=f"int-{tool_name}",
            value={
                "action_requests": [{"name": tool_name, "args": args}],
                "review_configs": [
                    {
                        "action_name": tool_name,
                        "allowed_decisions": ["approve", "reject"],
                    }
                ],
            },
        )
        return (namespace, "updates", {"__interrupt__": [interrupt]})

    async def test_outer_task_keeps_running_across_child_interrupt(self) -> None:
        """The outer `task` row is not paused when its child interrupts.

        The child `fetch_url` approval is unrelated to the `task` row, so at
        approval time (after the pause step) the task must still be `running`
        with an intact `_start_time`; before the fix the task was paused to
        `pending` with `_start_time` cleared.
        """
        snapshot: dict[str, Any] = {}

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            task_row = adapter._current_tool_messages["task-1"]
            snapshot["status"] = task_row._status
            snapshot["start_time"] = task_row._start_time
            future: asyncio.Future[object] = asyncio.Future()
            future.set_result({"type": "approve"})
            return future

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    self._task_chunk("task-1", "research the repo"),
                    self._child_interrupt_chunk(
                        ("task:task-1:0",),
                        "fetch_url",
                        {"url": "http://example.com"},
                    ),
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert snapshot["status"] == "running"
        assert snapshot["start_time"] is not None

    async def test_only_interrupting_sibling_task_is_unaffected(self) -> None:
        """Two concurrent tasks: a child interrupt leaves *both* timers running.

        The interrupt belongs to `task-1`'s subagent, but its child tool call
        is untracked, so neither the interrupting task nor the quiet sibling
        `task-2` may be paused.
        """
        snapshot: dict[str, Any] = {}

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            snapshot["task-1"] = (
                adapter._current_tool_messages["task-1"]._status,
                adapter._current_tool_messages["task-1"]._start_time,
            )
            snapshot["task-2"] = (
                adapter._current_tool_messages["task-2"]._status,
                adapter._current_tool_messages["task-2"]._start_time,
            )
            future: asyncio.Future[object] = asyncio.Future()
            future.set_result({"type": "approve"})
            return future

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    self._task_chunk("task-1", "task one"),
                    self._task_chunk("task-2", "task two"),
                    self._child_interrupt_chunk(
                        ("task:task-1:0",),
                        "execute",
                        {"command": "ls"},
                    ),
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert snapshot["task-1"][0] == "running"
        assert snapshot["task-1"][1] is not None
        assert snapshot["task-2"][0] == "running"
        assert snapshot["task-2"][1] is not None

    async def test_tool_awaiting_own_approval_is_paused_then_resumed(self) -> None:
        """A main-agent tool blocked on its own approval still pauses.

        Requirement 4: a tool waiting for its *own* initial approval must not
        misleadingly show "Running...". Its action request owns its row, so it
        is paused to `pending` (start time cleared) during the await, then
        resumed to `running` on approval.
        """
        snapshot: dict[str, Any] = {}

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            row = adapter._current_tool_messages["exec-1"]
            snapshot["status"] = row._status
            snapshot["start_time"] = row._start_time
            future: asyncio.Future[object] = asyncio.Future()
            future.set_result({"type": "approve"})
            return future

        execute_row_holder: list[ToolCallMessage] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            if isinstance(widget, ToolCallMessage):
                execute_row_holder.append(widget)

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    (
                        (),
                        "messages",
                        (
                            _tool_call_message(
                                "execute", {"command": "echo hi"}, "exec-1"
                            ),
                            {},
                        ),
                    ),
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": [
                                {"name": "execute", "args": {"command": "echo hi"}}
                            ],
                            "review_configs": [
                                {
                                    "action_name": "execute",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        }
                    ),
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        # Paused (not misleadingly "running") while awaiting its own approval.
        assert snapshot["status"] == "pending"
        assert snapshot["start_time"] is None
        # Resumed to running on approval; the empty resume stream leaves it there.
        assert execute_row_holder
        assert execute_row_holder[0]._status == "running"
        assert execute_row_holder[0]._start_time is not None

    async def test_completed_task_duration_spans_full_execution(self) -> None:
        """A task's `Took …` covers full run time, not just post-approval.

        The nested interrupt+approval must not reset the task's `_start_time`;
        otherwise the completed duration would only measure the sliver since the
        last approval. A far-past start time injected at approval time survives
        to completion and yields a large duration — before the fix, the approval
        reset it and the duration collapsed to ~0.
        """
        task_row_holder: list[ToolCallMessage] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            if isinstance(widget, ToolCallMessage):
                task_row_holder.append(widget)

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            task_row = adapter._current_tool_messages["task-1"]
            # The task must still be running (not paused) here; pin a far-past
            # start so a preserved timer produces a large, checkable duration.
            assert task_row._status == "running"
            assert task_row._start_time is not None
            task_row._start_time = time() - 100.0
            future: asyncio.Future[object] = asyncio.Future()
            future.set_result({"type": "approve"})
            return future

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    self._task_chunk("task-1", "long research"),
                    self._child_interrupt_chunk(
                        ("task:task-1:0",),
                        "fetch_url",
                        {"url": "http://example.com"},
                    ),
                ],
                [
                    (
                        (),
                        "messages",
                        (
                            ToolMessage(
                                content="subagent finished",
                                name="task",
                                tool_call_id="task-1",
                                status="success",
                            ),
                            {},
                        ),
                    ),
                ],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert task_row_holder
        task_row = task_row_holder[0]
        assert task_row._status == "success"
        assert task_row._duration is not None
        # Full span (~100s), not the near-zero interval since the approval.
        assert task_row._duration >= 99.0


class TestExecuteTaskTextualAskUser:
    """Tests for ask_user interrupt handling in the Textual adapter."""

    async def test_ask_user_interrupt_mounts_tool_call_row(self) -> None:
        """ask_user interrupts should mount the tool row before the prompt."""
        mounted: list[object] = []
        future: asyncio.Future[AskUserWidgetResult] = asyncio.Future()
        future.set_result({"type": "answered", "answers": ["Alice"]})

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return future

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        tool_rows = [
            widget for widget in mounted if isinstance(widget, ToolCallMessage)
        ]
        assert len(tool_rows) == 1
        tool_row = tool_rows[0]
        assert tool_row.tool_name == "ask_user"
        assert tool_row.has_expandable_args is True
        # Answered cleanup pops the row from `_current_tool_messages`.
        assert "tool-1" not in adapter._current_tool_messages

    async def test_ask_user_mount_failure_does_not_register_tool_id(self) -> None:
        """Mount failure should not poison `displayed_tool_ids` on the adapter."""

        async def mount_message(_widget: object) -> None:
            await asyncio.sleep(0)
            msg = "mount failed"
            raise RuntimeError(msg)

        future: asyncio.Future[AskUserWidgetResult] = asyncio.Future()
        future.set_result({"type": "answered", "answers": ["Alice"]})

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return future

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        # The flow continued, resumed with the answer, and never registered the
        # broken tool row.
        assert "tool-1" not in adapter._current_tool_messages

    async def test_ask_user_duplicate_interrupt_only_mounts_once(self) -> None:
        """Re-emitting the same `tool_call_id` should not double-mount."""
        mounted: list[object] = []
        future: asyncio.Future[AskUserWidgetResult] = asyncio.Future()
        future.set_result({"type": "answered", "answers": ["Alice"]})

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return future

        payload = {
            "type": "ask_user",
            "questions": [{"question": "Name?", "type": "text"}],
            "tool_call_id": "tool-dedup",
        }
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(payload),
                    _ask_user_interrupt_chunk(payload),
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        tool_rows = [w for w in mounted if isinstance(w, ToolCallMessage)]
        assert len(tool_rows) == 1

    async def test_ask_user_cancelled_marks_row_rejected_and_halts(self) -> None:
        """Cancelled result should reject the row and not resume generation."""
        mounted: list[object] = []
        token_events: list[str] = []
        future: asyncio.Future[AskUserWidgetResult] = asyncio.Future()
        future.set_result({"type": "cancelled"})

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return future

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )
        adapter._on_tokens_pending = lambda: token_events.append("pending")
        adapter._on_tokens_show = lambda *, approximate=False: token_events.append(
            f"show:{approximate}"
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert len(agent.stream_inputs) == 1
        assert "tool-1" not in adapter._current_tool_messages
        app_messages = [widget for widget in mounted if isinstance(widget, AppMessage)]
        assert len(app_messages) == 1
        assert "Question cancelled" in str(app_messages[0]._content)
        assert token_events == ["pending", "show:False"]

    async def test_hitl_rejection_restores_token_display_before_halt(self) -> None:
        """Rejected approval should restore tokens before returning early."""
        mounted: list[object] = []
        token_events: list[str] = []
        future: asyncio.Future[object] = asyncio.Future()
        future.set_result({"type": "reject"})

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            return future

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": [
                                {"name": "read_file", "args": {"path": "notes.txt"}}
                            ],
                            "review_configs": [
                                {
                                    "action_name": "read_file",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=request_approval,
        )
        adapter._on_tokens_pending = lambda: token_events.append("pending")
        adapter._on_tokens_show = lambda *, approximate=False: token_events.append(
            f"show:{approximate}"
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert len(agent.stream_inputs) == 1
        app_messages = [widget for widget in mounted if isinstance(widget, AppMessage)]
        assert len(app_messages) == 1
        assert "Command rejected" in str(app_messages[0]._content)
        assert token_events == ["pending", "show:False"]

    async def test_hitl_rejection_with_reason_resumes_agent(self) -> None:
        """Rejected approval with a reason should resume so the agent can react."""
        mounted: list[object] = []
        future: asyncio.Future[object] = asyncio.Future()
        future.set_result({"type": "reject", "message": "use a safer command"})

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            return future

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": [
                                {"name": "execute", "args": {"command": "rm file"}}
                            ],
                            "review_configs": [
                                {
                                    "action_name": "execute",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert len(agent.stream_inputs) == 2
        resume_cmd = agent.stream_inputs[1]
        assert isinstance(resume_cmd, Command)
        resume_payload = cast("dict[str, dict[str, Any]]", resume_cmd.resume)
        decisions = resume_payload["interrupt-1"]["decisions"]
        assert decisions == [{"type": "reject", "message": "use a safer command"}]
        app_messages = [widget for widget in mounted if isinstance(widget, AppMessage)]
        assert not any("Command rejected" in str(msg._content) for msg in app_messages)

    async def test_server_operation_bare_rejection_resumes_agent(self) -> None:
        """A criteria agent must receive a bare rejection and finish without context."""
        mounted: list[object] = []
        future: asyncio.Future[object] = asyncio.Future()
        future.set_result({"type": "reject"})

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            return future

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": [
                                {
                                    "name": "fetch_url",
                                    "args": {"url": "https://example.com"},
                                }
                            ],
                            "review_configs": [
                                {
                                    "action_name": "fetch_url",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=request_approval,
        )
        request = {
            "messages": [],
            "goal_criteria_request": {
                "request_id": "request-1",
                "kind": "create",
                "objective": "ship it",
            },
        }

        await execute_task_textual(
            user_input="",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
            graph_input=request,
        )

        assert len(agent.stream_inputs) == 2
        assert agent.stream_inputs[0] == request
        resume_cmd = agent.stream_inputs[1]
        assert isinstance(resume_cmd, Command)
        resume_payload = cast("dict[str, dict[str, Any]]", resume_cmd.resume)
        assert resume_payload["interrupt-1"]["decisions"] == [{"type": "reject"}]
        app_messages = [widget for widget in mounted if isinstance(widget, AppMessage)]
        assert not any("Command rejected" in str(msg._content) for msg in app_messages)

    async def test_ask_user_invalid_answers_payload_marks_row_error(self) -> None:
        """Non-list answers should mark row as error and pop it."""
        mounted: list[ToolCallMessage] = []
        error_calls: list[str] = []
        future: asyncio.Future[object] = asyncio.Future()
        future.set_result({"type": "answered", "answers": "not-a-list"})

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            if isinstance(widget, ToolCallMessage):
                original = widget.set_error

                def _capture(error: str) -> None:
                    error_calls.append(error)
                    original(error)

                widget.set_error = _capture  # ty: ignore
                mounted.append(widget)

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[object] | None:
            await asyncio.sleep(0)
            return future

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            # This test intentionally returns a malformed widget payload.
            request_ask_user=cast("Any", request_ask_user),
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        resume_cmd = agent.stream_inputs[1]
        assert isinstance(resume_cmd, Command)
        resume_payload = cast("dict[str, dict[str, Any]]", resume_cmd.resume)
        assert resume_payload["interrupt-1"]["status"] == "error"
        assert (
            resume_payload["interrupt-1"]["error"] == "invalid ask_user answers payload"
        )
        assert len(mounted) == 1
        assert "invalid ask_user answers payload" in error_calls
        assert "tool-1" not in adapter._current_tool_messages

    async def test_ask_user_unsupported_marks_row_error(self) -> None:
        """When no callback is registered, the mounted row gets an error."""
        mounted: list[ToolCallMessage] = []
        error_calls: list[str] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            if isinstance(widget, ToolCallMessage):
                original = widget.set_error

                def _capture(error: str) -> None:
                    error_calls.append(error)
                    original(error)

                widget.set_error = _capture  # ty: ignore
                mounted.append(widget)

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=None,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert len(mounted) == 1
        assert "ask_user not supported by this UI" in error_calls
        assert "tool-1" not in adapter._current_tool_messages

    async def test_ask_user_unsupported_dispatches_terminal_hooks(self) -> None:
        """The unsupported-UI ask_user branch closes its tool.use.

        With no `request_ask_user` callback the ask_user cannot run, so the
        branch must emit `tool.error` + an error `tool.result` carrying the
        canned unsupported message — otherwise its `tool.use` is unterminated.
        """
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=None,
        )

        with patch(
            "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
        ) as mock_dispatch:
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        events = [(c[0][0], c[0][1]) for c in mock_dispatch.call_args_list]
        assert ("tool.error", {"tool_names": ["ask_user"]}) in events
        result_payloads = [p for e, p in events if e == "tool.result"]
        assert len(result_payloads) == 1
        assert result_payloads[0]["tool_name"] == "ask_user"
        assert result_payloads[0]["tool_id"] == "tool-1"
        assert result_payloads[0]["tool_status"] == "error"
        assert result_payloads[0]["tool_output"] == "ask_user not supported by this UI"

    async def test_request_ask_user_returning_none_is_reported_as_error(self) -> None:
        """A `None` callback result should resume with explicit error status."""

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return None

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert len(agent.stream_inputs) >= 2
        resume_cmd = agent.stream_inputs[1]
        assert isinstance(resume_cmd, Command)
        resume_payload = cast("dict[str, dict[str, Any]]", resume_cmd.resume)
        ask_user_resume = resume_payload["interrupt-1"]
        assert ask_user_resume["status"] == "error"
        assert ask_user_resume["error"] == "ask_user callback returned no response"
        assert ask_user_resume["answers"] == [""]

    async def test_request_ask_user_mount_error_is_not_treated_as_cancel(self) -> None:
        """UI mount failures should resume with explicit error status."""

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            msg = "boom"
            raise RuntimeError(msg)

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        resume_cmd = agent.stream_inputs[1]
        assert isinstance(resume_cmd, Command)
        resume_payload = cast("dict[str, dict[str, Any]]", resume_cmd.resume)
        ask_user_resume = resume_payload["interrupt-1"]
        assert ask_user_resume["status"] == "error"
        assert ask_user_resume["error"] == "failed to display ask_user prompt"
        assert ask_user_resume["answers"] == [""]

    async def test_request_ask_user_missing_callback_is_reported_as_error(self) -> None:
        """ask_user interrupts without a UI callback should resume with error."""
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=None,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        resume_cmd = agent.stream_inputs[1]
        assert isinstance(resume_cmd, Command)
        resume_payload = cast("dict[str, dict[str, Any]]", resume_cmd.resume)
        ask_user_resume = resume_payload["interrupt-1"]
        assert ask_user_resume["status"] == "error"
        assert ask_user_resume["error"] == "ask_user not supported by this UI"
        assert ask_user_resume["answers"] == [""]

    async def test_spinner_reappears_after_ask_user_resume(self) -> None:
        """Spinner should re-show Thinking on each astream iteration.

        Regression for a gap where the model was working on the resume
        payload after an ask_user response but no spinner was visible.
        """
        statuses: list[str | None] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return None

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
            set_spinner=record_spinner,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        # Two astream iterations (interrupt, then resume) -> expect
        # Thinking set before each, and nothing above that count since
        # no tool calls stream in this test.
        assert len(agent.stream_inputs) == 2
        thinking_count = sum(1 for s in statuses if s == "Thinking")
        assert thinking_count == 2, (
            f"Expected Thinking spinner on each iteration; got {statuses}"
        )

    async def test_invalid_ask_user_interrupt_payload_raises_validation_error(
        self,
    ) -> None:
        """Missing required ask_user keys should fail validation at ingestion."""
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            # Missing required keys: `questions` and `tool_call_id`.
                        }
                    )
                ]
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with pytest.raises(ValidationError):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(
                    thread_id="thread-1",
                    auto_approve=False,
                ),
                adapter=adapter,
            )


# ---------------------------------------------------------------------------
# Helpers for dict-iteration safety tests
# ---------------------------------------------------------------------------


def _make_tool_widget(name: str = "tool", args: dict | None = None) -> MagicMock:
    """Create a MagicMock that mimics a ToolCallMessage widget."""
    widget = MagicMock()
    widget._tool_name = name
    widget._args = args or {}
    return widget


class _MutatingItemsDict(dict):  # noqa: FURB189  # must subclass dict to override C-level iteration
    """Dict whose `.items()` deletes another key mid-iteration.

    This deterministically reproduces the `RuntimeError: dictionary
    changed size during iteration` that occurs when async tool-result
    callbacks mutate `_current_tool_messages` while the HITL approval
    loop is iterating over it.

    We intentionally subclass `dict` (not `UserDict`) because we
    need to override the C-level iteration that triggers the error.
    """

    def items(self) -> Generator[tuple[str, Any], None, None]:  # ty: ignore
        """Yield items while mutating the dict mid-iteration."""
        it = iter(dict.items(self))
        first = next(it)
        # Remove a *different* key while iteration is in progress.
        remaining = [k for k in self if k != first[0]]
        if remaining:
            del self[remaining[0]]
        yield first
        yield from it


class _MutatingValuesDict(dict):  # noqa: FURB189  # must subclass dict to override C-level iteration
    """Dict whose `.values()` deletes a key mid-iteration.

    We intentionally subclass `dict` (not `UserDict`) because we
    need to override the C-level iteration that triggers the error.
    """

    def values(self) -> Generator[Any, None, None]:  # ty: ignore
        """Yield values while mutating the dict mid-iteration."""
        it = iter(dict.values(self))
        first = next(it)
        # Remove the first key to trigger size-change error.
        first_key = next(iter(self))
        del self[first_key]
        yield first
        yield from it


class TestDictIterationSafety:
    """Regression tests for #956.

    Parallel tool calls can modify `adapter._current_tool_messages`
    while another coroutine iterates over it, raising
    `RuntimeError: dictionary changed size during iteration`.

    The fix wraps every iteration with `list()` so a snapshot is
    taken before the loop body runs.  These tests prove the fix is
    necessary and sufficient.
    """

    # -- Test A: bare iteration over a mutating dict raises ----

    def test_items_iteration_fails_without_list(self) -> None:
        """Iterating .items() on a concurrently-mutated dict raises."""
        d = _MutatingItemsDict(
            {f"id_{i}": _make_tool_widget(f"t{i}") for i in range(3)}
        )
        with pytest.raises(RuntimeError, match="changed size"):
            for _ in d.items():
                pass

    def test_values_iteration_fails_without_list(self) -> None:
        """Iterating .values() on a concurrently-mutated dict raises."""
        d = _MutatingValuesDict(
            {f"id_{i}": _make_tool_widget(f"t{i}") for i in range(3)}
        )
        with pytest.raises(RuntimeError, match="changed size"):
            for _ in d.values():
                pass

    # -- Test B: list() snapshot protects iteration ----

    def test_items_iteration_safe_with_list(self) -> None:
        """`list(d.items())` snapshots before mutation can occur."""
        d: dict = {f"id_{i}": _make_tool_widget(f"t{i}") for i in range(5)}
        collected = []
        for key, _val in list(d.items()):
            collected.append(key)
            d.pop(key, None)  # mutate during loop body
        assert len(collected) == 5
        assert len(d) == 0

    def test_values_iteration_safe_with_list(self) -> None:
        """`list(d.values())` snapshots before mutation."""
        d: dict = {f"id_{i}": _make_tool_widget(f"t{i}") for i in range(5)}
        collected = []
        keys = list(d.keys())
        for val in list(d.values()):
            collected.append(val)
            if keys:
                d.pop(keys.pop(0), None)
        assert len(collected) == 5

    # -- Test C: _build_interrupted_ai_message uses list() ----

    def test_build_interrupted_ai_message_safe(self) -> None:
        """_build_interrupted_ai_message correctly builds an AIMessage.

        Verifies the function reconstructs tool calls and content from
        the provided widget dict. The `list()` snapshot inside the
        production code protects against external async mutation at
        `await` boundaries, which cannot be deterministically simulated
        in a synchronous unit test.
        """
        widgets = {
            f"id_{i}": _make_tool_widget(f"tool_{i}", {"k": i}) for i in range(4)
        }
        pending_text: dict[tuple, str] = {(): "hello"}
        result = _build_interrupted_ai_message(pending_text, widgets)
        assert result is not None
        assert result.content == "hello"
        assert len(result.tool_calls) == 4
        names = {tc["name"] for tc in result.tool_calls}
        assert names == {"tool_0", "tool_1", "tool_2", "tool_3"}

    def test_build_interrupted_ai_message_empty(self) -> None:
        """Returns None when there is no text and no tool calls."""
        result = _build_interrupted_ai_message({}, {})
        assert result is None


# ---------------------------------------------------------------------------
# tool.use / tool.result hook dispatch (textual path)
# ---------------------------------------------------------------------------


class TestToolHooksTextual:
    """Tests for tool.use and tool.result hook dispatch in execute_task_textual."""

    async def test_tool_use_hook_dispatched_before_mount(self) -> None:
        """tool.use fires (with name, id, args) before the ToolCallMessage mounts."""
        events: list[str] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            events.append(f"mount:{type(widget).__name__}")

        def record_dispatch(event: str, _payload: dict[str, Any]) -> None:
            events.append(f"dispatch:{event}")

        chunks = [
            (
                (),
                "messages",
                (_tool_call_message("read_file", {"path": "foo.py"}, "call-1"), {}),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with patch(
            "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget",
            side_effect=record_dispatch,
        ) as mock_dispatch:
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        # tool.use is the first dispatch. This fixture streams no ToolMessage, so
        # the clean-completion orphan close then emits terminal tool.error/
        # tool.result for the same call (covered by
        # test_clean_completion_closes_unresulted_tool_use); assert on the first
        # call rather than call count.
        assert mock_dispatch.call_args_list[0][0][0] == "tool.use"
        payload = mock_dispatch.call_args_list[0][0][1]
        assert payload["tool_name"] == "read_file"
        assert payload["tool_id"] == "call-1"
        assert payload["tool_args"] == {"path": "foo.py"}
        # The hook must fire before the widget mounts, not merely at some point.
        assert events.index("dispatch:tool.use") < events.index("mount:ToolCallMessage")

    async def test_tool_result_hook_dispatched_on_success(self) -> None:
        """tool.result fires with tool_status='success' after a successful tool run."""
        mounted: list[object] = []
        output = "x" * 5000

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        chunks = [
            (
                (),
                "messages",
                (_tool_call_message("read_file", {"path": "foo.py"}, "call-1"), {}),
            ),
            (
                (),
                "messages",
                (ToolMessage(content=output, tool_call_id="call-1"), {}),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        def fail_if_tool_result(event: str, _payload: dict[str, Any]) -> None:
            if event == "tool.result":
                msg = "tool.result hooks must not be awaited in the textual path"
                raise AssertionError(msg)

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
                side_effect=fail_if_tool_result,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        tool_result_calls = [
            c for c in mock_dispatch.call_args_list if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        payload = tool_result_calls[0][0][1]
        assert payload["tool_status"] == "success"
        assert payload["tool_name"] == "read_file"
        assert payload["tool_id"] == "call-1"
        # Capped to the limit and marked so a consumer can tell it was truncated;
        # the marker is counted within the cap so the total stays at 2000.
        assert len(payload["tool_output"]) == 2000
        assert payload["tool_output"].endswith(TOOL_OUTPUT_TRUNCATION_MARKER)
        assert payload["tool_output"].startswith("x" * 100)
        assert payload["tool_args"] == {"path": "foo.py"}
        tool_msg = next(
            widget for widget in mounted if isinstance(widget, ToolCallMessage)
        )
        assert tool_msg._output == output

    async def test_tool_result_sentinel_when_formatter_raises(self) -> None:
        """A formatter error still dispatches tool.result with the sentinel.

        The content-formatting guard must keep the terminal dispatch
        unconditional; otherwise a formatter (or pathological `__str__`) error
        would drop the tool.result for this tool and every later tool.
        """
        chunks = [
            (
                (),
                "messages",
                (_tool_call_message("read_file", {"path": "foo.py"}, "call-1"), {}),
            ),
            (
                (),
                "messages",
                (ToolMessage(content="unformattable", tool_call_id="call-1"), {}),
            ),
        ]
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        def boom(_content: object) -> str:
            msg = "formatter boom"
            raise RuntimeError(msg)

        with (
            patch(
                "deepagents_code.tui.textual_adapter.format_tool_message_content",
                side_effect=boom,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        result_payloads = [
            c[0][1] for c in mock_dispatch.call_args_list if c[0][0] == "tool.result"
        ]
        assert len(result_payloads) == 1
        assert result_payloads[0]["tool_output"] == UNRENDERABLE_TOOL_OUTPUT

    async def test_mount_failure_does_not_suppress_real_tool_result(self) -> None:
        """A widget mount failure still lets the real tool.result dispatch.

        tool.use fires when the tool call is parsed. If `_mount_message` raises,
        the pending call must stay tracked for correlation so the later real
        ToolMessage reports the actual status/output instead of being suppressed
        by a synthetic UI-mount error.
        """

        async def failing_mount(widget: object) -> None:
            await asyncio.sleep(0)
            # Simulate a mount failure only for the tool row (assistant/other
            # widgets still mount), so the guard under test is exercised.
            if isinstance(widget, ToolCallMessage):
                msg = "mount boom"
                raise RuntimeError(msg)  # noqa: TRY004  # simulated failure, not a type guard

        chunks = [
            (
                (),
                "messages",
                (_tool_call_message("read_file", {"path": "foo.py"}, "call-1"), {}),
            ),
            # The real result still arrives after the mount failed; it must be
            # the authoritative terminal tool.result.
            (
                (),
                "messages",
                (ToolMessage(content="ok", tool_call_id="call-1"), {}),
            ),
        ]
        adapter = TextualUIAdapter(
            mount_message=failing_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with patch(
            "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
        ) as mock_dispatch:
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        events = [(c[0][0], c[0][1]) for c in mock_dispatch.call_args_list]
        assert (
            "tool.use",
            {
                "tool_name": "read_file",
                "tool_id": "call-1",
                "tool_args": {"path": "foo.py"},
            },
        ) in events
        assert ("tool.error", {"tool_names": ["read_file"]}) not in events
        result_payloads = [p for e, p in events if e == "tool.result"]
        assert result_payloads == [
            {
                "tool_name": "read_file",
                "tool_id": "call-1",
                "tool_args": {"path": "foo.py"},
                "tool_status": "success",
                "tool_output": "ok",
            }
        ]

    async def test_set_success_failure_does_not_drop_later_tool_hooks(self) -> None:
        """A widget `set_success` failure must not abort the turn or drop hooks.

        The widget update is guarded and runs *after* the terminal dispatch, so
        even when the first tool's `set_success` raises, the stream loop keeps
        going and the second tool still emits its own tool.result. Without the
        guard the first exception would propagate and the second tool's hook
        would never fire.
        """
        chunks = [
            (
                (),
                "messages",
                (_tool_call_message("read_file", {"path": "a.py"}, "call-1"), {}),
            ),
            ((), "messages", (ToolMessage(content="a", tool_call_id="call-1"), {})),
            (
                (),
                "messages",
                (_tool_call_message("read_file", {"path": "b.py"}, "call-2"), {}),
            ),
            ((), "messages", (ToolMessage(content="b", tool_call_id="call-2"), {})),
        ]
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        calls = {"n": 0}

        def flaky_set_success(_self: ToolCallMessage, _result: str = "") -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                msg = "set_success boom"
                raise RuntimeError(msg)

        with (
            patch.object(ToolCallMessage, "set_success", flaky_set_success),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        result_payloads = [
            c[0][1] for c in mock_dispatch.call_args_list if c[0][0] == "tool.result"
        ]
        # Both tools emit a success tool.result even though the first widget
        # update raised — the guard swallows it and the loop proceeds.
        assert [p["tool_id"] for p in result_payloads] == ["call-1", "call-2"]
        assert all(p["tool_status"] == "success" for p in result_payloads)

    async def test_clean_completion_closes_unresulted_tool_use(self) -> None:
        """A tool.use with no ToolMessage is closed on a clean stream end.

        Parity with the headless orphan drain (`_dispatch_orphaned_tool_result_
        hooks`): a graph that ends the turn after emitting a tool call whose
        result never streams must still terminate the tool.use rather than
        leaving it dangling and the widget stuck "Running" across turns.
        """
        chunks = [
            (
                (),
                "messages",
                (_tool_call_message("read_file", {"path": "foo.py"}, "call-1"), {}),
            ),
        ]
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with patch(
            "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
        ) as mock_dispatch:
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        events = [(c[0][0], c[0][1]) for c in mock_dispatch.call_args_list]
        assert (
            "tool.use",
            {
                "tool_name": "read_file",
                "tool_id": "call-1",
                "tool_args": {"path": "foo.py"},
            },
        ) in events
        assert ("tool.error", {"tool_names": ["read_file"]}) in events
        result_payloads = [p for e, p in events if e == "tool.result"]
        assert result_payloads == [
            {
                "tool_name": "read_file",
                "tool_id": "call-1",
                "tool_args": {"path": "foo.py"},
                "tool_status": "error",
                "tool_output": "Stream ended before tool result",
            }
        ]
        # Tracking is cleared so the orphan can't leak into the next turn.
        assert adapter._current_tool_messages == {}

    async def test_uncorrelated_tool_result_logs_and_sends_empty_args(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A ToolMessage with no mounted widget emits tool.result with {} args.

        Mirrors the headless twin: an uncorrelated real-id result is logged at
        warning (degraded audit fidelity, greppable at default levels) and still
        dispatched so audit hooks observe every executed tool.
        """
        chunks = [
            (
                (),
                "messages",
                (
                    ToolMessage(
                        content="orphan", tool_call_id="ghost-1", name="read_file"
                    ),
                    {},
                ),
            ),
        ]
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with (
            caplog.at_level("WARNING", logger="deepagents_code.tui.textual_adapter"),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        result_payloads = [
            c[0][1] for c in mock_dispatch.call_args_list if c[0][0] == "tool.result"
        ]
        assert len(result_payloads) == 1
        assert result_payloads[0]["tool_id"] == "ghost-1"
        assert result_payloads[0]["tool_name"] == "read_file"
        assert result_payloads[0]["tool_args"] == {}
        assert any(
            "ghost-1" in record.message
            and "no correlated" in record.message
            and record.levelname == "WARNING"
            for record in caplog.records
        )

    async def test_ask_user_interrupt_dispatches_tool_hooks(self) -> None:
        """ask_user interrupt rows emit tool.use and tool.result hooks."""
        future: asyncio.Future[AskUserWidgetResult] = asyncio.Future()
        future.set_result({"type": "answered", "answers": ["Alice"]})

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return future

        questions: list[Question] = [{"question": "Name?", "type": "text"}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": questions,
                            "tool_call_id": "ask-1",
                        }
                    )
                ],
                [
                    (
                        (),
                        "messages",
                        (
                            ToolMessage(
                                content="answered via middleware",
                                tool_call_id="ask-1",
                                name="ask_user",
                            ),
                            {},
                        ),
                    ),
                ],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        mock_dispatch_background.assert_any_call(
            "tool.use",
            {
                "tool_name": "ask_user",
                "tool_id": "ask-1",
                "tool_args": {"questions": questions},
            },
        )

        tool_result_calls = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        assert tool_result_calls[0][0][1] == {
            "tool_name": "ask_user",
            "tool_id": "ask-1",
            "tool_args": {"questions": questions},
            "tool_status": "success",
            "tool_output": "User answered",
        }

    async def test_ask_user_mount_failure_skips_tool_use_hook(self) -> None:
        """A failed ask_user mount fires no tool.use, so nothing is orphaned.

        tool.use is gated on a successful mount, so a mount failure (e.g. a
        torn-down DOM) never emits a tool.use that a later cancel could leave
        unterminated. The question still resolves via the resolution loop, which
        dispatches the terminal tool.result independently of the widget.
        """

        async def mount_message(_widget: object) -> None:
            await asyncio.sleep(0)
            msg = "mount failed"
            raise RuntimeError(msg)

        future: asyncio.Future[AskUserWidgetResult] = asyncio.Future()
        future.set_result({"type": "answered", "answers": ["Alice"]})

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return future

        questions: list[Question] = [{"question": "Name?", "type": "text"}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": questions,
                            "tool_call_id": "ask-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        events = [c[0][0] for c in mock_dispatch_background.call_args_list]
        # No tool.use for the failed mount → nothing left dangling on a cancel.
        assert "tool.use" not in events
        # The question still resolved, so its terminal tool.result still fired.
        tool_result_calls = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        assert tool_result_calls[0][0][1]["tool_id"] == "ask-1"
        assert tool_result_calls[0][0][1]["tool_status"] == "success"

    async def test_ask_user_result_hook_survives_widget_success_failure(
        self,
    ) -> None:
        """ask_user result hooks dispatch even if the row update fails."""
        future: asyncio.Future[AskUserWidgetResult] = asyncio.Future()
        future.set_result({"type": "answered", "answers": ["Alice"]})

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return future

        async def mount_message(widget: object) -> None:
            if isinstance(widget, ToolCallMessage) and widget.tool_name == "ask_user":

                def fail_success(_output: str) -> None:
                    msg = "row was unmounted"
                    raise RuntimeError(msg)

                widget.set_success = fail_success  # ty: ignore
            await asyncio.sleep(0)

        questions: list[Question] = [{"question": "Name?", "type": "text"}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": questions,
                            "tool_call_id": "ask-1",
                        }
                    )
                ],
                [
                    (
                        (),
                        "messages",
                        (
                            ToolMessage(
                                content="answered via middleware",
                                tool_call_id="ask-1",
                                name="ask_user",
                            ),
                            {},
                        ),
                    ),
                ],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        tool_result_calls = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        assert tool_result_calls[0][0][1] == {
            "tool_name": "ask_user",
            "tool_id": "ask-1",
            "tool_args": {"questions": questions},
            "tool_status": "success",
            "tool_output": "User answered",
        }

    async def test_ask_user_interrupt_error_dispatches_tool_result_hook(self) -> None:
        """ask_user UI errors emit tool.result with tool_status='error'."""
        questions: list[Question] = [{"question": "Name?", "type": "text"}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": questions,
                            "tool_call_id": "ask-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=None,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        dispatched_events = [c[0][0] for c in mock_dispatch_background.call_args_list]
        assert "tool.error" in dispatched_events
        mock_dispatch_background.assert_any_call(
            "tool.use",
            {
                "tool_name": "ask_user",
                "tool_id": "ask-1",
                "tool_args": {"questions": questions},
            },
        )

        tool_result_calls = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        payload = tool_result_calls[0][0][1]
        assert payload["tool_name"] == "ask_user"
        assert payload["tool_id"] == "ask-1"
        assert payload["tool_args"] == {"questions": questions}
        assert payload["tool_status"] == "error"
        assert payload["tool_output"] == "ask_user not supported by this UI"

    async def test_tool_result_hook_dispatched_on_error(self) -> None:
        """tool.result fires with tool_status='error' and tool.error also fires."""
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        chunks = [
            (
                (),
                "messages",
                (_tool_call_message("write_file", {"path": "x.py"}, "call-2"), {}),
            ),
            (
                (),
                "messages",
                (
                    ToolMessage(
                        content="Permission denied",
                        tool_call_id="call-2",
                        status="error",
                    ),
                    {},
                ),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        dispatched_events = {c[0][0] for c in mock_dispatch_background.call_args_list}
        assert "tool.error" in dispatched_events

        tool_result_calls = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result"
        ]
        payload = tool_result_calls[0][0][1]
        assert payload["tool_status"] == "error"
        assert payload["tool_name"] == "write_file"

    async def test_hitl_bare_reject_dispatches_terminal_tool_hooks(self) -> None:
        """A bare HITL reject closes the prior tool.use hook before aborting."""
        action_requests = [{"name": "execute", "args": {"command": "echo hi"}}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    (
                        (),
                        "messages",
                        (
                            _tool_call_message(
                                "execute", {"command": "echo hi"}, "tool-1"
                            ),
                            {},
                        ),
                    ),
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": action_requests,
                            "review_configs": [
                                {
                                    "action_name": "execute",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        }
                    ),
                ],
                [],
            ]
        )

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            future: asyncio.Future[object] = asyncio.Future()
            future.set_result({"type": "reject"})
            return future

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        assert len(agent.stream_inputs) == 1
        assert [c[0][0] for c in mock_dispatch_background.call_args_list] == [
            "tool.use",
            "tool.error",
            "tool.result",
        ]
        tool_result_payload = mock_dispatch_background.call_args_list[2][0][1]
        assert tool_result_payload == {
            "tool_name": "execute",
            "tool_id": "tool-1",
            "tool_args": {"command": "echo hi"},
            "tool_status": "error",
            "tool_output": "Tool approval rejected",
        }

    async def test_hitl_reasoned_reject_preserves_tool_args_for_result(self) -> None:
        """A reasoned HITL reject keeps args until the resumed ToolMessage."""
        action_requests = [{"name": "execute", "args": {"command": "echo hi"}}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    (
                        (),
                        "messages",
                        (
                            _tool_call_message(
                                "execute", {"command": "echo hi"}, "tool-1"
                            ),
                            {},
                        ),
                    ),
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": action_requests,
                            "review_configs": [
                                {
                                    "action_name": "execute",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        }
                    ),
                ],
                [
                    (
                        (),
                        "messages",
                        (
                            ToolMessage(
                                content="Tool approval rejected",
                                tool_call_id="tool-1",
                                status="error",
                            ),
                            {},
                        ),
                    ),
                ],
            ]
        )

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            future: asyncio.Future[object] = asyncio.Future()
            future.set_result({"type": "reject", "message": "use another command"})
            return future

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        assert len(agent.stream_inputs) == 2
        tool_result_calls = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        assert tool_result_calls[0][0][1] == {
            "tool_name": "execute",
            "tool_id": "tool-1",
            "tool_args": {"command": "echo hi"},
            "tool_status": "error",
            "tool_output": "Tool approval rejected",
        }

    async def test_hitl_reasoned_reject_keeps_row_rejected(self) -> None:
        """A reasoned reject that resumes must not flip the row to Error.

        The resumed synthetic reject `ToolMessage` fires the terminal hook via
        the mounted branch, which also calls `set_error`; the widget must keep
        its rejected state because `set_error`/`set_success` no-op once a row is
        terminal-rejected. Guards against the row flipping "Rejected" -> "Error".
        """
        mounted: list[ToolCallMessage] = []

        async def capture_mount(widget: object) -> None:
            await asyncio.sleep(0)
            if isinstance(widget, ToolCallMessage):
                mounted.append(widget)

        action_requests = [{"name": "execute", "args": {"command": "echo hi"}}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    (
                        (),
                        "messages",
                        (
                            _tool_call_message(
                                "execute", {"command": "echo hi"}, "tool-1"
                            ),
                            {},
                        ),
                    ),
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": action_requests,
                            "review_configs": [
                                {
                                    "action_name": "execute",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        }
                    ),
                ],
                [
                    (
                        (),
                        "messages",
                        (
                            ToolMessage(
                                content="Tool approval rejected",
                                tool_call_id="tool-1",
                                status="error",
                            ),
                            {},
                        ),
                    ),
                ],
            ]
        )

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            future: asyncio.Future[object] = asyncio.Future()
            future.set_result({"type": "reject", "message": "use another command"})
            return future

        adapter = TextualUIAdapter(
            mount_message=capture_mount,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert len(agent.stream_inputs) == 2
        execute_widgets = [w for w in mounted if w.tool_name == "execute"]
        assert len(execute_widgets) == 1
        # Stayed rejected despite the resumed error ToolMessage driving set_error.
        assert execute_widgets[0]._status == "rejected"

    async def test_tool_use_dispatched_after_streaming_fragments(self) -> None:
        """tool.use reassembles streamed arg fragments and fires exactly once."""
        chunks = [
            _tool_chunk(name="execute", args='{"command": "uv run', chunk_id="call-1"),
            _tool_chunk(name=None, args=' pytest"}', chunk_id=None),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with patch(
            "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
        ) as mock_dispatch:
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        tool_use_calls = [
            c for c in mock_dispatch.call_args_list if c[0][0] == "tool.use"
        ]
        assert len(tool_use_calls) == 1
        assert tool_use_calls[0][0][1] == {
            "tool_name": "execute",
            "tool_id": "call-1",
            "tool_args": {"command": "uv run pytest"},
        }

    async def test_untracked_tool_message_dispatches_tool_result(self) -> None:
        """A ToolMessage whose id was never mounted still emits tool.result.

        Mirrors the headless path so audit hooks observe every executed tool,
        even when the tool call was never rendered (e.g. its args never parsed).
        """
        chunks = [
            (
                (),
                "messages",
                (
                    ToolMessage(
                        content="orphaned output",
                        tool_call_id="ghost-1",
                        name="read_file",
                        status="error",
                    ),
                    {},
                ),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        events = [c[0][0] for c in mock_dispatch_background.call_args_list]
        assert "tool.error" in events
        tool_result_calls = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        payload = tool_result_calls[0][0][1]
        assert payload["tool_name"] == "read_file"
        assert payload["tool_id"] == "ghost-1"
        assert payload["tool_args"] == {}
        assert payload["tool_status"] == "error"
        assert payload["tool_output"] == "orphaned output"

    async def test_untracked_tool_message_success_suppresses_tool_error(self) -> None:
        """An untracked *successful* ToolMessage emits tool.result, not tool.error.

        Complements the error variant above: the widget-less path must not fire a
        spurious tool.error for a tool that succeeded.
        """
        chunks = [
            (
                (),
                "messages",
                (
                    ToolMessage(
                        content="orphaned output",
                        tool_call_id="ghost-1",
                        name="read_file",
                        status="success",
                    ),
                    {},
                ),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        events = [c[0][0] for c in mock_dispatch_background.call_args_list]
        assert "tool.error" not in events
        tool_result_calls = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        assert tool_result_calls[0][0][1] == {
            "tool_name": "read_file",
            "tool_id": "ghost-1",
            "tool_args": {},
            "tool_status": "success",
            "tool_output": "orphaned output",
        }

    async def test_hitl_bare_reject_dispatches_hooks_for_every_tool(self) -> None:
        """A batch bare-reject closes each pending tool's tool.use, per tool.

        `_dispatch_terminal_tool_result_hooks` loops over every mounted tool, so
        rejecting a parallel batch must emit one tool.error + one tool.result for
        each tool, each carrying its own id and args — not just the first.
        """
        action_requests = [
            {"name": "execute", "args": {"command": "echo hi"}},
            {"name": "write_file", "args": {"path": "foo.py"}},
        ]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    (
                        (),
                        "messages",
                        (
                            _tool_call_message(
                                "execute", {"command": "echo hi"}, "tool-1"
                            ),
                            {},
                        ),
                    ),
                    (
                        (),
                        "messages",
                        (
                            _tool_call_message(
                                "write_file", {"path": "foo.py"}, "tool-2"
                            ),
                            {},
                        ),
                    ),
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": action_requests,
                            "review_configs": [
                                {
                                    "action_name": "execute",
                                    "allowed_decisions": ["approve", "reject"],
                                },
                                {
                                    "action_name": "write_file",
                                    "allowed_decisions": ["approve", "reject"],
                                },
                            ],
                        }
                    ),
                ],
                [],
            ]
        )

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            future: asyncio.Future[object] = asyncio.Future()
            future.set_result({"type": "reject"})
            return future

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        calls = mock_dispatch_background.call_args_list
        results = {
            c[0][1]["tool_id"]: c[0][1] for c in calls if c[0][0] == "tool.result"
        }
        errors = sorted(
            c[0][1]["tool_names"][0] for c in calls if c[0][0] == "tool.error"
        )
        assert set(results) == {"tool-1", "tool-2"}
        assert results["tool-1"]["tool_args"] == {"command": "echo hi"}
        assert results["tool-1"]["tool_status"] == "error"
        assert results["tool-2"]["tool_args"] == {"path": "foo.py"}
        assert results["tool-2"]["tool_status"] == "error"
        assert errors == ["execute", "write_file"]

    async def test_hitl_unexpected_decision_type_dispatches_terminal_hooks(
        self,
    ) -> None:
        """An unrecognized HITL decision type still closes the pending tool.use."""
        action_requests = [{"name": "execute", "args": {"command": "echo hi"}}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    (
                        (),
                        "messages",
                        (
                            _tool_call_message(
                                "execute", {"command": "echo hi"}, "tool-1"
                            ),
                            {},
                        ),
                    ),
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": action_requests,
                            "review_configs": [
                                {
                                    "action_name": "execute",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        }
                    ),
                ],
                [],
            ]
        )

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            future: asyncio.Future[object] = asyncio.Future()
            future.set_result({"type": "bogus"})
            return future

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        assert [c[0][0] for c in mock_dispatch_background.call_args_list] == [
            "tool.use",
            "tool.error",
            "tool.result",
        ]
        assert mock_dispatch_background.call_args_list[2][0][1] == {
            "tool_name": "execute",
            "tool_id": "tool-1",
            "tool_args": {"command": "echo hi"},
            "tool_status": "error",
            "tool_output": "Tool approval rejected",
        }

    async def test_hitl_non_dict_decision_dispatches_terminal_hooks(self) -> None:
        """A non-dict HITL decision still closes the pending tool.use."""
        action_requests = [{"name": "execute", "args": {"command": "echo hi"}}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    (
                        (),
                        "messages",
                        (
                            _tool_call_message(
                                "execute", {"command": "echo hi"}, "tool-1"
                            ),
                            {},
                        ),
                    ),
                    _hitl_interrupt_chunk(
                        {
                            "action_requests": action_requests,
                            "review_configs": [
                                {
                                    "action_name": "execute",
                                    "allowed_decisions": ["approve", "reject"],
                                }
                            ],
                        }
                    ),
                ],
                [],
            ]
        )

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            future: asyncio.Future[object] = asyncio.Future()
            future.set_result("reject")
            return future

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=request_approval,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        assert [c[0][0] for c in mock_dispatch_background.call_args_list] == [
            "tool.use",
            "tool.error",
            "tool.result",
        ]
        assert mock_dispatch_background.call_args_list[2][0][1] == {
            "tool_name": "execute",
            "tool_id": "tool-1",
            "tool_args": {"command": "echo hi"},
            "tool_status": "error",
            "tool_output": "Tool approval rejected",
        }

    async def test_malformed_complete_args_skip_tool_use_but_emit_result(self) -> None:
        """Interactive parity: complete-but-malformed args skip tool.use, keep result.

        Mirrors the headless `test_malformed_args_emit_result_without_use`: a
        streamed arg fragment that looks complete (bracketed and closed) but fails
        to parse produces no tool.use, yet the executed tool's tool.result still
        fires via the untracked path with `{}` args.
        """
        chunks = [
            _tool_chunk(name="execute", args='{"command": }', chunk_id="call-1"),
            (
                (),
                "messages",
                (
                    ToolMessage(
                        content="ran anyway",
                        tool_call_id="call-1",
                        name="execute",
                        status="success",
                    ),
                    {},
                ),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        events = [c[0][0] for c in mock_dispatch_background.call_args_list]
        assert "tool.use" not in events
        tool_result_calls = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        assert tool_result_calls[0][0][1] == {
            "tool_name": "execute",
            "tool_id": "call-1",
            "tool_args": {},
            "tool_status": "success",
            "tool_output": "ran anyway",
        }

    async def test_unexpected_tool_status_fails_closed_to_error(self) -> None:
        """Interactive parity: an unexpected ToolMessage.status fails closed to error.

        A mounted tool whose terminal ToolMessage carries a status outside the
        `success`/`error` domain must emit `tool.error` and an error
        `tool.result`, never a spurious success.
        """
        chunks = [
            (
                (),
                "messages",
                (_tool_call_message("read_file", {"path": "foo.py"}, "call-1"), {}),
            ),
            (
                (),
                "messages",
                (
                    ToolMessage.model_construct(
                        content="stopped",
                        tool_call_id="call-1",
                        name="read_file",
                        status="cancelled",
                    ),
                    {},
                ),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        events = [c[0][0] for c in mock_dispatch_background.call_args_list]
        assert "tool.error" in events
        tool_result_calls = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        payload = tool_result_calls[0][0][1]
        assert payload["tool_status"] == "error"
        assert payload["tool_name"] == "read_file"
        assert payload["tool_id"] == "call-1"
        assert payload["tool_args"] == {"path": "foo.py"}

    async def test_bare_reject_with_answered_ask_user_emits_single_result(
        self,
    ) -> None:
        """A bare reject beside an answered ask_user emits one execute tool.result.

        The turn still resumes to deliver the ask_user answer, so the middleware's
        synthetic reject ToolMessage streams back for the rejected `execute` call.
        Its terminal hooks already fired at reject time, so the resumed message
        must be deduped by tool id rather than emitting a second tool.result with
        mismatched `{}` args.
        """
        questions: list[Question] = [{"question": "Name?", "type": "text"}]
        action_requests = [{"name": "execute", "args": {"command": "echo hi"}}]
        ask_interrupt = SimpleNamespace(
            id="ask-int",
            value={
                "type": "ask_user",
                "questions": questions,
                "tool_call_id": "ask-1",
            },
        )
        hitl_interrupt = SimpleNamespace(
            id="hitl-int",
            value={
                "action_requests": action_requests,
                "review_configs": [
                    {
                        "action_name": "execute",
                        "allowed_decisions": ["approve", "reject"],
                    }
                ],
            },
        )
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    (
                        (),
                        "messages",
                        (
                            _tool_call_message(
                                "execute", {"command": "echo hi"}, "tool-1"
                            ),
                            {},
                        ),
                    ),
                    ((), "updates", {"__interrupt__": [ask_interrupt, hitl_interrupt]}),
                ],
                [
                    (
                        (),
                        "messages",
                        (
                            ToolMessage(
                                content="Tool approval rejected",
                                tool_call_id="tool-1",
                                status="error",
                            ),
                            {},
                        ),
                    ),
                ],
            ]
        )

        ask_future: asyncio.Future[AskUserWidgetResult] = asyncio.Future()
        ask_future.set_result({"type": "answered", "answers": ["Alice"]})

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return ask_future

        async def request_approval(
            _action_requests: list[dict[str, Any]],
            _assistant_id: str | None,
        ) -> asyncio.Future[object]:
            await asyncio.sleep(0)
            future: asyncio.Future[object] = asyncio.Future()
            future.set_result({"type": "reject"})
            return future

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=request_approval,
            request_ask_user=request_ask_user,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        # The turn resumed (to deliver the ask_user answer), so the synthetic
        # reject ToolMessage was streamed back on the second call.
        assert len(agent.stream_inputs) == 2
        execute_results = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result" and c[0][1]["tool_name"] == "execute"
        ]
        assert len(execute_results) == 1
        assert execute_results[0][0][1] == {
            "tool_name": "execute",
            "tool_id": "tool-1",
            "tool_args": {"command": "echo hi"},
            "tool_status": "error",
            "tool_output": "Tool approval rejected",
        }

    async def test_ask_user_interrupt_cancelled_dispatches_tool_result(self) -> None:
        """A cancelled ask_user emits tool.error and an error tool.result."""
        future: asyncio.Future[AskUserWidgetResult] = asyncio.Future()
        future.set_result({"type": "cancelled"})

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return future

        questions: list[Question] = [{"question": "Name?", "type": "text"}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": questions,
                            "tool_call_id": "ask-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        assert "tool.error" in [
            c[0][0] for c in mock_dispatch_background.call_args_list
        ]
        tool_result_calls = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        payload = tool_result_calls[0][0][1]
        assert payload["tool_name"] == "ask_user"
        assert payload["tool_id"] == "ask-1"
        assert payload["tool_status"] == "error"
        assert payload["tool_output"] == "Question cancelled"

    async def test_ask_user_interrupt_non_list_answers_dispatches_error(self) -> None:
        """A non-list answers payload emits tool.error and an error tool.result."""
        future: asyncio.Future[AskUserWidgetResult] = asyncio.Future()
        # Deliberately malformed (answers must be a list) to drive the error
        # branch; cast past the type checker since that is the whole point.
        future.set_result(
            cast("AskUserWidgetResult", {"type": "answered", "answers": "not-a-list"})
        )

        async def request_ask_user(
            _questions: list[Question],
        ) -> asyncio.Future[AskUserWidgetResult] | None:
            await asyncio.sleep(0)
            return future

        questions: list[Question] = [{"question": "Name?", "type": "text"}]
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": questions,
                            "tool_call_id": "ask-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )

        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook",
                new_callable=AsyncMock,
            ),
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
            ) as mock_dispatch_background,
        ):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        assert "tool.error" in [
            c[0][0] for c in mock_dispatch_background.call_args_list
        ]
        tool_result_calls = [
            c
            for c in mock_dispatch_background.call_args_list
            if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        payload = tool_result_calls[0][0][1]
        assert payload["tool_name"] == "ask_user"
        assert payload["tool_status"] == "error"
        assert payload["tool_output"] == "invalid ask_user answers payload"

    async def test_tool_use_not_dispatched_without_id(self) -> None:
        """tool.use waits for a tool id even when name and args are complete.

        Guards the interactive id gate: a call whose args parse and whose name is
        known must still not fire tool.use until an id arrives, so a later
        tool.result can always be correlated back to it. Mirrors the headless
        `test_tool_use_not_dispatched_without_id`.
        """
        chunks = [
            _tool_chunk(name="read_file", args='{"path": "foo.py"}', chunk_id=None)
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with patch(
            "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
        ) as mock_dispatch:
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        tool_use_calls = [
            c for c in mock_dispatch.call_args_list if c[0][0] == "tool.use"
        ]
        assert not tool_use_calls

    async def test_tool_use_dispatches_when_id_arrives_after_args(self) -> None:
        """Complete args are retained until a later chunk supplies the id."""
        chunks = [
            _tool_chunk(name="read_file", args='{"path": "foo.py"}', chunk_id=None),
            _tool_chunk(name=None, args="", chunk_id="call-1"),
            (
                (),
                "messages",
                (ToolMessage(content="ok", tool_call_id="call-1"), {}),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with patch(
            "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
        ) as mock_dispatch:
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        tool_use_calls = [
            c for c in mock_dispatch.call_args_list if c[0][0] == "tool.use"
        ]
        assert len(tool_use_calls) == 1
        assert tool_use_calls[0][0][1] == {
            "tool_name": "read_file",
            "tool_id": "call-1",
            "tool_args": {"path": "foo.py"},
        }

        tool_result_calls = [
            c for c in mock_dispatch.call_args_list if c[0][0] == "tool.result"
        ]
        assert len(tool_result_calls) == 1
        assert tool_result_calls[0][0][1] == {
            "tool_name": "read_file",
            "tool_id": "call-1",
            "tool_args": {"path": "foo.py"},
            "tool_status": "success",
            "tool_output": "ok",
        }

    async def test_tool_use_not_dispatched_without_name(self) -> None:
        """tool.use must not fire while a streamed call has args + id but no name.

        Mirrors the headless `test_tool_use_not_dispatched_when_no_name`.
        """
        chunks = [_tool_chunk(name=None, args='{"path": "foo.py"}', chunk_id="call-1")]

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with patch(
            "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"
        ) as mock_dispatch:
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        tool_use_calls = [
            c for c in mock_dispatch.call_args_list if c[0][0] == "tool.use"
        ]
        assert not tool_use_calls


# ---------------------------------------------------------------------------
# Cross-surface hook parity
# ---------------------------------------------------------------------------


def _normalize_hook_calls(
    calls: list[tuple[str, dict[str, Any]]],
) -> list[tuple[str, tuple[tuple[str, Any], ...]]]:
    """Turn captured (event, payload) calls into an order-independent form.

    Tool hooks are dispatched fire-and-forget and may be observed in any order
    (the parity contract only guarantees the *set* of events is identical across
    surfaces, not their arrival order), so sort by event + tool id/name and
    freeze each payload into a hashable, comparable tuple.
    """
    # Dict keys are unique, so tuples sort by key alone and the (possibly
    # non-comparable) values are never compared.
    frozen = [(event, tuple(sorted(payload.items()))) for event, payload in calls]
    return sorted(frozen, key=repr)


def _run_headless_surface(
    tool_call_blocks: list[dict[str, Any]],
    tool_message: ToolMessage | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Drive the headless surface and capture its fire-and-forget hook calls.

    Feeds the tool-call blocks through `_process_ai_message` (as one streamed
    `AIMessage`) then the terminal `ToolMessage` through `_process_message_chunk`
    — the same functions the real `-p` runner uses.
    """
    calls: list[tuple[str, dict[str, Any]]] = []

    def _capture(event: str, payload: Any) -> None:  # noqa: ANN401
        calls.append((event, dict(payload)))

    state = StreamState(quiet=True)
    console = Console(quiet=True)
    file_op_tracker = MagicMock()
    file_op_tracker.complete_with_message.return_value = None

    ai_msg = MagicMock(spec=AIMessage)
    ai_msg.content_blocks = tool_call_blocks

    with patch(
        "deepagents_code.client.non_interactive.dispatch_hook_fire_and_forget",
        side_effect=_capture,
    ):
        _process_ai_message(ai_msg, state, console)
        if tool_message is not None:
            _process_message_chunk((tool_message, {}), state, console, file_op_tracker)
    return calls


async def _run_textual_surface(
    stream_chunks: list[tuple[Any, ...]],
) -> list[tuple[str, dict[str, Any]]]:
    """Drive the TUI surface and capture its fire-and-forget hook calls.

    Runs the equivalent `messages` stream through `execute_task_textual` — the
    same entrypoint the interactive REPL uses.
    """
    calls: list[tuple[str, dict[str, Any]]] = []

    def _capture(event: str, payload: Any) -> None:  # noqa: ANN401
        calls.append((event, dict(payload)))

    adapter = TextualUIAdapter(
        mount_message=_mock_mount,
        update_status=_noop_status,
        request_approval=_mock_approval,
    )
    with patch(
        "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget",
        side_effect=_capture,
    ):
        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(stream_chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )
    return calls


class TestCrossSurfaceHookParity:
    """The headless and TUI surfaces emit identical hook payload *sets*.

    The dispatch/gating/correlation layers are implemented separately in each
    surface and kept in sync by hand (see the parity contract in `_tool_stream`).
    These tests feed one scenario through both real surfaces and assert the
    emitted `tool.use`/`tool.result`/`tool.error` payloads match, so a future
    edit to one surface's gating that is not mirrored in the other fails loudly
    rather than shipping green.
    """

    async def test_successful_tool_call_parity(self) -> None:
        """A parsed tool call + success result: tool.use then tool.result, both."""
        headless = _run_headless_surface(
            [
                {
                    "type": "tool_call",
                    "name": "read_file",
                    "id": "call-1",
                    "index": 0,
                    "args": {"path": "foo.py"},
                }
            ],
            ToolMessage(
                content="ok",
                tool_call_id="call-1",
                name="read_file",
                status="success",
            ),
        )
        textual = await _run_textual_surface(
            [
                (
                    (),
                    "messages",
                    (_tool_call_message("read_file", {"path": "foo.py"}, "call-1"), {}),
                ),
                (
                    (),
                    "messages",
                    (
                        ToolMessage(
                            content="ok",
                            tool_call_id="call-1",
                            name="read_file",
                            status="success",
                        ),
                        {},
                    ),
                ),
            ]
        )
        assert _normalize_hook_calls(headless) == _normalize_hook_calls(textual)
        # Sanity-check the shared expectation rather than only that they agree.
        assert _normalize_hook_calls(headless) == _normalize_hook_calls(
            [
                (
                    "tool.use",
                    {
                        "tool_name": "read_file",
                        "tool_id": "call-1",
                        "tool_args": {"path": "foo.py"},
                    },
                ),
                (
                    "tool.result",
                    {
                        "tool_name": "read_file",
                        "tool_id": "call-1",
                        "tool_args": {"path": "foo.py"},
                        "tool_status": "success",
                        "tool_output": "ok",
                    },
                ),
            ]
        )

    async def test_structured_content_output_parity(self) -> None:
        """List/structured tool output formats identically across surfaces.

        `tool_output` for multimodal / MCP content-block results must run through
        the same formatter on both surfaces; a regression to a raw `str(list)`
        repr on either would break this parity (and this scenario is the one the
        four scalar-content cases can't catch).
        """
        from deepagents_code.tool_display import format_tool_message_content

        content: list[Any] = [
            {"type": "text", "text": "line one"},
            {"type": "text", "text": "line two"},
        ]
        headless = _run_headless_surface(
            [
                {
                    "type": "tool_call",
                    "name": "read_file",
                    "id": "call-1",
                    "index": 0,
                    "args": {"path": "foo.py"},
                }
            ],
            ToolMessage(
                content=content,
                tool_call_id="call-1",
                name="read_file",
                status="success",
            ),
        )
        textual = await _run_textual_surface(
            [
                (
                    (),
                    "messages",
                    (_tool_call_message("read_file", {"path": "foo.py"}, "call-1"), {}),
                ),
                (
                    (),
                    "messages",
                    (
                        ToolMessage(
                            content=content,
                            tool_call_id="call-1",
                            name="read_file",
                            status="success",
                        ),
                        {},
                    ),
                ),
            ]
        )
        assert _normalize_hook_calls(headless) == _normalize_hook_calls(textual)
        # Guard the specific divergence risk: the formatted output, not a list
        # repr. Both surfaces must equal the shared formatter's result.
        expected = format_tool_message_content(content)
        result_output = next(
            payload["tool_output"]
            for event, payload in headless
            if event == "tool.result"
        )
        assert result_output == expected
        assert result_output != str(content)

    async def test_errored_tool_call_parity(self) -> None:
        """An error result co-fires tool.error alongside tool.result on both."""
        blocks = [
            {
                "type": "tool_call",
                "name": "run_shell",
                "id": "call-9",
                "index": 0,
                "args": {"command": "false"},
            }
        ]
        headless = _run_headless_surface(
            blocks,
            ToolMessage(
                content="boom",
                tool_call_id="call-9",
                name="run_shell",
                status="error",
            ),
        )
        textual = await _run_textual_surface(
            [
                (
                    (),
                    "messages",
                    (
                        _tool_call_message("run_shell", {"command": "false"}, "call-9"),
                        {},
                    ),
                ),
                (
                    (),
                    "messages",
                    (
                        ToolMessage(
                            content="boom",
                            tool_call_id="call-9",
                            name="run_shell",
                            status="error",
                        ),
                        {},
                    ),
                ),
            ]
        )
        assert _normalize_hook_calls(headless) == _normalize_hook_calls(textual)
        events = sorted(event for event, _ in headless)
        assert events == ["tool.error", "tool.result", "tool.use"]

    async def test_unparsed_args_result_parity(self) -> None:
        """Args that never parse: no tool.use, tool.result with empty args, both."""
        headless = _run_headless_surface(
            [
                {
                    "type": "tool_call_chunk",
                    "name": "read_file",
                    "id": "call-2",
                    "index": 0,
                    # Never closes, so the args never parse and no tool.use fires.
                    "args": '{"path": ',
                }
            ],
            ToolMessage(
                content="ok",
                tool_call_id="call-2",
                name="read_file",
                status="success",
            ),
        )
        textual = await _run_textual_surface(
            [
                _tool_chunk(name="read_file", args='{"path": ', chunk_id="call-2"),
                (
                    (),
                    "messages",
                    (
                        ToolMessage(
                            content="ok",
                            tool_call_id="call-2",
                            name="read_file",
                            status="success",
                        ),
                        {},
                    ),
                ),
            ]
        )
        assert _normalize_hook_calls(headless) == _normalize_hook_calls(textual)
        assert _normalize_hook_calls(headless) == _normalize_hook_calls(
            [
                (
                    "tool.result",
                    {
                        "tool_name": "read_file",
                        "tool_id": "call-2",
                        "tool_args": {},
                        "tool_status": "success",
                        "tool_output": "ok",
                    },
                ),
            ]
        )

    async def test_idless_tool_call_parity(self) -> None:
        """An empty (uncorrelatable) tool-call id: no tool.use, bare tool.result.

        A real `ToolMessage.tool_call_id` is always a string, so the "no usable
        id" case is exercised with an empty string — falsy on both surfaces, so
        neither correlates it and neither fires a `tool.use`.
        """
        headless = _run_headless_surface(
            [
                {
                    "type": "tool_call",
                    "name": "noop",
                    "id": "",
                    "index": 0,
                    "args": {"x": 1},
                }
            ],
            ToolMessage(content="done", tool_call_id="", name="noop"),
        )
        textual = await _run_textual_surface(
            [
                (
                    (),
                    "messages",
                    (_tool_call_message("noop", {"x": 1}, ""), {}),
                ),
                (
                    (),
                    "messages",
                    (ToolMessage(content="done", tool_call_id="", name="noop"), {}),
                ),
            ]
        )
        assert _normalize_hook_calls(headless) == _normalize_hook_calls(textual)
        assert _normalize_hook_calls(headless) == _normalize_hook_calls(
            [
                (
                    "tool.result",
                    {
                        "tool_name": "noop",
                        "tool_id": "",
                        "tool_args": {},
                        "tool_status": "success",
                        "tool_output": "done",
                    },
                ),
            ]
        )


class TestTextualEndOfStreamDiagnostics:
    """The TUI logs both end-of-stream diagnostics for unemitted buffers."""

    async def test_logs_unparsed_and_idless_buffers_at_stream_end(self, caplog) -> None:
        """Buffers that never mounted are classified into the two log lines.

        Drives the real `execute_task_textual` to a clean stream end with two
        buffers left behind: one whose args never parse (with an id) and one
        whose args parse but carry no id. Both never mount a widget or fire a
        `tool.use`, so they survive to the diagnostic block. Pins that the shared
        `count_unemitted_tool_calls` counts are wired to the correct TUI log
        lines — a swapped count, deleted branch, or garbled message fails here,
        which the helper-level unit test cannot catch. Distinct chunk indices
        keep the two fragments in separate buffers.
        """
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        chunks = [
            # Args never close -> unparsed, no tool.use, buffer retained.
            _tool_chunk(name="f", args='{"a": ', chunk_id="t1", index=0),
            # Args parse but no id -> tool.use gated out, buffer retained.
            _tool_chunk(name="g", args='{"b": 2}', chunk_id=None, index=1),
        ]
        with (
            patch("deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"),
            caplog.at_level("INFO", logger="deepagents_code.tui.textual_adapter"),
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        assert any("arguments never parsed" in r.message for r in caplog.records)
        assert any("carried no tool-call id" in r.message for r in caplog.records)

    async def test_logs_unemitted_buffer_on_midstream_error(self, caplog) -> None:
        """The diagnostic fires when the stream errors, not only on a clean end.

        The diagnostic lives in `execute_task_textual`'s `finally`, so a
        non-cancel mid-stream error (which skips the clean-end `else` branch)
        must still surface a buffered call whose args never parsed. Regression
        guard for the parity gap where this diagnostic previously ran only on the
        clean-end path and vanished on cancel/error.
        """
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        # Args never close -> unparsed, no tool.use, buffer retained to the exit.
        chunks = [_tool_chunk(name="f", args='{"a": ', chunk_id="t1", index=0)]
        with (
            patch("deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"),
            caplog.at_level("INFO", logger="deepagents_code.tui.textual_adapter"),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_RaisingAgent(chunks, RuntimeError("boom")),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        assert any("arguments never parsed" in r.message for r in caplog.records)

    async def test_logs_unemitted_buffer_on_cancel(self, caplog) -> None:
        """The diagnostic fires on a cancelled turn too (the other non-clean exit).

        Interrupt cleanup is patched out to isolate the assertion to the
        `finally` block: a `CancelledError` must still route through the diagnostic
        that was moved out of the clean-end branch.
        """
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        chunks = [_tool_chunk(name="f", args='{"a": ', chunk_id="t1", index=0)]
        cleanup = AsyncMock()
        with (
            patch("deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget"),
            patch(
                "deepagents_code.tui.textual_adapter._handle_interrupt_cleanup",
                cleanup,
            ),
            caplog.at_level("INFO", logger="deepagents_code.tui.textual_adapter"),
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_RaisingAgent(chunks, asyncio.CancelledError()),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        assert any("arguments never parsed" in r.message for r in caplog.records)
        assert cleanup.await_args is not None
        assert cleanup.await_args.kwargs["recover_interrupted_turn"] is True

    async def test_criteria_cancel_disables_chat_interruption_recovery(self) -> None:
        """Criteria graph input selects operation cleanup, not chat recovery."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        cleanup = AsyncMock()
        request = {
            "messages": [],
            "goal_criteria_request": {
                "request_id": "request-cancel",
                "kind": "create",
                "objective": "ship it",
            },
        }

        with patch(
            "deepagents_code.tui.textual_adapter._handle_interrupt_cleanup",
            cleanup,
        ):
            await execute_task_textual(
                user_input="",
                agent=_RaisingAgent([], asyncio.CancelledError()),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
                graph_input=request,
            )

        assert cleanup.await_args is not None
        assert cleanup.await_args.kwargs["recover_interrupted_turn"] is False


class TestTextualNonCleanExitTerminalHooks:
    """A non-cancel mid-stream error terminates pending `tool.use` hooks itself.

    The "every `tool.use` is closed by a terminal event" guarantee must be owned
    by `execute_task_textual` rather than depending on the caller's
    `finalize_pending_tools_with_error`. These drive `execute_task_textual`
    directly (no `app.py` caller), so a terminal `tool.result`/`tool.error` for a
    pending tool can only come from the surface's own `finally` backstop.
    """

    async def test_midstream_error_closes_pending_tool_use(self) -> None:
        """A stream error emits terminal hooks for a tool whose `tool.use` fired."""
        dispatched: list[tuple[str, dict[str, Any]]] = []

        def record_dispatch(event: str, payload: dict[str, Any]) -> None:
            dispatched.append((event, payload))

        chunks = [
            (
                (),
                "messages",
                (_tool_call_message("read_file", {"path": "f.py"}, "call-1"), {}),
            ),
        ]
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        with (
            patch(
                "deepagents_code.tui.textual_adapter.dispatch_hook_fire_and_forget",
                side_effect=record_dispatch,
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await execute_task_textual(
                user_input="hello",
                agent=_RaisingAgent(chunks, RuntimeError("boom")),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
                adapter=adapter,
            )

        events = [event for event, _ in dispatched]
        assert "tool.use" in events
        result_payloads = [p for event, p in dispatched if event == "tool.result"]
        assert any(
            p["tool_id"] == "call-1" and p["tool_status"] == "error"
            for p in result_payloads
        )
        error_payloads = [p for event, p in dispatched if event == "tool.error"]
        assert any("read_file" in p["tool_names"] for p in error_payloads)
        # The backstop terminated and cleared the pending widget, so a later
        # caller's finalize would find nothing to dispatch (no double tool.result).
        assert adapter._current_tool_messages == {}


# ---------------------------------------------------------------------------
# _read_mentioned_file inline embedding
# ---------------------------------------------------------------------------


class TestReadMentionedFile:
    """Tests for `_read_mentioned_file` inline embedding."""

    def test_embeds_small_file_in_text_fence(self, tmp_path: Path) -> None:
        """A small mentioned file is embedded in a ```text fenced block."""
        target = tmp_path / "note.txt"
        target.write_text("alpha\nbeta", encoding="utf-8")

        snippet = _read_mentioned_file(target, max_embed_bytes=1024)

        assert "```text\nalpha\nbeta\n```" in snippet
        assert f"Path: `{target}`" in snippet

    def test_oversized_file_returns_reference_without_fence(
        self, tmp_path: Path
    ) -> None:
        """A file over the embed threshold is referenced, not fenced."""
        target = tmp_path / "big.txt"
        target.write_text("x" * 4096, encoding="utf-8")

        snippet = _read_mentioned_file(target, max_embed_bytes=1024)

        assert "too large to embed" in snippet
        assert "```" not in snippet


# ---------------------------------------------------------------------------
# Rubric custom-stream events (textual path)
# ---------------------------------------------------------------------------


class TestExecuteTaskTextualRubricEvents:
    """Rubric custom-stream events surface only for the main agent."""

    async def test_main_agent_rubric_event_mounts_message(self) -> None:
        """A main-agent rubric verdict is rendered in the transcript."""
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        # (namespace, stream_mode, data); empty namespace == main agent.
        chunks = [
            ((), "custom", {"type": "rubric_evaluation_end", "result": "satisfied"}),
        ]
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        await execute_task_textual(
            user_input="hi",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        rubric_msgs = [
            m
            for m in mounted
            if isinstance(m, AppMessage)
            and "Acceptance criteria satisfied" in str(m._content)
        ]
        assert len(rubric_msgs) == 1

    async def test_subagent_rubric_event_is_not_mounted(self) -> None:
        """A rubric event from a subagent namespace must not reach the transcript."""
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        # Non-empty namespace == subagent; the is_main_agent gate suppresses it.
        chunks = [
            (
                ("subagent",),
                "custom",
                {"type": "rubric_evaluation_end", "result": "satisfied"},
            ),
        ]
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        await execute_task_textual(
            user_input="hi",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert not [
            m
            for m in mounted
            if isinstance(m, AppMessage) and "Rubric" in str(m._content)
        ]
