"""Unit tests for textual_adapter functions."""

import asyncio
import sys
from asyncio import Future
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from datetime import datetime
from io import StringIO
from pathlib import Path
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
from deepagents_code.approval_mode import APPROVAL_MODE_NAMESPACE, approval_mode_key
from deepagents_code.config import ASCII_GLYPHS, UNICODE_GLYPHS, build_stream_config
from deepagents_code.textual_adapter import (
    ModelStats,
    SessionStats,
    TextualUIAdapter,
    _build_interrupted_ai_message,
    _format_rubric_event,
    _handle_interrupt_cleanup,
    _is_summarization_chunk,
    _read_mentioned_file,
    execute_task_textual,
    format_token_count,
    print_usage_table,
)
from deepagents_code.widgets.messages import (
    AppMessage,
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
            "deepagents_code.textual_adapter.time.monotonic", return_value=101.0
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
        from deepagents_code.remote_client import RemoteAgent

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
        assert metadata["ls_agent_kind"] == "coding_agent"
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
            "deepagents_code.textual_adapter.get_glyphs",
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

    def test_needs_revision_includes_failed_criteria(self) -> None:
        """Failed criteria should be shown with actionable gaps."""
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
            == "↻ Changes need revision: missing coverage\n  ✗ tests pass — not run"
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
        """Hitting the iteration cap should warn the user it is unsatisfied."""
        assert (
            _format_rubric_event(
                {"type": "rubric_evaluation_end", "result": "max_iterations_reached"},
            )
            == "⚠ Acceptance criteria not satisfied (iteration limit reached)"
        )

    def test_grader_failure_results_render_warning(self) -> None:
        """Grader failures should surface as warnings with the explanation."""
        assert (
            _format_rubric_event(
                {
                    "type": "rubric_evaluation_end",
                    "result": "failed",
                    "explanation": "timeout",
                },
            )
            == "⚠ Rubric grader failed: timeout"
        )
        assert (
            _format_rubric_event(
                {"type": "rubric_evaluation_end", "result": "grader_error"},
            )
            == "⚠ Rubric grader error"
        )

    def test_unknown_terminal_result_renders_fallback(self) -> None:
        """An unrecognized terminal result must not be silently dropped."""
        assert (
            _format_rubric_event(
                {"type": "rubric_evaluation_end", "result": "something_new"},
            )
            == "⚠ Rubric grading ended"
        )

    def test_end_event_without_result_returns_none(self) -> None:
        """Partial end events should not render a spurious warning."""
        assert _format_rubric_event({"type": "rubric_evaluation_end"}) is None

    def test_unrelated_event_returns_none(self) -> None:
        """Only rubric events should render rubric messages."""
        assert _format_rubric_event({"type": "subagent_start"}) is None

    def test_ascii_mode_degrades_to_ascii_glyphs(self) -> None:
        """In ASCII mode the transcript glyphs must degrade, not stay Unicode."""
        with patch(
            "deepagents_code.textual_adapter.get_glyphs",
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
            f"{ASCII_GLYPHS.retry} Changes need revision\n"
            f"  {ASCII_GLYPHS.error} tests pass — not run"
        )
        assert satisfied == f"{ASCII_GLYPHS.checkmark} Acceptance criteria satisfied"
        assert failed == f"{ASCII_GLYPHS.warning} Rubric grader failed"


class _FakeAgent:
    """Minimal async stream agent used for adapter execution tests."""

    def __init__(self, chunks: list[tuple]) -> None:
        self._chunks = chunks

    async def astream(self, *_: Any, **__: Any) -> AsyncIterator[tuple[Any, ...]]:
        """Yield preconfigured stream chunks."""
        for chunk in self._chunks:
            yield chunk


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
        assert stream_input == {"messages": [{"role": "user", "content": "hi"}]}
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
            "messages": [{"role": "user", "content": "continue now"}]
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


def _server_tool_message(blocks: list[dict[str, Any]]) -> tuple[Any, ...]:
    """Build a message chunk carrying provider-side tool call blocks."""
    from langchain_core.messages import AIMessage

    content = cast("list[str | dict[Any, Any]]", blocks)
    return ((), "messages", (AIMessage(content=content), {}))


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

    async def test_server_tool_call_mounts_and_completes(self) -> None:
        """Hosted provider tool calls should render like local tool rows."""
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted.append(widget)

        chunks = [
            _server_tool_message(
                [
                    {
                        "type": "server_tool_call",
                        "name": "web_search",
                        "id": "ws_1",
                        "args": {
                            "type": "search",
                            "query": "fun things to do in nyc",
                            "queries": [
                                "fun things to do in nyc",
                                "nyc events this weekend",
                            ],
                        },
                    },
                    {
                        "type": "server_tool_result",
                        "tool_call_id": "ws_1",
                        "status": "success",
                    },
                ]
            )
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
        assert tool_msgs[0]._tool_name == "web_search"
        assert tool_msgs[0]._args["query"] == "fun things to do in nyc"
        assert tool_msgs[0]._status == "success"

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


class TestExecuteTaskTextualParallelToolSpinner:
    """Regression tests for #1796: premature spinner with parallel tools."""

    async def test_spinner_not_shown_until_all_parallel_tools_complete(self) -> None:
        """With two parallel tools, Thinking appears only at start and after last."""
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
        thinking_count = sum(1 for s in statuses if s == "Thinking")
        assert thinking_count == 2, (
            "Expected exactly 2 Thinking calls (start + after last tool); "
            f"got {thinking_count}: {statuses}"
        )

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

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
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

        tool_msg = adapter._current_tool_messages["tool-1"]
        assert tool_msg._status == "running"

    async def test_edit_file_does_not_get_per_tool_spinner_at_mount(self) -> None:
        """`edit_file` relies on the global Thinking spinner, not a per-tool one.

        Negative counterpart to the auto-executed case: tools in
        `_TOOL_CALLS_KEEP_THINKING_SPINNER` must NOT be flipped to "running" at
        mount, or they would show a duplicate spinner alongside "Thinking".
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

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
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

        tool_msg = adapter._current_tool_messages["tool-1"]
        assert tool_msg._status != "running"

    async def test_spinner_with_three_parallel_tools_out_of_order(self) -> None:
        """Three parallel tools completed out of order; Thinking after all."""
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

        thinking_count = sum(1 for s in statuses if s == "Thinking")
        assert thinking_count == 2, (
            "Expected exactly 2 Thinking calls (start + after last tool); "
            f"got {thinking_count}: {statuses}"
        )

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
            "deepagents_code.textual_adapter.AssistantMessage", return_value=fake_msg
        ):
            await execute_task_textual(
                user_input="hi",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
                adapter=adapter,
            )

        # Expected sequence:
        #   1. "Thinking" before astream
        #   2. "Thinking" after mounting the streaming AssistantMessage
        #      (re-anchor the spinner below the message so the user still
        #      sees activity if the model pauses before the tool call)
        #   3. None when the tool call mounts
        #   4. "Thinking" after the tool result
        assert statuses[0] == "Thinking"
        assert statuses[1] == "Thinking"
        assert None in statuses
        assert statuses[-1] == "Thinking"

        # The spinner must never be hidden before the tool call arrives.
        first_none = statuses.index(None)
        text_thinking_seen = statuses[:first_none].count("Thinking") >= 2
        assert text_thinking_seen, (
            f"Spinner was hidden during text streaming before tool call: {statuses}"
        )

    async def test_spinner_reanchors_for_text_after_tool_cycle(self) -> None:
        """Text -> tool_call -> tool_result -> text must re-anchor the spinner.

        After a tool cycle completes, the tool_call handler pops the previous
        AssistantMessage from `assistant_message_by_namespace`, so the next
        text chunk mounts a fresh widget. The new re-anchor call at
        `textual_adapter.py:780-784` must fire for that second text burst so
        the spinner stays visible between it and any follow-up tool call.
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
            "deepagents_code.textual_adapter.AssistantMessage", return_value=fake_msg
        ):
            await execute_task_textual(
                user_input="hi",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
                adapter=adapter,
            )

        # Expected Thinking calls:
        #   1. Before astream (line 517)
        #   2. After first AssistantMessage mount (re-anchor, line 784)
        #   3. After tool result (line 705)
        #   4. After second AssistantMessage mount (re-anchor again)
        thinking_count = sum(1 for s in statuses if s == "Thinking")
        assert thinking_count >= 4, (
            f"Expected at least 4 Thinking calls including re-anchors after "
            f"each text mount; got {thinking_count}: {statuses}"
        )

    async def test_spinner_reanchor_skipped_while_tools_pending(self) -> None:
        """The re-anchor must be gated on `not _current_tool_messages`.

        Contrived sequence: a tool call mounts (populating
        `_current_tool_messages`), then a text chunk arrives before the tool
        result. The new re-anchor logic must NOT call `_set_spinner("Thinking")`
        in that window — the tool-call widget is the dominant progress
        indicator.
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
            "deepagents_code.textual_adapter.AssistantMessage", return_value=fake_msg
        ):
            await execute_task_textual(
                user_input="hi",
                agent=_FakeAgent(chunks),
                assistant_id="assistant",
                session_state=SimpleNamespace(thread_id="thread-1", auto_approve=True),
                adapter=adapter,
            )

        # Thinking calls should be:
        #   1. Before astream
        #   2. After tool result (guard is back to empty)
        # The re-anchor must NOT fire while the tool is in flight.
        thinking_count = sum(1 for s in statuses if s == "Thinking")
        assert thinking_count == 2, (
            f"Expected 2 Thinking calls (start + after tool); got "
            f"{thinking_count}: {statuses}"
        )


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
                "deepagents_code.textual_adapter.AssistantMessage",
                side_effect=FakeAssistantMessage,
            ),
            patch(
                "deepagents_code.textual_adapter.get_glyphs",
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
        app_text = [str(widget._content) for widget in app_messages]
        assert app_text == [
            (
                f"{UNICODE_GLYPHS.hourglass} Checking acceptance criteria"
                f"{UNICODE_GLYPHS.ellipsis}"
            ),
            f"{UNICODE_GLYPHS.retry} Changes need revision: say yellow",
        ]


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
# SessionStats tests
# ---------------------------------------------------------------------------


class TestSessionStats:
    """Tests for `SessionStats` recording and merging."""

    def test_record_request_named_model(self) -> None:
        """record_request updates totals and per_model for a named model."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)

        assert stats.request_count == 1
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert ("", "gpt-4") in stats.per_model
        assert stats.per_model["", "gpt-4"].request_count == 1
        assert stats.per_model["", "gpt-4"].input_tokens == 100
        assert stats.per_model["", "gpt-4"].output_tokens == 50

    def test_record_request_empty_model(self) -> None:
        """record_request with empty model skips per_model entry."""
        stats = SessionStats()
        stats.record_request("", 200, 80)

        assert stats.request_count == 1
        assert stats.input_tokens == 200
        assert stats.output_tokens == 80
        assert stats.per_model == {}

    def test_record_request_multiple_models(self) -> None:
        """Multiple models create separate per_model entries."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        stats.record_request("claude-opus-4-6", 200, 80)

        assert stats.request_count == 2
        assert stats.input_tokens == 300
        assert stats.output_tokens == 130
        assert len(stats.per_model) == 2
        assert stats.per_model["", "gpt-4"].request_count == 1
        assert stats.per_model["", "claude-opus-4-6"].request_count == 1

    def test_record_request_splits_same_model_by_provider(self) -> None:
        """Provider-specific model names should not collapse into one entry."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50, provider="openai")
        stats.record_request("gpt-4", 200, 80, provider="azure")

        assert len(stats.per_model) == 2
        assert stats.per_model["openai", "gpt-4"].request_count == 1
        assert stats.per_model["azure", "gpt-4"].request_count == 1

    def test_merge(self) -> None:
        """merge() folds another SessionStats into self."""
        a = SessionStats(
            request_count=1, input_tokens=100, output_tokens=50, wall_time_seconds=1.0
        )
        a.per_model["", "gpt-4"] = ModelStats(
            request_count=1, input_tokens=100, output_tokens=50, model_name="gpt-4"
        )

        b = SessionStats(
            request_count=2, input_tokens=300, output_tokens=120, wall_time_seconds=2.5
        )
        b.per_model["", "claude-opus-4-6"] = ModelStats(
            request_count=2,
            input_tokens=300,
            output_tokens=120,
            model_name="claude-opus-4-6",
        )

        a.merge(b)

        assert a.request_count == 3
        assert a.input_tokens == 400
        assert a.output_tokens == 170
        assert a.wall_time_seconds == pytest.approx(3.5)
        assert len(a.per_model) == 2
        assert a.per_model["", "claude-opus-4-6"].request_count == 2

    def test_merge_overlapping_models(self) -> None:
        """merge() combines per_model entries for the same model."""
        a = SessionStats()
        a.record_request("gpt-4", 100, 50)

        b = SessionStats()
        b.record_request("gpt-4", 200, 80)

        a.merge(b)

        assert a.request_count == 2
        assert a.input_tokens == 300
        assert a.output_tokens == 130
        assert a.per_model["", "gpt-4"].request_count == 2
        assert a.per_model["", "gpt-4"].input_tokens == 300
        assert a.per_model["", "gpt-4"].output_tokens == 130

    def test_merge_splits_same_model_by_provider(self) -> None:
        """merge() preserves provider-specific entries for the same model."""
        a = SessionStats()
        a.record_request("gpt-4", 100, 50, provider="openai")

        b = SessionStats()
        b.record_request("gpt-4", 200, 80, provider="azure")

        a.merge(b)

        assert len(a.per_model) == 2
        assert a.per_model["openai", "gpt-4"].input_tokens == 100
        assert a.per_model["azure", "gpt-4"].input_tokens == 200


# ---------------------------------------------------------------------------
# format_token_count tests
# ---------------------------------------------------------------------------


class TestFormatTokenCount:
    """Tests for `format_token_count` shared formatter."""

    def test_small_count(self) -> None:
        assert format_token_count(500) == "500"

    def test_thousands(self) -> None:
        assert format_token_count(12_500) == "12.5K"

    def test_millions(self) -> None:
        assert format_token_count(1_200_000) == "1.2M"

    def test_exact_thousand(self) -> None:
        assert format_token_count(1000) == "1.0K"

    def test_zero(self) -> None:
        assert format_token_count(0) == "0"


# ---------------------------------------------------------------------------
# print_usage_table tests
# ---------------------------------------------------------------------------


class TestPrintUsageTable:
    """Tests for `print_usage_table` output."""

    def test_no_model_called_skips_unknown_row(self) -> None:
        """When no model was called, the table should not show 'unknown'."""
        stats = SessionStats()
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=1.5, console=console)
        output = buf.getvalue()
        assert "unknown" not in output
        assert "Usage Stats" not in output
        assert "Agent active" in output

    def test_single_model_shows_name(self) -> None:
        """Single-model session should display the model name."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "gpt-4" in output
        assert "unknown" not in output

    def test_shows_provider_name(self) -> None:
        """The table should include the provider for each model."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50, provider="openai")
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "Provider" in output
        assert "openai" in output
        assert "gpt-4" in output

    def test_multi_model_shows_all_names_and_total(self) -> None:
        """Multi-model session should show each model and a Total row."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        stats.record_request("claude-opus-4-6", 200, 80)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "gpt-4" in output
        assert "claude-opus-4-6" in output
        assert "Total" in output
        assert "unknown" not in output

    def test_same_model_with_different_providers_shows_separate_rows(self) -> None:
        """Same-name models from different providers should render separately."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50, provider="openai")
        stats.record_request("gpt-4", 200, 80, provider="azure")
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "openai" in output
        assert "azure" in output
        assert "Total" in output
        # Two distinct rows, not a collapsed one: each provider's per-row token
        # counts must appear (100/50 and 200/80), alongside the 300/130 totals.
        assert "100" in output
        assert "50" in output
        assert "200" in output
        assert "80" in output

    def test_tokens_with_no_wall_time_omits_timing_line(self) -> None:
        """Token table should print but timing line should be absent."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=0.0, console=console)
        output = buf.getvalue()
        assert "gpt-4" in output
        assert "Agent active" not in output

    def test_no_requests_no_time_prints_nothing(self) -> None:
        """Empty stats with negligible wall time should print nothing."""
        stats = SessionStats()
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=0.01, console=console)
        output = buf.getvalue()
        assert output.strip() == ""


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
