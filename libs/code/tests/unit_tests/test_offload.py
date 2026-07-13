"""Unit tests for /offload slash command."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import BaseMessage

from deepagents_code._session_stats import format_token_count
from deepagents_code.app import DeepAgentsApp
from deepagents_code.command_registry import SLASH_COMMANDS
from deepagents_code.offload import (
    OffloadModelError,
    OffloadResult,
    OffloadThresholdNotMet,
    _fallback_offload_backend,
    _offload_fallback_root,
    format_offload_limit,
    offload_messages_to_backend,
)
from deepagents_code.tui.widgets.messages import AppMessage, ErrorMessage

# Patch targets for lower-level offload_messages_to_backend tests
_GET_BUFFER_STRING_PATH = "deepagents_code.offload.get_buffer_string"


def _make_messages(n: int) -> list[MagicMock]:
    """Create a list of mock messages with unique IDs."""
    messages = []
    for i in range(n):
        msg = MagicMock()
        msg.id = f"msg-{i}"
        msg.content = f"Message {i}"
        msg.additional_kwargs = {}
        messages.append(msg)
    return messages


def _make_dict_messages(n: int) -> list[dict[str, Any]]:
    """Create serialized message payloads matching remote state snapshots."""
    messages: list[dict[str, Any]] = []
    for i in range(n):
        message_type = "human" if i % 2 == 0 else "ai"
        payload: dict[str, Any] = {
            "content": f"Message {i}",
            "additional_kwargs": {},
            "response_metadata": {},
            "type": message_type,
            "name": None,
            "id": f"msg-{i}",
        }
        if message_type == "ai":
            payload["tool_calls"] = []
        messages.append(payload)
    return messages


def _make_dict_summary_message() -> dict[str, Any]:
    """Create a serialized summary message payload from remote state."""
    return {
        "content": "Old summary.",
        "additional_kwargs": {"lc_source": "summarization"},
        "response_metadata": {},
        "type": "human",
        "name": None,
        "id": "summary-1",
    }


def _summary_event(
    cutoff: int, *, file_path: str | None = "/conversation_history/test-thread.md"
) -> dict[str, Any]:
    """Build a persisted `_summarization_event` mapping for server-state tests."""
    return {
        "cutoff_index": cutoff,
        "summary_message": _make_dict_summary_message(),
        "file_path": file_path,
    }


def _state_values(
    messages: list[Any], event: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a thread state-values dict (as returned by _get_thread_state_values)."""
    values: dict[str, Any] = {"messages": messages}
    if event is not None:
        values["_summarization_event"] = event
    return values


def _setup_server_offload_app(app: DeepAgentsApp) -> None:
    """Configure a `DeepAgentsApp` for server-side offload unit tests.

    The server-side path reads state via `_get_thread_state_values` and drives
    the tool via `_drive_server_side_compaction`; tests patch those seams
    directly, so only the plain identity/flags are set here.
    """
    app._agent = MagicMock()
    app._backend = None
    app._lc_thread_id = "test-thread"
    app._agent_running = False


class TestOffloadInAutocomplete:
    """Verify /offload is registered in the autocomplete system."""

    def test_offload_in_slash_commands(self) -> None:
        """The /offload command should be in the SLASH_COMMANDS list."""
        labels = [entry.name for entry in SLASH_COMMANDS]
        assert "/offload" in labels

    def test_offload_sorted_alphabetically(self) -> None:
        """The /offload entry should appear between /model and /quit."""
        labels = [entry.name for entry in SLASH_COMMANDS]
        model_idx = labels.index("/model")
        offload_idx = labels.index("/offload")
        quit_idx = labels.index("/quit")
        assert model_idx < offload_idx < quit_idx


class TestOffloadGuards:
    """Test guard conditions that prevent offloading."""

    async def test_no_agent_shows_error(self) -> None:
        """Should show error when there is no active agent."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = None
            app._lc_thread_id = None

            await app._handle_offload()
            await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("Nothing to offload" in str(w._content) for w in msgs)

    async def test_agent_running_shows_error(self) -> None:
        """Should show error when agent is currently running."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._backend = MagicMock()
            app._lc_thread_id = "test-thread"
            app._agent_running = True

            await app._handle_offload()
            await pilot.pause()

            msgs = app.query(AppMessage)
            assert any(
                "Cannot offload while agent is running" in str(w._content) for w in msgs
            )

    async def test_nothing_to_compact_noop(self) -> None:
        """Show a no-op message when server-side compaction changed nothing.

        With `force=True` the eligibility gate is bypassed, so the only no-op
        left is "cutoff == 0" — the persisted event is unchanged.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(3))
            after = _state_values(_make_dict_messages(3))

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert any(
                "the conversation is already compact" in str(w._content) for w in msgs
            )

    async def test_empty_state_shows_error(self) -> None:
        """Should show error when state has no values."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._backend = MagicMock()
            app._lc_thread_id = "test-thread"
            app._agent_running = False

            mock_state = MagicMock()
            mock_state.values = {}
            app._agent.aget_state = AsyncMock(return_value=mock_state)

            await app._handle_offload()
            await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("Nothing to offload" in str(w._content) for w in msgs)

    async def test_state_read_failure_shows_error(self) -> None:
        """Should show error when reading state raises an exception."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._backend = MagicMock()
            app._lc_thread_id = "test-thread"
            app._agent_running = False

            app._agent.aget_state = AsyncMock(
                side_effect=RuntimeError("connection lost")
            )

            await app._handle_offload()
            await pilot.pause()

            msgs = app.query(ErrorMessage)
            assert any("Failed to read state" in str(w._content) for w in msgs)


class TestOffloadSuccess:
    """Test successful offload flow."""

    async def test_successful_offload_drives_server_tool(self) -> None:
        """Should trigger server-side compaction and render persisted state."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))
            after = _state_values(
                _make_dict_messages(12),
                _summary_event(6),
            )

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    return_value=None,
                ) as mock_drive,
            ):
                await app._handle_offload()
                await pilot.pause()

            # The client drives the server-side tool exactly once and never
            # writes `_summarization_event` itself — the tool owns that write.
            mock_drive.assert_awaited_once()

            msgs = app.query(AppMessage)
            # Offloaded count is the new cutoff of six minus a prior cutoff of zero.
            assert any("Offloaded 6 older messages" in str(w._content) for w in msgs)

    async def test_offload_shows_feedback_message(self) -> None:
        """Should display feedback with message count and token change."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))
            after = _state_values(_make_dict_messages(12), _summary_event(4))

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            msgs = app.query(AppMessage)
            # Offloaded count is the new cutoff of four minus a prior cutoff of zero.
            assert any("Offloaded 4 older messages" in str(w._content) for w in msgs)
            # Kept count is the ten before-messages minus the new cutoff of four.
            assert any("6 messages kept" in str(w._content) for w in msgs)

    async def test_offload_updates_context_tokens(self) -> None:
        """Should update `_context_tokens` to the post-compaction count."""
        from langchain_core.messages.utils import count_tokens_approximately

        from deepagents_code.app import _effective_conversation

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            after_messages = _make_dict_messages(12)
            after_event = _summary_event(4)
            before = _state_values(_make_dict_messages(10))
            after = _state_values(after_messages, after_event)

            expected = count_tokens_approximately(
                _effective_conversation(after_messages, after_event)
            )

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            assert app._context_tokens == expected

    async def test_no_ui_clear_reload(self) -> None:
        """Should NOT clear/reload UI since messages stay in state."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))
            after = _state_values(_make_dict_messages(12), _summary_event(4))

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch.object(
                    app, "_clear_messages", new_callable=AsyncMock
                ) as mock_clear,
                patch.object(
                    app, "_load_thread_history", new_callable=AsyncMock
                ) as mock_load,
            ):
                await app._handle_offload()
                await pilot.pause()

            mock_clear.assert_not_called()
            mock_load.assert_not_called()


class TestOffloadEdgeCases:
    """Test edge cases in the offload logic."""

    async def test_noop_does_not_report_offloaded(self) -> None:
        """A no-op (event unchanged) shows the no-op message, not a success."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            # Prior event present; after-state cutoff unchanged -> nothing moved.
            event = _summary_event(6)
            before = _state_values(_make_dict_messages(8), event)
            after = _state_values(_make_dict_messages(8), event)

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert any(
                "the conversation is already compact" in str(w._content) for w in msgs
            )
            assert not any("Offloaded " in str(w._content) for w in msgs)

    async def test_cutoff_one_offloads_single_message(self) -> None:
        """A cutoff of 1 reports a single offloaded message."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(7))
            after = _state_values(_make_dict_messages(9), _summary_event(1))

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("Offloaded 1 older messages" in str(w._content) for w in msgs)


class TestReOffload:
    """Test offload when a prior _summarization_event already exists."""

    async def test_reoffload_uses_absolute_cutoff_delta(self) -> None:
        """Re-offload counts only the newly offloaded messages.

        With a prior cutoff of 5 and a new absolute cutoff of 7, exactly two
        additional messages were offloaded this run.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            prior_event = _summary_event(5, file_path=None)
            before = _state_values(_make_dict_messages(15), prior_event)
            after = _state_values(_make_dict_messages(17), _summary_event(7))

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            msgs = app.query(AppMessage)
            # Offloaded count is the new cutoff of seven minus a prior cutoff of five.
            assert any("Offloaded 2 older messages" in str(w._content) for w in msgs)


class TestAgentRunningGuard:
    """Test that _handle_offload sets _agent_running to prevent races."""

    async def test_agent_running_set_during_offload(self) -> None:
        """Should set _agent_running=True during offload and reset after."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))
            after = _state_values(_make_dict_messages(12), _summary_event(4))

            running_during_offload: list[bool] = []

            def capture_running(_config: object) -> None:
                running_during_offload.append(app._agent_running)

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    side_effect=capture_running,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            # _agent_running should have been True while the tool ran
            assert running_during_offload == [True]
            # And reset after completion
            assert app._agent_running is False

    async def test_agent_running_reset_after_failure(self) -> None:
        """Should reset _agent_running=False even when offload fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("stream down"),
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            assert app._agent_running is False


class TestOffloadErrorHandling:
    """Test error handling during offload."""

    async def test_tool_reported_compaction_failure_shows_error(self) -> None:
        """A "Compaction failed" ToolMessage surfaces as an `ErrorMessage`."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))
            tool_error = (
                "Compaction failed: an error occurred while generating the "
                "summary (RuntimeError: model unavailable)."
            )

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    return_value=tool_error,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            error_msgs = app.query(ErrorMessage)
            assert any("Compaction failed" in str(w._content) for w in error_msgs)
            # A no-success guarantee: the offloaded feedback is not shown.
            assert not any(
                "Offloaded " in str(w._content) for w in app.query(AppMessage)
            )

    async def test_compaction_run_failure_shows_error(self) -> None:
        """Should show error and leave state untouched when the run raises."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("stream unavailable"),
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            error_msgs = app.query(ErrorMessage)
            assert any("Offload failed" in str(w._content) for w in error_msgs)

    async def test_spinner_hidden_after_failure(self) -> None:
        """Should hide spinner even when offload fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("backend down"),
                ),
                patch.object(
                    app, "_set_spinner", new_callable=AsyncMock
                ) as mock_spinner,
            ):
                await app._handle_offload()
                await pilot.pause()

            # Spinner should be shown then hidden
            assert mock_spinner.call_count == 2
            mock_spinner.assert_any_call("Offloading")
            mock_spinner.assert_any_call(None)


class TestOffloadMessagesToBackend:
    """Test offload_messages_to_backend code paths."""

    async def test_filters_summary_messages(self) -> None:
        """Should use middleware._filter_summary_messages to exclude summaries."""
        mock_mw = MagicMock()
        messages = _make_messages(3)
        mock_mw._filter_summary_messages.return_value = [messages[0], messages[2]]

        resp = MagicMock()
        resp.content = None
        resp.error = None
        mock_backend = MagicMock()
        mock_backend.adownload_files = AsyncMock(return_value=[resp])
        write_result = MagicMock()
        write_result.error = None
        mock_backend.awrite = AsyncMock(return_value=write_result)

        with patch(_GET_BUFFER_STRING_PATH, return_value="msg text"):
            result = await offload_messages_to_backend(
                messages,
                mock_mw,
                thread_id="test-thread",
                backend=mock_backend,
            )

        mock_mw._filter_summary_messages.assert_called_once_with(messages)
        assert result is not None
        assert result != ""

    async def test_all_summary_messages_returns_empty(self) -> None:
        """Should return empty string when all messages are summaries."""
        mock_mw = MagicMock()
        mock_mw._filter_summary_messages.return_value = []

        mock_backend = MagicMock()

        result = await offload_messages_to_backend(
            _make_messages(2),
            mock_mw,
            thread_id="test-thread",
            backend=mock_backend,
        )

        assert result == ""

    async def test_appends_to_existing_content(self) -> None:
        """Should append new section to existing history file."""
        mock_mw = MagicMock()
        messages = _make_messages(2)
        mock_mw._filter_summary_messages.return_value = messages

        existing = b"## Prior section\n\nold content\n\n"
        resp = MagicMock()
        resp.content = existing
        resp.error = None
        mock_backend = MagicMock()
        mock_backend.adownload_files = AsyncMock(return_value=[resp])
        edit_result = MagicMock()
        edit_result.error = None
        mock_backend.aedit = AsyncMock(return_value=edit_result)

        with patch(_GET_BUFFER_STRING_PATH, return_value="msg text"):
            result = await offload_messages_to_backend(
                messages,
                mock_mw,
                thread_id="test-thread",
                backend=mock_backend,
            )

        assert result is not None
        # Should have called aedit (not awrite) since existing content exists
        mock_backend.aedit.assert_called_once()

    async def test_creates_new_file_when_none_exists(self) -> None:
        """Should call awrite when no existing file is found."""
        mock_mw = MagicMock()
        messages = _make_messages(2)
        mock_mw._filter_summary_messages.return_value = messages

        resp = MagicMock()
        resp.content = None
        resp.error = None
        mock_backend = MagicMock()
        mock_backend.adownload_files = AsyncMock(return_value=[resp])
        write_result = MagicMock()
        write_result.error = None
        mock_backend.awrite = AsyncMock(return_value=write_result)

        with patch(_GET_BUFFER_STRING_PATH, return_value="msg text"):
            result = await offload_messages_to_backend(
                messages,
                mock_mw,
                thread_id="test-thread",
                backend=mock_backend,
            )

        assert result is not None
        mock_backend.awrite.assert_called_once()

    async def test_read_failure_returns_none(self) -> None:
        """Should return None when reading existing file fails."""
        mock_mw = MagicMock()
        mock_mw._filter_summary_messages.return_value = _make_messages(2)

        mock_backend = MagicMock()
        mock_backend.adownload_files = AsyncMock(
            side_effect=RuntimeError("storage unavailable")
        )

        with patch(_GET_BUFFER_STRING_PATH, return_value="msg text"):
            result = await offload_messages_to_backend(
                _make_messages(2),
                mock_mw,
                thread_id="test-thread",
                backend=mock_backend,
            )

        assert result is None

    async def test_write_failure_returns_none(self) -> None:
        """Should return None when writing to backend fails."""
        mock_mw = MagicMock()
        mock_mw._filter_summary_messages.return_value = _make_messages(2)

        resp = MagicMock()
        resp.content = None
        resp.error = None
        mock_backend = MagicMock()
        mock_backend.adownload_files = AsyncMock(return_value=[resp])
        mock_backend.awrite = AsyncMock(side_effect=RuntimeError("disk full"))

        with patch(_GET_BUFFER_STRING_PATH, return_value="msg text"):
            result = await offload_messages_to_backend(
                _make_messages(2),
                mock_mw,
                thread_id="test-thread",
                backend=mock_backend,
            )

        assert result is None

    async def test_write_error_result_returns_none(self) -> None:
        """Should return None when write result contains an error."""
        mock_mw = MagicMock()
        mock_mw._filter_summary_messages.return_value = _make_messages(2)

        resp = MagicMock()
        resp.content = None
        resp.error = None
        mock_backend = MagicMock()
        mock_backend.adownload_files = AsyncMock(return_value=[resp])
        write_result = MagicMock()
        write_result.error = "permission denied"
        mock_backend.awrite = AsyncMock(return_value=write_result)

        with patch(_GET_BUFFER_STRING_PATH, return_value="msg text"):
            result = await offload_messages_to_backend(
                _make_messages(2),
                mock_mw,
                thread_id="test-thread",
                backend=mock_backend,
            )

        assert result is None


class TestFallbackOffloadBackend:
    """Cover the `backend is None` fallback used by `/offload` in server mode.

    In server mode the app has no handle to the agent's routed backend, so
    `perform_offload` builds its own. A non-virtual backend would resolve the
    absolute `/conversation_history/...` path against the real filesystem root
    (unwritable on normal accounts), silently failing every persist. The
    fallback must instead be a virtual-mode backend rooted at a writable
    per-user directory.
    """

    def test_fallback_root_prefers_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`~/.deepagents` is preferred when the home directory resolves."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _offload_fallback_root() == tmp_path / ".deepagents"

    def test_fallback_backend_is_virtual(self) -> None:
        """Fallback backend must use virtual mode so absolute paths stay rooted."""
        backend = _fallback_offload_backend()
        # `virtual_mode=False` is the bug: it lets `/conversation_history/...`
        # escape to the real filesystem root.
        assert backend.virtual_mode is True

    async def test_fallback_backend_writes_under_writable_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Offloading through the fallback lands the file under the root, not `/`."""
        from langchain_core.messages import AIMessage, HumanMessage

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        mock_mw = MagicMock()
        messages = [HumanMessage("hello"), AIMessage("hi back")]
        mock_mw._filter_summary_messages.return_value = messages

        backend = _fallback_offload_backend()
        result = await offload_messages_to_backend(
            messages,
            mock_mw,
            thread_id="thread-abc",
            backend=backend,
        )

        # The returned/stored path stays the stable virtual path.
        assert result == "/conversation_history/thread-abc.md"
        # The bytes actually land under the writable per-user root, kept inside
        # the virtual root rather than the real filesystem root.
        on_disk = tmp_path / ".deepagents" / "conversation_history" / "thread-abc.md"
        assert on_disk.exists()
        assert "hello" in on_disk.read_text()


class TestOffloadRouting:
    """Test that /offload is routed through _handle_command."""

    async def test_offload_routed_from_handle_command(self) -> None:
        """'/offload' should be correctly routed through _handle_command."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = None
            app._lc_thread_id = None

            await app._handle_command("/offload")
            await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("Nothing to offload" in str(w._content) for w in msgs)

    async def test_compact_alias_routed_from_handle_command(self) -> None:
        """'/compact' should still route through _handle_command for backward compat."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = None
            app._lc_thread_id = None

            await app._handle_command("/compact")
            await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("Nothing to offload" in str(w._content) for w in msgs)


class TestDriveServerSideCompaction:
    """Unit-test the server-side `compact_conversation` trigger mechanism."""

    @staticmethod
    def _fake_remote_agent(tool_content: str) -> tuple[Any, list[Any]]:
        """Build a fake `RemoteAgent` that interrupts then returns a ToolMessage.

        First `astream(None)` surfaces a HITL approval interrupt; the resume
        stream (`Command(resume=...)`) yields a `ToolMessage` with the supplied
        content so callers can exercise both the success and failure branches.
        """
        from langchain_core.messages import ToolMessage

        from deepagents_code.client.remote_client import RemoteAgent

        astream_inputs: list[Any] = []

        class _Interrupt:
            id = "interrupt-1"

        async def _astream(stream_input: object, **_kwargs: object):  # noqa: RUF029, ANN202
            astream_inputs.append(stream_input)
            if stream_input is None:
                yield ((), "updates", {"__interrupt__": [_Interrupt()]})
            else:
                yield (
                    (),
                    "messages",
                    (ToolMessage(content=tool_content, tool_call_id="x"), {}),
                )

        agent = MagicMock(spec=RemoteAgent)
        agent.aensure_thread = AsyncMock()
        agent.aupdate_state = AsyncMock()
        agent.astream = _astream
        return agent, astream_inputs

    async def test_seeds_tool_call_and_resumes_interrupt(self) -> None:
        """Seeds a forced `compact_conversation` call and approves the interrupt."""
        from langgraph.types import Command

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            agent, astream_inputs = self._fake_remote_agent(
                "Conversation compacted. Summarized 2 messages into a concise summary."
            )
            app._agent = agent
            app._lc_thread_id = "test-thread"

            config = {"configurable": {"thread_id": "test-thread"}}
            result = await app._drive_server_side_compaction(config)  # ty: ignore
            await pilot.pause()

            assert result is None

            # Seed is attributed to the model node so the tool-call routing
            # reaches the ToolNode.
            agent.aupdate_state.assert_awaited_once()
            seed_values = agent.aupdate_state.call_args.args[1]
            (seed_msg,) = seed_values["messages"]
            (tool_call,) = seed_msg.tool_calls
            assert tool_call["name"] == "compact_conversation"
            assert tool_call["args"] == {"force": True}
            assert agent.aupdate_state.call_args.kwargs["as_node"] == "model"

            # Stream is advanced with None, then resumed after the interrupt.
            assert astream_inputs[0] is None
            assert isinstance(astream_inputs[1], Command)
            resume = astream_inputs[1].resume
            assert "interrupt-1" in resume

    async def test_reports_tool_failure(self) -> None:
        """Returns the tool's error text when compaction fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            agent, _inputs = self._fake_remote_agent(
                "Compaction failed: an error occurred while generating the summary."
            )
            app._agent = agent
            app._lc_thread_id = "test-thread"

            config = {"configurable": {"thread_id": "test-thread"}}
            result = await app._drive_server_side_compaction(config)  # ty: ignore
            await pilot.pause()

            assert result is not None
            assert result.startswith("Compaction failed")


class TestFormatTokenCount:
    """Test the format_token_count helper function."""

    def test_zero(self) -> None:
        assert format_token_count(0) == "0"

    def test_below_threshold(self) -> None:
        assert format_token_count(999) == "999"

    def test_at_threshold(self) -> None:
        assert format_token_count(1000) == "1.0K"

    def test_above_threshold(self) -> None:
        assert format_token_count(1500) == "1.5K"

    def test_large_value(self) -> None:
        assert format_token_count(200000) == "200.0K"

    def test_millions(self) -> None:
        assert format_token_count(1_000_000) == "1.0M"

    def test_above_million(self) -> None:
        assert format_token_count(2_500_000) == "2.5M"


class TestFormatOffloadLimit:
    """Test the format_offload_limit helper function."""

    def test_format_messages_limit(self) -> None:
        assert format_offload_limit(("messages", 6), None) == "last 6 messages"

    def test_format_tokens_limit(self) -> None:
        assert format_offload_limit(("tokens", 12_345), None) == "12.3K tokens"

    def test_format_fraction_limit_with_context(self) -> None:
        assert format_offload_limit(("fraction", 0.1), 200_000) == "20.0K tokens"

    def test_format_fraction_limit_without_context(self) -> None:
        assert format_offload_limit(("fraction", 0.1), None) == "10% of context window"

    def test_format_messages_singular(self) -> None:
        assert format_offload_limit(("messages", 1), None) == "last 1 message"

    def test_format_unknown_keep_type(self) -> None:
        result = format_offload_limit(("unknown", 42), None)
        assert result == "current retention threshold"

    def test_format_fraction_with_zero_context(self) -> None:
        assert format_offload_limit(("fraction", 0.5), 0) == "1 tokens"


# ---------------------------------------------------------------------------
# Patch targets for perform_offload direct tests
# ---------------------------------------------------------------------------
_CREATE_MODEL_PATH = "deepagents_code.offload.create_model"
_COMPUTE_DEFAULTS_PATH = (
    "deepagents.middleware.summarization.compute_summarization_defaults"
)
_MW_CLASS_PATH = "deepagents.middleware.summarization.SummarizationMiddleware"
_TOKEN_COUNT_PATH = "deepagents_code.offload.count_tokens_approximately"
_OFFLOAD_BACKEND_PATH = "deepagents_code.offload.offload_messages_to_backend"


def _mock_perform_deps(
    *,
    cutoff: int = 4,
    summary: str = "Summary.",
) -> tuple[MagicMock, MagicMock]:
    """Return (mock_model_result, mock_middleware) for perform_offload tests."""
    mock_model = MagicMock()
    mock_model.profile = {"max_input_tokens": 200_000}
    mock_result = MagicMock()
    mock_result.model = mock_model

    mock_mw = MagicMock()
    mock_mw._apply_event_to_messages.side_effect = lambda msgs, _ev: list(msgs)
    mock_mw._determine_cutoff_index.return_value = cutoff
    mock_mw._partition_messages.side_effect = lambda msgs, idx: (
        msgs[:idx],
        msgs[idx:],
    )
    mock_mw._acreate_summary = AsyncMock(return_value=summary)

    summary_msg = MagicMock()
    summary_msg.content = summary
    summary_msg.additional_kwargs = {"lc_source": "summarization"}
    mock_mw._build_new_messages_with_path.return_value = [summary_msg]
    mock_mw._compute_state_cutoff.side_effect = lambda _ev, c: c
    mock_mw._filter_summary_messages.side_effect = lambda msgs: msgs

    return mock_result, mock_mw


class TestPerformOffload:
    """Direct unit tests for the perform_offload business logic."""

    async def test_success_returns_offload_result(self) -> None:
        """Happy path returns OffloadResult with correct fields."""
        from deepagents_code.offload import perform_offload

        model_result, mock_mw = _mock_perform_deps(cutoff=3)
        messages = _make_messages(10)

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw),
            patch(_TOKEN_COUNT_PATH, return_value=100),
            patch(_OFFLOAD_BACKEND_PATH, new_callable=AsyncMock, return_value="/p.md"),
        ):
            result = await perform_offload(
                messages=messages,
                prior_event=None,
                thread_id="t1",
                model_spec="openai:gpt-4",
                profile_overrides=None,
                context_limit=None,
                total_context_tokens=0,
                backend=MagicMock(),
            )

        assert isinstance(result, OffloadResult)
        assert result.messages_offloaded == 3
        assert result.messages_kept == 7
        assert result.new_event["cutoff_index"] == 3

    async def test_cutoff_zero_returns_threshold_not_met(self) -> None:
        """When cutoff is 0, returns OffloadThresholdNotMet."""
        from deepagents_code.offload import perform_offload

        model_result, mock_mw = _mock_perform_deps(cutoff=0)

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw),
            patch(_TOKEN_COUNT_PATH, return_value=50),
        ):
            result = await perform_offload(
                messages=_make_messages(5),
                prior_event=None,
                thread_id="t1",
                model_spec="openai:gpt-4",
                profile_overrides=None,
                context_limit=200_000,
                total_context_tokens=500,
                backend=MagicMock(),
            )

        assert isinstance(result, OffloadThresholdNotMet)
        assert result.conversation_tokens == 50
        assert result.total_context_tokens == 500
        assert result.context_limit == 200_000

    async def test_model_creation_failure_raises_offload_model_error(self) -> None:
        """When create_model fails, OffloadModelError is raised."""
        from deepagents_code.offload import perform_offload

        with (
            patch(_CREATE_MODEL_PATH, side_effect=ValueError("bad key")),
            pytest.raises(OffloadModelError, match="working model configuration"),
        ):
            await perform_offload(
                messages=_make_messages(5),
                prior_event=None,
                thread_id="t1",
                model_spec="openai:gpt-4",
                profile_overrides=None,
                context_limit=None,
                total_context_tokens=0,
                backend=MagicMock(),
            )

    async def test_context_limit_patches_model_profile(self) -> None:
        """When context_limit differs from native, profile is patched."""
        from deepagents_code.offload import perform_offload

        model_result, mock_mw = _mock_perform_deps(cutoff=0)
        model = model_result.model
        model.profile = {"max_input_tokens": 200_000}

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw),
            patch(_TOKEN_COUNT_PATH, return_value=50),
        ):
            await perform_offload(
                messages=_make_messages(5),
                prior_event=None,
                thread_id="t1",
                model_spec="openai:gpt-4",
                profile_overrides=None,
                context_limit=4096,
                total_context_tokens=0,
                backend=MagicMock(),
            )

        assert model.profile["max_input_tokens"] == 4096

    async def test_context_limit_none_skips_patching(self) -> None:
        """When context_limit is None, profile is not modified."""
        from deepagents_code.offload import perform_offload

        model_result, mock_mw = _mock_perform_deps(cutoff=0)
        original_profile = {"max_input_tokens": 200_000}
        model_result.model.profile = original_profile.copy()

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw),
            patch(_TOKEN_COUNT_PATH, return_value=50),
        ):
            await perform_offload(
                messages=_make_messages(5),
                prior_event=None,
                thread_id="t1",
                model_spec="openai:gpt-4",
                profile_overrides=None,
                context_limit=None,
                total_context_tokens=0,
                backend=MagicMock(),
            )

        assert model_result.model.profile == original_profile

    async def test_no_model_profile_creates_new_dict(self) -> None:
        """When model has no profile dict, a new one is created."""
        from deepagents_code.offload import perform_offload

        model_result, mock_mw = _mock_perform_deps(cutoff=0)
        model_result.model.profile = None

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw),
            patch(_TOKEN_COUNT_PATH, return_value=50),
        ):
            await perform_offload(
                messages=_make_messages(5),
                prior_event=None,
                thread_id="t1",
                model_spec="openai:gpt-4",
                profile_overrides=None,
                context_limit=4096,
                total_context_tokens=0,
                backend=MagicMock(),
            )

        assert model_result.model.profile == {"max_input_tokens": 4096}

    async def test_backend_none_uses_filesystem_backend(self) -> None:
        """When backend is None, FilesystemBackend is used."""
        from deepagents_code.offload import perform_offload

        model_result, mock_mw = _mock_perform_deps(cutoff=0)

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw) as mw_cls,
            patch(_TOKEN_COUNT_PATH, return_value=50),
            patch("deepagents.backends.filesystem.FilesystemBackend") as mock_fs,
        ):
            await perform_offload(
                messages=_make_messages(5),
                prior_event=None,
                thread_id="t1",
                model_spec="openai:gpt-4",
                profile_overrides=None,
                context_limit=None,
                total_context_tokens=0,
                backend=None,
            )

        mock_fs.assert_called_once()
        # Verify the fallback backend was passed to SummarizationMiddleware
        _, call_kwargs = mw_cls.call_args
        assert call_kwargs["backend"] is mock_fs.return_value

    async def test_backend_write_failure_sets_offload_warning(self) -> None:
        """When backend write fails, offload_warning is set on result."""
        from deepagents_code.offload import perform_offload

        model_result, mock_mw = _mock_perform_deps(cutoff=3)

        with (
            patch(_CREATE_MODEL_PATH, return_value=model_result),
            patch(_COMPUTE_DEFAULTS_PATH, return_value={"keep": ("fraction", 0.1)}),
            patch(_MW_CLASS_PATH, return_value=mock_mw),
            patch(_TOKEN_COUNT_PATH, return_value=100),
            patch(
                _OFFLOAD_BACKEND_PATH,
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await perform_offload(
                messages=_make_messages(10),
                prior_event=None,
                thread_id="t1",
                model_spec="openai:gpt-4",
                profile_overrides=None,
                context_limit=None,
                total_context_tokens=0,
                backend=MagicMock(),
            )

        assert isinstance(result, OffloadResult)
        assert result.offload_warning is not None
        assert "could not be saved" in result.offload_warning
