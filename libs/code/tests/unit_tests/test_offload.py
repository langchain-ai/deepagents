"""Unit tests for /offload slash command."""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import nullcontext
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

    async def test_committed_offload_survives_trailing_model_failure(self) -> None:
        """A checkpointed tool update wins over a later stream failure."""
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
                    side_effect=RuntimeError("trailing model unavailable"),
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            assert any(
                "Offloaded 4 older messages" in str(widget._content)
                for widget in app.query(AppMessage)
            )
            assert not any(
                "Offload failed" in str(widget._content)
                for widget in app.query(ErrorMessage)
            )

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
        """Should update `_context_tokens` to the post-compaction count.

        The count is taken from the pre-seed conversation plus the new event, so
        it excludes the tool's own machinery (the seeded call, the tool result,
        and the trailing model turn) that the post-run state carries. Using
        distinct before/after message lists guards against regressing to the
        post-run state, which would understate the reduction.
        """
        from langchain_core.messages.utils import count_tokens_approximately

        from deepagents_code.app import _effective_conversation

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before_messages = _make_dict_messages(10)
            after_messages = _make_dict_messages(12)
            after_event = _summary_event(4)
            before = _state_values(before_messages)
            after = _state_values(after_messages, after_event)

            expected = count_tokens_approximately(
                _effective_conversation(before_messages, after_event)
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
            quiescent_during_offload: list[bool] = []

            def capture_running(_config: object, _seed_id: object = None) -> None:
                running_during_offload.append(app._agent_running)
                quiescent_during_offload.append(app._agent_quiescent.is_set())

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
            assert quiescent_during_offload == [False]
            # And reset after completion
            assert app._agent_running is False
            assert app._agent_quiescent.is_set()

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

    async def test_missing_archive_path_warns_about_unrecoverable_history(
        self,
    ) -> None:
        """A successful summary with a failed backend write is not silent."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))
            after = _state_values(
                _make_dict_messages(12), _summary_event(4, file_path=None)
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

            error_messages = app.query(ErrorMessage)
            assert any(
                "Older messages will not be recoverable" in str(widget._content)
                for widget in error_messages
            )
            assert any(
                "Offloaded 4 older messages" in str(widget._content)
                for widget in app.query(AppMessage)
            )

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

    async def test_stale_compaction_failure_is_not_reported(self) -> None:
        """A no-op ignores failure messages committed by an earlier run."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            messages = [
                *_make_dict_messages(3),
                {
                    "type": "tool",
                    "content": "Compaction failed: old failure",
                    "tool_call_id": "old-call",
                },
            ]
            before = _state_values(messages)
            after = _state_values([*messages, *_make_dict_messages(1)])

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

            assert not any(
                "old failure" in str(widget._content)
                for widget in app.query(ErrorMessage)
            )
            assert any(
                "the conversation is already compact" in str(widget._content)
                for widget in app.query(AppMessage)
            )

    async def test_current_durable_compaction_failure_is_reported(self) -> None:
        """A failure appended by this invocation survives a missed stream event."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            messages = _make_dict_messages(3)
            before = _state_values(messages)
            after = _state_values(
                [
                    *messages,
                    {
                        "type": "tool",
                        "content": "Compaction failed: current failure",
                        "tool_call_id": "current-call",
                    },
                ]
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

            assert any(
                "current failure" in str(widget._content)
                for widget in app.query(ErrorMessage)
            )

    async def test_failed_run_removes_dangling_seed(self) -> None:
        """A raising run cleans up the committed seed before surfacing failure.

        When the drive raises and the committed cutoff has not advanced, the
        seeded (and now unanswered) tool call must be removed so it does not
        wedge the next turn; the failure is still surfaced to the user.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(6))
            reconciled = _state_values(_make_dict_messages(6))  # cutoff unchanged

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, reconciled],
                ),
                patch.object(
                    app,
                    "_drive_server_side_compaction",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("stream boom"),
                ),
                patch.object(
                    app,
                    "_remove_unanswered_offload_seed",
                    new_callable=AsyncMock,
                ) as cleanup,
            ):
                await app._handle_offload()
                await pilot.pause()

            cleanup.assert_awaited_once()
            assert any(
                "Offload failed" in str(widget._content)
                for widget in app.query(ErrorMessage)
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

    def test_fallback_root_prefers_home_and_tightens_permissions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`~/.deepagents` is preferred and made private when writable."""
        root = tmp_path / ".deepagents"
        root.mkdir(mode=0o755)
        root.chmod(0o755)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert _offload_fallback_root() == root
        assert stat.S_IMODE(root.stat().st_mode) == 0o700

    def test_fallback_root_uses_temp_when_home_is_read_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resolved but read-only home directory falls back to temp storage."""
        home_root = tmp_path / "home" / ".deepagents"
        home_root.mkdir(parents=True)
        temp_dir = tmp_path / "tmp"
        probe = MagicMock(
            side_effect=[PermissionError("read-only home"), nullcontext()]
        )
        uid = os.getuid()

        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_dir))
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", probe)

        root = _offload_fallback_root()

        assert root == temp_dir / f"deepagents-{uid}"
        assert root.is_dir()
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert probe.call_count == 2

    def test_fallback_root_avoids_foreign_or_invalid_per_user_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unusable predictable path falls back to a private unique one."""
        home_root = tmp_path / "home" / ".deepagents"
        home_root.mkdir(parents=True)
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        uid = os.getuid()
        reserved = temp_dir / f"deepagents-{uid}"
        reserved.write_text("owned by another account")
        probe = MagicMock(
            side_effect=[PermissionError("read-only home"), nullcontext()]
        )

        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_dir))
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", probe)

        root = _offload_fallback_root()

        assert root != reserved
        assert root.name.startswith(f"deepagents-{uid}-")
        assert stat.S_IMODE(root.stat().st_mode) == 0o700

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
    def _fake_remote_agent(
        tool_content: str,
    ) -> tuple[Any, list[Any], list[object]]:
        """Build a fake `RemoteAgent` that interrupts then returns a ToolMessage.

        First `astream(None)` surfaces a HITL approval interrupt; the resume
        stream (`Command(resume=...)`) yields a `ToolMessage` with the supplied
        content so callers can exercise both the success and failure branches.
        """
        from langchain_core.messages import ToolMessage

        from deepagents_code.client.remote_client import RemoteAgent

        astream_inputs: list[Any] = []
        astream_contexts: list[object] = []

        class _Interrupt:
            id = "interrupt-1"
            value = {  # noqa: RUF012  # test stub; immutability irrelevant
                "action_requests": [
                    {"name": "compact_conversation", "args": {"force": True}}
                ]
            }

        async def _astream(stream_input: object, **kwargs: object):  # noqa: RUF029, ANN202
            astream_inputs.append(stream_input)
            astream_contexts.append(kwargs.get("context"))
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
        return agent, astream_inputs, astream_contexts

    async def test_seeds_tool_call_and_resumes_interrupt(self) -> None:
        """Seeds a forced `compact_conversation` call and approves the interrupt."""
        from langgraph.types import Command

        from deepagents_code.config import settings

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            agent, astream_inputs, astream_contexts = self._fake_remote_agent(
                "Conversation compacted. Summarized 2 messages into a concise summary."
            )
            app._agent = agent
            app._lc_thread_id = "test-thread"
            app._model_override = "provider:active-model"
            app._model_params_override = {"temperature": 0}
            app._profile_override = {"max_input_tokens": 4096}

            config = {"configurable": {"thread_id": "test-thread"}}
            with patch.object(settings, "model_context_limit", 4096):
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
            assert astream_contexts == [
                {
                    "model": "provider:active-model",
                    "model_params": {"temperature": 0},
                    "profile_overrides": {"max_input_tokens": 4096},
                    "model_context_limit": 4096,
                    "thread_id": "test-thread",
                },
                {
                    "model": "provider:active-model",
                    "model_params": {"temperature": 0},
                    "profile_overrides": {"max_input_tokens": 4096},
                    "model_context_limit": 4096,
                    "thread_id": "test-thread",
                },
            ]

    async def test_reports_tool_failure(self) -> None:
        """Returns the tool's error text when compaction fails."""
        from deepagents_code.offload_middleware import COMPACTION_FAILURE_PREFIX

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            agent, _inputs, _contexts = self._fake_remote_agent(
                f"{COMPACTION_FAILURE_PREFIX}: an error occurred during compaction."
            )
            app._agent = agent
            app._lc_thread_id = "test-thread"

            config = {"configurable": {"thread_id": "test-thread"}}
            result = await app._drive_server_side_compaction(config)  # ty: ignore
            await pilot.pause()

            assert result is not None
            assert result.startswith(COMPACTION_FAILURE_PREFIX)

    async def test_forwards_startup_model_profile_to_compaction(self) -> None:
        """Profile data is usable even without a session `/model` override."""
        from deepagents_code.config import settings

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            agent, _inputs, contexts = self._fake_remote_agent(
                "Conversation compacted. Summarized 2 messages."
            )
            app._agent = agent
            app._lc_thread_id = "test-thread"
            app._model_override = None
            app._profile_override = {"max_input_tokens": 4096}

            config = {"configurable": {"thread_id": "test-thread"}}
            with (
                patch.object(settings, "model_provider", "provider"),
                patch.object(settings, "model_name", "startup-model"),
                patch.object(settings, "model_context_limit", 4096),
            ):
                await app._drive_server_side_compaction(config)  # ty: ignore
            await pilot.pause()

        assert contexts
        assert all(
            context
            == {
                "model": "provider:startup-model",
                "model_params": {},
                "profile_overrides": {"max_input_tokens": 4096},
                "model_context_limit": 4096,
                "thread_id": "test-thread",
            }
            for context in contexts
        )

    async def test_rejects_interrupt_without_identifiable_action(self) -> None:
        """Malformed interrupt payloads fail closed instead of being approved."""
        from langgraph.types import Command

        from deepagents_code.client.remote_client import RemoteAgent

        astream_inputs: list[Any] = []

        class _Interrupt:
            id = "interrupt-unknown"
            value: dict[str, Any] = {}  # noqa: RUF012  # test stub

        async def _astream(  # noqa: RUF029, ANN202
            stream_input: object, **_kwargs: object
        ):
            astream_inputs.append(stream_input)
            if stream_input is None:
                yield ((), "updates", {"__interrupt__": [_Interrupt()]})

        agent = MagicMock(spec=RemoteAgent)
        agent.aensure_thread = AsyncMock()
        agent.aupdate_state = AsyncMock()
        agent.astream = _astream

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = agent
            app._lc_thread_id = "test-thread"

            config = {"configurable": {"thread_id": "test-thread"}}
            result = await app._drive_server_side_compaction(config)  # ty: ignore
            await pilot.pause()

        assert result is None
        assert len(astream_inputs) == 2
        assert isinstance(astream_inputs[1], Command)
        decision = astream_inputs[1].resume["interrupt-unknown"]["decisions"][0]
        assert decision["type"] == "reject"

    async def test_rejects_trailing_gated_tool_call(self) -> None:
        """Approves compaction but rejects a trailing model turn's gated tool.

        Driving the graph loops back to the model after compaction, so the
        trailing turn can request its own gated tools. Those must NOT be
        auto-approved (the user never saw them) — they are rejected so the run
        finishes cleanly without side effects and without stranding an
        interrupt.
        """
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command

        from deepagents_code.client.remote_client import RemoteAgent

        astream_inputs: list[Any] = []

        class _Interrupt:
            def __init__(self, iid: str, tool_name: str) -> None:
                self.id = iid
                self.value = {"action_requests": [{"name": tool_name, "args": {}}]}

        async def _astream(stream_input: object, **_kwargs: object):  # noqa: RUF029, ANN202
            idx = len(astream_inputs)
            astream_inputs.append(stream_input)
            if idx == 0:
                compact = _Interrupt("i-compact", "compact_conversation")
                yield ((), "updates", {"__interrupt__": [compact]})
            elif idx == 1:
                # Trailing model turn requests a gated tool.
                write = _Interrupt("i-write", "write_file")
                yield ((), "updates", {"__interrupt__": [write]})
            else:
                yield (
                    (),
                    "messages",
                    (
                        ToolMessage(
                            content="Conversation compacted. Summarized 2 messages "
                            "into a concise summary.",
                            tool_call_id="x",
                        ),
                        {},
                    ),
                )

        agent = MagicMock(spec=RemoteAgent)
        agent.aensure_thread = AsyncMock()
        agent.aupdate_state = AsyncMock()
        agent.astream = _astream

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = agent
            app._lc_thread_id = "test-thread"

            config = {"configurable": {"thread_id": "test-thread"}}
            result = await app._drive_server_side_compaction(config)  # ty: ignore
            await pilot.pause()

        assert result is None
        # Initial drain + two resumes (compaction, then trailing tool).
        assert len(astream_inputs) == 3
        assert isinstance(astream_inputs[1], Command)
        assert isinstance(astream_inputs[2], Command)
        # Compaction was approved.
        compact_decision = astream_inputs[1].resume["i-compact"]["decisions"][0]
        assert compact_decision["type"] == "approve"
        # The trailing gated tool call was rejected, not approved.
        write_decision = astream_inputs[2].resume["i-write"]["decisions"][0]
        assert write_decision["type"] == "reject"

    async def test_bounds_resume_loop_and_reports_abandoned_drain(self) -> None:
        """A model that keeps requesting tools cannot spin `/offload` forever.

        Every stream yields a fresh gated interrupt, so the resume loop never
        drains cleanly. It must stop at the `max_resume_rounds` cap (initial
        drain + 10 resumes = 11 streams) and surface a user-visible notice that
        the run was left paused, rather than looping indefinitely.
        """
        from deepagents_code.client.remote_client import RemoteAgent

        astream_inputs: list[Any] = []

        class _Interrupt:
            def __init__(self, iid: str) -> None:
                self.id = iid
                self.value = {"action_requests": [{"name": "write_file", "args": {}}]}

        async def _astream(stream_input: object, **_kwargs: object):  # noqa: RUF029, ANN202
            idx = len(astream_inputs)
            astream_inputs.append(stream_input)
            # Never terminate: each round surfaces another gated interrupt.
            yield ((), "updates", {"__interrupt__": [_Interrupt(f"i-{idx}")]})

        agent = MagicMock(spec=RemoteAgent)
        agent.aensure_thread = AsyncMock()
        agent.aupdate_state = AsyncMock()
        agent.astream = _astream

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = agent
            app._lc_thread_id = "test-thread"

            config = {"configurable": {"thread_id": "test-thread"}}
            result = await app._drive_server_side_compaction(config)  # ty: ignore
            await pilot.pause()

            # No compaction failure was reported, so the run returns cleanly.
            assert result is None
            # Initial drain + exactly 10 resume rounds, then the cap breaks.
            assert len(astream_inputs) == 11
            assert any(
                "could not be fully drained" in str(widget._content)
                for widget in app.query(ErrorMessage)
            )


class TestRemoveUnansweredOffloadSeed:
    """Cleanup of a committed-but-unanswered `/offload` seed after a failure."""

    @staticmethod
    def _seed_message(tool_call_id: str) -> dict[str, Any]:
        """Serialized seed AIMessage carrying the forced compaction tool call."""
        return {
            "type": "ai",
            "content": "",
            "id": f"offload-seed-{tool_call_id}",
            "tool_calls": [
                {
                    "name": "compact_conversation",
                    "args": {"force": True},
                    "id": tool_call_id,
                }
            ],
        }

    async def test_removes_dangling_seed(self) -> None:
        """An unanswered seed is removed so it cannot wedge the next turn."""
        from langchain_core.messages import RemoveMessage

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            agent = MagicMock()
            agent.aupdate_state = AsyncMock()
            app._agent = agent
            state = _state_values(
                [*_make_dict_messages(2), self._seed_message("seed-call")]
            )
            with patch.object(
                app,
                "_get_thread_state_values",
                new_callable=AsyncMock,
                return_value=state,
            ):
                await app._remove_unanswered_offload_seed(
                    {"configurable": {"thread_id": "test-thread"}}, "seed-call"
                )

            agent.aupdate_state.assert_awaited_once()
            update = agent.aupdate_state.call_args.args[1]
            (removal,) = update["messages"]
            assert isinstance(removal, RemoveMessage)
            assert removal.id == "offload-seed-seed-call"

    async def test_keeps_answered_seed(self) -> None:
        """A seed answered by a ToolMessage is a valid pair and is left intact."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            agent = MagicMock()
            agent.aupdate_state = AsyncMock()
            app._agent = agent
            answered = {
                "type": "tool",
                "content": "Nothing to compact yet.",
                "tool_call_id": "seed-call",
            }
            state = _state_values(
                [*_make_dict_messages(2), self._seed_message("seed-call"), answered]
            )
            with patch.object(
                app,
                "_get_thread_state_values",
                new_callable=AsyncMock,
                return_value=state,
            ):
                await app._remove_unanswered_offload_seed(
                    {"configurable": {"thread_id": "test-thread"}}, "seed-call"
                )

            agent.aupdate_state.assert_not_awaited()

    async def test_noop_when_seed_absent(self) -> None:
        """Nothing is removed when no seed with the id is present."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            agent = MagicMock()
            agent.aupdate_state = AsyncMock()
            app._agent = agent
            state = _state_values(_make_dict_messages(2))
            with patch.object(
                app,
                "_get_thread_state_values",
                new_callable=AsyncMock,
                return_value=state,
            ):
                await app._remove_unanswered_offload_seed(
                    {"configurable": {"thread_id": "test-thread"}}, "seed-call"
                )

            agent.aupdate_state.assert_not_awaited()


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


class TestOffloadHelpers:
    """Pure helpers backing `/offload` accounting and failure detection."""

    def test_summarization_cutoff_reads_int(self) -> None:
        from deepagents_code.app import _summarization_cutoff

        assert _summarization_cutoff({"cutoff_index": 4}) == 4

    def test_summarization_cutoff_defaults_zero_on_malformed(self) -> None:
        from deepagents_code.app import _summarization_cutoff

        assert _summarization_cutoff(None) == 0
        assert _summarization_cutoff({"cutoff_index": "x"}) == 0
        assert _summarization_cutoff({}) == 0
        assert _summarization_cutoff("not-a-dict") == 0

    def test_effective_conversation_applies_event(self) -> None:
        from deepagents_code.app import _effective_conversation

        messages = [f"m{i}" for i in range(5)]
        event = {"summary_message": "S", "cutoff_index": 2}
        assert _effective_conversation(messages, event) == ["S", "m2", "m3", "m4"]

    def test_effective_conversation_degrades_on_malformed(self) -> None:
        from deepagents_code.app import _effective_conversation

        messages = ["m0", "m1"]
        # No event, non-dict event, missing summary, and non-int cutoff all
        # return the messages unchanged rather than raising or emitting a None.
        assert _effective_conversation(messages, None) == messages
        assert _effective_conversation(messages, "x") == messages
        assert _effective_conversation(messages, {"cutoff_index": 1}) == messages
        assert _effective_conversation(messages, {"summary_message": "S"}) == messages

    def test_effective_conversation_cutoff_past_end(self) -> None:
        from deepagents_code.app import _effective_conversation

        event = {"summary_message": "S", "cutoff_index": 9}
        assert _effective_conversation(["m0"], event) == ["S"]

    def test_message_text_handles_str_and_block_list(self) -> None:
        from deepagents_code.app import _message_text

        assert _message_text(MagicMock(content="hello")) == "hello"
        # A block-list content is concatenated, not stringified to "[{...}]".
        blocks = [
            {"type": "text", "text": "Compaction "},
            {"type": "text", "text": "failed"},
        ]
        assert _message_text({"content": blocks}) == "Compaction failed"
        assert _message_text({"content": None}) == ""

    def test_find_compaction_failure_scans_durable_state(self) -> None:
        from langchain_core.messages import HumanMessage, ToolMessage

        from deepagents_code.app import _find_compaction_failure
        from deepagents_code.offload_middleware import COMPACTION_FAILURE_PREFIX

        failing = ToolMessage(
            content=f"{COMPACTION_FAILURE_PREFIX}: boom",
            tool_call_id="tc",
        )
        messages = [HumanMessage("hi"), failing]
        assert (
            _find_compaction_failure(messages) == f"{COMPACTION_FAILURE_PREFIX}: boom"
        )

    def test_find_compaction_failure_ignores_success(self) -> None:
        from langchain_core.messages import ToolMessage

        from deepagents_code.app import _find_compaction_failure

        ok = ToolMessage(content="Conversation compacted.", tool_call_id="tc")
        assert _find_compaction_failure([ok]) is None
        # Serialized-dict tool message form is handled too.
        assert _find_compaction_failure([{"type": "tool", "content": "ok"}]) is None
