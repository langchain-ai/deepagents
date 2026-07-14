"""Unit tests for /offload slash command."""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import nullcontext
from pathlib import Path, PureWindowsPath
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends.utils import validate_path

from deepagents_code._session_stats import format_token_count
from deepagents_code.app import DeepAgentsApp
from deepagents_code.command_registry import SLASH_COMMANDS
from deepagents_code.offload import (
    _artifacts_root,
    _filesystem_tool_path,
    _offload_fallback_root,
)
from deepagents_code.tui.widgets.messages import AppMessage, ErrorMessage


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


def _setup_server_offload_app(app: DeepAgentsApp) -> MagicMock:
    """Configure a `DeepAgentsApp` for server-side offload unit tests.

    The server-side path reads state via `_get_thread_state_values` and drives
    the tool via `_drive_server_side_compaction`; tests patch those seams
    directly, so only the plain identity/flags are set here.
    """
    agent = MagicMock()
    agent.aupdate_state = AsyncMock()
    app._agent = agent
    app._backend = None
    app._lc_thread_id = "test-thread"
    app._agent_running = False
    return agent


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

    async def test_committed_offload_survives_stream_failure(self) -> None:
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
                    side_effect=RuntimeError("stream unavailable"),
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
        """A no-op restores history and shows the no-op message, not success."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            agent = _setup_server_offload_app(app)

            # Prior event present; after-state cutoff unchanged -> nothing moved.
            event = _summary_event(6)
            messages = _make_dict_messages(8)
            artifacts = [
                {
                    "type": "ai",
                    "content": "",
                    "id": "offload-seed-test",
                    "tool_calls": [
                        {
                            "name": "compact_conversation",
                            "args": {"force": True},
                            "id": "seed-call",
                        }
                    ],
                },
                {
                    "type": "tool",
                    "content": "Nothing to compact yet.",
                    "id": "offload-result-test",
                    "tool_call_id": "seed-call",
                },
                {
                    "type": "ai",
                    "content": "Trailing response",
                    "id": "offload-trailing-test",
                    "tool_calls": [],
                },
            ]
            before = _state_values(messages, event)
            after = _state_values([*messages, *artifacts], event)

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
            agent.aupdate_state.assert_awaited_once()
            update = agent.aupdate_state.call_args.args[1]
            assert [message.id for message in update["messages"]] == [
                "offload-seed-test",
                "offload-result-test",
                "offload-trailing-test",
            ]

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

    async def test_reoffload_noop_restores_prior_summary(self) -> None:
        """A summary-only re-offload restores the prior summarization event."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            agent = _setup_server_offload_app(app)

            prior_event = _summary_event(5, file_path=None)
            replacement_event = _summary_event(5)
            replacement_event["summary_message"]["content"] = "Replacement summary."
            before_messages = _make_dict_messages(11)
            after_messages = [*before_messages, *_make_dict_messages(2)]
            after_messages[-2]["id"] = "offload-seed"
            after_messages[-1]["id"] = "offload-result"
            before = _state_values(before_messages, prior_event)
            after = _state_values(after_messages, replacement_event)

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

            agent.aupdate_state.assert_awaited_once()
            update = agent.aupdate_state.call_args.args[1]
            assert update["_summarization_event"] is prior_event
            assert [message.id for message in update["messages"]] == [
                "offload-seed",
                "offload-result",
            ]
            assert any(
                "Nothing to offload" in str(widget._content)
                for widget in app.query(AppMessage)
            )


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
        """A failed backend write surfaces in a single, non-contradictory message.

        The reduction and the unrecoverable-archive warning are combined into one
        `ErrorMessage` rather than a warning immediately followed by a separate
        success line.
        """
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

            # Both the reduction and the archive-failure warning land in one
            # ErrorMessage.
            assert any(
                "Offloaded 4 older messages" in str(widget._content)
                and "could not be saved to storage" in str(widget._content)
                for widget in app.query(ErrorMessage)
            )
            # No separate success line is emitted alongside the warning.
            assert not any(
                "Offloaded" in str(widget._content) for widget in app.query(AppMessage)
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

    async def test_double_failure_warns_thread_may_be_inconsistent(self) -> None:
        """Stream failure + failed reconcile + failed cleanup warns the user.

        When the drive raises, the reconcile state-read also fails, and the
        best-effort seed cleanup cannot confirm removal (returns False), the
        user is warned the thread may be inconsistent -- in addition to the
        surfaced "Offload failed" error -- so a later cryptic `tool_use`
        rejection is not their only signal.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(6))

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, RuntimeError("reconcile read boom")],
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
                    return_value=False,
                ) as cleanup,
            ):
                await app._handle_offload()
                await pilot.pause()

            cleanup.assert_awaited_once()
            error_text = " ".join(
                str(widget._content) for widget in app.query(ErrorMessage)
            )
            assert "inconsistent state" in error_text
            assert "Offload failed" in error_text

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


class TestOffloadFallbackRoot:
    """Cover writable local storage for offloaded conversation history."""

    def test_fallback_root_prefers_home_and_tightens_only_archive_subdir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`~/.deepagents` is preferred; only the archive subdir is hardened.

        The shared config root must keep its own permissions (it houses
        `config.toml`, `hooks.json`, `.env`, etc.); only the offload-specific
        `conversation_history` subdirectory is tightened to `0o700`.
        """
        root = tmp_path / ".deepagents"
        root.mkdir(mode=0o755)
        root.chmod(0o755)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert _offload_fallback_root() == root
        # The shared config root's permissions are left untouched.
        assert stat.S_IMODE(root.stat().st_mode) == 0o755
        # Only the archive subdirectory is made private.
        archive_dir = root / "conversation_history"
        assert archive_dir.is_dir()
        assert stat.S_IMODE(archive_dir.stat().st_mode) == 0o700

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
        getuid = getattr(os, "getuid", None)
        uid = getuid() if getuid is not None else os.getpid()

        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_dir))
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", probe)

        root = _offload_fallback_root()

        assert root == temp_dir / f"deepagents-{uid}"
        assert root.is_dir()
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert probe.call_count == 2

    def test_fallback_root_avoids_file_at_predictable_per_user_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-directory at the predictable temp path falls back to a unique one.

        A plain file where `deepagents-<uid>` is expected makes
        `mkdir(exist_ok=True)` raise `FileExistsError` (an `OSError`), so the
        resolver creates a private unique directory instead.
        """
        home_root = tmp_path / "home" / ".deepagents"
        home_root.mkdir(parents=True)
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        getuid = getattr(os, "getuid", None)
        uid = getuid() if getuid is not None else os.getpid()
        reserved = temp_dir / f"deepagents-{uid}"
        reserved.write_text("not a directory")
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

    def test_fallback_root_rejects_foreign_owned_per_user_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A predictable temp dir owned by another user is rejected for a unique one.

        Exercises the `st_uid != getuid()` ownership guard: `lstat` is stubbed to
        report a foreign owner for the predictable per-user dir only, so it is
        rejected while the freshly-created unique dir (real ownership) passes.
        """
        from types import SimpleNamespace

        getuid = getattr(os, "getuid", None)
        if getuid is None:
            pytest.skip("uid ownership check requires os.getuid")

        home_root = tmp_path / "home" / ".deepagents"
        home_root.mkdir(parents=True)
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        uid = getuid()
        reserved = temp_dir / f"deepagents-{uid}"
        reserved.mkdir()  # a real, us-owned directory; lstat is faked below
        probe = MagicMock(
            side_effect=[PermissionError("read-only home"), nullcontext()]
        )

        real_lstat = Path.lstat

        def fake_lstat(self: Path) -> Any:  # noqa: ANN401
            info = real_lstat(self)
            if self == reserved:
                # Report a foreign owner for the predictable dir only.
                return SimpleNamespace(st_mode=info.st_mode, st_uid=info.st_uid + 1)
            return info

        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_dir))
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", probe)
        monkeypatch.setattr(Path, "lstat", fake_lstat)

        root = _offload_fallback_root()

        assert root != reserved
        assert root.name.startswith(f"deepagents-{uid}-")

    def test_fallback_root_rejects_symlinked_archive_subdir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `conversation_history` that is itself a symlink is rejected (S_ISDIR).

        The `lstat`/`S_ISDIR` guard does not follow the link, so a symlinked
        archive subdirectory (even one pointing at a real, us-owned directory)
        makes the persistent path fail and offload falls back to temp storage.
        """
        home = tmp_path / "home"
        base = home / ".deepagents"
        base.mkdir(parents=True)
        real_target = tmp_path / "elsewhere"
        real_target.mkdir()
        (base / "conversation_history").symlink_to(real_target)
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        getuid = getattr(os, "getuid", None)
        uid = getuid() if getuid is not None else os.getpid()
        # Only the temp fallback's write-probe should run; the symlinked archive
        # subdir is rejected by S_ISDIR before the user dir is probed.
        probe = MagicMock(return_value=nullcontext())

        monkeypatch.setattr(Path, "home", lambda: home)
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_dir))
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", probe)

        root = _offload_fallback_root()

        assert root == temp_dir / f"deepagents-{uid}"
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        # The temp fallback is not persistent; the flag reflects that.
        from deepagents_code.offload import offload_storage_is_ephemeral

        assert offload_storage_is_ephemeral() is True

    def test_fallback_root_tightens_preexisting_loose_archive_subdir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An existing `conversation_history` with loose perms is tightened to 0o700.

        `mkdir(mode=...)` does not tighten an existing directory, so the explicit
        `chmod(0o700)` is what protects a pre-existing world-readable archive
        dir. Removing that call would regress this test.
        """
        root = tmp_path / ".deepagents"
        root.mkdir()
        archive_dir = root / "conversation_history"
        archive_dir.mkdir(mode=0o755)
        archive_dir.chmod(0o755)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert _offload_fallback_root() == root
        assert stat.S_IMODE(archive_dir.stat().st_mode) == 0o700
        # The persistent per-user location is not ephemeral.
        from deepagents_code.offload import offload_storage_is_ephemeral

        assert offload_storage_is_ephemeral() is False


class TestArtifactsRoot:
    """Cover the real-filesystem artifacts root for offloaded tool results."""

    def test_artifacts_root_is_stable_and_hardened(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The per-user artifacts dir is predictable, private, and reused."""
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        getuid = getattr(os, "getuid", None)
        uid = getuid() if getuid is not None else os.getpid()

        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_dir))

        storage = _artifacts_root()
        root_path = Path(storage.root)

        assert storage.large_results_dir is None
        assert root_path.samefile(temp_dir / f"dcode-artifacts-{uid}")
        assert stat.S_IMODE(root_path.stat().st_mode) == 0o700
        # Stable across calls (paths embedded in resumed threads stay resolvable).
        assert _artifacts_root() == storage

    def test_windows_artifacts_root_is_accepted_by_filesystem_tools(self) -> None:
        """A Windows temp path retains its drive without a rejected drive prefix."""
        disk_root = PureWindowsPath(
            "C:/Users/test/AppData/Local/Temp/dcode-artifacts-123"
        )

        root = _filesystem_tool_path(disk_root)
        result_path = f"{root}/large_tool_results/tool-call-id"

        assert root == "//?/C:/Users/test/AppData/Local/Temp/dcode-artifacts-123"
        assert PureWindowsPath(root).is_absolute()
        assert validate_path(result_path) == result_path

    def test_artifacts_root_falls_back_when_predictable_path_foreign_owned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A predictable dir owned by another user is rejected for a unique one."""
        from types import SimpleNamespace

        getuid = getattr(os, "getuid", None)
        if getuid is None:
            pytest.skip("uid ownership check requires os.getuid")

        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        uid = getuid()
        reserved = temp_dir / f"dcode-artifacts-{uid}"
        reserved.mkdir()  # a real, us-owned directory; lstat is faked below

        real_lstat = Path.lstat

        def fake_lstat(self: Path) -> Any:  # noqa: ANN401
            info = real_lstat(self)
            if self == reserved:
                return SimpleNamespace(st_mode=info.st_mode, st_uid=info.st_uid + 1)
            return info

        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_dir))
        monkeypatch.setattr(Path, "lstat", fake_lstat)

        storage = _artifacts_root()
        next_storage = _artifacts_root()

        assert storage.root == "/dcode-artifacts-fallback"
        assert next_storage.root == storage.root
        assert storage.large_results_dir is not None
        assert next_storage.large_results_dir is not None
        assert not storage.large_results_dir.samefile(reserved)
        assert storage.large_results_dir.name.startswith(f"dcode-artifacts-{uid}-")
        assert stat.S_IMODE(storage.large_results_dir.stat().st_mode) == 0o700
        assert next_storage.large_results_dir != storage.large_results_dir


class TestOffloadStorageCaveat:
    """Surface the persistence caveat when offload uses ephemeral storage."""

    async def test_ephemeral_storage_appends_caveat_to_success(self) -> None:
        """A successful offload into temp storage warns it may not persist."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))
            after = _state_values(_make_dict_messages(12), _summary_event(6))

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
                patch(
                    "deepagents_code.offload.offload_storage_is_ephemeral",
                    return_value=True,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("Offloaded 6 older messages" in str(w._content) for w in msgs)
            assert any("may not survive a restart" in str(w._content) for w in msgs)

    async def test_persistent_storage_omits_caveat(self) -> None:
        """A successful offload into persistent storage adds no caveat."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))
            after = _state_values(_make_dict_messages(12), _summary_event(6))

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
                patch(
                    "deepagents_code.offload.offload_storage_is_ephemeral",
                    return_value=False,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("Offloaded 6 older messages" in str(w._content) for w in msgs)
            assert not any("may not survive a restart" in str(w._content) for w in msgs)


class TestNoopArtifactCleanup:
    """A failed no-op restoration must not be reported as an offload failure."""

    async def test_cleanup_failure_keeps_noop_report(self) -> None:
        """When restoration fails, still report the no-op — not "Offload failed"."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            agent = _setup_server_offload_app(app)
            # The no-op branch restores state via aupdate_state; make it fail.
            agent.aupdate_state = AsyncMock(side_effect=RuntimeError("write failed"))

            before = _state_values(_make_dict_messages(4))
            after = _state_values(_make_dict_messages(6))

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
                "the conversation is already compact" in str(w._content)
                for w in app.query(AppMessage)
            )
            assert not any(
                "Offload failed" in str(w._content) for w in app.query(ErrorMessage)
            )


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


class TestOffloadToolGuard:
    """Server-side tool execution guard for hidden `/offload` turns."""

    @pytest.mark.parametrize(
        "tool_call",
        [
            {"name": "write_file", "args": {"path": "x"}, "id": "model-call"},
            {
                "name": "compact_conversation",
                "args": {"force": True},
                # Even reusing the authorized ID cannot turn a later model
                # message into the one server-seeded call.
                "id": "seed-call",
            },
        ],
    )
    async def test_blocks_every_call_except_seed(
        self, tool_call: dict[str, Any]
    ) -> None:
        """Unrelated and repeated tools never reach their execution handler."""
        from langchain_core.messages import ToolMessage

        from deepagents_code.offload_middleware import CLICompactionMiddleware

        middleware = object.__new__(CLICompactionMiddleware)
        request = MagicMock()
        request.runtime.context = {"offload_tool_call_id": "seed-call"}
        request.tool_call = tool_call
        request.state = {"messages": [{"id": "model-generated-message"}]}
        handler = AsyncMock()

        result = await middleware.awrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        handler.assert_not_awaited()

    async def test_allows_exact_seeded_compaction(self) -> None:
        """The one forced call seeded by `/offload` reaches the tool handler."""
        from langchain_core.messages import ToolMessage

        from deepagents_code.offload_middleware import CLICompactionMiddleware

        middleware = object.__new__(CLICompactionMiddleware)
        request = MagicMock()
        request.runtime.context = {"offload_tool_call_id": "seed-call"}
        request.tool_call = {
            "name": "compact_conversation",
            "args": {"force": True},
            "id": "seed-call",
        }
        request.state = {"messages": [{"id": "offload-seed-seed-call"}]}
        expected = ToolMessage(content="done", tool_call_id="seed-call")
        handler = AsyncMock(return_value=expected)

        result = await middleware.awrap_tool_call(request, handler)

        assert result is expected
        handler.assert_awaited_once_with(request)

    async def test_ordinary_runs_are_unchanged(self) -> None:
        """Without `/offload` context, normal tools pass through the guard."""
        from langchain_core.messages import ToolMessage

        from deepagents_code.offload_middleware import CLICompactionMiddleware

        middleware = object.__new__(CLICompactionMiddleware)
        request = MagicMock()
        request.runtime.context = {}
        request.tool_call = {"name": "write_file", "args": {}, "id": "normal-call"}
        expected = ToolMessage(content="done", tool_call_id="normal-call")
        handler = AsyncMock(return_value=expected)

        result = await middleware.awrap_tool_call(request, handler)

        assert result is expected
        handler.assert_awaited_once_with(request)


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
                    "offload_tool_call_id": tool_call["id"],
                },
                {
                    "model": "provider:active-model",
                    "model_params": {"temperature": 0},
                    "profile_overrides": {"max_input_tokens": 4096},
                    "model_context_limit": 4096,
                    "thread_id": "test-thread",
                    "offload_tool_call_id": tool_call["id"],
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
        seed_values = agent.aupdate_state.call_args.args[1]
        (seed_msg,) = seed_values["messages"]
        (tool_call,) = seed_msg.tool_calls
        assert all(
            context
            == {
                "model": "provider:startup-model",
                "model_params": {},
                "profile_overrides": {"max_input_tokens": 4096},
                "model_context_limit": 4096,
                "thread_id": "test-thread",
                "offload_tool_call_id": tool_call["id"],
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

    async def test_approves_only_first_forced_compaction(self) -> None:
        """A repeated forced compaction request is rejected, not approved."""
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command

        from deepagents_code.client.remote_client import RemoteAgent

        astream_inputs: list[Any] = []
        guard_ids: list[object] = []

        class _Interrupt:
            def __init__(self, iid: str, tool_name: str, args: dict[str, Any]) -> None:
                self.id = iid
                self.value = {"action_requests": [{"name": tool_name, "args": args}]}

        async def _astream(stream_input: object, **kwargs: object):  # noqa: RUF029, ANN202
            idx = len(astream_inputs)
            astream_inputs.append(stream_input)
            context = kwargs.get("context")
            guard_ids.append(
                context.get("offload_tool_call_id")
                if isinstance(context, dict)
                else None
            )
            if idx == 0:
                compact = _Interrupt(
                    "i-compact", "compact_conversation", {"force": True}
                )
                yield ((), "updates", {"__interrupt__": [compact]})
            elif idx == 1:
                # Model a trailing turn that asks to compact again.
                repeated = _Interrupt(
                    "i-repeated", "compact_conversation", {"force": True}
                )
                yield ((), "updates", {"__interrupt__": [repeated]})
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
        assert len(set(guard_ids)) == 1
        assert isinstance(guard_ids[0], str)
        # Compaction was approved.
        compact_decision = astream_inputs[1].resume["i-compact"]["decisions"][0]
        assert compact_decision["type"] == "approve"
        # A second compaction request is not the seeded call and is rejected.
        repeated_decision = astream_inputs[2].resume["i-repeated"]["decisions"][0]
        assert repeated_decision["type"] == "reject"

    async def test_sets_tool_guard_context_without_hitl(self) -> None:
        """The per-run tool guard is set even when no HITL interrupt exists."""
        from langchain_core.messages import ToolMessage

        from deepagents_code.client.remote_client import RemoteAgent

        guard_ids: list[object] = []

        async def _astream(_stream_input: object, **kwargs: object):  # noqa: RUF029, ANN202
            context = kwargs.get("context")
            guard_ids.append(
                context.get("offload_tool_call_id")
                if isinstance(context, dict)
                else None
            )
            yield (
                (),
                "messages",
                (
                    ToolMessage(
                        content="Conversation compacted. Summarized 2 messages.",
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
        seed_values = agent.aupdate_state.call_args.args[1]
        (seed_msg,) = seed_values["messages"]
        (tool_call,) = seed_msg.tool_calls
        assert guard_ids == [tool_call["id"]]

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

    async def test_returns_true_when_seed_removed(self) -> None:
        """Successful removal reports the thread is clean."""
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
                cleaned = await app._remove_unanswered_offload_seed(
                    {"configurable": {"thread_id": "test-thread"}}, "seed-call"
                )

            assert cleaned is True

    async def test_returns_false_when_state_read_fails(self) -> None:
        """A failed state read cannot confirm cleanup, so it reports unclean."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            agent = MagicMock()
            agent.aupdate_state = AsyncMock()
            app._agent = agent
            with patch.object(
                app,
                "_get_thread_state_values",
                new_callable=AsyncMock,
                side_effect=RuntimeError("state read boom"),
            ):
                cleaned = await app._remove_unanswered_offload_seed(
                    {"configurable": {"thread_id": "test-thread"}}, "seed-call"
                )

            assert cleaned is False
            # The dangling seed could not be removed, so nothing was written.
            agent.aupdate_state.assert_not_awaited()

    async def test_returns_false_when_removal_write_fails(self) -> None:
        """A failed removal write leaves the seed and reports unclean."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            agent = MagicMock()
            agent.aupdate_state = AsyncMock(side_effect=RuntimeError("write boom"))
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
                cleaned = await app._remove_unanswered_offload_seed(
                    {"configurable": {"thread_id": "test-thread"}}, "seed-call"
                )

            assert cleaned is False


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
