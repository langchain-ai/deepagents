"""Unit tests for /offload slash command."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.app import DeepAgentsApp
from deepagents_cli.command_registry import SLASH_COMMANDS
from deepagents_cli.config import settings
from deepagents_cli.offload import (
    OffloadModelError,
    OffloadResult,
    OffloadThresholdNotMet,
    format_offload_limit,
    offload_messages_to_backend,
)
from deepagents_cli.textual_adapter import format_token_count
from deepagents_cli.widgets.messages import AppMessage, ErrorMessage

# Patch target for perform_offload (business logic)
_PERFORM_OFFLOAD_PATH = "deepagents_cli.offload.perform_offload"

# Patch targets for lower-level offload_messages_to_backend tests
_GET_BUFFER_STRING_PATH = "deepagents_cli.offload.get_buffer_string"


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


def _make_offload_result(
    *,
    messages_offloaded: int = 4,
    messages_kept: int = 6,
    tokens_before: int = 1000,
    tokens_after: int = 500,
    pct_decrease: int = 50,
    offload_warning: str | None = None,
    cutoff_index: int = 4,
    file_path: str | None = "/conversation_history/test-thread.md",
) -> OffloadResult:
    """Build an `OffloadResult` with sensible defaults for UI tests."""
    summary_msg = MagicMock()
    summary_msg.content = "Summary of the conversation."
    summary_msg.additional_kwargs = {"lc_source": "summarization"}
    new_event: dict[str, Any] = {
        "cutoff_index": cutoff_index,
        "summary_message": summary_msg,
        "file_path": file_path,
    }
    return OffloadResult(
        new_event=new_event,  # ty: ignore[invalid-argument-type]
        messages_offloaded=messages_offloaded,
        messages_kept=messages_kept,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        pct_decrease=pct_decrease,
        offload_warning=offload_warning,
    )


def _make_threshold_not_met(
    *,
    conversation_tokens: int = 100,
    total_context_tokens: int = 0,
    context_limit: int | None = None,
    budget_str: str = "last 6 messages",
) -> OffloadThresholdNotMet:
    """Build an `OffloadThresholdNotMet` with sensible defaults."""
    return OffloadThresholdNotMet(
        conversation_tokens=conversation_tokens,
        total_context_tokens=total_context_tokens,
        context_limit=context_limit,
        budget_str=budget_str,
    )


def _setup_offload_app(
    app: DeepAgentsApp,
    n_messages: int = 10,
    *,
    prior_event: dict[str, Any] | None = None,
) -> list[MagicMock]:
    """Set up app state for an offload test.

    Args:
        app: The app instance to configure.
        n_messages: Number of mock messages to create.
        prior_event: Optional prior `_summarization_event` to include in state.

    Returns:
        The list of mock messages.
    """
    messages = _make_messages(n_messages)
    mock_state = MagicMock()
    values: dict[str, Any] = {"messages": messages}
    if prior_event is not None:
        values["_summarization_event"] = prior_event
    mock_state.values = values

    app._agent = MagicMock()
    app._agent.aget_state = AsyncMock(return_value=mock_state)
    app._agent.aupdate_state = AsyncMock()
    app._backend = MagicMock()
    app._lc_thread_id = "test-thread"
    app._agent_running = False
    return messages


class TestOffloadInAutocomplete:
    """Verify /offload is registered in the autocomplete system."""

    def test_offload_in_slash_commands(self) -> None:
        """The /offload command should be in the SLASH_COMMANDS list."""
        labels = [label for label, *_ in SLASH_COMMANDS]
        assert "/offload" in labels

    def test_offload_sorted_alphabetically(self) -> None:
        """The /offload entry should appear between /model and /quit."""
        labels = [label for label, *_ in SLASH_COMMANDS]
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

    async def test_cutoff_zero_shows_not_enough(self) -> None:
        """Should show info when perform_offload returns threshold not met."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app, n_messages=3)

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                return_value=_make_threshold_not_met(
                    conversation_tokens=45,
                    budget_str="last 6 messages",
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("within the retention budget" in str(w._content) for w in msgs)

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

    async def test_successful_offload_sets_event(self) -> None:
        """Should set _summarization_event with cutoff and summary message."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app)

            result = _make_offload_result(
                cutoff_index=4,
                file_path="/conversation_history/test-thread.md",
            )

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                return_value=result,
            ):
                await app._handle_offload()
                await pilot.pause()

            mock_agent = app._agent
            assert mock_agent.aupdate_state.call_count == 1  # type: ignore[union-attr]

            update_values = mock_agent.aupdate_state.call_args_list[0][0][1]  # type: ignore[union-attr]
            event = update_values["_summarization_event"]
            assert event["cutoff_index"] == 4
            assert event["summary_message"] is not None
            assert event["file_path"] == "/conversation_history/test-thread.md"

    async def test_offload_shows_feedback_message(self) -> None:
        """Should display feedback with message count and token change."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app)

            result = _make_offload_result(messages_offloaded=4)

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                return_value=result,
            ):
                await app._handle_offload()
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("Offloaded 4 older messages" in str(w._content) for w in msgs)

    async def test_offload_updates_token_tracker(self) -> None:
        """Should update token tracker after offload."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app)
            app._token_tracker = MagicMock()

            result = _make_offload_result(tokens_after=500)

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                return_value=result,
            ):
                await app._handle_offload()
                await pilot.pause()

            app._token_tracker.add.assert_called_once_with(500)

    async def test_no_ui_clear_reload(self) -> None:
        """Should NOT clear/reload UI since messages stay in state."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app)

            result = _make_offload_result()

            with (
                patch(
                    _PERFORM_OFFLOAD_PATH,
                    new_callable=AsyncMock,
                    return_value=result,
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

    async def test_cutoff_zero_does_not_update_state(self) -> None:
        """When perform_offload returns threshold-not-met, no state update."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app, n_messages=6)

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                return_value=_make_threshold_not_met(
                    conversation_tokens=45,
                    budget_str="last 6 messages",
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("within the retention budget" in str(w._content) for w in msgs)
            app._agent.aupdate_state.assert_not_called()  # type: ignore[union-attr]

    async def test_cutoff_zero_overhead_dominated(self) -> None:
        """Show overhead message when context exceeds limit."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app, n_messages=3)

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                return_value=_make_threshold_not_met(
                    conversation_tokens=45,
                    total_context_tokens=14_000,
                    context_limit=4_096,
                    budget_str="last 6 messages",
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("can't be offloaded" in str(w._content) for w in msgs)

    async def test_cutoff_one_offloads_single_message(self) -> None:
        """With cutoff=1, event should have cutoff_index=1."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app, n_messages=7)

            result = _make_offload_result(
                cutoff_index=1,
                messages_offloaded=1,
                messages_kept=6,
            )

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                return_value=result,
            ):
                await app._handle_offload()
                await pilot.pause()

            mock_agent = app._agent
            update_values = mock_agent.aupdate_state.call_args_list[0][0][1]  # type: ignore[union-attr]
            event = update_values["_summarization_event"]
            assert event["cutoff_index"] == 1

    async def test_perform_offload_called_with_correct_args(self) -> None:
        """Should pass correct args from app state to perform_offload."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            messages = _setup_offload_app(app, n_messages=10)
            app._token_tracker = MagicMock()
            app._token_tracker.current_context = 7500
            app._profile_override = {"temperature": 0.5}

            result = _make_offload_result()

            with (
                patch(
                    _PERFORM_OFFLOAD_PATH,
                    new_callable=AsyncMock,
                    return_value=result,
                ) as mock_perform,
                patch.object(settings, "model_provider", "openai"),
                patch.object(settings, "model_name", "gpt-4"),
                patch.object(settings, "model_context_limit", 128_000),
            ):
                await app._handle_offload()
                await pilot.pause()

            mock_perform.assert_called_once()
            kwargs = mock_perform.call_args.kwargs
            assert kwargs["messages"] == messages
            assert kwargs["prior_event"] is None
            assert kwargs["thread_id"] == "test-thread"
            assert kwargs["model_spec"] == "openai:gpt-4"
            assert kwargs["profile_overrides"] == {"temperature": 0.5}
            assert kwargs["context_limit"] == 128_000
            assert kwargs["total_context_tokens"] == 7500
            assert kwargs["backend"] is app._backend


class TestReOffload:
    """Test offload when a prior _summarization_event already exists."""

    async def test_reoffload_calculates_absolute_cutoff(self) -> None:
        """Re-offload should pass prior_event to perform_offload.

        The actual cutoff calculation is in offload.py; here we verify
        the UI layer forwards state correctly and applies the returned event.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            prior_summary = MagicMock()
            prior_summary.content = "Old summary."
            prior_summary.additional_kwargs = {"lc_source": "summarization"}
            prior_event = {
                "cutoff_index": 5,
                "summary_message": prior_summary,
                "file_path": None,
            }
            _setup_offload_app(app, n_messages=15, prior_event=prior_event)

            # offload.py would compute old_cutoff(5) + new_cutoff(3) - 1 = 7
            result = _make_offload_result(
                cutoff_index=7,
                messages_offloaded=3,
                messages_kept=12,
            )

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                return_value=result,
            ) as mock_perform:
                await app._handle_offload()
                await pilot.pause()

            mock_agent = app._agent
            assert mock_agent.aupdate_state.call_count == 1  # type: ignore[union-attr]

            update_values = mock_agent.aupdate_state.call_args_list[0][0][1]  # type: ignore[union-attr]
            event = update_values["_summarization_event"]
            assert event["cutoff_index"] == 7

            # Verify prior_event was passed through
            kwargs = mock_perform.call_args.kwargs
            assert kwargs["prior_event"] is prior_event


class TestAgentRunningGuard:
    """Test that _handle_offload sets _agent_running to prevent races."""

    async def test_agent_running_set_during_offload(self) -> None:
        """Should set _agent_running=True during offload and reset after."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app)

            running_during_offload: list[bool] = []

            def capture_running(**_kwargs: Any) -> OffloadResult:
                running_during_offload.append(app._agent_running)
                return _make_offload_result()

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                side_effect=capture_running,
            ):
                await app._handle_offload()
                await pilot.pause()

            # _agent_running should have been True during perform_offload
            assert running_during_offload == [True]
            # And reset after completion
            assert app._agent_running is False

    async def test_agent_running_reset_after_failure(self) -> None:
        """Should reset _agent_running=False even when offload fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app)

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                side_effect=RuntimeError("model down"),
            ):
                await app._handle_offload()
                await pilot.pause()

            assert app._agent_running is False


class TestOffloadErrorHandling:
    """Test error handling during offload."""

    async def test_offload_failure_proceeds_without_path(self) -> None:
        """Should display warning when offload_warning is set."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app)

            result = _make_offload_result(
                file_path=None,
                offload_warning=(
                    "Warning: conversation history could not be saved to "
                    "storage. Older messages will not be recoverable."
                ),
            )

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                return_value=result,
            ):
                await app._handle_offload()
                await pilot.pause()

            mock_agent = app._agent
            assert mock_agent.aupdate_state.call_count == 1  # type: ignore[union-attr]

            update_values = mock_agent.aupdate_state.call_args_list[0][0][1]  # type: ignore[union-attr]
            event = update_values["_summarization_event"]
            assert event["file_path"] is None

    async def test_summary_generation_failure_shows_error(self) -> None:
        """Should show error and leave state untouched when perform_offload raises."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app)

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                side_effect=RuntimeError("model unavailable"),
            ):
                await app._handle_offload()
                await pilot.pause()

            # State should not have been updated
            app._agent.aupdate_state.assert_not_called()  # type: ignore[union-attr]

            error_msgs = app.query(ErrorMessage)
            assert any("Offload failed" in str(w._content) for w in error_msgs)

    async def test_state_update_failure_shows_error(self) -> None:
        """Should show error when aupdate_state raises."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app)
            app._agent.aupdate_state = AsyncMock(  # type: ignore[union-attr]
                side_effect=RuntimeError("state write failed")
            )

            result = _make_offload_result()

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                return_value=result,
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
            _setup_offload_app(app)

            with (
                patch(
                    _PERFORM_OFFLOAD_PATH,
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


class TestCreateModelFailure:
    """Test that _handle_offload handles OffloadModelError from perform_offload."""

    async def test_create_model_failure_shows_error(self) -> None:
        """Should show error when perform_offload raises OffloadModelError."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app)

            with patch(
                _PERFORM_OFFLOAD_PATH,
                new_callable=AsyncMock,
                side_effect=OffloadModelError(
                    "Offload requires a working model configuration: no API key"
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            error_msgs = app.query(ErrorMessage)
            assert any(
                "working model configuration" in str(w._content) for w in error_msgs
            )
            # State should not have been modified
            app._agent.aupdate_state.assert_not_called()  # type: ignore[union-attr]
            # _agent_running must be reset so the UI doesn't lock up
            assert app._agent_running is False


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


class TestOffloadRemoteFallback:
    """Verify `/offload` handles resumed remote threads."""

    async def test_resumed_remote_thread_uses_checkpointer_state(self) -> None:
        """Should offload using checkpoint fallback when remote state is empty."""
        from deepagents_cli.remote_client import RemoteAgent

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            messages = _make_messages(15)
            prior_summary = MagicMock()
            prior_summary.content = "Old summary."
            prior_summary.additional_kwargs = {"lc_source": "summarization"}
            prior_event = {
                "cutoff_index": 5,
                "summary_message": prior_summary,
                "file_path": None,
            }

            empty_state = MagicMock()
            empty_state.values = {}

            app._agent = MagicMock(spec=RemoteAgent)
            app._agent.aget_state = AsyncMock(return_value=empty_state)
            app._agent.aensure_thread = AsyncMock()
            app._agent.aupdate_state = AsyncMock()
            app._backend = MagicMock()
            app._lc_thread_id = "test-thread"
            app._agent_running = False

            # offload.py computes old_cutoff(5) + new_cutoff(3) - 1 = 7
            result = _make_offload_result(
                cutoff_index=7,
                messages_offloaded=3,
                messages_kept=12,
            )

            with (
                patch.object(
                    DeepAgentsApp,
                    "_read_channel_values_from_checkpointer",
                    return_value={
                        "messages": messages,
                        "_summarization_event": prior_event,
                    },
                ) as checkpointer_mock,
                patch(
                    _PERFORM_OFFLOAD_PATH,
                    new_callable=AsyncMock,
                    return_value=result,
                ) as mock_perform,
            ):
                await app._handle_offload()
                await pilot.pause()

            checkpointer_mock.assert_awaited_once_with("test-thread")

            # Verify perform_offload was called with the fallback state
            kwargs = mock_perform.call_args.kwargs
            assert kwargs["messages"] == messages
            assert kwargs["prior_event"] is prior_event

            app._agent.aensure_thread.assert_awaited_once_with(
                {"configurable": {"thread_id": "test-thread"}}
            )

            update_values = app._agent.aupdate_state.call_args_list[0][0][1]
            event = update_values["_summarization_event"]
            assert event["cutoff_index"] == 7


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


class TestOffloadProfileOverride:
    """Verify /offload respects profile overrides (--profile-override / config.toml).

    Since profile-override logic now lives in offload.py, these tests verify
    the UI layer passes the correct kwargs to `perform_offload`.
    """

    async def test_offload_passes_context_limit_to_perform_offload(self) -> None:
        """Settings.model_context_limit should be forwarded to perform_offload."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app, n_messages=5)

            result = _make_threshold_not_met()

            with (
                patch(
                    _PERFORM_OFFLOAD_PATH,
                    new_callable=AsyncMock,
                    return_value=result,
                ) as mock_perform,
                patch.object(settings, "model_context_limit", 4096),
            ):
                await app._handle_offload()
                await pilot.pause()

            kwargs = mock_perform.call_args.kwargs
            assert kwargs["context_limit"] == 4096

    async def test_offload_passes_matching_context_limit(self) -> None:
        """When override matches native profile value, same value is forwarded."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app, n_messages=5)

            result = _make_threshold_not_met()

            with (
                patch(
                    _PERFORM_OFFLOAD_PATH,
                    new_callable=AsyncMock,
                    return_value=result,
                ) as mock_perform,
                patch.object(settings, "model_context_limit", 200_000),
            ):
                await app._handle_offload()
                await pilot.pause()

            kwargs = mock_perform.call_args.kwargs
            assert kwargs["context_limit"] == 200_000

    async def test_offload_override_triggers_offload(self) -> None:
        """With a small override, perform_offload returns OffloadResult."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app, n_messages=8)

            result = _make_offload_result(
                cutoff_index=4,
                messages_offloaded=4,
                messages_kept=4,
            )

            with (
                patch(
                    _PERFORM_OFFLOAD_PATH,
                    new_callable=AsyncMock,
                    return_value=result,
                ) as mock_perform,
                patch.object(settings, "model_context_limit", 4096),
            ):
                await app._handle_offload()
                await pilot.pause()

            # State should have been updated (offload happened)
            app._agent.aupdate_state.assert_called_once()  # type: ignore[union-attr]
            kwargs = mock_perform.call_args.kwargs
            assert kwargs["context_limit"] == 4096

    async def test_offload_override_none_passes_none(self) -> None:
        """When model_context_limit is None, None is forwarded to perform_offload."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app, n_messages=5)

            result = _make_threshold_not_met()

            with (
                patch(
                    _PERFORM_OFFLOAD_PATH,
                    new_callable=AsyncMock,
                    return_value=result,
                ) as mock_perform,
                patch.object(settings, "model_context_limit", None),
            ):
                await app._handle_offload()
                await pilot.pause()

            kwargs = mock_perform.call_args.kwargs
            assert kwargs["context_limit"] is None

    async def test_offload_passes_profile_overrides(self) -> None:
        """Profile overrides from _profile_override should be forwarded."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_offload_app(app, n_messages=5)
            app._profile_override = {"max_input_tokens": 4096}

            result = _make_threshold_not_met()

            with (
                patch(
                    _PERFORM_OFFLOAD_PATH,
                    new_callable=AsyncMock,
                    return_value=result,
                ) as mock_perform,
            ):
                await app._handle_offload()
                await pilot.pause()

            kwargs = mock_perform.call_args.kwargs
            assert kwargs["profile_overrides"] == {"max_input_tokens": 4096}


# ---------------------------------------------------------------------------
# Patch targets for perform_offload direct tests
# ---------------------------------------------------------------------------
_CREATE_MODEL_PATH = "deepagents_cli.offload.create_model"
_COMPUTE_DEFAULTS_PATH = (
    "deepagents.middleware.summarization.compute_summarization_defaults"
)
_MW_CLASS_PATH = "deepagents.middleware.summarization.SummarizationMiddleware"
_TOKEN_COUNT_PATH = "deepagents_cli.offload.count_tokens_approximately"
_OFFLOAD_BACKEND_PATH = "deepagents_cli.offload.offload_messages_to_backend"


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
        from deepagents_cli.offload import perform_offload

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
        from deepagents_cli.offload import perform_offload

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
        from deepagents_cli.offload import OffloadModelError, perform_offload

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
        from deepagents_cli.offload import perform_offload

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
        from deepagents_cli.offload import perform_offload

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
        from deepagents_cli.offload import perform_offload

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
        from deepagents_cli.offload import perform_offload

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
        from deepagents_cli.offload import perform_offload

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
