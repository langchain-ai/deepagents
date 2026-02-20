"""Unit tests for /compact slash command."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.app import DeepAgentsApp, _format_token_count
from deepagents_cli.widgets.autocomplete import SLASH_COMMANDS
from deepagents_cli.widgets.messages import AppMessage, ErrorMessage

if TYPE_CHECKING:
    from collections.abc import Generator

# Patch target for count_tokens_approximately used inside _handle_compact
_TOKEN_COUNT_PATH = "langchain_core.messages.utils.count_tokens_approximately"

# Patch targets for middleware-based partitioning in _handle_compact
_CREATE_MODEL_PATH = "deepagents_cli.app.create_model"
_COMPUTE_DEFAULTS_PATH = (
    "deepagents.middleware.summarization.compute_summarization_defaults"
)
_LC_MIDDLEWARE_PATH = "deepagents.middleware.summarization.SummarizationMiddleware"
_GET_BUFFER_STRING_PATH = "langchain_core.messages.get_buffer_string"


def _real_build_summary_msg(summary: str, file_path: str | None) -> list[Any]:
    """Build a real HumanMessage matching SummarizationMiddleware format."""
    from langchain_core.messages import HumanMessage

    if file_path is not None:
        content = (
            "You are in the middle of a conversation "
            "that has been summarized.\n\n"
            "The full conversation history has been "
            f"saved to {file_path} "
            "should you need to refer back to it "
            "for details.\n\n"
            "A condensed summary follows:\n\n"
            f"<summary>\n{summary}\n</summary>"
        )
    else:
        content = "Here is a summary of the conversation to date:\n\n" + summary

    return [
        HumanMessage(
            content=content,
            additional_kwargs={"lc_source": "summarization"},
        )
    ]


@contextmanager
def _mock_middleware(
    *,
    cutoff: int,
    summary: str = "Summary of the conversation.",
) -> Generator[MagicMock, None, None]:
    """Patch `create_model`, defaults, and `SummarizationMiddleware`.

    Args:
        cutoff: Value returned by `_determine_cutoff_index`.
        summary: Text returned by `_acreate_summary`.

    Yields:
        The mock middleware instance.
    """
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.model = mock_model

    mock_mw = MagicMock()
    mock_mw._determine_cutoff_index.return_value = cutoff
    mock_mw._partition_messages.side_effect = lambda msgs, idx: (
        msgs[:idx],
        msgs[idx:],
    )
    mock_mw._acreate_summary = AsyncMock(return_value=summary)
    mock_mw._build_new_messages_with_path.side_effect = _real_build_summary_msg
    mock_mw._apply_event_to_messages.side_effect = lambda msgs, event: (
        list(msgs)
        if event is None
        else [event["summary_message"], *msgs[event["cutoff_index"] :]]
    )
    mock_mw._compute_state_cutoff.side_effect = lambda event, effective_cutoff: (
        effective_cutoff
        if event is None
        else event["cutoff_index"] + effective_cutoff - 1
    )

    with (
        patch(_CREATE_MODEL_PATH, return_value=mock_result),
        patch(
            _COMPUTE_DEFAULTS_PATH,
            return_value={"keep": ("fraction", 0.10)},
        ),
        patch(_LC_MIDDLEWARE_PATH, return_value=mock_mw),
    ):
        yield mock_mw


class TestCompactInAutocomplete:
    """Verify /compact is registered in the autocomplete system."""

    def test_compact_in_slash_commands(self) -> None:
        """The /compact command should be in the SLASH_COMMANDS list."""
        labels = [label for label, _ in SLASH_COMMANDS]
        assert "/compact" in labels

    def test_compact_sorted_alphabetically(self) -> None:
        """The /compact entry should appear between /clear and /docs."""
        labels = [label for label, _ in SLASH_COMMANDS]
        clear_idx = labels.index("/clear")
        compact_idx = labels.index("/compact")
        docs_idx = labels.index("/docs")
        assert clear_idx < compact_idx < docs_idx


class TestCompactGuards:
    """Test guard conditions that prevent compaction."""

    @pytest.mark.asyncio
    async def test_no_agent_shows_error(self) -> None:
        """Should show error when there is no active agent."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = None
            app._lc_thread_id = None

            await app._handle_compact()
            await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("No active session" in str(w._content) for w in msgs)

    @pytest.mark.asyncio
    async def test_agent_running_shows_error(self) -> None:
        """Should show error when agent is currently running."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._backend = MagicMock()
            app._lc_thread_id = "test-thread"
            app._agent_running = True

            await app._handle_compact()
            await pilot.pause()

            msgs = app.query(AppMessage)
            assert any(
                "Cannot compact while agent is running" in str(w._content) for w in msgs
            )

    @pytest.mark.asyncio
    async def test_cutoff_zero_shows_not_enough(self) -> None:
        """Should show error when middleware cutoff is zero."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app, n_messages=3)

            with _mock_middleware(cutoff=0):
                await app._handle_compact()
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("Nothing to compact yet" in str(w._content) for w in msgs)

    @pytest.mark.asyncio
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

            await app._handle_compact()
            await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("No active session" in str(w._content) for w in msgs)

    @pytest.mark.asyncio
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

            await app._handle_compact()
            await pilot.pause()

            msgs = app.query(ErrorMessage)
            assert any("Failed to read state" in str(w._content) for w in msgs)


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


def _setup_compact_app(
    app: DeepAgentsApp,
    n_messages: int = 10,
    *,
    prior_event: dict[str, Any] | None = None,
) -> list[MagicMock]:
    """Set up app state for a successful compaction test.

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


class TestCompactSuccess:
    """Test successful compaction flow."""

    @pytest.mark.asyncio
    async def test_successful_compaction_sets_event(self) -> None:
        """Should set _summarization_event with cutoff and summary message."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            with (
                _mock_middleware(cutoff=4),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value="/conversation_history/test-thread.md",
                ),
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                await app._handle_compact()
                await pilot.pause()

            mock_agent = app._agent
            # Single aupdate_state call to set the event
            assert mock_agent.aupdate_state.call_count == 1  # type: ignore[union-attr]

            update_values = mock_agent.aupdate_state.call_args_list[0][0][1]  # type: ignore[union-attr]
            event = update_values["_summarization_event"]
            assert event["cutoff_index"] == 4
            assert event["summary_message"] is not None
            assert event["file_path"] == "/conversation_history/test-thread.md"

            # Summary message should have correct content
            summary_msg = event["summary_message"]
            assert summary_msg.additional_kwargs.get("lc_source") == "summarization"
            assert "/conversation_history/test-thread.md" in summary_msg.content
            assert "<summary>" in summary_msg.content
            assert "Summary of the conversation." in summary_msg.content

    @pytest.mark.asyncio
    async def test_compaction_shows_feedback_message(self) -> None:
        """Should display feedback with message count and token change."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            with (
                _mock_middleware(cutoff=4, summary="Summary."),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                await app._handle_compact()
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("Compacted 4 messages" in str(w._content) for w in msgs)

    @pytest.mark.asyncio
    async def test_compaction_updates_token_tracker(self) -> None:
        """Should update token tracker after compaction."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)
            app._token_tracker = MagicMock()

            with (
                _mock_middleware(cutoff=4, summary="Summary."),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                await app._handle_compact()
                await pilot.pause()

            app._token_tracker.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_ui_clear_reload(self) -> None:
        """Should NOT clear/reload UI since messages stay in state."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            with (
                _mock_middleware(cutoff=4, summary="Summary."),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch.object(
                    app, "_clear_messages", new_callable=AsyncMock
                ) as mock_clear,
                patch.object(
                    app, "_load_thread_history", new_callable=AsyncMock
                ) as mock_load,
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                await app._handle_compact()
                await pilot.pause()

            mock_clear.assert_not_called()
            mock_load.assert_not_called()


class TestCompactEdgeCases:
    """Test edge cases in the compaction logic."""

    @pytest.mark.asyncio
    async def test_cutoff_zero_does_not_update_state(self) -> None:
        """When middleware returns cutoff=0, state should not be modified."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app, n_messages=6)

            with _mock_middleware(cutoff=0):
                await app._handle_compact()
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("Nothing to compact yet" in str(w._content) for w in msgs)
            app._agent.aupdate_state.assert_not_called()  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_cutoff_one_compacts_single_message(self) -> None:
        """With cutoff=1, event should have cutoff_index=1."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app, n_messages=7)

            with (
                _mock_middleware(cutoff=1, summary="Summary."),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch(_TOKEN_COUNT_PATH, return_value=100),
            ):
                await app._handle_compact()
                await pilot.pause()

            mock_agent = app._agent
            update_values = mock_agent.aupdate_state.call_args_list[0][0][1]  # type: ignore[union-attr]
            event = update_values["_summarization_event"]
            assert event["cutoff_index"] == 1

    @pytest.mark.asyncio
    async def test_middleware_cutoff_called_with_effective_messages(self) -> None:
        """Should pass effective messages to middleware cutoff logic."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            messages = _setup_compact_app(app, n_messages=10)

            with (
                _mock_middleware(cutoff=4, summary="Summary.") as mock_mw,
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                await app._handle_compact()
                await pilot.pause()

            # _apply_event_to_messages should be called
            mock_mw._apply_event_to_messages.assert_called_once_with(messages, None)
            # cutoff called with effective messages (same as raw when no event)
            mock_mw._determine_cutoff_index.assert_called_once_with(messages)
            mock_mw._partition_messages.assert_called_once_with(messages, 4)


class TestReCompaction:
    """Test compaction when a prior _summarization_event already exists."""

    @pytest.mark.asyncio
    async def test_recompact_calculates_absolute_cutoff(self) -> None:
        """Re-compaction should compute state_cutoff = old_cutoff + new_cutoff - 1."""
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
            _setup_compact_app(app, n_messages=15, prior_event=prior_event)

            with (
                _mock_middleware(cutoff=3, summary="New summary."),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                await app._handle_compact()
                await pilot.pause()

            mock_agent = app._agent
            assert mock_agent.aupdate_state.call_count == 1  # type: ignore[union-attr]

            update_values = mock_agent.aupdate_state.call_args_list[0][0][1]  # type: ignore[union-attr]
            event = update_values["_summarization_event"]
            # old_cutoff(5) + new_cutoff(3) - 1 = 7
            assert event["cutoff_index"] == 7


class TestAgentRunningGuard:
    """Test that _handle_compact sets _agent_running to prevent races."""

    @pytest.mark.asyncio
    async def test_agent_running_set_during_compaction(self) -> None:
        """Should set _agent_running=True during compaction and reset after."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            running_during_compact = []

            original_acreate = AsyncMock(return_value="Summary.")

            async def capture_running(*args: Any, **kwargs: Any) -> str:
                running_during_compact.append(app._agent_running)
                return await original_acreate(*args, **kwargs)

            with (
                _mock_middleware(cutoff=4) as mock_mw,
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                mock_mw._acreate_summary = AsyncMock(side_effect=capture_running)
                await app._handle_compact()
                await pilot.pause()

            # _agent_running should have been True during summary generation
            assert running_during_compact == [True]
            # And reset after completion
            assert app._agent_running is False

    @pytest.mark.asyncio
    async def test_agent_running_reset_after_failure(self) -> None:
        """Should reset _agent_running=False even when compaction fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            with _mock_middleware(cutoff=4) as mock_mw:
                mock_mw._acreate_summary = AsyncMock(
                    side_effect=RuntimeError("model down")
                )
                await app._handle_compact()
                await pilot.pause()

            assert app._agent_running is False


class TestCompactErrorHandling:
    """Test error handling during compaction."""

    @pytest.mark.asyncio
    async def test_offload_failure_proceeds_without_path(self) -> None:
        """Should proceed with compaction even if offload fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            with (
                _mock_middleware(cutoff=4, summary="Summary."),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                await app._handle_compact()
                await pilot.pause()

            mock_agent = app._agent
            assert mock_agent.aupdate_state.call_count == 1  # type: ignore[union-attr]

            update_values = mock_agent.aupdate_state.call_args_list[0][0][1]  # type: ignore[union-attr]
            event = update_values["_summarization_event"]
            assert event["file_path"] is None

            # Summary should NOT have file path reference
            summary_content = event["summary_message"].content
            assert "conversation history has been saved" not in summary_content

    @pytest.mark.asyncio
    async def test_summary_generation_failure_shows_error(self) -> None:
        """Should show error and leave state untouched when summary fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            with _mock_middleware(cutoff=4) as mock_mw:
                mock_mw._acreate_summary = AsyncMock(
                    side_effect=RuntimeError("model unavailable")
                )
                await app._handle_compact()
                await pilot.pause()

            # State should not have been updated
            app._agent.aupdate_state.assert_not_called()  # type: ignore[union-attr]

            error_msgs = app.query(ErrorMessage)
            assert any("Compaction failed" in str(w._content) for w in error_msgs)

    @pytest.mark.asyncio
    async def test_state_update_failure_shows_error(self) -> None:
        """Should show error when aupdate_state raises."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)
            app._agent.aupdate_state = AsyncMock(  # type: ignore[union-attr]
                side_effect=RuntimeError("state write failed")
            )

            with (
                _mock_middleware(cutoff=4, summary="Summary."),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                await app._handle_compact()
                await pilot.pause()

            error_msgs = app.query(ErrorMessage)
            assert any("Compaction failed" in str(w._content) for w in error_msgs)

    @pytest.mark.asyncio
    async def test_spinner_hidden_after_failure(self) -> None:
        """Should hide spinner even when compaction fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            with (
                _mock_middleware(cutoff=4, summary="Summary."),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("backend down"),
                ),
                patch.object(
                    app, "_set_spinner", new_callable=AsyncMock
                ) as mock_spinner,
            ):
                await app._handle_compact()
                await pilot.pause()

            # Spinner should be shown then hidden
            assert mock_spinner.call_count == 2
            mock_spinner.assert_any_call("Compacting")
            mock_spinner.assert_any_call(None)


class TestCreateModelFailure:
    """Test that _handle_compact handles create_model() failures."""

    @pytest.mark.asyncio
    async def test_create_model_failure_shows_error(self) -> None:
        """Should show error when create_model() raises."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            with patch(_CREATE_MODEL_PATH, side_effect=ValueError("no API key")):
                await app._handle_compact()
                await pilot.pause()

            error_msgs = app.query(ErrorMessage)
            assert any(
                "working model configuration" in str(w._content) for w in error_msgs
            )
            # State should not have been modified
            app._agent.aupdate_state.assert_not_called()  # type: ignore[union-attr]
            # _agent_running must be reset so the UI doesn't lock up
            assert app._agent_running is False


class TestOffloadMessagesForCompact:
    """Test _offload_messages_for_compact code paths."""

    @pytest.mark.asyncio
    async def test_filters_summary_messages(self) -> None:
        """Should use middleware._filter_summary_messages to exclude summaries."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            mock_mw = MagicMock()
            # Mix of real and summary messages — filter keeps only non-summaries
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
            app._backend = mock_backend

            with patch(_GET_BUFFER_STRING_PATH, return_value="msg text"):
                result = await app._offload_messages_for_compact(messages, mock_mw)

            mock_mw._filter_summary_messages.assert_called_once_with(messages)
            assert result is not None
            assert result != ""

    @pytest.mark.asyncio
    async def test_all_summary_messages_returns_empty(self) -> None:
        """Should return empty string when all messages are summaries."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            mock_mw = MagicMock()
            mock_mw._filter_summary_messages.return_value = []

            result = await app._offload_messages_for_compact(_make_messages(2), mock_mw)

            assert result == ""

    @pytest.mark.asyncio
    async def test_appends_to_existing_content(self) -> None:
        """Should append new section to existing history file."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

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
            app._backend = mock_backend

            with patch(_GET_BUFFER_STRING_PATH, return_value="msg text"):
                result = await app._offload_messages_for_compact(messages, mock_mw)

            assert result is not None
            # Should have called aedit (not awrite) since existing content exists
            mock_backend.aedit.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_new_file_when_none_exists(self) -> None:
        """Should call awrite when no existing file is found."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

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
            app._backend = mock_backend

            with patch(_GET_BUFFER_STRING_PATH, return_value="msg text"):
                result = await app._offload_messages_for_compact(messages, mock_mw)

            assert result is not None
            mock_backend.awrite.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_failure_returns_none(self) -> None:
        """Should return None when reading existing file fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            mock_mw = MagicMock()
            mock_mw._filter_summary_messages.return_value = _make_messages(2)

            mock_backend = MagicMock()
            mock_backend.adownload_files = AsyncMock(
                side_effect=RuntimeError("storage unavailable")
            )
            app._backend = mock_backend

            with patch(_GET_BUFFER_STRING_PATH, return_value="msg text"):
                result = await app._offload_messages_for_compact(
                    _make_messages(2), mock_mw
                )

            assert result is None

    @pytest.mark.asyncio
    async def test_write_failure_returns_none(self) -> None:
        """Should return None when writing to backend fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            mock_mw = MagicMock()
            mock_mw._filter_summary_messages.return_value = _make_messages(2)

            resp = MagicMock()
            resp.content = None
            resp.error = None
            mock_backend = MagicMock()
            mock_backend.adownload_files = AsyncMock(return_value=[resp])
            mock_backend.awrite = AsyncMock(side_effect=RuntimeError("disk full"))
            app._backend = mock_backend

            with patch(_GET_BUFFER_STRING_PATH, return_value="msg text"):
                result = await app._offload_messages_for_compact(
                    _make_messages(2), mock_mw
                )

            assert result is None

    @pytest.mark.asyncio
    async def test_write_error_result_returns_none(self) -> None:
        """Should return None when write result contains an error."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

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
            app._backend = mock_backend

            with patch(_GET_BUFFER_STRING_PATH, return_value="msg text"):
                result = await app._offload_messages_for_compact(
                    _make_messages(2), mock_mw
                )

            assert result is None


class TestCompactRouting:
    """Test that /compact is routed through _handle_command."""

    @pytest.mark.asyncio
    async def test_compact_routed_from_handle_command(self) -> None:
        """'/compact' should be correctly routed through _handle_command."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = None
            app._lc_thread_id = None

            await app._handle_command("/compact")
            await pilot.pause()

            msgs = app.query(AppMessage)
            assert any("No active session" in str(w._content) for w in msgs)


class TestFormatTokenCount:
    """Test the _format_token_count helper function."""

    def test_zero(self) -> None:
        assert _format_token_count(0) == "0"

    def test_below_threshold(self) -> None:
        assert _format_token_count(999) == "999"

    def test_at_threshold(self) -> None:
        assert _format_token_count(1000) == "1.0K"

    def test_above_threshold(self) -> None:
        assert _format_token_count(1500) == "1.5K"

    def test_large_value(self) -> None:
        assert _format_token_count(200000) == "200.0K"

    def test_millions(self) -> None:
        assert _format_token_count(1_000_000) == "1.0M"

    def test_above_million(self) -> None:
        assert _format_token_count(2_500_000) == "2.5M"
