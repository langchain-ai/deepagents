"""Unit tests for /compact slash command."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.app import DeepAgentsApp
from deepagents_cli.widgets.autocomplete import SLASH_COMMANDS
from deepagents_cli.widgets.messages import AppMessage, ErrorMessage

if TYPE_CHECKING:
    from collections.abc import Generator

# Patch target for count_tokens_approximately used inside _handle_compact
_TOKEN_COUNT_PATH = "langchain_core.messages.utils.count_tokens_approximately"

# Patch targets for middleware-based partitioning in _handle_compact
_CREATE_MODEL_PATH = "deepagents_cli.app.create_model"
_COMPUTE_DEFAULTS_PATH = (
    "deepagents.middleware.summarization._compute_summarization_defaults"
)
_LC_MIDDLEWARE_PATH = (
    "langchain.agents.middleware.summarization.SummarizationMiddleware"
)


@contextmanager
def _mock_middleware(*, cutoff: int) -> Generator[MagicMock, None, None]:
    """Patch `create_model`, defaults, and `LCSummarizationMiddleware`.

    Args:
        cutoff: Value returned by `_determine_cutoff_index`.

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
) -> list[MagicMock]:
    """Set up app state for a successful compaction test.

    Args:
        app: The app instance to configure.
        n_messages: Number of mock messages to create.

    Returns:
        The list of mock messages.
    """
    messages = _make_messages(n_messages)
    mock_state = MagicMock()
    mock_state.values = {"messages": messages}

    app._agent = MagicMock()
    app._agent.aget_state = AsyncMock(return_value=mock_state)
    app._agent.aupdate_state = AsyncMock()
    app._lc_thread_id = "test-thread"
    app._agent_running = False
    return messages


class TestCompactSuccess:
    """Test successful compaction flow."""

    @pytest.mark.asyncio
    async def test_successful_compaction(self) -> None:
        """Should remove old messages, add summary, and refresh UI."""
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
                patch.object(
                    app,
                    "_generate_compact_summary",
                    new_callable=AsyncMock,
                    return_value="Summary of the conversation.",
                ),
                patch.object(app, "_clear_messages", new_callable=AsyncMock),
                patch.object(app, "_load_thread_history", new_callable=AsyncMock),
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                await app._handle_compact()
                await pilot.pause()

            # aupdate_state called twice: remove+summary, then event reset
            assert app._agent.aupdate_state.call_count == 2

            # First call: remove ops + summary message
            first_values = app._agent.aupdate_state.call_args_list[0][0][1]
            update_messages = first_values["messages"]
            from langchain_core.messages import HumanMessage, RemoveMessage

            remove_ops = [m for m in update_messages if isinstance(m, RemoveMessage)]
            summaries = [m for m in update_messages if isinstance(m, HumanMessage)]
            # Middleware cutoff=4 → summarize first 4 messages
            assert len(remove_ops) == 4
            assert len(summaries) == 1
            assert summaries[0].additional_kwargs.get("lc_source") == "summarization"

            # Verify file path and summary content are embedded
            assert "/conversation_history/test-thread.md" in summaries[0].content
            assert "<summary>" in summaries[0].content
            assert "Summary of the conversation." in summaries[0].content

            # Second call: reset _summarization_event
            second_values = app._agent.aupdate_state.call_args_list[1][0][1]
            assert second_values == {"_summarization_event": None}

    @pytest.mark.asyncio
    async def test_compaction_shows_feedback_message(self) -> None:
        """Should display feedback with message count and token change."""
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
                    return_value=None,
                ),
                patch.object(
                    app,
                    "_generate_compact_summary",
                    new_callable=AsyncMock,
                    return_value="Summary.",
                ),
                patch.object(app, "_clear_messages", new_callable=AsyncMock),
                patch.object(app, "_load_thread_history", new_callable=AsyncMock),
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
                _mock_middleware(cutoff=4),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch.object(
                    app,
                    "_generate_compact_summary",
                    new_callable=AsyncMock,
                    return_value="Summary.",
                ),
                patch.object(app, "_clear_messages", new_callable=AsyncMock),
                patch.object(app, "_load_thread_history", new_callable=AsyncMock),
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                await app._handle_compact()
                await pilot.pause()

            app._token_tracker.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_compaction_clears_and_reloads_ui(self) -> None:
        """Should clear messages and reload thread history."""
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
                    return_value=None,
                ),
                patch.object(
                    app,
                    "_generate_compact_summary",
                    new_callable=AsyncMock,
                    return_value="Summary.",
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

            mock_clear.assert_called_once()
            mock_load.assert_called_once()


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
            app._agent.aupdate_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_cutoff_one_compacts_single_message(self) -> None:
        """With cutoff=1, only the first message should be summarized."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app, n_messages=7)

            with (
                _mock_middleware(cutoff=1),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch.object(
                    app,
                    "_generate_compact_summary",
                    new_callable=AsyncMock,
                    return_value="Summary.",
                ),
                patch.object(app, "_clear_messages", new_callable=AsyncMock),
                patch.object(app, "_load_thread_history", new_callable=AsyncMock),
                patch(_TOKEN_COUNT_PATH, return_value=100),
            ):
                await app._handle_compact()
                await pilot.pause()

            from langchain_core.messages import RemoveMessage

            first_values = app._agent.aupdate_state.call_args_list[0][0][1]
            remove_ops = [
                m for m in first_values["messages"] if isinstance(m, RemoveMessage)
            ]
            assert len(remove_ops) == 1

    @pytest.mark.asyncio
    async def test_middleware_cutoff_called_with_messages(self) -> None:
        """Should pass the full message list to middleware cutoff logic."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            messages = _setup_compact_app(app, n_messages=10)

            with (
                _mock_middleware(cutoff=4) as mock_mw,
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch.object(
                    app,
                    "_generate_compact_summary",
                    new_callable=AsyncMock,
                    return_value="Summary.",
                ),
                patch.object(app, "_clear_messages", new_callable=AsyncMock),
                patch.object(app, "_load_thread_history", new_callable=AsyncMock),
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                await app._handle_compact()
                await pilot.pause()

            mock_mw._determine_cutoff_index.assert_called_once_with(messages)
            mock_mw._partition_messages.assert_called_once_with(messages, 4)


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
                _mock_middleware(cutoff=4),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch.object(
                    app,
                    "_generate_compact_summary",
                    new_callable=AsyncMock,
                    return_value="Summary.",
                ),
                patch.object(app, "_clear_messages", new_callable=AsyncMock),
                patch.object(app, "_load_thread_history", new_callable=AsyncMock),
                patch(_TOKEN_COUNT_PATH, return_value=500),
            ):
                await app._handle_compact()
                await pilot.pause()

            # Should still have called aupdate_state successfully
            assert app._agent.aupdate_state.call_count == 2

            # Summary should NOT have file path reference
            from langchain_core.messages import HumanMessage

            first_values = app._agent.aupdate_state.call_args_list[0][0][1]
            summaries = [
                m for m in first_values["messages"] if isinstance(m, HumanMessage)
            ]
            assert len(summaries) == 1
            assert "conversation history has been saved" not in summaries[0].content

    @pytest.mark.asyncio
    async def test_summary_generation_failure_shows_error(self) -> None:
        """Should show error and leave state untouched when summary fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_compact_app(app)

            with (
                _mock_middleware(cutoff=4),
                patch.object(
                    app,
                    "_generate_compact_summary",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("model unavailable"),
                ),
            ):
                await app._handle_compact()
                await pilot.pause()

            # State should not have been updated
            app._agent.aupdate_state.assert_not_called()

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
                _mock_middleware(cutoff=4),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch.object(
                    app,
                    "_generate_compact_summary",
                    new_callable=AsyncMock,
                    return_value="Summary.",
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
                _mock_middleware(cutoff=4),
                patch.object(
                    app,
                    "_offload_messages_for_compact",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("backend down"),
                ),
                patch.object(
                    app,
                    "_generate_compact_summary",
                    new_callable=AsyncMock,
                    return_value="Summary.",
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
