"""Unit tests for /offload slash command."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Callable  # noqa: TC003
from contextlib import nullcontext
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends.utils import validate_path
from langgraph.graph.message import add_messages

from deepagents_code import offload
from deepagents_code._session_stats import format_token_count
from deepagents_code.app import DeepAgentsApp
from deepagents_code.command_registry import get_slash_commands
from deepagents_code.hooks.manager import HooksManager
from deepagents_code.offload import (
    _artifacts_root,
    _filesystem_tool_path,
    _offload_fallback_root,
    delete_offloaded_history,
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


def _make_dict_message(
    content: str, *, message_id: str | None = None
) -> dict[str, Any]:
    """Create one serialized human-message payload with a stable id."""
    return {
        "content": content,
        "additional_kwargs": {},
        "response_metadata": {},
        "type": "human",
        "name": None,
        "id": message_id or f"msg-{content}",
    }


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
    """Configure a `DeepAgentsApp` as a server-backed agent for offload tests.

    The operation-graph path reads state via `_get_thread_state_values` and
    drives the graph via `_drive_offload_operation_graph`; tests patch those
    seams directly, so only the remote identity/flags are set here. The agent
    is specced as a `RemoteAgent` so `_remote_agent()` narrows to it.
    """
    from deepagents_code.client.remote_client import RemoteAgent

    agent = MagicMock(spec=RemoteAgent)
    agent.aupdate_state = AsyncMock()
    app._agent = agent
    app._backend = None
    app._lc_thread_id = "test-thread"
    app._agent_running = False
    return agent


def _setup_local_offload_app(app: DeepAgentsApp) -> MagicMock:
    """Configure a `DeepAgentsApp` as a local in-process agent for offload tests.

    A plain `MagicMock` agent is *not* a `RemoteAgent`, so `_remote_agent()`
    returns `None` and `_handle_offload` takes the seeded in-process path
    (`_drive_local_seeded_compaction`) instead of the operation graph.
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
        """The /offload command should be in the get_slash_commands() list."""
        labels = [entry.name for entry in get_slash_commands()]
        assert "/offload" in labels

    def test_offload_sorted_alphabetically(self) -> None:
        """The /offload entry should appear between /model and /quit."""
        labels = [entry.name for entry in get_slash_commands()]
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
            after["_session_cost_usd"] = 0.75
            app._set_session_cost(0.5)
            app._add_provisional_cost(0.75)

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_offload_operation_graph",
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
            # The graph prices the compaction run's own model call, so the
            # committed total replaces the client's provisional estimate.
            assert app._session_cost_usd == pytest.approx(0.75)
            assert app._displayed_cost_usd == pytest.approx(0.75)

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
                    "_drive_offload_operation_graph",
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

    async def test_rebind_warning_accompanies_a_reported_success(self) -> None:
        """A rebind failure is surfaced where the success is reported.

        The driver cannot mount it itself: it does not know whether the caller
        is about to report success. Reporting the offload as finished while
        staying silent about the mis-bound thread leaves a later `/goal` or
        `/rubric` to fail with nothing connecting it to `/offload`.
        """
        from deepagents_code.app import _OFFLOAD_REBIND_WARNING

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))
            after = _state_values(_make_dict_messages(12), _summary_event(6))

            async def drive(*_args: object, **_kwargs: object) -> None:  # noqa: RUF029
                app._offload_rebind_failed = True

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(app, "_drive_offload_operation_graph", side_effect=drive),
            ):
                await app._handle_offload()
                await pilot.pause()

            assert any(
                "Offloaded 6 older messages" in str(w._content)
                for w in app.query(AppMessage)
            )
            assert any(
                _OFFLOAD_REBIND_WARNING in str(w._content)
                for w in app.query(ErrorMessage)
            )

    async def test_rebind_warning_is_withheld_from_a_reported_failure(self) -> None:
        """A drain error must not be paired with "Offload finished, but...".

        Both `drain_error` breaks exit the stream loop normally, so a flag that
        only tracked "the stream did not raise" would mount the rebind warning
        alongside the failure — two messages asserting opposite outcomes.
        """
        from deepagents_code.app import _OFFLOAD_REBIND_WARNING

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))

            async def drive(*_args: object, **_kwargs: object) -> str:  # noqa: RUF029
                app._offload_rebind_failed = True
                return "Offload could not complete: a configured hook kept ..."

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before],
                ),
                patch.object(app, "_drive_offload_operation_graph", side_effect=drive),
            ):
                await app._handle_offload()
                await pilot.pause()

            errors = [str(w._content) for w in app.query(ErrorMessage)]

        assert any("could not complete" in text for text in errors)
        assert not any(_OFFLOAD_REBIND_WARNING in text for text in errors)

    async def test_unreadable_state_is_not_reported_as_already_compact(self) -> None:
        """An empty state read must not render as a benign no-op.

        On the operation-graph path the state re-read is the only evidence of
        the outcome — there is no `ToolMessage` to fall back on — and
        `_get_thread_state_values` collapses a missing snapshot (a 404 after the
        run rebound the thread, a server restart) to `{}`. Reporting that as
        "already compact" tells the user nothing happened when the conversation
        may well have been compacted.
        """
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
                    side_effect=[before, {}],
                ),
                patch.object(
                    app,
                    "_drive_offload_operation_graph",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            errors = [str(w._content) for w in app.query(ErrorMessage)]
            infos = [str(w._content) for w in app.query(AppMessage)]

        assert any("could not be confirmed" in text for text in errors)
        assert not any("already compact" in text for text in infos)

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
                    "_drive_offload_operation_graph",
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
                    "_drive_offload_operation_graph",
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
                    "_drive_offload_operation_graph",
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
                    "_drive_offload_operation_graph",
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
        """A no-op restores history and shows the no-op message, not success.

        The local seeded driver commits its synthetic seed, tool result, and
        trailing turn on a no-op, so `_handle_offload` removes those artifacts
        via `aupdate_state`.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # A plain (non-remote) mock agent drives the seeded in-process
            # path, whose no-op branch restores state via `aupdate_state`.
            agent = MagicMock()
            agent.aupdate_state = AsyncMock()
            app._agent = agent
            app._backend = None
            app._lc_thread_id = "test-thread"
            app._agent_running = False

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
                    "_drive_local_seeded_compaction",
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
            # The seeded no-op restores the pre-run conversation by removing
            # the committed artifacts.
            agent.aupdate_state.assert_awaited()

    async def test_noop_operation_graph_writes_nothing(self) -> None:
        """The operation graph commits no synthetic artifacts on a no-op."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            agent = _setup_server_offload_app(app)

            # Prior event present; after-state cutoff unchanged -> nothing moved.
            event = _summary_event(6)
            messages = _make_dict_messages(8)
            before = _state_values(messages, event)
            after = _state_values(messages, event)

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_offload_operation_graph",
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
            agent.aupdate_state.assert_not_awaited()

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
                    "_drive_offload_operation_graph",
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
                    "_drive_offload_operation_graph",
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
                    "_drive_offload_operation_graph",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            agent.aupdate_state.assert_not_awaited()
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
                    "_drive_offload_operation_graph",
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
                    "_drive_offload_operation_graph",
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
                    "_drive_offload_operation_graph",
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
                    "_drive_offload_operation_graph",
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
                    "_drive_offload_operation_graph",
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
                    "_drive_offload_operation_graph",
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

    async def test_failed_operation_graph_run_has_no_seed_to_clean_up(self) -> None:
        """A failed operation-graph run needs no seed cleanup.

        The operation graph commits no synthetic tool call, so there is nothing
        that could wedge the next turn with a dangling `tool_use` and the cleanup
        must not run. The failure is still surfaced to the user. The seeded
        driver's counterpart is
        `test_failed_seeded_run_removes_dangling_seed`.
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
                    "_drive_offload_operation_graph",
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

            cleanup.assert_not_awaited()
            assert any(
                "Offload failed" in str(widget._content)
                for widget in app.query(ErrorMessage)
            )

    async def test_operation_graph_double_failure_surfaces_one_error(self) -> None:
        """Stream failure plus a failed reconcile still reports exactly once.

        The seeded driver additionally warns that the thread may be inconsistent
        when it cannot confirm seed removal (see
        `test_seeded_double_failure_warns_thread_may_be_inconsistent`). The
        operation graph has no seed, so no cleanup runs and no wedge warning is
        appropriate -- the user should see the "Offload failed" error alone
        rather than a second, inapplicable warning.
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
                    "_drive_offload_operation_graph",
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

            cleanup.assert_not_awaited()
            error_text = " ".join(
                str(widget._content) for widget in app.query(ErrorMessage)
            )
            assert "Offload failed" in error_text
            assert "inconsistent state" not in error_text

    async def test_failed_seeded_run_removes_dangling_seed(self) -> None:
        """A failed local seeded run must not leave an unanswered tool call.

        The seeded driver commits a synthetic assistant `tool_use` before the
        tool runs, so a run that fails without compacting has to remove it or the
        model API rejects the next turn.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_local_offload_app(app)

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
                    "_drive_local_seeded_compaction",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("stream boom"),
                ),
                patch.object(
                    app,
                    "_remove_unanswered_offload_seed",
                    new_callable=AsyncMock,
                    return_value=True,
                ) as cleanup,
            ):
                await app._handle_offload()
                await pilot.pause()

            cleanup.assert_awaited_once()
            assert any(
                "Offload failed" in str(widget._content)
                for widget in app.query(ErrorMessage)
            )

    async def test_seeded_double_failure_warns_thread_may_be_inconsistent(self) -> None:
        """Unconfirmed seed removal warns the user the thread may be wedged.

        When the drive raises, the reconcile state-read also fails, and the
        best-effort seed cleanup cannot confirm removal (returns `False`), the
        user is warned -- in addition to the surfaced "Offload failed" error --
        so a later cryptic `tool_use` rejection is not their only signal.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_local_offload_app(app)

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
                    "_drive_local_seeded_compaction",
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
            assert "Offload failed" in error_text
            assert "inconsistent state" in error_text

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
                    "_drive_offload_operation_graph",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("stream unavailable"),
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            error_msgs = app.query(ErrorMessage)
            assert any("Offload failed" in str(w._content) for w in error_msgs)

    async def test_remote_exception_is_unwrapped_not_shown_as_a_dict(self) -> None:
        """A server-side failure must read as prose, not a Python dict repr.

        The operation graph reports failure by raising, so a server-backed
        `/offload` surfaces a `RemoteException` whose sole arg is the server's
        error payload dict. `str()` on that renders `{'error': ..., 'message':
        ...}`; only `format_agent_exception` unwraps it. Every other test on this
        path raises a plain `RuntimeError`, for which the two are identical.
        """
        from langgraph.pregel.remote import RemoteException

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)

            before = _state_values(_make_dict_messages(10))
            remote_exc = RemoteException(
                {
                    "error": "RuntimeError",
                    "message": "Compaction failed: OSError: disk full.",
                }
            )

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, before],
                ),
                patch.object(
                    app,
                    "_drive_offload_operation_graph",
                    new_callable=AsyncMock,
                    side_effect=remote_exc,
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            error_text = " ".join(str(w._content) for w in app.query(ErrorMessage))
            assert "disk full" in error_text
            assert "{'error'" not in error_text

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
                    "_drive_offload_operation_graph",
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
        monkeypatch.setattr(offload, "_UNIQUE_OFFLOAD_FALLBACK_ROOT", None)

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
        monkeypatch.setattr(offload, "_UNIQUE_OFFLOAD_FALLBACK_ROOT", None)

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


class TestDeleteOffloadedHistory:
    """Cover cleanup of a thread's offloaded conversation-history archive."""

    def test_removes_persistent_archive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The per-thread archive under `~/.deepagents` is removed."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        archive_dir = tmp_path / ".deepagents" / "conversation_history"
        archive_dir.mkdir(parents=True)
        archive = archive_dir / "thread-1.md"
        archive.write_text("history")
        keep = archive_dir / "thread-2.md"
        keep.write_text("other")

        assert delete_offloaded_history("thread-1") is True
        assert not archive.exists()
        # Unrelated threads' archives are left untouched.
        assert keep.exists()

    def test_removes_archive_from_reused_unique_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cleanup reuses the random root selected when the archive was written."""
        home_root = tmp_path / "home" / ".deepagents"
        home_root.mkdir(parents=True)
        temp_dir = tmp_path / "tmp"
        temp_dir.mkdir()
        getuid = getattr(os, "getuid", None)
        uid = getuid() if getuid is not None else os.getpid()
        (temp_dir / f"deepagents-{uid}").write_text("not a directory")
        probe = MagicMock(
            side_effect=[PermissionError("read-only home"), nullcontext()]
        )

        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_dir))
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", probe)
        monkeypatch.setattr(offload, "_UNIQUE_OFFLOAD_FALLBACK_ROOT", None)

        root = _offload_fallback_root()
        archive = root / "conversation_history" / "thread-1.md"
        archive.parent.mkdir(parents=True)
        archive.write_text("history")

        assert delete_offloaded_history("thread-1") is True
        assert not archive.exists()
        assert probe.call_count == 2

    def test_missing_archive_reports_nothing_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting a thread with no archive reports nothing removed."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert delete_offloaded_history("thread-1") is False

    def test_empty_thread_id_is_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty thread id never touches the filesystem."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert delete_offloaded_history("") is False

    def test_unlink_failure_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing `unlink` is logged and reported as nothing removed."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(offload, "_UNIQUE_OFFLOAD_FALLBACK_ROOT", None)
        archive_dir = tmp_path / ".deepagents" / "conversation_history"
        archive_dir.mkdir(parents=True)
        archive = archive_dir / "thread-1.md"
        archive.write_text("history")
        monkeypatch.setattr(
            Path, "unlink", MagicMock(side_effect=PermissionError("read-only mount"))
        )

        assert delete_offloaded_history("thread-1") is False
        # The archive survives the failed deletion rather than being lost.
        assert archive.exists()

    def test_unresolvable_root_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unresolvable offload root is swallowed, not raised."""
        monkeypatch.setattr(
            offload,
            "_offload_fallback_root",
            MagicMock(side_effect=OSError("no writable location")),
        )

        assert delete_offloaded_history("thread-1") is False

    def test_rejects_thread_id_path_traversal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crafted thread id cannot escape the archive directory."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / ".deepagents" / "conversation_history").mkdir(parents=True)
        # A relative escape resolves to `.deepagents/config.md`, so a decoy there
        # is load-bearing: were the guard removed, `unlink` would delete it.
        relative_decoy = tmp_path / ".deepagents" / "config.md"
        relative_decoy.write_text("secret")
        # An absolute thread id resets the join, escaping the archive tree
        # entirely; place its decoy where that reset lands.
        outside = tmp_path / "outside.md"
        outside.write_text("secret")

        assert delete_offloaded_history("../config") is False
        assert delete_offloaded_history(str(tmp_path / "outside")) is False
        # An embedded separator lands in a subdirectory, not `archive_dir`.
        assert delete_offloaded_history("sub/thread") is False
        assert relative_decoy.exists()
        assert outside.exists()


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
                    "_drive_offload_operation_graph",
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
                    "_drive_offload_operation_graph",
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
            # A plain (non-remote) mock agent drives the seeded in-process
            # path, whose no-op branch restores state via `aupdate_state`;
            # make that write fail.
            agent = MagicMock()
            agent.aupdate_state = AsyncMock(side_effect=RuntimeError("write failed"))
            app._agent = agent
            app._backend = None
            app._lc_thread_id = "test-thread"
            app._agent_running = False

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
                    "_drive_local_seeded_compaction",
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


class TestOffloadSessionStartHook:
    """`SessionStart(COMPACT)` fires once, from whichever driver ran."""

    @staticmethod
    def _offloaded_state() -> tuple[dict[str, Any], dict[str, Any]]:
        """Build before/after state where compaction advanced the cutoff.

        Returns:
            The pre-run and post-run thread state.
        """
        before = _state_values(_make_dict_messages(6))
        after = _state_values(_make_dict_messages(6))
        after["_summarization_event"] = {
            "cutoff_index": 4,
            "summary_message": {"type": "ai", "content": "summary"},
            "file_path": "/conversation_history/t.md",
        }
        return before, after

    async def test_operation_graph_path_fires_the_compact_boundary(self) -> None:
        """A configured `SessionStart` hook must see `/offload` on this path too.

        The seeded driver fires this from inside its drain, keyed on the
        compaction tool result. The operation graph produces no tool result, so
        without an explicit call here a configured hook would silently never run
        for server-backed `/offload` — and every success test would still pass.
        """
        from deepagents_code.hooks.models.domain import SessionStartCause

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            before, after = self._offloaded_state()

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_offload_operation_graph",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch.object(
                    app,
                    "_run_session_start_hook",
                    new_callable=AsyncMock,
                    return_value=True,
                ) as hook,
            ):
                await app._handle_offload()
                await pilot.pause()

            hook.assert_awaited_once_with(SessionStartCause.COMPACT)

    async def test_seeded_path_does_not_fire_it_twice(self) -> None:
        """The seeded driver owns the boundary, so `_handle_offload` must not.

        Both firing would run a user's `SessionStart` hook twice for one
        `/offload`.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_local_offload_app(app)
            before, after = self._offloaded_state()

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_local_seeded_compaction",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch.object(
                    app,
                    "_run_session_start_hook",
                    new_callable=AsyncMock,
                    return_value=True,
                ) as hook,
            ):
                await app._handle_offload()
                await pilot.pause()

            hook.assert_not_awaited()

    async def test_hook_stop_still_reports_the_committed_offload(self) -> None:
        """A stopping hook must not hide an offload that already committed.

        Compaction is durable by the time this hook runs, so returning early
        would leave the user with only "stopped by a hook" while their
        conversation *was* compacted and the status bar kept pre-offload counts.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            before, after = self._offloaded_state()
            tokens: list[int] = []

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new_callable=AsyncMock,
                    side_effect=[before, after],
                ),
                patch.object(
                    app,
                    "_drive_offload_operation_graph",
                    new_callable=AsyncMock,
                    return_value=None,
                ),
                patch.object(
                    app,
                    "_run_session_start_hook",
                    new_callable=AsyncMock,
                    return_value=False,
                ),
                patch.object(app, "_on_tokens_update", side_effect=tokens.append),
            ):
                await app._handle_offload()
                await pilot.pause()

            text = " ".join(str(w._content) for w in app.query(AppMessage))
            assert "Offloaded" in text
            # The status bar is refreshed rather than left pre-offload.
            assert tokens


class TestServerOperationOffload:
    """The slash command uses the explicit server operation graph."""

    async def test_aborts_when_the_state_refresh_fails(self) -> None:
        """A failed re-read aborts the offload instead of replaying the stale snapshot.

        The run input replaces the thread's `messages` channel, so falling back
        to the caller's pre-run snapshot — taken before `_set_agent_running`
        blocked new turns — would delete any message committed in the gap
        (shared/external threads, a concurrent completion). No stream may
        start when the fresh state cannot be obtained.
        """
        app = DeepAgentsApp()
        operation = MagicMock()
        streamed = False

        async def stream(*_args: object, **_kwargs: object):  # noqa: ANN202, RUF029
            nonlocal streamed
            streamed = True
            yield (), "updates", {"force_compact": {}}

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                side_effect=RuntimeError("checkpoint store unreachable")
            )
            app._lc_thread_id = "test-thread"
            with (
                patch.object(app, "_remote_agent", return_value=remote),
                pytest.raises(RuntimeError, match="Could not refresh thread state"),
            ):
                await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    {"messages": [{"type": "human", "content": "hi"}]},
                )

        assert streamed is False

    async def test_aborts_when_the_state_refresh_comes_back_empty(self) -> None:
        """An empty re-read aborts the offload rather than replaying stale state.

        `_get_thread_state_values` collapses a missing snapshot (a 404 after a
        rebind, a server restart, an un-flushed checkpoint) to `{}`, which the
        old `or state_values` fallback silently replaced with the pre-run
        snapshot — replaying that would truncate the live conversation.
        """
        app = DeepAgentsApp()
        operation = MagicMock()
        streamed = False

        async def stream(*_args: object, **_kwargs: object):  # noqa: ANN202, RUF029
            nonlocal streamed
            streamed = True
            yield (), "updates", {"force_compact": {}}

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(return_value=SimpleNamespace(values={}))
            app._lc_thread_id = "test-thread"
            with (
                patch.object(app, "_remote_agent", return_value=remote),
                pytest.raises(RuntimeError, match="came back empty"),
            ):
                await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    {"messages": [{"type": "human", "content": "hi"}]},
                )

        assert streamed is False

    async def test_streams_named_offload_graph_without_seed_context(self) -> None:
        """The operation has no synthetic model call or HITL resume loop."""
        app = DeepAgentsApp()
        operation = MagicMock()
        stream_args: list[object] = []
        stream_kwargs: dict[str, object] = {}

        async def stream(*args: object, **kwargs: object):  # noqa: ANN202, RUF029
            stream_args.extend(args)
            stream_kwargs.update(kwargs)
            yield (), "updates", {"force_compact": {}}

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation

        messages = [{"type": "human", "content": "hi"}]
        # A realistic thread carries far more than `messages`. Every extra
        # channel here must be withheld from the run input.
        state_values = {
            "messages": messages,
            "_summarization_event": {"cutoff_index": 2, "summary_message": {}},
            "_session_cost_usd": 0.42,
            "_session_cost_transfers": {"scope": {"total": 1.0}},
            "_goal_objective": "ship it",
            "todos": [{"content": "t", "status": "pending"}],
        }
        async with app.run_test() as pilot:
            await pilot.pause()
            # The driver re-reads thread state before streaming (a stale replay
            # would overwrite the live conversation), and that read is what
            # registers the thread server-side.
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                return_value=SimpleNamespace(values=state_values)
            )
            app._lc_thread_id = "test-thread"
            with patch.object(app, "_remote_agent", return_value=remote):
                await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}}, state_values
                )

        remote.aensure_thread.assert_awaited_once_with(
            {"configurable": {"thread_id": "test-thread"}}
        )
        remote.for_graph.assert_called_once_with("offload")
        # `messages` is replayed so the node sees the real conversation rather
        # than an emptied list -- and *nothing else* is. Replaying
        # `_session_cost_usd` would double the thread's persisted spend on every
        # `/offload` (it reduces with `operator.add`), and a writable
        # `_summarization_event` would let the caller set the compaction cutoff.
        # Asserted by exact key set so a newly-replayed channel fails here.
        assert stream_args == [{"messages": messages}]
        context = stream_kwargs["context"]
        assert isinstance(context, dict)
        assert "offload_tool_call_id" not in context

    async def test_forwards_the_active_model_selection_to_the_offload_run(
        self,
    ) -> None:
        """The run context must carry the model the summarizer should use.

        `_runtime_model_config` reads exactly these fields off the context to
        build the summarizer, so dropping one silently summarizes with the
        startup default under a `/model` or profile override. The seeded driver
        has an equivalent test; this path had none, and the integration test's
        own `model_context_limit` fix shows the failure mode is live.
        """
        app = DeepAgentsApp()
        operation = MagicMock()
        stream_kwargs: dict[str, object] = {}

        async def stream(*_args: object, **kwargs: object):  # noqa: ANN202, RUF029
            stream_kwargs.update(kwargs)
            yield (), "updates", {"force_compact": {}}

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                return_value=SimpleNamespace(
                    values=_state_values([_make_dict_message("hi")])
                )
            )
            app._lc_thread_id = "test-thread"
            app._model_params_override = {"temperature": 0.1}
            app._profile_override = {"reasoning": "high"}
            with (
                patch.object(app, "_remote_agent", return_value=remote),
                patch.object(
                    app, "_effective_model_spec", return_value="anthropic:some-model"
                ),
            ):
                await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    {"messages": [{"type": "human", "content": "hi"}]},
                )

        context = cast("dict[str, Any]", stream_kwargs["context"])
        assert context["model"] == "anthropic:some-model"
        assert context["model_params"] == {"temperature": 0.1}
        assert context["profile_overrides"] == {"reasoning": "high"}
        assert context["thread_id"] == "test-thread"
        from deepagents_code.config import settings

        assert context["model_context_limit"] == settings.model_context_limit

    async def test_an_unanswerable_interrupt_wins_over_fulfillable_ones(self) -> None:
        """A mixed round must report, not resume with a partial answer.

        One round can yield both a hook interrupt this client can fulfill and a
        `HITLRequest` it cannot. Resuming with only the fulfillable half leaves
        the other pending forever, so the unanswerable one has to decide the
        round. Plausibly correct but unpinned before this test — a reordering
        would flip it silently.
        """
        app = DeepAgentsApp()
        operation = MagicMock()
        hook_interrupt = MagicMock()
        hook_interrupt.id = "interrupt-1"
        hook_interrupt.value = {"type": "hook_invocation", "invocation_id": "inv-1"}
        alien = MagicMock()
        alien.id = "interrupt-2"
        alien.value = {"type": "hitl_request"}
        streams: list[object] = []

        async def stream(stream_input: object, **_kwargs: object):  # noqa: ANN202, RUF029
            streams.append(stream_input)
            yield (), "updates", {"__interrupt__": [hook_interrupt, alien]}

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                return_value=SimpleNamespace(
                    values=_state_values([_make_dict_message("hi")])
                )
            )
            app._lc_thread_id = "test-thread"
            manager = MagicMock(spec=HooksManager)
            manager.apply_graph_context = MagicMock()
            manager.fulfill_interrupt = AsyncMock(return_value={"decision": "ok"})
            with (
                patch.object(app, "_remote_agent", return_value=remote),
                self._hooks_patch(manager),
            ):
                result = await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    {"messages": [{"type": "human", "content": "hi"}]},
                )

        assert result is not None
        assert "cannot answer" in result
        # No resume round: the fulfillable half is deliberately discarded.
        assert len(streams) == 1

    async def test_restores_main_graph_association(self) -> None:
        """A `graph_id` rebind from the named-graph run is reset before returning.

        The server records the last-run graph on the thread; leaving it at
        `offload` would send later out-of-run `as_node="model"` state writes to
        a graph with no model node. The driver rebinds the thread's `graph_id`
        metadata through the main client. An empty `aupdate_state` cannot do
        this -- it resolves against the thread's *current* graph association --
        so the rebind must go through `arebind_thread`.
        """
        app = DeepAgentsApp()
        operation = MagicMock()

        async def stream(*_args: object, **_kwargs: object):  # noqa: ANN202, RUF029
            yield (), "updates", {"force_compact": {}}

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                return_value=SimpleNamespace(
                    values=_state_values([_make_dict_message("hi")])
                )
            )
            app._lc_thread_id = "test-thread"
            with patch.object(app, "_remote_agent", return_value=remote):
                await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    {"messages": [{"type": "human", "content": "hi"}]},
                )

        remote.arebind_thread.assert_awaited_once_with(
            {"configurable": {"thread_id": "test-thread"}}
        )

    async def test_graph_restore_failure_warns_without_failing_the_offload(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failed rebind is recorded for the caller, not mounted here.

        The offload itself succeeded, so this must not surface as a failure. It
        must not be silent either: the thread stays bound to the `offload`
        graph, so an unrelated later `/goal` or `/rubric` would fail with no
        explanation. The driver only records it -- `_OFFLOAD_REBIND_WARNING`
        says the offload finished, and only `_handle_offload` knows whether it
        is about to report that.
        """
        app = DeepAgentsApp()
        operation = MagicMock()

        async def stream(*_args: object, **_kwargs: object):  # noqa: ANN202, RUF029
            yield (), "updates", {"force_compact": {}}

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock(side_effect=RuntimeError("server gone"))
        remote.for_graph.return_value = operation

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                return_value=SimpleNamespace(
                    values=_state_values([_make_dict_message("hi")])
                )
            )
            app._lc_thread_id = "test-thread"
            with (
                patch.object(app, "_remote_agent", return_value=remote),
                caplog.at_level("WARNING"),
            ):
                result = await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    {"messages": [{"type": "human", "content": "hi"}]},
                )
                await pilot.pause()

                # Not reported as a drain failure -- the offload still succeeded.
                assert result is None
                assert app._offload_rebind_failed is True
                error_text = " ".join(
                    str(widget._content) for widget in app.query(ErrorMessage)
                )

        assert "Failed to restore the thread's main graph association" in caplog.text
        # Recorded, not rendered: the caller decides when this is safe to say.
        assert error_text == ""

    async def test_stream_error_is_not_masked_by_a_failing_rebind(self) -> None:
        """A rebind failure in the `finally` must not replace the stream error.

        The caller distinguishes a committed-but-interrupted offload from a
        failed one by the exception it sees; swallowing the real error in favor
        of the bookkeeping failure would lose that. The driver mounts nothing
        either way — it records the rebind failure and lets `_handle_offload`
        decide, which is what lets a stream error that still committed the
        compaction be reconciled into a success *and* carry the warning.
        """
        from deepagents_code.app import _OFFLOAD_REBIND_WARNING

        app = DeepAgentsApp()
        operation = MagicMock()

        async def stream(*_args: object, **_kwargs: object):  # noqa: ANN202, RUF029
            yield (), "updates", {"force_compact": {}}
            msg = "stream died"
            raise RuntimeError(msg)

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock(side_effect=RuntimeError("server gone"))
        remote.for_graph.return_value = operation

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                return_value=SimpleNamespace(
                    values=_state_values([_make_dict_message("hi")])
                )
            )
            app._lc_thread_id = "test-thread"
            with (
                patch.object(app, "_remote_agent", return_value=remote),
                pytest.raises(RuntimeError, match="stream died"),
            ):
                await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    {"messages": [{"type": "human", "content": "hi"}]},
                )
            await pilot.pause()
            error_text = " ".join(
                str(widget._content) for widget in app.query(ErrorMessage)
            )

        assert _OFFLOAD_REBIND_WARNING not in error_text
        # Still recorded, so a caller that reconciles this into a success can
        # surface it there.
        assert app._offload_rebind_failed is True

    @staticmethod
    def _hooks_patch(manager: MagicMock):  # noqa: ANN205
        """Swap the read-only `_hooks` property for a stubbed manager."""
        return patch.object(
            DeepAgentsApp,
            "_hooks",
            new_callable=lambda: property(lambda _app: manager),
        )

    async def test_fulfills_hook_interrupts_and_resumes(self) -> None:
        """A hook interrupt is answered through the hook engine and the graph resumed.

        With a configured `PreCompact`/`PreToolUse` hook the node interrupts at
        the hook boundary instead of returning; without this loop `/offload`
        would park there and report "Nothing to offload".
        """
        from deepagents_code.hooks.manager import HooksManager

        app = DeepAgentsApp()
        operation = MagicMock()
        streams: list[object] = []
        hook_payload = {"type": "hook_invocation", "invocation_id": "inv-1"}
        hook_interrupt = MagicMock()
        hook_interrupt.id = "interrupt-1"
        hook_interrupt.value = hook_payload

        async def stream(input_: object, **_kwargs: object):  # noqa: ANN202, RUF029
            streams.append(input_)
            if len(streams) == 1:
                yield (), "updates", {"__interrupt__": [hook_interrupt]}
            else:
                yield (), "updates", {"force_compact": {}}

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                return_value=SimpleNamespace(
                    values=_state_values([_make_dict_message("hi")])
                )
            )
            app._lc_thread_id = "test-thread"
            manager = MagicMock(spec=HooksManager)
            manager.apply_graph_context = MagicMock()
            manager.fulfill_interrupt = AsyncMock(return_value={"decision": "ok"})
            with (
                patch.object(app, "_remote_agent", return_value=remote),
                self._hooks_patch(manager),
            ):
                result = await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    {"messages": [{"type": "human", "content": "hi"}]},
                )

        assert result is None
        manager.fulfill_interrupt.assert_awaited_once_with(hook_payload)
        assert len(streams) == 2
        resume = streams[1]
        # The resume command carries the fulfilled hook decision keyed by the
        # server-supplied interrupt id.
        assert getattr(resume, "resume", None) == {"interrupt-1": {"decision": "ok"}}

    async def test_unbounded_hook_interrupts_stop_at_resume_cap(self) -> None:
        """A hook interrupting every round is bounded *and* reported.

        Reporting is the load-bearing half. The driver returns `None` for a
        successful run, and the caller reads an unchanged `_summarization_event`
        as "nothing to offload" -- so a silent give-up would tell the user their
        conversation is already compact while the run sits paused mid-interrupt
        and nothing was compacted at all.
        """
        app = DeepAgentsApp()
        operation = MagicMock()
        hook_interrupt = MagicMock()
        hook_interrupt.id = "interrupt-1"
        hook_interrupt.value = {"type": "hook_invocation", "invocation_id": "inv-1"}
        rounds = 0

        async def stream(_input: object, **_kwargs: object):  # noqa: ANN202, RUF029
            nonlocal rounds
            rounds += 1
            yield (), "updates", {"__interrupt__": [hook_interrupt]}

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                return_value=SimpleNamespace(
                    values=_state_values([_make_dict_message("hi")])
                )
            )
            app._lc_thread_id = "test-thread"
            manager = MagicMock(spec=HooksManager)
            manager.apply_graph_context = MagicMock()
            manager.fulfill_interrupt = AsyncMock(return_value={})
            with (
                patch.object(app, "_remote_agent", return_value=remote),
                self._hooks_patch(manager),
            ):
                result = await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    {"messages": [{"type": "human", "content": "hi"}]},
                )

        from deepagents_code.app import _OFFLOAD_MAX_RESUME_ROUNDS

        assert rounds == _OFFLOAD_MAX_RESUME_ROUNDS + 1
        assert result is not None
        assert "could not complete" in result
        assert str(_OFFLOAD_MAX_RESUME_ROUNDS) in result

    async def test_unanswerable_interrupt_is_reported_not_silently_dropped(
        self,
    ) -> None:
        """An approval-shaped interrupt this client cannot answer is an error.

        A `PreToolUse` hook returning an `ask` permission makes the hook
        middleware raise a plain `HITLRequest`, which is not a hook-invocation
        payload. The operation graph has no HITL middleware to route it, so the
        run stays paused; dropping it silently would surface as "the
        conversation is already compact".
        """
        app = DeepAgentsApp()
        operation = MagicMock()
        approval = MagicMock()
        approval.id = "interrupt-1"
        approval.value = {"action_request": {"action": "compact_conversation"}}
        rounds = 0

        async def stream(_input: object, **_kwargs: object):  # noqa: ANN202, RUF029
            nonlocal rounds
            rounds += 1
            yield (), "updates", {"__interrupt__": [approval]}

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                return_value=SimpleNamespace(
                    values=_state_values([_make_dict_message("hi")])
                )
            )
            app._lc_thread_id = "test-thread"
            manager = MagicMock(spec=HooksManager)
            manager.apply_graph_context = MagicMock()
            manager.fulfill_interrupt = AsyncMock()
            with (
                patch.object(app, "_remote_agent", return_value=remote),
                self._hooks_patch(manager),
            ):
                result = await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    {"messages": [{"type": "human", "content": "hi"}]},
                )

        # Bails on the first round rather than burning the whole resume budget,
        # and never tries to fulfill a payload the hook engine cannot parse.
        assert rounds == 1
        assert result is not None
        assert "approval" in result
        manager.fulfill_interrupt.assert_not_awaited()


class TestDriveLegacySeededCompaction:
    """Unit-test the seeded in-process `compact_conversation` trigger.

    This driver serves local `Pregel` agents, which have no server operation
    graph; server-backed agents use the dedicated `offload` operation instead.
    """

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
                result = await app._drive_local_seeded_compaction(config)  # ty: ignore
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
            expected = {
                "model": "provider:active-model",
                "model_params": {"temperature": 0},
                "profile_overrides": {"max_input_tokens": 4096},
                "model_context_limit": 4096,
                "thread_id": "test-thread",
                "offload_tool_call_id": tool_call["id"],
            }
            assert len(astream_contexts) == 2
            for context in astream_contexts:
                assert isinstance(context, dict)
                normalized = {str(key): value for key, value in context.items()}
                assert {key: normalized[key] for key in expected} == expected

    async def test_records_summary_and_trailing_usage_in_cost_breakdown(self) -> None:
        """Manual offload usage reconciles by type and serving model."""
        from langchain_core.messages import AIMessage, ToolMessage

        from deepagents_code.client.remote_client import RemoteAgent

        class _Interrupt:
            id = "interrupt-1"
            value = {  # noqa: RUF012  # test stub; immutability irrelevant
                "action_requests": [
                    {"name": "compact_conversation", "args": {"force": True}}
                ]
            }

        summary = AIMessage(
            content="summary",
            id="summary-request",
            usage_metadata={
                "input_tokens": 200,
                "output_tokens": 20,
                "total_tokens": 220,
            },
            response_metadata={
                "model_name": "summary-model",
                "model_provider": "anthropic",
            },
        )
        trailing = AIMessage(
            content="done",
            id="trailing-request",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
            },
            response_metadata={
                "model_name": "active-model",
                "model_provider": "openai",
            },
        )

        async def _astream(  # noqa: ANN202, RUF029
            stream_input: object, **_kwargs: object
        ):
            if stream_input is None:
                yield ((), "updates", {"__interrupt__": [_Interrupt()]})
                return
            yield (
                (),
                "messages",
                (summary, {"lc_source": "summarization"}),
            )
            yield (
                (),
                "messages",
                (
                    ToolMessage(
                        content="Conversation compacted. Summarized 2 messages.",
                        tool_call_id="compact-call",
                    ),
                    {},
                ),
            )
            yield ((), "messages", (trailing, {}))

        agent = MagicMock(spec=RemoteAgent)
        agent.aensure_thread = AsyncMock()
        agent.aupdate_state = AsyncMock()
        agent.astream = _astream
        app = DeepAgentsApp()
        app._model_override = "openai:active-model"
        app._set_session_cost(0.50)
        for stats in (app._thread_stats, app._session_stats):
            stats.record_request(
                "active-model",
                1_000,
                100,
                provider="openai",
                cost_usd=0.50,
            )

        def _cost(
            _usage: object,
            model_name: str,
            _provider: str = "",
        ) -> float:
            return {"summary-model": 0.20, "active-model": 0.05}[model_name]

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = agent
            app._lc_thread_id = "test-thread"
            with patch(
                "deepagents_code.cost_tracking.estimate_cost", side_effect=_cost
            ):
                result = await app._drive_local_seeded_compaction(
                    {"configurable": {"thread_id": "test-thread"}}
                )
            await pilot.pause()

        assert result is None
        assert app._thread_stats.request_count == 3
        assert app._session_stats.request_count == 3
        assert app._thread_stats.per_kind["assistant"].cost_usd == pytest.approx(0.50)
        assert app._thread_stats.per_kind["offload"].request_count == 2
        assert app._thread_stats.per_kind["offload"].input_tokens == 300
        assert app._thread_stats.per_kind["offload"].output_tokens == 30
        assert app._thread_stats.per_kind["offload"].cost_usd == pytest.approx(0.25)
        assert app._thread_stats.per_model[
            "anthropic", "summary-model"
        ].cost_usd == pytest.approx(0.20)
        assert app._thread_stats.per_model[
            "openai", "active-model"
        ].cost_usd == pytest.approx(0.55)

        # The estimates keep the running total aligned until the graph-owned
        # checkpoint total arrives and clears the provisional amount.
        assert app._session_cost_usd == pytest.approx(0.50)
        assert app._displayed_cost_usd == pytest.approx(0.75)
        app._set_session_cost(0.75)
        assert app._displayed_cost_usd == pytest.approx(0.75)
        summary_text = app._format_cost_summary()
        assert "Estimated thread cost: $0.75" in summary_text
        assert "Assistant: $0.50" in summary_text
        assert "Offload: $0.25" in summary_text
        assert "anthropic:summary-model: $0.20" in summary_text
        assert "openai:active-model: $0.55" in summary_text
        assert "detailed usage metadata was unavailable" not in summary_text

    async def test_resume_replay_records_usage_once(self) -> None:
        """A usage message replayed after an interrupt is not double-counted."""
        from langchain_core.messages import AIMessage, ToolMessage

        from deepagents_code.client.remote_client import RemoteAgent

        class _Interrupt:
            id = "interrupt-1"
            value = {  # noqa: RUF012  # test stub; immutability irrelevant
                "action_requests": [
                    {"name": "compact_conversation", "args": {"force": True}}
                ]
            }

        usage_message = AIMessage(
            content="summary",
            id="replayed-request",
            usage_metadata={
                "input_tokens": 200,
                "output_tokens": 20,
                "total_tokens": 220,
            },
            response_metadata={"model_name": "summary-model"},
        )

        async def _astream(  # noqa: ANN202, RUF029
            stream_input: object, **_kwargs: object
        ):
            yield (
                (),
                "messages",
                (usage_message, {"lc_source": "summarization"}),
            )
            if stream_input is None:
                yield ((), "updates", {"__interrupt__": [_Interrupt()]})
            else:
                yield (
                    (),
                    "messages",
                    (ToolMessage(content="Nothing to compact", tool_call_id="x"), {}),
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
            with patch(
                "deepagents_code.cost_tracking.estimate_cost", return_value=0.20
            ):
                result = await app._drive_local_seeded_compaction(
                    {"configurable": {"thread_id": "test-thread"}}
                )
            await pilot.pause()

        assert result is None
        assert app._thread_stats.request_count == 1
        assert app._session_stats.request_count == 1
        assert app._thread_stats.per_kind["offload"].cost_usd == pytest.approx(0.20)
        assert app._session_cost_usd == pytest.approx(0.0)
        assert app._displayed_cost_usd == pytest.approx(0.20)
        summary = app._format_cost_summary()
        assert "Estimated thread cost: $0.20" in summary
        assert "Offload: $0.20" in summary

    async def test_stream_failure_keeps_usage_recorded_once(self) -> None:
        """Usage completed before a stream failure is still merged once."""
        from langchain_core.messages import AIMessage

        from deepagents_code.client.remote_client import RemoteAgent

        usage_message = AIMessage(
            content="summary",
            id="failed-request",
            usage_metadata={
                "input_tokens": 200,
                "output_tokens": 20,
                "total_tokens": 220,
            },
            response_metadata={"model_name": "summary-model"},
        )

        async def _astream(  # noqa: ANN202, RUF029
            _stream_input: object, **_kwargs: object
        ):
            for _ in range(2):
                yield (
                    (),
                    "messages",
                    (usage_message, {"lc_source": "summarization"}),
                )
            msg = "stream failed"
            raise RuntimeError(msg)

        agent = MagicMock(spec=RemoteAgent)
        agent.aensure_thread = AsyncMock()
        agent.aupdate_state = AsyncMock()
        agent.astream = _astream
        app = DeepAgentsApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = agent
            app._lc_thread_id = "test-thread"
            with (
                patch("deepagents_code.cost_tracking.estimate_cost", return_value=0.20),
                pytest.raises(RuntimeError, match="stream failed"),
            ):
                await app._drive_local_seeded_compaction(
                    {"configurable": {"thread_id": "test-thread"}}
                )
            await pilot.pause()

        assert app._thread_stats.request_count == 1
        assert app._session_stats.request_count == 1
        assert app._thread_stats.per_kind["offload"].cost_usd == pytest.approx(0.20)
        assert app._session_cost_usd == pytest.approx(0.0)
        assert app._displayed_cost_usd == pytest.approx(0.20)
        summary = app._format_cost_summary()
        assert "Estimated thread cost: $0.20" in summary
        assert "Offload: $0.20" in summary

    async def test_fulfills_precompact_before_manual_approval(self) -> None:
        """A precompact hook is fulfilled before the compaction approval."""
        from types import SimpleNamespace

        from langchain_core.messages import ToolMessage
        from langgraph.types import Command

        from deepagents_code.client.remote_client import RemoteAgent
        from deepagents_code.hooks.interrupt import HOOK_INVOCATION_INTERRUPT_TYPE

        streams: list[object] = []

        async def _astream(  # noqa: ANN202, RUF029
            value: object, **_kwargs: object
        ):
            index = len(streams)
            streams.append(value)
            if index == 0:
                interrupt = SimpleNamespace(
                    id="hook-interrupt",
                    value={"type": HOOK_INVOCATION_INTERRUPT_TYPE},
                )
            elif index == 1:
                interrupt = SimpleNamespace(
                    id="approval-interrupt",
                    value={
                        "action_requests": [
                            {
                                "name": "compact_conversation",
                                "args": {"force": True},
                            }
                        ]
                    },
                )
            else:
                yield (
                    (),
                    "messages",
                    (ToolMessage(content="compacted", tool_call_id="compact-call"), {}),
                )
                return
            yield ((), "updates", {"__interrupt__": [interrupt]})

        agent = MagicMock(
            spec=RemoteAgent,
            aensure_thread=AsyncMock(),
            aupdate_state=AsyncMock(),
            astream=_astream,
        )
        app = DeepAgentsApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            runtime = MagicMock(snapshot_id="snapshot")
            runtime.configured_server_events.return_value = ("PreCompact",)
            assert app._session_state is not None
            app._session_state.hooks = HooksManager.adopting(
                runtime,
                identity=app._session_state.hook_identity,
            )
            app._agent = agent
            app._lc_thread_id = "test-thread"
            fulfill = AsyncMock(return_value={"hook": "approved"})
            with patch("deepagents_code.hooks.client.fulfill_hook_interrupt", fulfill):
                result = await app._drive_local_seeded_compaction(
                    {"configurable": {"thread_id": "test-thread"}}
                )

        assert result is None
        fulfill.assert_awaited_once()
        assert len(streams) == 3
        assert isinstance(streams[1], Command)
        assert streams[1].resume == {"hook-interrupt": {"hook": "approved"}}
        assert isinstance(streams[2], Command)
        approval = streams[2].resume
        assert isinstance(approval, dict)
        assert "approval-interrupt" in approval

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
            result = await app._drive_local_seeded_compaction(config)  # ty: ignore
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
                await app._drive_local_seeded_compaction(config)  # ty: ignore
            await pilot.pause()

        assert contexts
        seed_values = agent.aupdate_state.call_args.args[1]
        (seed_msg,) = seed_values["messages"]
        (tool_call,) = seed_msg.tool_calls
        expected = {
            "model": "provider:startup-model",
            "model_params": {},
            "profile_overrides": {"max_input_tokens": 4096},
            "model_context_limit": 4096,
            "thread_id": "test-thread",
            "offload_tool_call_id": tool_call["id"],
        }
        for context in contexts:
            assert isinstance(context, dict)
            normalized = {str(key): value for key, value in context.items()}
            assert {key: normalized[key] for key in expected} == expected

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
            result = await app._drive_local_seeded_compaction(config)  # ty: ignore
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
            result = await app._drive_local_seeded_compaction(config)  # ty: ignore
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
            result = await app._drive_local_seeded_compaction(config)  # ty: ignore
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
            result = await app._drive_local_seeded_compaction(config)  # ty: ignore
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


def _deny_dispatched_call(
    reason: str | None,
) -> Callable[[Any, Any], dict[str, Any]]:
    """Build an `aafter_model` stub that denies whichever call was dispatched.

    Keys the outcome on the tool-call id the node actually generated rather than
    a fixed literal. The node derives that id per run (so a hook fulfillment
    cannot be memoized across two `/offload`s in one turn), so a hardcoded key
    here would silently stop matching and the denial would be read as "no
    outcome" instead of failing loudly.

    Also asserts the dispatched call's `name`/`args`, which are the values
    `ServerHooksMiddleware._after_model` gates on: without `compact_conversation`
    it never raises `PreCompact` at all, and without `force: True` it raises the
    event as `CompactTrigger.AUTO`, silently exempting `/offload` from a hook
    scoped to manual compaction. Uses the middleware's own state key for the
    same reason -- a re-spelling would make the node read `{}` and compact
    straight through this denial.

    Args:
        reason: Denial reason, or `None` to omit it.

    Returns:
        A side-effect callable for an `AsyncMock`.
    """
    from deepagents_code.hooks.server_middleware import _PRE_TOOL_STATE_KEY

    def deny(state: Any, _runtime: Any) -> dict[str, Any]:  # noqa: ANN401
        call = state["messages"][0].tool_calls[0]
        assert call["name"] == "compact_conversation"
        assert call["args"] == {"force": True}
        outcome: dict[str, Any] = {"behavior": "deny"}
        if reason is not None:
            outcome["reason"] = reason
        return {_PRE_TOOL_STATE_KEY: {call["id"]: outcome}}

    return deny


class TestForcedCompactionGraph:
    """Lifecycle and cost guarantees of the dedicated `/offload` graph."""

    async def test_precompact_denial_skips_forced_compaction(self) -> None:
        """A configured `PreCompact` hook can deny manual `/offload`.

        The denial has to raise rather than return an empty update: an empty
        update is indistinguishable from "nothing old enough to compact", so the
        client would render a hook veto as "the conversation is already compact"
        and the reason would never reach the user.
        """
        from deepagents_code._cli_context import CLIContext
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock()
        hooks = MagicMock()
        hooks.aafter_model = AsyncMock(
            side_effect=_deny_dispatched_call("policy forbids compaction")
        )

        graph = create_forced_compaction_graph(middleware, hooks_middleware=hooks)
        with pytest.raises(RuntimeError, match="policy forbids compaction"):
            await cast("Any", graph).ainvoke({"messages": []}, context=CLIContext())

        hooks.aafter_model.assert_awaited_once()
        middleware.arun_forced_compaction_update.assert_not_awaited()

    async def test_precompact_denial_without_reason_still_surfaces(self) -> None:
        """A denial carrying no reason still reports something actionable."""
        from deepagents_code._cli_context import CLIContext
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock()
        hooks = MagicMock()
        hooks.aafter_model = AsyncMock(side_effect=_deny_dispatched_call(None))

        graph = create_forced_compaction_graph(middleware, hooks_middleware=hooks)
        with pytest.raises(RuntimeError, match="Blocked by a compaction hook"):
            await cast("Any", graph).ainvoke({"messages": []}, context=CLIContext())

        middleware.arun_forced_compaction_update.assert_not_awaited()

    async def test_compaction_failure_is_raised_with_a_preserved_message(self) -> None:
        """A node failure raises `RuntimeError` so its text survives the server.

        The LangGraph server preserves an exception's message only for an
        allowlist of builtin types and replaces every other one with "An
        internal error occurred", so an `OSError` from the archive write has to
        be re-raised as an allowlisted type to stay diagnosable.
        """
        from deepagents_code._cli_context import CLIContext
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock(
            side_effect=OSError("disk is full")
        )

        graph = create_forced_compaction_graph(middleware, hooks_middleware=None)
        with pytest.raises(RuntimeError, match="OSError: disk is full") as exc_info:
            await cast("Any", graph).ainvoke({"messages": []}, context=CLIContext())

        assert "Your conversation is unchanged." in str(exc_info.value)

    async def test_failed_compaction_leaves_summary_spend_undrained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure must not drain cost it is about to discard.

        The drain is destructive and the raise discards any update built on this
        path, so draining here would lose the summarizer's spend outright.
        Undrained records are charged on the next turn's first step instead.
        """
        from deepagents_code import offload_middleware
        from deepagents_code._cli_context import CLIContext

        cost_tracking = MagicMock()
        cost_tracking.after_agent = MagicMock(return_value={"_session_cost_usd": 0.25})
        monkeypatch.setattr(
            offload_middleware, "CostTrackingMiddleware", lambda: cost_tracking
        )
        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock(
            side_effect=OSError("disk is full")
        )

        graph = offload_middleware.create_forced_compaction_graph(
            middleware, hooks_middleware=None
        )
        with pytest.raises(RuntimeError):
            await cast("Any", graph).ainvoke({"messages": []}, context=CLIContext())

        cost_tracking.after_agent.assert_not_called()

    def test_only_messages_is_a_writable_input_channel(self) -> None:
        """The run input surface is restricted by the graph, not by the client.

        `PrivateStateAttr` / `OmitFromInput` on the state schema are honored by
        `create_agent`, *not* by a raw `StateGraph`, so without an explicit
        `input_schema` every channel `_OffloadState` declares would be writable
        by any local caller on the same `noop`-auth port (see THREAT_MODEL TB10):
        `_summarization_event` would let them set the compaction cutoff, and
        `_session_cost_usd` (an `operator.add` channel) would let them inflate
        the thread's recorded spend.
        """
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        graph = create_forced_compaction_graph(MagicMock(), hooks_middleware=None)

        assert set(graph.get_input_jsonschema()["properties"]) == {"messages"}

    async def test_private_channel_in_input_is_dropped_not_applied(self) -> None:
        """An injected private channel must not reach the node or the checkpoint.

        The schema check above is the mechanism; this is the behavior. A replayed
        `_session_cost_usd` would *add* to the thread's total, so echoing the
        checkpointed value back would double the recorded spend on every
        `/offload` and compound across runs.
        """
        from langchain_core.messages import HumanMessage

        from deepagents_code._cli_context import CLIContext
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        seen: dict[str, Any] = {}
        middleware = MagicMock()

        async def capture(state: Any, _runtime: Any) -> dict[str, Any]:  # noqa: ANN401, RUF029
            seen.update(state)
            return {}

        middleware.arun_forced_compaction_update = AsyncMock(side_effect=capture)
        graph = create_forced_compaction_graph(middleware, hooks_middleware=None)

        out = await cast("Any", graph).ainvoke(
            {
                "messages": [HumanMessage("hi", id="m1")],
                "_session_cost_usd": 0.42,
                "_summarization_event": {"cutoff_index": 99},
            },
            context=CLIContext(),
        )

        assert seen.get("_session_cost_usd") == pytest.approx(0.0)
        assert seen.get("_summarization_event") is None
        assert [m.content for m in seen["messages"]] == ["hi"]
        assert out.get("_session_cost_usd") == pytest.approx(0.0)

    async def test_forced_hook_call_id_is_unique_per_invocation(self) -> None:
        """Two `/offload`s in one turn must not share a hook invocation id.

        `ServerHooksMiddleware` derives its `invocation_id` from this id plus the
        thread, hook snapshot, and prompt id — and the prompt id only rotates on
        user-prompt submit. A constant id would therefore collide across two
        `/offload`s in one turn and replay the first decision, so a first-run
        *denial* would silently deny the second run too.
        """
        from deepagents_code._cli_context import CLIContext
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        seen_ids: list[str] = []
        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock(return_value={})
        hooks = MagicMock()

        async def record(state: Any, _runtime: Any) -> dict[str, Any]:  # noqa: ANN401, RUF029
            seen_ids.append(state["messages"][0].tool_calls[0]["id"])
            return {}

        hooks.aafter_model = AsyncMock(side_effect=record)
        graph = create_forced_compaction_graph(middleware, hooks_middleware=hooks)

        await cast("Any", graph).ainvoke({"messages": []}, context=CLIContext())
        await cast("Any", graph).ainvoke({"messages": []}, context=CLIContext())

        assert len(seen_ids) == 2
        assert seen_ids[0] != seen_ids[1]

    async def test_hook_interrupt_survives_a_real_resume_round_trip(
        self, tmp_path: Path
    ) -> None:
        """A fulfilled hook interrupt must let the offload finish.

        The one test that exercises the whole loop against the real
        `ServerHooksMiddleware` rather than a stub. Answering an `interrupt()`
        re-executes the node **from the top**, so anything the node derives
        before dispatching must survive that replay. It previously minted the
        forced tool-call id with `uuid4()`; the middleware folds that id into
        its hook `invocation_id`, so the resumed execution computed a different
        one and `parse_hook_resume_value` rejected the client's answer as
        fatal — making `/offload` fail outright for every user with a
        `PreCompact`/`PreToolUse` hook configured, and rendering the client's
        entire fulfill/resume loop unreachable.

        Mocking either side hides this: the bug lives in the interaction between
        node replay and invocation-id derivation.
        """
        from uuid import uuid4

        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        from deepagents_code._cli_context import CLIContextSchema
        from deepagents_code.hooks.interrupt import (
            build_hook_resume_value,
            parse_hook_interrupt_payload,
        )
        from deepagents_code.hooks.models.domain import HookEvent, PreCompactDecision
        from deepagents_code.hooks.models.transport import HookInvocationResponse
        from deepagents_code.hooks.server_middleware import ServerHooksMiddleware
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock(
            return_value={"_summarization_event": {"cutoff_index": 4}}
        )
        # The graph compiles without a checkpointer (the LangGraph server owns
        # durability in production), but an interrupt cannot resume without one.
        built = create_forced_compaction_graph(
            middleware, hooks_middleware=ServerHooksMiddleware(cwd=tmp_path)
        )
        graph = cast("Any", built).builder.compile(checkpointer=InMemorySaver())

        context = CLIContextSchema(
            hooks_snapshot_id="snap",
            hooks_server_events=[HookEvent.PRE_COMPACT.value],
            thread_id="t1",
        )
        config = {"configurable": {"thread_id": str(uuid4())}}

        first = await graph.ainvoke(
            {"messages": [HumanMessage("hi", id="m1")]},
            config=config,
            context=context,
        )
        interrupts = first["__interrupt__"]
        assert len(interrupts) == 1
        request = parse_hook_interrupt_payload(interrupts[0].value)
        assert request is not None

        resume_value = build_hook_resume_value(
            HookInvocationResponse(
                protocol_version=1,
                invocation_id=request.invocation_id,
                snapshot_id=request.snapshot_id,
                decision=PreCompactDecision(event=HookEvent.PRE_COMPACT),
            )
        )
        result = await graph.ainvoke(
            Command(resume={interrupts[0].id: resume_value}),
            config=config,
            context=context,
        )

        # The resume was accepted and the compaction actually ran.
        assert result["_summarization_event"]["cutoff_index"] == 4
        middleware.arun_forced_compaction_update.assert_awaited_once()

    async def test_hook_dispatch_failure_is_reported_as_a_hook_failure(self) -> None:
        """A hook-layer crash must not reach the user as "internal error".

        The server replaces the message of any exception outside its builtin
        allowlist, so this has to be re-raised as `RuntimeError` like the
        compaction failure — but worded so it is not read as a compaction bug.
        """
        from deepagents_code._cli_context import CLIContext
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock()
        hooks = MagicMock()
        hooks.aafter_model = AsyncMock(side_effect=OSError("hook socket died"))
        graph = create_forced_compaction_graph(middleware, hooks_middleware=hooks)

        with pytest.raises(RuntimeError, match=r"Offload hooks failed.*socket died"):
            await cast("Any", graph).ainvoke({"messages": []}, context=CLIContext())

        middleware.arun_forced_compaction_update.assert_not_awaited()

    async def test_hook_interrupt_bubbles_to_the_operation_graph(self) -> None:
        """A hook approval pause must reach the server's interrupt stream."""
        from langgraph.errors import GraphInterrupt
        from langgraph.types import Interrupt

        from deepagents_code._cli_context import CLIContext
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock()
        hooks = MagicMock()
        hooks.aafter_model = AsyncMock(
            side_effect=GraphInterrupt((Interrupt(value={"type": "hook_invocation"}),))
        )
        graph = create_forced_compaction_graph(middleware, hooks_middleware=hooks)

        result = await cast("Any", graph).ainvoke(
            {"messages": []}, context=CLIContext()
        )

        assert result["__interrupt__"]
        middleware.arun_forced_compaction_update.assert_not_awaited()

    async def test_cost_drain_failure_does_not_discard_the_compaction(self) -> None:
        """A bookkeeping failure must not throw away a committed archive write.

        By the time the drain runs, the archive section is already written. Raising
        here would report "your conversation is unchanged" while leaving an
        orphaned section no `_summarization_event` references.
        """
        from deepagents_code import offload_middleware
        from deepagents_code._cli_context import CLIContext

        event = {"cutoff_index": 3, "summary_message": None, "file_path": "/a.md"}
        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock(
            return_value={"_summarization_event": event}
        )

        with patch.object(
            offload_middleware.CostTrackingMiddleware,
            "after_agent",
            new=MagicMock(side_effect=RuntimeError("pricing down")),
        ):
            graph = offload_middleware.create_forced_compaction_graph(
                middleware, hooks_middleware=None
            )
            out = await cast("Any", graph).ainvoke(
                {"messages": []}, context=CLIContext()
            )

        assert out["_summarization_event"] == event

    async def test_operation_graph_preserves_agent_only_channels(self) -> None:
        """The narrow operation schema must not drop the agent's other state.

        Both graphs run against one thread, but `_OffloadState` declares only
        the two channels the operation needs. A schema that dropped or replaced
        the rest would silently destroy conversation state the agent owns, so
        this pins that a real checkpoint round-trip leaves it intact.
        """
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph

        from deepagents_code._cli_context import CLIContext
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        checkpointer = InMemorySaver()
        config = {"configurable": {"thread_id": "shared-thread"}}

        # Stand in for the agent graph: a wider schema on the same thread.
        class _WideState(TypedDict, total=False):
            messages: Annotated[list, add_messages]
            todos: list[str]
            _summarization_event: dict

        def _seed(state: _WideState) -> dict:  # noqa: ARG001  # node input unused
            return {
                "messages": [HumanMessage(content="secret history", id="m0")],
                "todos": ["keep me"],
            }

        wide = StateGraph(cast("Any", _WideState))
        wide.add_node("seed", cast("Any", _seed))
        wide.add_edge(START, "seed")
        wide.add_edge("seed", END)
        await cast("Any", wide.compile(checkpointer=checkpointer)).ainvoke({}, config)

        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock(
            return_value={"_summarization_event": {"cutoff_index": 1}}
        )
        offload = create_forced_compaction_graph(middleware, hooks_middleware=None)
        # `create_forced_compaction_graph` compiles without a checkpointer
        # because the LangGraph server attaches its own to every registered
        # graph. Do the same here so both graphs share one thread.
        cast("Any", offload).checkpointer = checkpointer
        await cast("Any", offload).ainvoke({}, config, context=CLIContext())

        # Read back through the wide schema, as the agent graph would.
        state = await cast("Any", wide.compile(checkpointer=checkpointer)).aget_state(
            config
        )
        assert state.values["todos"] == ["keep me"]
        assert [message.id for message in state.values["messages"]] == ["m0"]
        assert state.values["_summarization_event"] == {"cutoff_index": 1}

    async def test_checkpointed_summarization_event_reaches_the_node(self) -> None:
        """The prior event must arrive from the checkpoint, not from input.

        `_OffloadInput` deliberately keeps `_summarization_event` out of the run
        input, and a separate test pins that. This is the other half: the node
        still has to *read* it from `_OffloadState`. Narrowing that schema — an
        easy-looking cleanup, given `_OffloadInput` sits right beside it — would
        hand the node `event=None`, so a second `/offload` would re-summarize
        already-archived messages and compute its cutoff from the wrong base.
        Both graph-level tests would still pass.
        """
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph

        from deepagents_code._cli_context import CLIContext
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        checkpointer = InMemorySaver()
        config = {"configurable": {"thread_id": "resumed-thread"}}
        prior_event = {"cutoff_index": 7, "summary_message": None, "file_path": "/a.md"}

        class _WideState(TypedDict, total=False):
            messages: Annotated[list, add_messages]
            _summarization_event: dict

        def _seed(state: _WideState) -> dict:  # noqa: ARG001  # node input unused
            return {
                "messages": [HumanMessage(content="older", id="m0")],
                "_summarization_event": prior_event,
            }

        wide = StateGraph(cast("Any", _WideState))
        wide.add_node("seed", cast("Any", _seed))
        wide.add_edge(START, "seed")
        wide.add_edge("seed", END)
        await cast("Any", wide.compile(checkpointer=checkpointer)).ainvoke({}, config)

        seen: dict[str, Any] = {}

        async def capture(state: Any, _runtime: Any) -> dict[str, Any]:  # noqa: ANN401, RUF029
            seen.update(state)
            return {}

        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock(side_effect=capture)
        offload = create_forced_compaction_graph(middleware, hooks_middleware=None)
        cast("Any", offload).checkpointer = checkpointer
        await cast("Any", offload).ainvoke(
            {"messages": []}, config, context=CLIContext()
        )

        assert seen["_summarization_event"] == prior_event

    async def test_replayed_input_reaches_the_node_without_duplicating(self) -> None:
        """The driver's state replay must arrive intact and not double up.

        `_drive_offload_operation_graph` replays the thread's messages as the run
        input because an empty input leaves a *server-backed* run with nothing to
        compact. The `add_messages` reducer is what makes that safe: replaying
        messages that are already checkpointed merges them by ID rather than
        appending a second copy.
        """
        from langchain_core.messages import HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph

        from deepagents_code._cli_context import CLIContext
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        checkpointer = InMemorySaver()
        config = {"configurable": {"thread_id": "shared-thread"}}
        seeded = [
            HumanMessage(content="history", id="m0"),
            HumanMessage(content="more", id="m1"),
        ]

        class _WideState(TypedDict, total=False):
            messages: Annotated[list, add_messages]

        def _seed(state: _WideState) -> dict:  # noqa: ARG001  # node input unused
            return {"messages": seeded}

        wide = StateGraph(cast("Any", _WideState))
        wide.add_node("seed", cast("Any", _seed))
        wide.add_edge(START, "seed")
        wide.add_edge("seed", END)
        await cast("Any", wide.compile(checkpointer=checkpointer)).ainvoke({}, config)

        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock(return_value=None)
        offload = create_forced_compaction_graph(middleware, hooks_middleware=None)
        cast("Any", offload).checkpointer = checkpointer
        await cast("Any", offload).ainvoke(
            {"messages": seeded}, config, context=CLIContext()
        )

        await_args = middleware.arun_forced_compaction_update.await_args
        assert await_args is not None
        state_arg = await_args.args[0]
        assert [message.id for message in state_arg["messages"]] == ["m0", "m1"]

    async def test_nothing_to_compact_returns_an_empty_update(self) -> None:
        """A `None` update must not become a partial write."""
        from deepagents_code._cli_context import CLIContext
        from deepagents_code.offload_middleware import create_forced_compaction_graph

        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock(return_value=None)

        graph = create_forced_compaction_graph(middleware, hooks_middleware=None)
        result = await cast("Any", graph).ainvoke(
            {"messages": []}, context=CLIContext()
        )

        assert "_summarization_event" not in result

    async def test_summary_cost_is_drained_into_graph_update(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The graph checkpoints summarizer spend before `/offload` returns.

        The stub implements only `after_agent`, exactly like the real
        `CostTrackingMiddleware`, and inherits nothing. Awaiting `aafter_agent`
        instead therefore raises `AttributeError` into the drain's `except` and
        the spend silently vanishes — which is what a `MagicMock` stub hid:
        `AgentMiddleware.aafter_agent` exists as an empty base method, so
        `await ...aafter_agent(...)` returns `None` in production while a mock
        materializes a working one in the test.
        """
        from deepagents_code import offload_middleware
        from deepagents_code._cli_context import CLIContext

        calls: list[str] = []

        class _OnlySyncDrain:
            """Mirrors `CostTrackingMiddleware`'s real method surface."""

            def after_agent(self, _state: object, _runtime: object) -> dict[str, float]:
                calls.append("after_agent")
                return {"_session_cost_usd": 0.25}

        monkeypatch.setattr(
            offload_middleware, "CostTrackingMiddleware", _OnlySyncDrain
        )
        middleware = MagicMock()
        middleware.arun_forced_compaction_update = AsyncMock(
            return_value={"_summarization_event": {"cutoff_index": 2}}
        )

        graph = offload_middleware.create_forced_compaction_graph(
            middleware, hooks_middleware=None
        )
        result = await cast("Any", graph).ainvoke(
            {"messages": []}, context=CLIContext()
        )

        assert calls == ["after_agent"]
        assert result["_session_cost_usd"] == pytest.approx(0.25)


class TestOffloadDriverSelection:
    """`_handle_offload` must pick its driver by agent kind, not by accident.

    Every other test in this file asserts routing *indirectly*, by patching the
    driver it expects and letting an unpatched one blow up against a mock. That
    fails loudly but misleadingly. These pin the predicate itself.
    """

    async def test_server_agent_uses_only_the_operation_graph(self) -> None:
        """A `RemoteAgent` must never take the seeded in-process path."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    AsyncMock(return_value=_state_values([_make_dict_message("hi")])),
                ),
                patch.object(
                    app, "_drive_offload_operation_graph", AsyncMock(return_value=None)
                ) as operation,
                patch.object(
                    app, "_drive_local_seeded_compaction", AsyncMock(return_value=None)
                ) as seeded,
            ):
                await app._handle_offload()

        operation.assert_awaited_once()
        seeded.assert_not_awaited()

    async def test_custom_server_graph_falls_back_to_seeded_compaction(self) -> None:
        """A custom graph without `offload` retains the pre-operation behavior."""
        from deepagents_code.app import _MissingOffloadGraphError

        app = DeepAgentsApp()
        before = _state_values(_make_dict_messages(2))
        after = _state_values(_make_dict_messages(2), _summary_event(1))
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    AsyncMock(side_effect=[before, after]),
                ),
                patch.object(
                    app,
                    "_drive_offload_operation_graph",
                    AsyncMock(side_effect=_MissingOffloadGraphError()),
                ) as operation,
                patch.object(
                    app, "_drive_local_seeded_compaction", AsyncMock(return_value=None)
                ) as seeded,
            ):
                await app._handle_offload()

        operation.assert_awaited_once()
        seeded.assert_awaited_once()

    async def test_local_agent_uses_only_the_seeded_driver(self) -> None:
        """A local in-process `Pregel` agent must never stream the named graph."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_local_offload_app(app)
            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    AsyncMock(return_value=_state_values([_make_dict_message("hi")])),
                ),
                patch.object(
                    app, "_drive_offload_operation_graph", AsyncMock(return_value=None)
                ) as operation,
                patch.object(
                    app, "_drive_local_seeded_compaction", AsyncMock(return_value=None)
                ) as seeded,
            ):
                await app._handle_offload()

        seeded.assert_awaited_once()
        operation.assert_not_awaited()


class TestOffloadReplaySafety:
    """The run input replaces the `messages` channel, so it must be current.

    Against a real LangGraph server the `/offload` run input is authoritative
    for `messages`: streaming `{"messages": []}` empties an eight-message thread
    (see `test_offload_server_side.py`). A stale or partial replay is therefore
    not a stale read but a destructive write.
    """

    @staticmethod
    def _remote_with_stream(chunks: list[Any]) -> MagicMock:
        """Build a remote whose offload graph yields `chunks`."""
        operation = MagicMock()

        async def stream(*_args: object, **_kwargs: object):  # noqa: ANN202, RUF029
            for chunk in chunks:
                yield chunk

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation
        return remote

    async def test_replays_freshly_read_state_not_the_callers_snapshot(self) -> None:
        """A turn committed after the caller's snapshot must not be dropped.

        `_handle_offload` reads state *before* `_set_agent_running(True)`, so a
        run committed in that window is missing from its snapshot. Replaying
        that snapshot would write the shorter list over the live conversation
        and delete the newer turn.
        """
        app = DeepAgentsApp()
        stale = [_make_dict_message("one")]
        fresh = [_make_dict_message("one"), _make_dict_message("two")]
        captured: list[object] = []

        operation = MagicMock()

        async def stream(*args: object, **_kwargs: object):  # noqa: ANN202, RUF029
            captured.extend(args)
            yield (), "updates", {"force_compact": {}}

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                return_value=SimpleNamespace(values=_state_values(fresh))
            )
            app._lc_thread_id = "test-thread"
            with patch.object(app, "_remote_agent", return_value=remote):
                await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    _state_values(stale),
                )

        assert captured == [{"messages": fresh}]

    async def test_empty_replay_is_refused_rather_than_streamed(self) -> None:
        """An unreadable state must abort, not wipe the conversation.

        If both the re-read and the caller's snapshot yield no messages, running
        anyway would truncate the thread to zero and report "already compact".
        """
        app = DeepAgentsApp()
        remote = self._remote_with_stream([((), "updates", {"force_compact": {}})])

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                return_value=SimpleNamespace(values={"messages": []})
            )
            app._lc_thread_id = "test-thread"
            with (
                patch.object(app, "_remote_agent", return_value=remote),
                pytest.raises(RuntimeError, match="could not be read back"),
            ):
                await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}}, {"messages": []}
                )

        remote.for_graph.assert_not_called()


class TestOffloadDrainCompletion:
    """A stream that never reports the node's update is not a success."""

    @staticmethod
    def _app_with_chunks(app: DeepAgentsApp, chunks: list[Any]) -> MagicMock:
        """Point `app` at a remote whose offload stream yields `chunks`."""
        operation = MagicMock()

        async def stream(*_args: object, **_kwargs: object):  # noqa: ANN202, RUF029
            for chunk in chunks:
                yield chunk

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation
        app._agent = MagicMock()
        app._agent.aget_state = AsyncMock(
            return_value=SimpleNamespace(
                values=_state_values([_make_dict_message("hi")])
            )
        )
        app._lc_thread_id = "test-thread"
        return remote

    async def test_missing_node_update_is_reported_as_a_failure(self) -> None:
        """Chunk-shape drift must not read as "already compact".

        Interrupt detection is all the drain loop does, so a chunk shape it
        cannot parse is indistinguishable from a clean run: no interrupts, no
        error, and a run still paused server-side. The caller would then read an
        unadvanced event and tell the user their conversation is already
        compact.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # 2-tuples: the shape the loop's `len(chunk) != 3` filter discards.
            remote = self._app_with_chunks(app, [("updates", {"force_compact": {}})])
            with patch.object(app, "_remote_agent", return_value=remote):
                result = await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    _state_values([_make_dict_message("hi")]),
                )

        assert result is not None
        assert "without reporting a result" in result
        assert "conversation is unchanged" in result

    async def test_node_update_marks_the_run_complete(self) -> None:
        """The positive control: a real node update reports success."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = self._app_with_chunks(
                app, [((), "updates", {"force_compact": {}})]
            )
            with patch.object(app, "_remote_agent", return_value=remote):
                result = await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    _state_values([_make_dict_message("hi")]),
                )

        assert result is None

    async def test_paused_run_keeps_its_offload_graph_binding(self) -> None:
        """A drain failure leaves the run suspended, so do not rebind the thread.

        Rebinding would re-point the thread away from the graph the paused run
        belongs to, leaving it unaddressable.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = self._app_with_chunks(app, [("updates", {})])
            with patch.object(app, "_remote_agent", return_value=remote):
                result = await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    _state_values([_make_dict_message("hi")]),
                )

        assert result is not None
        remote.arebind_thread.assert_not_awaited()


class TestOffloadHookStopReporting:
    """A hook that stops the client mid-`/offload` must still say something."""

    async def test_operation_path_reports_a_hook_stop(self) -> None:
        """The operation graph's fulfillment mounts nothing on its own."""
        from deepagents_code.hooks.client_lifecycle import ClientHookStopError

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    AsyncMock(return_value=_state_values([_make_dict_message("hi")])),
                ),
                patch.object(
                    app,
                    "_drive_offload_operation_graph",
                    AsyncMock(side_effect=ClientHookStopError("stopped")),
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            errors = [str(w._content) for w in app.query(ErrorMessage)]

        assert any("Offload stopped by a hook." in text for text in errors)

    async def test_seeded_path_stays_silent(self) -> None:
        """The seeded driver mounts its own stop reason before raising.

        Mounting a second, generic line here would duplicate it.
        """
        from deepagents_code.hooks.client_lifecycle import ClientHookStopError

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_local_offload_app(app)
            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    AsyncMock(return_value=_state_values([_make_dict_message("hi")])),
                ),
                patch.object(
                    app,
                    "_drive_local_seeded_compaction",
                    AsyncMock(side_effect=ClientHookStopError("stopped")),
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            errors = [str(w._content) for w in app.query(ErrorMessage)]

        assert not any("Offload stopped by a hook." in text for text in errors)


class TestOffloadRebindWarningCoverage:
    """Every path that reports a finished offload must surface a failed rebind.

    A dropped warning leaves an unrelated later `/goal` or `/rubric` to fail
    with an opaque `as_node="model"` error and nothing tying it to the offload.
    """

    async def test_unconfirmed_result_still_warns(self) -> None:
        """The empty-state branch says "Offload finished" — so it must warn."""
        app = DeepAgentsApp()

        async def _fail_rebind(*_args: object, **_kwargs: object) -> None:  # noqa: RUF029
            # `_handle_offload` clears the flag before dispatching, so the
            # driver has to be the one that sets it -- as the real one does in
            # its `finally`.
            app._offload_rebind_failed = True

        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            states = [_state_values([_make_dict_message("hi")]), {}]
            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    AsyncMock(side_effect=states),
                ),
                patch.object(
                    app,
                    "_drive_offload_operation_graph",
                    AsyncMock(side_effect=_fail_rebind),
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            errors = [str(w._content) for w in app.query(ErrorMessage)]

        assert any("could not be confirmed" in text for text in errors)
        assert any("re-associated with the main agent" in text for text in errors)

    async def test_hook_stop_still_warns(self) -> None:
        """A hook stop after a failed rebind must not swallow the warning."""
        from deepagents_code.hooks.client_lifecycle import ClientHookStopError

        app = DeepAgentsApp()

        async def _stop(*_args: object, **_kwargs: object) -> None:  # noqa: RUF029
            app._offload_rebind_failed = True
            msg = "stopped"
            raise ClientHookStopError(msg)

        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    AsyncMock(return_value=_state_values([_make_dict_message("hi")])),
                ),
                patch.object(
                    app, "_drive_offload_operation_graph", AsyncMock(side_effect=_stop)
                ),
            ):
                await app._handle_offload()
                await pilot.pause()

            errors = [str(w._content) for w in app.query(ErrorMessage)]

        assert any("re-associated with the main agent" in text for text in errors)


class TestOffloadResourceWiring:
    """`/offload` must archive into the agent's own backend."""

    def test_mismatched_backend_is_rejected_at_construction(self) -> None:
        """A compaction bound elsewhere would archive where the agent cannot read.

        The symptom — history that silently is not there — surfaces long after
        the mis-wiring, so fail where the pair is built instead.
        """
        from deepagents_code.offload_middleware import (
            OffloadServerResources,
            attach_offload_resources,
        )

        backend = SimpleNamespace()
        compaction = MagicMock()
        compaction._summarization._backend = SimpleNamespace()

        with pytest.raises(ValueError, match="different backend"):
            attach_offload_resources(
                cast("Any", backend),
                OffloadServerResources(
                    compaction=cast("Any", compaction), hooks=cast("Any", MagicMock())
                ),
            )

    def test_matching_backend_is_published(self) -> None:
        """The positive control for the check above."""
        from deepagents_code.offload_middleware import (
            OffloadServerResources,
            attach_offload_resources,
            offload_resources_from,
        )

        backend = SimpleNamespace()
        compaction = MagicMock()
        compaction._summarization._backend = backend
        hooks = MagicMock()

        attach_offload_resources(
            cast("Any", backend),
            OffloadServerResources(
                compaction=cast("Any", compaction), hooks=cast("Any", hooks)
            ),
        )

        resources = offload_resources_from(cast("Any", backend))
        assert resources is not None
        assert resources.compaction is compaction
        assert resources.hooks is hooks


class TestForcedOffloadCallId:
    """The hook dispatch's call id must be stable across a run's resumes."""

    def test_missing_checkpoint_namespace_is_logged_not_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A run without a usable `checkpoint_ns` breaks hook resumes.

        The random fallback makes the id differ between the request and the
        resume, which `parse_hook_resume_value` rejects as fatal — so `/offload`
        dies with "the client answered a different request", but only for users
        with hooks configured. Without a log line there is nothing to point at.
        """
        from deepagents_code import offload_middleware

        with (
            patch.object(
                offload_middleware,
                "get_config",
                return_value={"configurable": {}},
            ),
            caplog.at_level("WARNING"),
        ):
            call_id = offload_middleware._forced_offload_call_id()

        assert call_id.startswith("offload-precompact-")
        assert "checkpoint_ns" in caplog.text

    def test_no_runnable_context_is_not_warned_about(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A direct call outside a graph is expected, not a misconfiguration.

        Nothing can interrupt or resume such a call, so the random id is
        correct there and must not be reported as a problem.
        """
        from deepagents_code import offload_middleware

        with (
            patch.object(
                offload_middleware, "get_config", side_effect=RuntimeError("no context")
            ),
            caplog.at_level("WARNING"),
        ):
            call_id = offload_middleware._forced_offload_call_id()

        assert call_id.startswith("offload-precompact-")
        assert "checkpoint_ns" not in caplog.text

    def test_same_namespace_yields_the_same_id(self) -> None:
        """Answering a hook interrupt replays the node from the top."""
        from deepagents_code import offload_middleware

        config = {"configurable": {"checkpoint_ns": "force_compact:abc123"}}
        with patch.object(offload_middleware, "get_config", return_value=config):
            first = offload_middleware._forced_offload_call_id()
            second = offload_middleware._forced_offload_call_id()

        assert first == second


class TestOffloadStreamShape:
    """The operation driver must request the chunk shape it parses."""

    async def test_streams_with_subgraphs_enabled(self) -> None:
        """`subgraphs=True` is what makes the yielded chunk a real 3-tuple.

        `RemoteAgent.astream` only yields the documented
        `(namespace, mode, data)` shape when `subgraphs` is set. Without it a
        live server yields `("updates", {...}, None)`, so the driver's unpacking
        binds `mode` to the payload dict, every chunk falls through the
        `mode != "updates"` filter, and interrupts are dropped in silence — the
        run stays paused while the user is told their conversation is already
        compact. Unit tests feed the loop chunks directly and cannot see that,
        so pin the request instead.
        """
        app = DeepAgentsApp()
        stream_kwargs: dict[str, object] = {}
        operation = MagicMock()

        async def stream(*_args: object, **kwargs: object):  # noqa: ANN202, RUF029
            stream_kwargs.update(kwargs)
            yield (), "updates", {"force_compact": {}}

        operation.astream = stream
        remote = MagicMock()
        remote.aensure_thread = AsyncMock()
        remote.arebind_thread = AsyncMock()
        remote.for_graph.return_value = operation

        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent = MagicMock()
            app._agent.aget_state = AsyncMock(
                return_value=SimpleNamespace(
                    values=_state_values([_make_dict_message("hi")])
                )
            )
            app._lc_thread_id = "test-thread"
            with patch.object(app, "_remote_agent", return_value=remote):
                await app._drive_offload_operation_graph(
                    {"configurable": {"thread_id": "test-thread"}},
                    _state_values([_make_dict_message("hi")]),
                )

        assert stream_kwargs["subgraphs"] is True


class TestSeededDriverAgainstALocalAgent:
    """The seeded driver's *only* production shape is a local `Pregel` agent.

    Every other test in `TestDriveLocalSeededCompaction` builds its agent with
    `MagicMock(spec=RemoteAgent)`, but `_handle_offload` now routes remote
    agents to the operation graph — so the driver is exercised exclusively in
    the shape it no longer serves. These use a non-`RemoteAgent` double.
    """

    @staticmethod
    def _local_agent(tool_content: str) -> tuple[MagicMock, dict[str, object]]:
        """Build a local (non-`RemoteAgent`) agent double.

        Returns:
            The agent and a dict recording the kwargs it was streamed with.
        """
        from langchain_core.messages import ToolMessage

        stream_kwargs: dict[str, object] = {}

        async def _astream(*_args: object, **kwargs: object):  # noqa: ANN202, RUF029
            stream_kwargs.update(kwargs)
            yield (
                (),
                "messages",
                (ToolMessage(content=tool_content, tool_call_id="x"), {}),
            )

        agent = MagicMock()
        agent.aupdate_state = AsyncMock()
        agent.astream = _astream
        return agent, stream_kwargs

    async def test_local_agent_is_driven_without_thread_registration(self) -> None:
        """A local agent has no server thread to register.

        `aensure_thread` exists only on `RemoteAgent`; calling it here would
        raise, and the tool result must still be detected from the stream.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            agent, stream_kwargs = self._local_agent(
                "Conversation compacted. Summarized 2 messages into a summary."
            )
            app._agent = agent
            app._lc_thread_id = "test-thread"

            result = await app._drive_local_seeded_compaction(  # ty: ignore
                {"configurable": {"thread_id": "test-thread"}}
            )

        assert result is None
        assert not hasattr(agent.aensure_thread, "assert_awaited")  # not a RemoteAgent
        # The driver parses `(namespace, mode, data)`, so it must ask for it.
        assert stream_kwargs["subgraphs"] is True

    async def test_local_agent_failure_is_detected_from_the_tool_message(self) -> None:
        """The driver's only failure signal is the `ToolMessage` text."""
        from deepagents_code.offload_middleware import COMPACTION_FAILURE_PREFIX

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            agent, _ = self._local_agent(
                f"{COMPACTION_FAILURE_PREFIX}: OSError: disk is full."
            )
            app._agent = agent
            app._lc_thread_id = "test-thread"

            result = await app._drive_local_seeded_compaction(  # ty: ignore
                {"configurable": {"thread_id": "test-thread"}}
            )

        assert result is not None
        assert "disk is full" in result


class TestOffloadSessionCostSync:
    """Operation-graph spend reaches `/cost` only via the post-run state read."""

    async def test_summary_spend_is_synced_from_the_committed_state(self) -> None:
        """The summarizer runs outside the agent loop, so nothing else reports it.

        `stream_mode=["updates"]` gives this path no per-kind itemization, so
        `_sync_session_cost_from_state` on the state read back after the run is
        the only route from the graph's `CostTrackingMiddleware` to the session
        total. Dropping or reordering that call silently under-reports every
        server-side `/offload` until the next turn.
        """
        app = DeepAgentsApp()
        after = _state_values([_make_dict_message("hi")], _summary_event(2)) | {
            "_session_cost_usd": 1.25
        }

        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_server_offload_app(app)
            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    AsyncMock(
                        side_effect=[
                            _state_values([_make_dict_message("hi")]),
                            after,
                        ]
                    ),
                ),
                patch.object(
                    app, "_drive_offload_operation_graph", AsyncMock(return_value=None)
                ),
                patch.object(app, "_sync_session_cost_from_state") as sync,
                patch.object(app, "_run_session_start_hook", AsyncMock()),
            ):
                await app._handle_offload()
                await pilot.pause()

        sync.assert_called_once_with(after)
