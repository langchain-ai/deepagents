"""Unit tests for /offload slash command."""

from __future__ import annotations

import asyncio
import os
import stat
import tempfile
from collections.abc import AsyncIterator, Callable  # noqa: TC003
from contextlib import nullcontext
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Coroutine

import pytest
from deepagents.backends.utils import validate_path
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from textual.worker import WorkerCancelled

from deepagents_code import offload
from deepagents_code._cli_context import CLIContextSchema
from deepagents_code._session_stats import format_token_count
from deepagents_code._tracing import RESUME_TRACE_TAG
from deepagents_code.app import DeepAgentsApp
from deepagents_code.command_registry import get_slash_commands
from deepagents_code.hooks.manager import HooksManager
from deepagents_code.offload import (
    _artifacts_root,
    _filesystem_tool_path,
    _offload_fallback_root,
    delete_offloaded_history,
)
from deepagents_code.tui.widgets.chat_input import ChatInput
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


def _compacted_result() -> dict[str, Any]:
    """Build a successful server-owned offload result."""
    return {
        "status": "compacted",
        "messages_offloaded": 6,
        "messages_kept": 4,
        "tokens_before": 1000,
        "tokens_after": 250,
        "archive_path": "/conversation_history/test-thread.md",
        "archive_ephemeral": False,
        "error": None,
    }


def _setup_server_offload_app(app: DeepAgentsApp) -> MagicMock:
    """Configure a `DeepAgentsApp` as a server-backed agent for offload tests.

    The agent is specced as a `RemoteAgent` so `_remote_agent()` narrows to it.
    """
    from deepagents_code.client.remote_client import RemoteAgent

    agent = MagicMock(spec=RemoteAgent)
    agent.aupdate_state = AsyncMock()
    agent.asupports_offload = AsyncMock(return_value=True)
    agent.aoffload = AsyncMock()
    app._agent = agent
    app._backend = None
    app._lc_thread_id = "test-thread"
    app._agent_running = False
    return agent


def _setup_local_offload_app(app: DeepAgentsApp) -> MagicMock:
    """Configure a `DeepAgentsApp` as a local in-process agent for offload tests.

    A plain `MagicMock` agent is *not* a `RemoteAgent`, so `_remote_agent()`
    returns `None` and `_handle_offload` takes the seeded in-process path
    (`_drive_local_seeded_compaction`) instead of the server operation.
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


class TestOffloadCommand:
    """The TUI requests a typed operation and does not manage server state."""

    async def test_no_agent_shows_error(self) -> None:
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._handle_offload()
            assert any(
                "Nothing to offload" in str(w._content) for w in app.query(AppMessage)
            )

    async def test_offload_while_busy_queues_instead_of_overlapping(self) -> None:
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = _setup_server_offload_app(app)
            drive_started = asyncio.Event()
            release_drive = asyncio.Event()
            drive_calls = 0

            async def block_offload(**_kwargs: Any) -> dict[str, Any]:
                nonlocal drive_calls
                drive_calls += 1
                drive_started.set()
                if drive_calls == 1:
                    await release_drive.wait()
                return _compacted_result()

            remote.aoffload = AsyncMock(side_effect=block_offload)
            with (
                patch.object(
                    app, "_sync_session_cost_from_checkpoint", new=AsyncMock()
                ),
                patch.object(app, "_run_session_start_hook", new=AsyncMock()),
            ):
                app.post_message(ChatInput.Submitted("/offload", "command"))
                await asyncio.wait_for(drive_started.wait(), timeout=1)

                assert app._agent_running is True
                app.post_message(ChatInput.Submitted("/offload", "command"))
                await pilot.pause()
                assert drive_calls == 1
                assert len(app._pending_messages) == 1

                release_drive.set()
                worker = app._offload_worker
                assert worker is not None
                await worker.wait()
                await pilot.pause()

            assert drive_calls == 2
            assert app._agent_running is False
            assert app._offload_worker is None
            assert not app._pending_messages

    async def test_server_result_is_rendered_without_reading_checkpoint_state(
        self,
    ) -> None:
        app = DeepAgentsApp()
        result = {
            "status": "compacted",
            "messages_offloaded": 6,
            "messages_kept": 4,
            "tokens_before": 1000,
            "tokens_after": 250,
            "archive_path": "/conversation_history/test-thread.md",
            "archive_ephemeral": False,
            "error": None,
        }
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = _setup_server_offload_app(app)
            remote.aoffload = AsyncMock(return_value=result)
            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new=AsyncMock(side_effect=AssertionError("client state read")),
                ),
                patch.object(
                    app, "_sync_session_cost_from_checkpoint", new=AsyncMock()
                ),
                patch.object(app, "_run_session_start_hook", new=AsyncMock()),
            ):
                await app._handle_offload()
                text = "\n".join(str(w._content) for w in app.query(AppMessage))
                assert "Offloaded 6 older messages" in text
                assert "4 messages kept" in text

            remote.aoffload.assert_awaited_once()
            await_args = remote.aoffload.await_args
            assert await_args is not None
            kwargs = await_args.kwargs
            assert kwargs["config"] == {"configurable": {"thread_id": "test-thread"}}
            assert "messages" not in kwargs["context"]

    async def test_failing_session_start_hook_does_not_erase_the_result(self) -> None:
        """A hook raising after a committed compaction must not hide the outcome.

        The compaction is already durable server-side by this point, so letting
        the hook's exception reach the generic handler would leave the user with
        only "Offload failed" while their conversation really was compacted and
        the status bar kept pre-offload counts.
        """
        app = DeepAgentsApp()
        result = {
            "status": "compacted",
            "messages_offloaded": 6,
            "messages_kept": 4,
            "tokens_before": 1000,
            "tokens_after": 250,
            "archive_path": "/conversation_history/test-thread.md",
            "archive_ephemeral": False,
            "error": None,
        }
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = _setup_server_offload_app(app)
            remote.aoffload = AsyncMock(return_value=result)
            tokens = MagicMock()
            with (
                patch.object(
                    app, "_sync_session_cost_from_checkpoint", new=AsyncMock()
                ),
                patch.object(
                    app,
                    "_run_session_start_hook",
                    new=AsyncMock(side_effect=RuntimeError("hook spawn failed")),
                ),
                patch.object(app, "_on_tokens_update", new=tokens),
            ):
                await app._handle_offload()

            text = "\n".join(str(w._content) for w in app.query(AppMessage))
            assert "Offloaded 6 older messages" in text
            errors = "\n".join(str(w._content) for w in app.query(ErrorMessage))
            assert "SessionStart hook failed" in errors
            assert "Offload failed" not in errors
            tokens.assert_called_once_with(250, approximate=True)

    async def test_session_start_hook_fires_after_a_committed_offload(self) -> None:
        """The `COMPACT` lifecycle event still reaches configured hooks."""
        from deepagents_code.hooks.models.domain import SessionStartCause

        app = DeepAgentsApp()
        result = {
            "status": "compacted",
            "messages_offloaded": 2,
            "messages_kept": 1,
            "tokens_before": 100,
            "tokens_after": 50,
            "archive_path": "/conversation_history/test-thread.md",
            "archive_ephemeral": False,
            "error": None,
        }
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = _setup_server_offload_app(app)
            remote.aoffload = AsyncMock(return_value=result)
            hook = AsyncMock()
            with (
                patch.object(
                    app, "_sync_session_cost_from_checkpoint", new=AsyncMock()
                ),
                patch.object(app, "_run_session_start_hook", new=hook),
            ):
                await app._handle_offload()

            hook.assert_awaited_once_with(SessionStartCause.COMPACT)

    async def test_probe_failure_falls_back_instead_of_refusing(self) -> None:
        """An unreachable capability probe must not disable a working /offload.

        A timeout or a gateway 401/403/405 says nothing about whether offload
        works, and the seeded path drives the agent's own tool against any
        server -- so degrade to it, visibly, rather than refuse outright.
        """
        app = DeepAgentsApp()
        before = _state_values([_make_dict_message("hi")])
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = _setup_server_offload_app(app)
            remote.asupports_offload = AsyncMock(
                side_effect=RuntimeError("gateway timeout")
            )
            with (
                patch.object(
                    app, "_get_thread_state_values", new=AsyncMock(return_value=before)
                ),
                patch.object(
                    app,
                    "_drive_local_seeded_compaction",
                    new=AsyncMock(return_value="Compaction failed: nope"),
                ) as seeded,
                patch.object(
                    app, "_sync_session_cost_from_checkpoint", new=AsyncMock()
                ),
            ):
                await app._handle_offload()

            seeded.assert_awaited_once()
            notices = "\n".join(str(w._content) for w in app.query(AppMessage))
            assert "compatibility path" in notices

    async def test_server_failure_is_rendered_from_typed_result(self) -> None:
        app = DeepAgentsApp()
        result = {
            "status": "failed",
            "messages_offloaded": 0,
            "messages_kept": 4,
            "tokens_before": 100,
            "tokens_after": 100,
            "archive_path": None,
            "archive_ephemeral": False,
            "error": "summary unavailable",
        }
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = _setup_server_offload_app(app)
            remote.aoffload = AsyncMock(return_value=result)
            with (
                patch.object(
                    app, "_sync_session_cost_from_checkpoint", new=AsyncMock()
                ),
            ):
                await app._handle_offload()
                assert any(
                    "summary unavailable" in str(w._content)
                    for w in app.query(ErrorMessage)
                )

    async def test_custom_graph_uses_seeded_fallback(self) -> None:
        app = DeepAgentsApp()
        before = _state_values([_make_dict_message("hi")])
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = _setup_server_offload_app(app)
            remote.asupports_offload = AsyncMock(return_value=False)
            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new=AsyncMock(return_value=before),
                ),
                patch.object(
                    app,
                    "_drive_local_seeded_compaction",
                    new=AsyncMock(return_value="fallback stopped"),
                ) as seeded,
            ):
                await app._handle_offload()
                assert any(
                    "fallback stopped" in str(w._content)
                    for w in app.query(ErrorMessage)
                )

            seeded.assert_awaited_once()
            remote.aoffload.assert_not_awaited()


class TestServerOffloadReporting:
    """The server path must describe its estimates on the seeded path's terms."""

    @staticmethod
    def _result(**overrides: object) -> dict[str, object]:
        """Build a `compacted` server result."""
        return {
            "status": "compacted",
            "messages_offloaded": 6,
            "messages_kept": 4,
            "tokens_before": 1000,
            "tokens_after": 250,
            "archive_path": "/conversation_history/test-thread.md",
            "archive_ephemeral": False,
            "error": None,
        } | overrides

    async def _render(self, app: DeepAgentsApp, result: dict[str, object]) -> str:
        """Drive `/offload` against a server result and return the rendered text."""
        remote = _setup_server_offload_app(app)
        remote.aoffload = AsyncMock(return_value=result)
        with (
            patch.object(app, "_sync_session_cost_from_checkpoint", new=AsyncMock()),
            patch.object(app, "_run_session_start_hook", new=AsyncMock()),
        ):
            await app._handle_offload()
        return "\n".join(str(w._content) for w in app.query(AppMessage)) + "\n".join(
            str(w._content) for w in app.query(ErrorMessage)
        )

    async def test_estimates_are_labelled_conversation_and_marked(self) -> None:
        """Server figures are conversation-scale estimates, not context totals.

        "Conversation" excludes the system/tool overhead that "Context"
        includes, so labelling an estimate "Context" invites the user to compare
        two percentages that are not comparable across offloads.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._context_tokens = 0
            app._tokens_approximate = True
            text = await self._render(app, self._result())

        assert "Conversation: ~1.0K → ~250 tokens (75% decrease)" in text
        assert "Context:" not in text

    async def test_a_larger_summary_never_reports_a_negative_decrease(self) -> None:
        """A summary can exceed what it replaced; that is an increase, not -14%."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._context_tokens = 0
            app._tokens_approximate = True
            text = await self._render(
                app, self._result(tokens_before=100, tokens_after=114)
            )

        assert "(increase)" in text
        assert "-" not in text.split("tokens")[1].split(",")[0]
        assert "summary was larger than the messages it replaced" in text

    async def test_a_real_provider_total_promotes_the_report_to_context(self) -> None:
        """A cached count from a real turn is the provider's own total.

        The delta is subtracted from that total rather than rebuilt as
        `overhead + after`, so both figures stay on the provider's scale.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._context_tokens = 5000
            app._tokens_approximate = False
            text = await self._render(app, self._result())

        # 5000 - (1000 - 250) = 4250; `before` is exact, only `after` estimated.
        assert "Context: 5.0K → ~4.2K tokens (15% decrease)" in text

    async def test_ephemeral_storage_is_disclosed(self) -> None:
        """History in a temp fallback must not be presented as durable."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._context_tokens = 0
            app._tokens_approximate = True
            text = await self._render(app, self._result(archive_ephemeral=True))

        assert "may not survive a restart" in text

    async def test_durable_storage_adds_no_caveat(self) -> None:
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._context_tokens = 0
            app._tokens_approximate = True
            text = await self._render(app, self._result(archive_ephemeral=False))

        assert "may not survive" not in text

    async def test_a_failed_archive_write_reports_unrecoverable_messages(self) -> None:
        """Context was freed but the history is gone; both facts must be said.

        This is data-loss messaging: reporting plain success here would tell the
        user their conversation is archived when it is not.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._context_tokens = 0
            app._tokens_approximate = True
            text = await self._render(app, self._result(archive_path=None))
            errors = [str(w._content) for w in app.query(ErrorMessage)]

        assert "not recoverable" in text
        # An error, not a success message: the offload did not fully succeed.
        assert errors
        assert "not recoverable" in "\n".join(errors)

    async def test_singular_message_labels(self) -> None:
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._context_tokens = 0
            app._tokens_approximate = True
            text = await self._render(
                app, self._result(messages_offloaded=1, messages_kept=1)
            )

        assert "1 older message," in text
        assert "1 message kept" in text


class TestLocalOffloadReporting:
    """Preserve current reporting behavior on the seeded fallback path."""

    @staticmethod
    async def _run(
        before: dict[str, Any], after: dict[str, Any]
    ) -> tuple[list[str], int, bool]:
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_local_offload_app(app)
            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new=AsyncMock(side_effect=[before, after]),
                ),
                patch.object(
                    app,
                    "_drive_local_seeded_compaction",
                    new=AsyncMock(return_value=None),
                ),
            ):
                await app._handle_offload()
                await pilot.pause()
            contents = [str(widget._content) for widget in app.query(AppMessage)]
            return contents, app._context_tokens, app._tokens_approximate

    @staticmethod
    async def _run_errors(
        before: dict[str, Any], after: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """Drive the seeded path and return its app and error message text."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_local_offload_app(app)
            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new=AsyncMock(side_effect=[before, after]),
                ),
                patch.object(
                    app,
                    "_drive_local_seeded_compaction",
                    new=AsyncMock(return_value=None),
                ),
            ):
                await app._handle_offload()
                await pilot.pause()
            return (
                [str(w._content) for w in app.query(AppMessage)],
                [str(w._content) for w in app.query(ErrorMessage)],
            )

    async def test_ephemeral_storage_is_disclosed(self) -> None:
        """A temp-fallback archive must not be presented as durable.

        The write succeeded and context was freed, so this is a success message
        -- but the history may not outlive a restart, which the user has to know
        before relying on it.
        """
        before_messages = _make_dict_messages(4)
        with patch(
            "deepagents_code.offload.offload_storage_is_ephemeral",
            return_value=True,
        ):
            contents, _ = await self._run_errors(
                _state_values(before_messages),
                _state_values(
                    [*before_messages, *_make_dict_messages(2)], _summary_event(2)
                ),
            )

        assert "may not survive a restart" in "\n".join(contents)

    async def test_durable_storage_adds_no_caveat(self) -> None:
        before_messages = _make_dict_messages(4)
        with patch(
            "deepagents_code.offload.offload_storage_is_ephemeral",
            return_value=False,
        ):
            contents, _ = await self._run_errors(
                _state_values(before_messages),
                _state_values(
                    [*before_messages, *_make_dict_messages(2)], _summary_event(2)
                ),
            )

        assert "may not survive" not in "\n".join(contents)

    async def test_a_failed_archive_write_reports_unrecoverable_messages(self) -> None:
        """Context was freed but the archive write failed; say both.

        Data-loss messaging: an event with no `file_path` means the offloaded
        messages are gone, so reporting plain success would tell the user their
        conversation is archived when it is not.
        """
        before_messages = _make_dict_messages(4)
        _contents, errors = await self._run_errors(
            _state_values(before_messages),
            _state_values(
                [*before_messages, *_make_dict_messages(2)],
                _summary_event(2, file_path=None),
            ),
        )

        assert errors
        assert "not recoverable" in "\n".join(errors)

    async def test_turn_counts_ignore_tools_and_internal_messages(self) -> None:
        before_messages = [
            {"type": "human", "content": "Old prompt", "id": "old-human"},
            {"type": "ai", "content": "", "id": "old-tool-call"},
            {
                "type": "tool",
                "content": "Old result",
                "id": "old-tool",
                "tool_call_id": "old-call",
            },
            {"type": "ai", "content": "Old answer", "id": "old-ai"},
            {"role": "user", "content": "Kept prompt", "id": "kept-human"},
            {"role": "assistant", "content": "", "id": "kept-tool-call"},
            {
                "type": "tool",
                "content": "Kept result",
                "id": "kept-tool",
                "tool_call_id": "kept-call",
            },
            {"role": "assistant", "content": "Kept answer", "id": "kept-ai"},
            {
                "type": "human",
                "content": "Internal state",
                "id": "internal-human",
                "additional_kwargs": {"lc_source": "goal_state"},
            },
        ]
        contents, _, _ = await self._run(
            _state_values(before_messages),
            _state_values(
                [*before_messages, *_make_dict_messages(2)], _summary_event(4)
            ),
        )

        assert any(
            "Offloaded 4 older messages (1 conversation turn)" in content
            for content in contents
        )
        assert any(
            "5 messages (1 conversation turn) kept" in content for content in contents
        )

    async def test_offloaded_turns_ignore_internal_messages(self) -> None:
        before_messages = [
            {"type": "human", "content": "Old prompt", "id": "old-human"},
            {
                "type": "human",
                "content": "Internal state",
                "id": "offloaded-internal",
                "additional_kwargs": {"lc_source": "goal_state"},
            },
            {
                "type": "human",
                "content": "[SYSTEM] Task interrupted by user.",
                "id": "offloaded-system-prefix",
            },
            {"type": "ai", "content": "Old answer", "id": "old-ai"},
            {"type": "human", "content": "Kept prompt", "id": "kept-human"},
            {"type": "ai", "content": "Kept answer", "id": "kept-ai"},
        ]
        contents, _, _ = await self._run(
            _state_values(before_messages),
            _state_values(
                [*before_messages, *_make_dict_messages(2)], _summary_event(4)
            ),
        )

        assert any(
            "Offloaded 4 older messages (1 conversation turn)" in content
            for content in contents
        )

    async def test_singular_labels_and_zero_offloaded_turns(self) -> None:
        before_messages = [
            {"type": "ai", "content": "", "id": "old-tool-call"},
            {
                "type": "tool",
                "content": "Old result",
                "id": "old-tool",
                "tool_call_id": "old-call",
            },
            {"type": "ai", "content": "Old answer", "id": "old-ai"},
            {"type": "human", "content": "Kept prompt", "id": "kept-human"},
        ]
        contents, _, _ = await self._run(
            _state_values(before_messages),
            _state_values(
                [*before_messages, *_make_dict_messages(2)], _summary_event(3)
            ),
        )

        assert any(
            "Offloaded 3 older messages (0 conversation turns)" in content
            for content in contents
        )
        assert any(
            "1 message (1 conversation turn) kept" in content for content in contents
        )

    async def test_zero_kept_turns_when_cutoff_reaches_last_human(self) -> None:
        before_messages = [
            {"type": "human", "content": "Old prompt", "id": "old-human"},
            {"type": "ai", "content": "Old answer", "id": "old-ai"},
            {"type": "ai", "content": "Trailing", "id": "trailing-ai"},
        ]
        contents, _, _ = await self._run(
            _state_values(before_messages),
            _state_values(
                [*before_messages, *_make_dict_messages(2)], _summary_event(2)
            ),
        )

        assert any(
            "1 message (0 conversation turns) kept" in content for content in contents
        )

    async def test_preserves_fixed_overhead_in_context_report(self) -> None:
        from langchain_core.messages.utils import count_tokens_approximately

        from deepagents_code.app import _effective_conversation

        before_messages = _make_dict_messages(10)
        after_event = _summary_event(4)
        conversation_before = count_tokens_approximately(before_messages)
        conversation_after = count_tokens_approximately(
            _effective_conversation(before_messages, after_event)
        )
        fixed_tokens = 50_000
        reported_before = conversation_before + fixed_tokens
        expected_after = conversation_after + fixed_tokens
        before = _state_values(before_messages)
        before["_context_tokens"] = reported_before
        contents, context_tokens, approximate = await self._run(
            before,
            _state_values([*before_messages, *_make_dict_messages(2)], after_event),
        )

        expected_report = (
            f"Context: {format_token_count(reported_before)} → "
            f"~{format_token_count(expected_after)} tokens"
        )
        assert any(expected_report in content for content in contents)
        assert context_tokens == expected_after
        assert approximate is True

    async def test_report_stays_on_provider_scale_when_total_is_low(self) -> None:
        from langchain_core.messages.utils import count_tokens_approximately

        from deepagents_code.app import _effective_conversation

        before_messages = _make_dict_messages(10)
        after_event = _summary_event(4)
        conversation_before = count_tokens_approximately(before_messages)
        conversation_after = count_tokens_approximately(
            _effective_conversation(before_messages, after_event)
        )
        reported_before = conversation_before // 2
        expected_after = reported_before - (conversation_before - conversation_after)
        assert expected_after > 0
        before = _state_values(before_messages)
        before["_context_tokens"] = reported_before
        contents, context_tokens, _ = await self._run(
            before,
            _state_values([*before_messages, *_make_dict_messages(2)], after_event),
        )

        expected_report = (
            f"Context: {format_token_count(reported_before)} → "
            f"~{format_token_count(expected_after)} tokens"
        )
        assert any(expected_report in content for content in contents)
        assert not any("(100% decrease)" in content for content in contents)
        assert context_tokens == expected_after

    async def test_oversized_summary_is_reported_as_an_increase(self) -> None:
        before_messages = _make_dict_messages(10)
        after_event = _summary_event(4)
        after_event["summary_message"]["content"] = "verbose summary " * 500
        contents, _, _ = await self._run(
            _state_values(before_messages),
            _state_values([*before_messages, *_make_dict_messages(2)], after_event),
        )

        assert any("Offloaded " in content for content in contents)
        assert any("(increase)" in content for content in contents)
        assert not any(
            "freeing up context window space" in content for content in contents
        )
        assert any("context increased" in content for content in contents)


class TestOffloadInterrupt:
    """Test that Escape can cancel `/offload` through the real App dispatch."""

    async def test_command_reserves_turn_before_worker_starts(self) -> None:
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            worker = MagicMock()
            scheduled: list[Coroutine[Any, Any, None]] = []

            def defer_worker(
                work: Coroutine[Any, Any, None], **_kwargs: object
            ) -> MagicMock:
                scheduled.append(work)
                return worker

            with patch.object(app, "run_worker", side_effect=defer_worker):
                await app._handle_command("/offload")

            assert app._agent_running is True
            assert app._offload_worker is worker
            assert app._offload_task_started is False
            assert len(scheduled) == 1

            coroutine = scheduled[0]
            try:
                await app._submit_input("hello", "normal")
                assert len(app._pending_messages) == 1
                app._cancel_worker(worker)
            finally:
                coroutine.close()

            worker.cancel.assert_called_once_with()
            assert app._agent_running is False
            assert app._offload_worker is None
            assert not app._pending_messages

    async def test_escape_cancels_server_owned_offload(self) -> None:
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = _setup_server_offload_app(app)
            drive_started = asyncio.Event()
            drive_cancelled = asyncio.Event()

            async def block_offload(**_kwargs: Any) -> dict[str, Any]:
                drive_started.set()
                try:
                    await asyncio.Future()
                finally:
                    drive_cancelled.set()
                return _compacted_result()

            remote.aoffload = AsyncMock(side_effect=block_offload)
            app.post_message(ChatInput.Submitted("/offload", "command"))
            await asyncio.wait_for(drive_started.wait(), timeout=1)

            worker = app._offload_worker
            assert worker is not None
            assert app._agent_running is True

            await pilot.press("escape")
            await asyncio.wait_for(drive_cancelled.wait(), timeout=1)
            with pytest.raises(WorkerCancelled):
                await worker.wait()

            assert worker.is_cancelled
            assert app._agent_running is False
            assert app._agent_quiescent.is_set()
            assert app._loading_widget is None

    async def test_escape_cancels_local_fallback(self) -> None:
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_local_offload_app(app)
            before = _state_values(_make_dict_messages(6))
            reconciled = _state_values(_make_dict_messages(6))
            drive_started = asyncio.Event()
            drive_cancelled = asyncio.Event()

            async def block_drive(_config: object, _seed_id: object = None) -> None:
                drive_started.set()
                try:
                    await asyncio.Future()
                finally:
                    drive_cancelled.set()

            with (
                patch.object(
                    app,
                    "_get_thread_state_values",
                    new=AsyncMock(side_effect=[before, reconciled]),
                ),
                patch.object(
                    app,
                    "_drive_local_seeded_compaction",
                    new=AsyncMock(side_effect=block_drive),
                ),
                patch.object(
                    app,
                    "_remove_unanswered_offload_seed",
                    new=AsyncMock(return_value=True),
                ) as cleanup,
            ):
                app.post_message(ChatInput.Submitted("/offload", "command"))
                await asyncio.wait_for(drive_started.wait(), timeout=1)

                worker = app._offload_worker
                assert worker is not None
                await pilot.press("escape")
                await asyncio.wait_for(drive_cancelled.wait(), timeout=1)
                with pytest.raises(WorkerCancelled):
                    await worker.wait()

            cleanup.assert_awaited_once()
            assert app._agent_running is False
            assert app._loading_widget is None

    async def test_offload_blocks_queued_prompt_until_done(self) -> None:
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = _setup_server_offload_app(app)
            drive_started = asyncio.Event()
            release_drive = asyncio.Event()

            async def block_offload(**_kwargs: Any) -> dict[str, Any]:
                drive_started.set()
                await release_drive.wait()
                return _compacted_result()

            remote.aoffload = AsyncMock(side_effect=block_offload)
            dispatch = AsyncMock()
            with (
                patch.object(
                    app, "_sync_session_cost_from_checkpoint", new=AsyncMock()
                ),
                patch.object(app, "_run_session_start_hook", new=AsyncMock()),
            ):
                app.post_message(ChatInput.Submitted("/offload", "command"))
                await asyncio.wait_for(drive_started.wait(), timeout=1)

                with patch.object(app, "_dispatch_queued_message", new=dispatch):
                    app.post_message(ChatInput.Submitted("hello", "prompt"))
                    await pilot.pause()
                    assert app._agent_running is True
                    assert len(app._pending_messages) == 1
                    dispatch.assert_not_awaited()

                    release_drive.set()
                    worker = app._offload_worker
                    assert worker is not None
                    await worker.wait()
                    await pilot.pause()

            dispatch.assert_awaited_once()
            assert app._agent_running is False
            assert app._offload_worker is None
            assert not app._pending_messages

    async def test_server_failure_releases_busy_state_and_spinner(self) -> None:
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = _setup_server_offload_app(app)
            remote.aoffload = AsyncMock(side_effect=RuntimeError("server unavailable"))

            with patch.object(app, "_set_spinner", new_callable=AsyncMock) as spinner:
                await app._handle_offload()

            spinner.assert_any_await("Offloading")
            spinner.assert_awaited_with(None)
            assert app._agent_running is False
            assert app._agent_quiescent.is_set()


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


class TestDriveLegacySeededCompaction:
    """Unit-test the seeded in-process `compact_conversation` trigger.

    This driver serves local `Pregel` agents, which have no server operation
    graph; server-backed agents use the dedicated HTTP operation instead.
    """

    @staticmethod
    def _fake_remote_agent(
        tool_content: str,
    ) -> tuple[Any, list[Any], list[object], list[Any]]:
        """Build a fake `RemoteAgent` that interrupts then returns a ToolMessage.

        First `astream(None)` surfaces a HITL approval interrupt; the resume
        stream (`Command(resume=...)`) yields a `ToolMessage` with the supplied
        content so callers can exercise both the success and failure branches.

        Args:
            tool_content: Body of the `ToolMessage` the resume stream yields.

        Returns:
            The agent plus one list per recorded `astream` keyword -- inputs,
                contexts, and configs -- each appended to in call order.
        """
        from langchain_core.messages import ToolMessage

        from deepagents_code.client.remote_client import RemoteAgent

        astream_inputs: list[Any] = []
        astream_contexts: list[object] = []
        astream_configs: list[Any] = []

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
            astream_configs.append(kwargs.get("config"))
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
        return agent, astream_inputs, astream_contexts, astream_configs

    async def test_seeds_tool_call_and_resumes_interrupt(self) -> None:
        """Seeds a forced `compact_conversation` call and approves the interrupt."""
        from langgraph.types import Command

        from deepagents_code.config import settings

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            agent, astream_inputs, astream_contexts, astream_configs = (
                self._fake_remote_agent(
                    "Conversation compacted. Summarized 2 messages into a "
                    "concise summary."
                )
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

            initial_config, resume_config = astream_configs
            assert RESUME_TRACE_TAG not in initial_config.get("tags", [])
            assert RESUME_TRACE_TAG in resume_config["tags"]
            assert initial_config["configurable"] == resume_config["configurable"]

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
            agent, _inputs, _contexts, _configs = self._fake_remote_agent(
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
            agent, _inputs, contexts, _configs = self._fake_remote_agent(
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


class TestOffloadOperation:
    """The server service owns checkpoint state and compaction policy."""

    @staticmethod
    def _runtime() -> Runtime[CLIContextSchema]:
        return Runtime(context=CLIContextSchema())

    @staticmethod
    def _middleware(
        *, hook_update: dict[str, object] | None = None
    ) -> tuple[Any, MagicMock, MagicMock]:
        from deepagents_code.offload_middleware import OffloadOperation

        compaction = MagicMock()
        compaction.arun_forced_compaction_update = AsyncMock()
        compaction._summarization._apply_event_to_messages.side_effect = (
            lambda messages, _event: messages
        )
        hooks = MagicMock()
        # Default to the shape `ServerHooksMiddleware._after_model` really
        # returns: every one of its return paths carries the pre-tool channel,
        # including the "no hook events enabled" path. The operation fails closed
        # when the channel is absent, so a mock returning a bare `{}` would
        # assert a contract the middleware never produces.
        from deepagents_code.hooks.server_middleware import _PRE_TOOL_STATE_KEY

        hooks.aafter_model = AsyncMock(
            return_value=hook_update
            if hook_update is not None
            else {_PRE_TOOL_STATE_KEY: {}}
        )
        return OffloadOperation(compaction, hooks), compaction, hooks

    async def test_compacts_checkpoint_state_without_message_input(self) -> None:
        event = _summary_event(2)
        middleware, compaction, _hooks = self._middleware()
        compaction.arun_forced_compaction_update = AsyncMock(
            return_value={
                "_summarization_event": event,
                "_summarization_session_id": "archive-1",
            }
        )
        state = {
            "messages": _make_dict_messages(4),
        }

        execution = await middleware.execute(state, self._runtime())

        compaction.arun_forced_compaction_update.assert_awaited_once()
        await_args = compaction.arun_forced_compaction_update.await_args
        assert await_args is not None
        state_arg = await_args.args[0]
        assert state_arg is state
        assert "messages" not in execution.update
        assert execution.update["_summarization_session_id"] == "archive-1"
        assert execution.result["status"] == "compacted"
        assert execution.result["messages_offloaded"] == 2

    async def test_reoffload_reports_the_absolute_cutoff_delta(self) -> None:
        """Counts are deltas against the prior event, not absolute cutoffs.

        With a prior cutoff of 0 the two are indistinguishable, so this drives a
        chained offload: 6 messages, prior cutoff 2, new cutoff 5 must report 3
        offloaded and 1 kept. Reporting `new_cutoff` directly would say 5.
        """
        middleware, compaction, _hooks = self._middleware()
        compaction.arun_forced_compaction_update = AsyncMock(
            return_value={
                "_summarization_event": _summary_event(5),
                "_summarization_session_id": "archive-1",
            }
        )
        state = {
            "messages": _make_dict_messages(6),
            "_summarization_event": _summary_event(2),
        }

        execution = await middleware.execute(state, self._runtime())

        assert execution.result["status"] == "compacted"
        assert execution.result["messages_offloaded"] == 3
        assert execution.result["messages_kept"] == 1

    async def test_non_compacted_counts_never_go_negative(self) -> None:
        """A stale cutoff beyond the message count must not report a negative."""
        middleware, compaction, _hooks = self._middleware()
        compaction.arun_forced_compaction_update = AsyncMock(return_value=None)
        state = {
            "messages": _make_dict_messages(2),
            "_summarization_event": _summary_event(9),
        }

        execution = await middleware.execute(state, self._runtime())

        assert execution.result["status"] == "noop"
        assert execution.result["messages_kept"] == 0
        assert execution.update == {}

    async def test_hook_denial_skips_compaction(self) -> None:
        """A `PreToolUse` denial must stop the compaction.

        Keys the outcome on the id the node really generated and asserts the
        dispatched call's `name`/`args`, so a re-spelled tool name or a dropped
        `force` flag -- either of which silently exempts `/offload` from the
        user's hook -- fails here instead of reading as "no outcome".
        """
        middleware, compaction, hooks = self._middleware()
        hooks.aafter_model = AsyncMock(side_effect=_deny_dispatched_call("policy"))

        execution = await middleware.execute(
            {"messages": _make_dict_messages(4)}, self._runtime()
        )

        assert execution.result["status"] == "denied"
        assert execution.result["error"] == "policy"
        compaction.arun_forced_compaction_update.assert_not_awaited()

    async def test_hook_denial_without_a_reason_still_stops_compaction(self) -> None:
        """A denial carrying no reason must not read as an allow."""
        middleware, compaction, hooks = self._middleware()
        hooks.aafter_model = AsyncMock(side_effect=_deny_dispatched_call(None))

        execution = await middleware.execute(
            {"messages": _make_dict_messages(4)}, self._runtime()
        )

        assert execution.result["status"] == "denied"
        compaction.arun_forced_compaction_update.assert_not_awaited()

    async def test_a_missing_hook_channel_refuses_instead_of_allowing(self) -> None:
        """A hook decision that cannot be read must not be treated as an allow.

        Every `_after_model` return path carries the pre-tool channel, so its
        absence means the channel, the id derivation, or the outcome shape
        drifted. Reading that through a `.get(..., {})` chain would turn a user's
        denial into "no outcome" and compact straight through it, with no log.
        """
        middleware, compaction, hooks = self._middleware(hook_update={})

        execution = await middleware.execute(
            {"messages": _make_dict_messages(4)}, self._runtime()
        )

        assert execution.result["status"] == "failed"
        assert "hook decision" in (execution.result["error"] or "")
        compaction.arun_forced_compaction_update.assert_not_awaited()
        assert "messages" not in execution.update
        hooks.aafter_model.assert_awaited_once()

    async def test_failure_returns_result_without_rewriting_messages(self) -> None:
        middleware, compaction, _hooks = self._middleware()
        compaction.arun_forced_compaction_update = AsyncMock(
            side_effect=OSError("archive unavailable")
        )

        execution = await middleware.execute(
            {"messages": _make_dict_messages(4)}, self._runtime()
        )

        assert execution.result["status"] == "failed"
        assert "archive unavailable" in (execution.result["error"] or "")
        assert "messages" not in execution.update


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


class TestSeededDriverAgainstALocalAgent:
    """The seeded driver's *only* production shape is a local `Pregel` agent.

    Every other test in `TestDriveLocalSeededCompaction` builds its agent with
    `MagicMock(spec=RemoteAgent)`, but `_handle_offload` now routes remote
    agents to the server operation — so the driver is exercised exclusively in
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
