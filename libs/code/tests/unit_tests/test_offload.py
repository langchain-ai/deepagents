"""Unit tests for /offload slash command."""

from __future__ import annotations

import asyncio
import logging
import os
import stat
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

import pytest
from deepagents.backends.utils import validate_path
from langgraph.runtime import Runtime
from textual.worker import WorkerCancelled

from deepagents_code import offload
from deepagents_code._cli_context import CLIContextSchema
from deepagents_code._session_stats import format_token_count
from deepagents_code.app import DeepAgentsApp
from deepagents_code.command_registry import get_slash_commands
from deepagents_code.configuration.types import TomlSnapshot
from deepagents_code.hooks.manager import HooksManager
from deepagents_code.offload import (
    _artifacts_root,
    _filesystem_tool_path,
    _offload_fallback_root,
    delete_offloaded_history,
    sweep_offloaded_history,
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
    agent.aoffload = AsyncMock()
    app._agent = agent
    app._backend = None
    app._lc_thread_id = "test-thread"
    app._agent_running = False
    return agent


def _setup_local_offload_app(app: DeepAgentsApp) -> MagicMock:
    """Configure a `DeepAgentsApp` with a local in-process agent."""
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

    async def test_context_carries_the_session_approval_mode(self) -> None:
        """Hooks must see the session's real mode during `/offload`.

        The server defaults a missing `approval_mode` to `manual`, so omitting
        it would show a configured `PreCompact`/`PreToolUse` hook Manual even in
        YOLO -- a different mode than the same hook sees on every interactive
        turn.
        """
        from deepagents_code.approval_mode import ApprovalMode

        app = DeepAgentsApp()
        result = {
            "status": "noop",
            "messages_offloaded": 0,
            "messages_kept": 1,
            "tokens_before": 10,
            "tokens_after": 10,
            "archive_path": None,
            "archive_ephemeral": False,
            "error": None,
        }
        async with app.run_test() as pilot:
            await pilot.pause()
            remote = _setup_server_offload_app(app)
            remote.aoffload = AsyncMock(return_value=result)
            app._approval_mode = ApprovalMode.YOLO
            app._auto_approve = True
            with patch.object(
                app, "_sync_session_cost_from_checkpoint", new=AsyncMock()
            ):
                await app._handle_offload()

            await_args = remote.aoffload.await_args
            assert await_args is not None
            context = await_args.kwargs["context"]
            assert context["approval_mode"] == "yolo"
            assert context["auto_approve"] is True

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

    async def test_failing_report_does_not_report_a_committed_offload_as_failed(
        self,
    ) -> None:
        """A rendering failure after the commit must not say "Offload failed".

        Everything between `aoffload` returning and the SessionStart hook is
        local reporting over a conversation the server has already compacted.
        Routing a failure there into the generic handler would tell the user to
        offload again, compacting an already-compacted conversation.
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
            with (
                patch.object(
                    app, "_sync_session_cost_from_checkpoint", new=AsyncMock()
                ),
                patch.object(
                    app,
                    "_on_tokens_update",
                    new=MagicMock(side_effect=RuntimeError("status bar exploded")),
                ),
            ):
                await app._handle_offload()

            errors = "\n".join(str(w._content) for w in app.query(ErrorMessage))
            assert "could not be displayed" in errors
            assert "Offload failed" not in errors

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

    async def test_local_agent_shows_unsupported_message(self) -> None:
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            _setup_local_offload_app(app)
            await app._handle_offload()
            assert any(
                "not supported for local agents" in str(widget._content)
                for widget in app.query(AppMessage)
            )


def test_a_reasonless_refusal_cannot_be_built() -> None:
    """A `denied`/`failed` result with no reason must not be constructible.

    The wire shape is one flat object, so `error` is `str | None` on every
    status and the checker cannot make "a refusal has a reason" a compile-time
    fact. A reasonless refusal reaches the user as the client's generic "the
    server rejected the operation", which says nothing, so the single
    construction point enforces it.
    """
    from deepagents_code.offload_middleware import unchanged_offload_result

    for status in ("denied", "failed"):
        with pytest.raises(ValueError, match="must carry a reason"):
            unchanged_offload_result(status, messages=1, tokens=2)  # ty: ignore[invalid-argument-type]

    # Unchanged, non-refusal outcomes legitimately carry no reason.
    assert unchanged_offload_result("empty", messages=0, tokens=0)["error"] is None


class TestServerOffloadReporting:
    """The server path reports its estimates with explicit metric labels."""

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


class TestSweepOffloadedHistory:
    """Cover startup cleanup of expired conversation-history archives."""

    @staticmethod
    def _setup(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str = ""
    ) -> Path:
        root = tmp_path / "offload"
        archive_dir = root / offload.CONVERSATION_HISTORY_DIRNAME
        archive_dir.mkdir(parents=True)
        config_path = tmp_path / "config.toml"
        if config:
            config_path.write_text(config)
        monkeypatch.setattr(offload, "_offload_fallback_root", lambda: root)
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH", config_path
        )
        # Isolate from the developer's shell: resolution must see only the
        # test's config.toml, never a real managed snapshot or exported env var.
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_managed_config_toml",
            lambda **_: {},
        )
        monkeypatch.delenv("DEEPAGENTS_CODE_HISTORY_RETENTION_DAYS", raising=False)
        return archive_dir

    def test_deletes_old_file_and_keeps_fresh_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only archives older than the configured retention are removed."""
        archive_dir = self._setup(tmp_path, monkeypatch)
        old = archive_dir / "old.md"
        fresh = archive_dir / "fresh.md"
        old.write_text("old")
        fresh.write_text("fresh")
        old_time = time.time() - 31 * 86_400
        os.utime(old, (old_time, old_time))

        assert sweep_offloaded_history() == 1
        assert not old.exists()
        assert fresh.exists()

    def test_nonzero_retention_override_is_applied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid config value replaces the default retention window."""
        archive_dir = self._setup(
            tmp_path, monkeypatch, "[history]\nretention_days = 1\n"
        )
        archive = archive_dir / "old.md"
        archive.write_text("old")
        old_time = time.time() - 2 * 86_400
        os.utime(archive, (old_time, old_time))

        assert sweep_offloaded_history() == 1
        assert not archive.exists()

    def test_ignores_non_markdown_and_non_regular_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sweep ignores non-markdown files and markdown directories."""
        archive_dir = self._setup(tmp_path, monkeypatch)
        text_file = archive_dir / "old.txt"
        markdown_dir = archive_dir / "old.md"
        text_file.write_text("keep")
        markdown_dir.mkdir()
        old_time = time.time() - 31 * 86_400
        os.utime(text_file, (old_time, old_time))
        os.utime(markdown_dir, (old_time, old_time))

        assert sweep_offloaded_history() == 0
        assert text_file.exists()
        assert markdown_dir.exists()

    def test_missing_archive_directory_is_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing archive directory does not raise."""
        root = tmp_path / "offload"
        monkeypatch.setattr(offload, "_offload_fallback_root", lambda: root)
        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_CONFIG_PATH",
            tmp_path / "missing.toml",
        )
        monkeypatch.setattr(
            "deepagents_code.config_manifest.load_managed_config_toml",
            lambda **_: {},
        )
        monkeypatch.delenv("DEEPAGENTS_CODE_HISTORY_RETENTION_DAYS", raising=False)

        assert sweep_offloaded_history() == 0

    def test_zero_retention_disables_sweep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A zero-day retention setting avoids resolving archive storage."""
        self._setup(tmp_path, monkeypatch, "[history]\nretention_days = 0\n")
        resolver = MagicMock(side_effect=AssertionError("storage should not resolve"))
        monkeypatch.setattr(offload, "_offload_fallback_root", resolver)

        assert sweep_offloaded_history() == 0
        resolver.assert_not_called()

    def test_invalid_retention_uses_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Invalid retention config warns and falls back to 30 days."""
        archive_dir = self._setup(
            tmp_path, monkeypatch, '[history]\nretention_days = "forever"\n'
        )
        archive = archive_dir / "old.md"
        archive.write_text("old")
        old_time = time.time() - 31 * 86_400
        os.utime(archive, (old_time, old_time))

        assert sweep_offloaded_history() == 1
        assert "retention_days" in caplog.text

    def test_env_var_overrides_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The env var wins over `[history].retention_days` in config.toml."""
        archive_dir = self._setup(
            tmp_path, monkeypatch, "[history]\nretention_days = 30\n"
        )
        monkeypatch.setenv("DEEPAGENTS_CODE_HISTORY_RETENTION_DAYS", "1")
        archive = archive_dir / "old.md"
        archive.write_text("old")
        old_time = time.time() - 2 * 86_400
        os.utime(archive, (old_time, old_time))

        assert sweep_offloaded_history() == 1
        assert not archive.exists()

    def test_managed_config_takes_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A managed `retention_days` outranks env var and config.toml."""
        archive_dir = self._setup(
            tmp_path, monkeypatch, "[history]\nretention_days = 1\n"
        )
        monkeypatch.setenv("DEEPAGENTS_CODE_HISTORY_RETENTION_DAYS", "1")
        monkeypatch.setattr(
            "deepagents_code.configuration.service.get_managed_snapshot",
            lambda **_: TomlSnapshot.from_table(
                "managed config", {"history": {"retention_days": 30}}
            ),
        )
        archive = archive_dir / "old.md"
        archive.write_text("old")
        old_time = time.time() - 2 * 86_400
        os.utime(archive, (old_time, old_time))

        assert sweep_offloaded_history() == 0
        assert archive.exists()

    def test_unlink_failure_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unlink failure leaves the archive and does not raise."""
        archive_dir = self._setup(tmp_path, monkeypatch)
        archive = archive_dir / "old.md"
        archive.write_text("old")
        old_time = time.time() - 31 * 86_400
        os.utime(archive, (old_time, old_time))
        monkeypatch.setattr(
            Path, "unlink", MagicMock(side_effect=PermissionError("read-only mount"))
        )

        assert sweep_offloaded_history() == 0
        assert archive.exists()

    def test_archive_refreshed_between_iterdir_and_unlink_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An archive rewritten after the sweep lists it must not be deleted.

        Simulates a second `dcode` process refreshing an expired archive after
        this process's `iterdir()` has already enumerated it: the pre-unlink
        `fstat` observes the refreshed mtime and keeps the file, so the rewrite
        is not orphaned by a stale expiry decision.
        """
        archive_dir = self._setup(tmp_path, monkeypatch)
        archive = archive_dir / "old.md"
        archive.write_text("old")
        old_time = time.time() - 31 * 86_400
        os.utime(archive, (old_time, old_time))

        real_fstat = os.fstat
        refreshed = False

        def fstat_with_refresh(fd: int) -> os.stat_result:
            nonlocal refreshed
            if not refreshed:
                refreshed = True
                # The racing writer rewrites the archive before the sweep's
                # fstat lands, making it fresh again.
                archive.write_text("refreshed")
                fresh_time = time.time()
                os.utime(archive, (fresh_time, fresh_time))
            return real_fstat(fd)

        monkeypatch.setattr(os, "fstat", fstat_with_refresh)

        assert sweep_offloaded_history() == 0
        assert archive.exists()
        assert archive.read_text() == "refreshed"


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


class TestEventCutoff:
    """`_event_cutoff` feeds the offloaded/kept counts, so it must not guess.

    A wrong cutoff shifts `messages_offloaded`/`messages_kept` and the
    already-compacted short circuit, so every malformed shape has to read as
    zero rather than as a plausible index.
    """

    @pytest.mark.parametrize(
        ("event", "expected"),
        [
            ({"cutoff_index": 3}, 3),
            ({"cutoff_index": 0}, 0),
            (None, 0),
            ("not-a-dict", 0),
            ({}, 0),
            ({"cutoff_index": None}, 0),
            ({"cutoff_index": "3"}, 0),
            ({"cutoff_index": 3.5}, 0),
            # `bool` is an `int` subclass, so an unguarded isinstance check
            # would read this as cutoff 1.
            ({"cutoff_index": True}, 0),
            ({"cutoff_index": False}, 0),
        ],
    )
    def test_only_a_real_int_cutoff_is_honoured(
        self, event: object, expected: int
    ) -> None:
        from deepagents_code.offload_middleware import _event_cutoff

        assert _event_cutoff(event) == expected


class TestOffloadHelpers:
    """Pure helpers for effective-conversation reconstruction."""

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
        assert (
            _effective_conversation(
                messages,
                {"summary_message": "S", "cutoff_index": -1},
            )
            == messages
        )

    def test_effective_conversation_logs_a_discarded_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Dropping the event drops the summary, so it must not be silent.

        The middleware already logs this. The client is the side that reads
        possibly-malformed remote snapshot dicts, so it meets a corrupt event
        sooner, and the next request silently re-sends the whole untrimmed
        history.
        """
        from deepagents_code.app import _effective_conversation

        with caplog.at_level(
            logging.WARNING, logger="deepagents_code.goal_state_notice"
        ):
            assert _effective_conversation(["m0", "m1"], {"cutoff_index": "x"}) == [
                "m0",
                "m1",
            ]

        assert "Discarding malformed `_summarization_event`" in caplog.text

    def test_effective_conversation_does_not_log_without_an_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No event is the normal case, not a discard."""
        from deepagents_code.app import _effective_conversation

        with caplog.at_level(
            logging.WARNING, logger="deepagents_code.goal_state_notice"
        ):
            _effective_conversation(["m0"], None)

        assert "Discarding malformed" not in caplog.text

    def test_malformed_event_log_bounds_a_huge_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-Mapping event is repr'd, so it must not spill into the log."""
        from deepagents_code.goal_state_notice import (
            log_malformed_summarization_event,
        )

        with caplog.at_level(
            logging.WARNING, logger="deepagents_code.goal_state_notice"
        ):
            log_malformed_summarization_event(["x" * 5_000], 3)

        assert "(truncated)" in caplog.text
        assert len(caplog.text) < 1_000

    def test_effective_conversation_cutoff_past_end(self) -> None:
        """An out-of-bounds cutoff deliberately diverges from the SDK.

        `_apply_event_to_messages` reads a cutoff past the end as "everything
        was summarized" and returns `[summary]`. A shorter list than the cutoff
        means messages were removed after the summary was written, so the
        survivors are live turns; returning `[summary]` would hide them from
        the context sizing and dangling-tool-call checks that call this.
        """
        from deepagents_code.app import _effective_conversation

        event = {"summary_message": "S", "cutoff_index": 9}
        assert _effective_conversation(["m0"], event) == ["m0"]
        # Not the SDK's reading, which would be `["S"]`.
        assert _effective_conversation(["m0"], event) != ["S"]


def _deny_dispatched_call(
    reason: str | None,
) -> Callable[[Any, Any], dict[str, Any]]:
    """Build an `aafter_model` stub that denies the dispatched compact call."""
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
        compaction._aplan_forced_compaction_update = AsyncMock()
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

    @staticmethod
    def _plan(update: dict[str, object]) -> SimpleNamespace:
        """Build the narrow compaction-plan shape consumed by the operation."""
        return SimpleNamespace(update=lambda _path: update, archive=MagicMock())

    async def test_compacts_checkpoint_state_without_message_input(self) -> None:
        event = _summary_event(2)
        middleware, compaction, _hooks = self._middleware()
        compaction._aplan_forced_compaction_update = AsyncMock(
            return_value=self._plan(
                {
                    "_summarization_event": event,
                    "_summarization_session_id": "archive-1",
                }
            )
        )
        state = {
            "messages": _make_dict_messages(4),
        }

        execution = await middleware.execute(state, self._runtime())

        compaction._aplan_forced_compaction_update.assert_awaited_once()
        await_args = compaction._aplan_forced_compaction_update.await_args
        assert await_args is not None
        state_arg = await_args.args[0]
        assert state_arg is state
        assert "messages" not in execution.update
        assert execution.update["_summarization_session_id"] == "archive-1"
        assert execution.result["status"] == "compacted"
        assert execution.result["messages_offloaded"] == 2

    async def test_a_hook_interrupt_propagates_instead_of_failing(self) -> None:
        """A hook request must reach the client, not become a `failed` result.

        Two independent mechanisms protect this: the `BaseException` base, which
        the compaction chain's broad `except Exception` handlers cannot catch,
        and the explicit re-raise in `execute`. Either alone is sufficient, so
        this asserts the outcome rather than a mechanism -- losing *both* turns
        every interrupt into "Compaction failed:
        HookTransportInterruptError", silently breaking `/offload` for hook
        users only. Verified by mutating both. The boundary test mocks the whole
        operation, so it cannot cover this.
        """
        from uuid import uuid4

        from deepagents_code.hooks.server_middleware import (
            HookTransportInterruptError,
        )

        middleware, compaction, _hooks = self._middleware()
        request = SimpleNamespace(invocation_id=uuid4())
        compaction._aplan_forced_compaction_update = AsyncMock(
            side_effect=HookTransportInterruptError(cast("Any", request))
        )

        with pytest.raises(HookTransportInterruptError) as raised:
            await middleware.execute(
                {"messages": _make_dict_messages(4)}, self._runtime()
            )

        assert raised.value.request is request

    async def test_reoffload_reports_the_absolute_cutoff_delta(self) -> None:
        """Counts are deltas against the prior event, not absolute cutoffs.

        With a prior cutoff of 0 the two are indistinguishable, so this drives a
        chained offload: 6 messages, prior cutoff 2, new cutoff 5 must report 3
        offloaded and 1 kept. Reporting `new_cutoff` directly would say 5.
        """
        middleware, compaction, _hooks = self._middleware()
        compaction._aplan_forced_compaction_update = AsyncMock(
            return_value=self._plan(
                {
                    "_summarization_event": _summary_event(5),
                    "_summarization_session_id": "archive-1",
                }
            )
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
        compaction._aplan_forced_compaction_update = AsyncMock(return_value=None)
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
        compaction._aplan_forced_compaction_update.assert_not_awaited()

    async def test_hook_denial_without_a_reason_still_stops_compaction(self) -> None:
        """A denial carrying no reason must not read as an allow."""
        middleware, compaction, hooks = self._middleware()
        hooks.aafter_model = AsyncMock(side_effect=_deny_dispatched_call(None))

        execution = await middleware.execute(
            {"messages": _make_dict_messages(4)}, self._runtime()
        )

        assert execution.result["status"] == "denied"
        compaction._aplan_forced_compaction_update.assert_not_awaited()

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
        compaction._aplan_forced_compaction_update.assert_not_awaited()
        assert "messages" not in execution.update
        hooks.aafter_model.assert_awaited_once()

    async def test_failure_returns_result_without_rewriting_messages(self) -> None:
        middleware, compaction, _hooks = self._middleware()
        compaction._aplan_forced_compaction_update = AsyncMock(
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
