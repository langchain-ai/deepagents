"""Unit tests for in-TUI `/threads -r` resume and previous-thread tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from textual.widget import MountError

from deepagents_code.app import (
    DeepAgentsApp,
    TextualSessionState,
    _ThreadsResumeTarget,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestSessionStatePreviousThread:
    """`reset_thread` should record the outgoing thread as `previous_thread_id`."""


def _make_app() -> DeepAgentsApp:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-1")
    app._mount_message = AsyncMock()  # ty: ignore
    app._show_thread_selector = AsyncMock()  # ty: ignore
    app._resume_thread = AsyncMock()  # ty: ignore
    return app


class TestHandleThreadsCommand:
    """`/threads` dispatch: bare opens the selector, `-r` resumes in place."""

    async def test_no_resume_when_target_none(self) -> None:
        app = _make_app()
        app._resolve_threads_resume_target = AsyncMock(return_value=None)  # ty: ignore
        await app._handle_threads_command("/threads -r missing")
        app._resume_thread.assert_not_awaited()  # ty: ignore

    async def test_unknown_flag_shows_usage(self) -> None:
        app = _make_app()
        await app._handle_threads_command("/threads --nope")
        app._show_thread_selector.assert_not_awaited()  # ty: ignore
        app._resume_thread.assert_not_awaited()  # ty: ignore
        message = app._mount_message.await_args.args[0]  # ty: ignore
        assert "Usage: /threads" in str(message._content)

    async def test_too_many_args_shows_usage(self) -> None:
        app = _make_app()
        app._resolve_threads_resume_target = AsyncMock()  # ty: ignore
        await app._handle_threads_command("/threads -r a b")
        app._resolve_threads_resume_target.assert_not_awaited()  # ty: ignore
        app._resume_thread.assert_not_awaited()  # ty: ignore
        message = app._mount_message.await_args.args[0]  # ty: ignore
        assert "at most one thread ID" in str(message._content)


class TestResolveResumeTarget:
    """`-r` argument resolution against the checkpoint store and session state."""

    async def test_specific_id_for_another_agent_resolves_with_owner(self) -> None:
        app = _make_app()
        app._assistant_id = "coder"
        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=True),
            ),
            patch(
                "deepagents_code.sessions.get_thread_agent",
                AsyncMock(return_value="researcher"),
            ),
        ):
            target = await app._resolve_threads_resume_target("abc")
        assert target == _ThreadsResumeTarget("abc", "researcher")
        app._mount_message.assert_not_awaited()  # ty: ignore

    async def test_specific_id_missing_notifies(self) -> None:
        app = _make_app()
        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=False),
            ),
            patch(
                "deepagents_code.sessions.find_similar_threads",
                AsyncMock(return_value=[]),
            ),
        ):
            target = await app._resolve_threads_resume_target("abc")
        assert target is None
        message = app._mount_message.await_args.args[0]  # ty: ignore
        assert "Thread 'abc' not found." in str(message._content)

    async def test_specific_id_missing_suggests_similar(self) -> None:
        """A near-miss id surfaces the `find_similar_threads` suggestions."""
        app = _make_app()
        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=False),
            ),
            patch(
                "deepagents_code.sessions.find_similar_threads",
                AsyncMock(return_value=["abc123", "abc456"]),
            ),
        ):
            target = await app._resolve_threads_resume_target("abc")
        assert target is None
        message = str(app._mount_message.await_args.args[0]._content)  # ty: ignore
        assert "Did you mean: abc123, abc456?" in message

    async def test_specific_id_database_failure_notifies(self) -> None:
        """An expected thread-store error asks the user to retry."""
        import sqlite3

        app = _make_app()
        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=False),
            ),
            patch(
                "deepagents_code.sessions.find_similar_threads",
                AsyncMock(side_effect=sqlite3.OperationalError("db locked")),
            ),
        ):
            target = await app._resolve_threads_resume_target("abc")
        assert target is None
        message = app._mount_message.await_args.args[0]  # ty: ignore
        assert "Could not look up thread history" in str(message._content)

    async def test_specific_id_unexpected_error_notifies(self) -> None:
        """A non-DB error is surfaced distinctly from a lookup failure."""
        app = _make_app()
        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=False),
            ),
            patch(
                "deepagents_code.sessions.find_similar_threads",
                AsyncMock(side_effect=RuntimeError("boom")),
            ),
        ):
            target = await app._resolve_threads_resume_target("abc")
        assert target is None
        message = app._mount_message.await_args.args[0]  # ty: ignore
        assert "Something went wrong resolving that thread." in str(message._content)

    async def test_bare_prefers_previous_thread(self) -> None:
        app = _make_app()
        state = TextualSessionState(thread_id="cur")
        state.previous_thread_id = "prev"
        app._session_state = state
        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=True),
            ),
            patch(
                "deepagents_code.sessions.get_thread_agent",
                AsyncMock(return_value="agent"),
            ),
        ):
            target = await app._resolve_threads_resume_target(None)
        assert target == _ThreadsResumeTarget("prev", "agent")

    async def test_bare_none_when_no_threads(self) -> None:
        app = _make_app()
        app._session_state = TextualSessionState(thread_id="cur")
        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=False),
            ),
            patch(
                "deepagents_code.sessions.get_most_recent",
                AsyncMock(return_value=None),
            ),
        ):
            target = await app._resolve_threads_resume_target(None)
        assert target is None
        message = app._mount_message.await_args.args[0]  # ty: ignore
        assert "No previous threads for 'agent' to resume." in str(message._content)

    async def test_bare_database_failure_notifies(self) -> None:
        app = _make_app()
        app._session_state = TextualSessionState(thread_id="cur")
        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=False),
            ),
            patch(
                "deepagents_code.sessions.get_most_recent",
                AsyncMock(side_effect=RuntimeError("db unavailable")),
            ),
        ):
            target = await app._resolve_threads_resume_target(None)
        assert target is None
        app._mount_message.assert_awaited_once()  # ty: ignore


class TestCrossAgentResume:
    """Confirmation and orchestration for a cross-agent resume target."""

    async def test_cancel_keeps_current_session_untouched(self, tmp_path: Path) -> None:
        """Esc exits before history, cwd, or server state is mutated."""
        app = _make_app()
        app._server_kwargs = {"assistant_id": "agent"}
        app._server_proc = MagicMock()
        (tmp_path / "researcher").mkdir()
        app._push_screen_wait = AsyncMock(return_value="cancel")  # ty: ignore
        fetch = AsyncMock()
        app._fetch_thread_history_data = fetch  # ty: ignore

        with (
            patch("deepagents_code.config.credentials") as settings,
            patch(
                "deepagents_code.app.user_deepagents_dir",
                return_value=tmp_path,
            ),
        ):
            settings.user_deepagents_dir = tmp_path
            await app._confirm_then_resume_cross_agent_thread(
                _ThreadsResumeTarget("research-thread", "researcher")
            )

        fetch.assert_not_awaited()
        message = app._mount_message.await_args.args[0]  # ty: ignore
        assert "canceled" in str(message._content)

    async def test_remote_session_gets_relaunch_instruction(self) -> None:
        """Remote sessions receive an actionable fallback instead of a modal."""
        app = _make_app()
        app._server_kwargs = None
        app._server_proc = None

        await app._confirm_then_resume_cross_agent_thread(
            _ThreadsResumeTarget("research-thread", "researcher")
        )

        message = app._mount_message.await_args.args[0]  # ty: ignore
        content = str(message._content)
        assert "cannot switch its remote server" in content
        assert "dcode -r research-thread" in content


class TestPreviousThreadHintOwnership:
    """The advertised resume action must be executable in this session."""

    async def test_unmountable_hint_reports_failure(self) -> None:
        """A hint that raised while mounting has not been shown.

        Reporting success here would suppress the agent swap's relaunch
        fallback, leaving the user with no way back at all.
        """
        app = _make_app()
        app._assistant_id = "coder"
        app._server_kwargs = {"assistant_id": "coder"}
        app._mount_message = AsyncMock(  # ty: ignore
            side_effect=MountError("container is detached")
        )

        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=True),
            ),
            patch(
                "deepagents_code.sessions.get_thread_agent",
                AsyncMock(return_value="coder"),
            ),
            patch.object(app, "_schedule_thread_message_link"),
        ):
            hinted = await app._mount_previous_thread_hint(
                "research-thread", had_agent_output=True
            )

        assert hinted is False
