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

    def test_previous_thread_starts_none(self) -> None:
        state = TextualSessionState(thread_id="thread-a")
        assert state.previous_thread_id is None

    def test_reset_thread_records_previous(self) -> None:
        state = TextualSessionState(thread_id="thread-a")
        first = state.thread_id
        new = state.reset_thread()
        assert state.previous_thread_id == first
        assert new != first
        assert state.thread_id == new

    def test_reset_thread_updates_previous_each_time(self) -> None:
        state = TextualSessionState(thread_id="thread-a")
        second = state.reset_thread()
        assert state.previous_thread_id == "thread-a"
        state.reset_thread()
        assert state.previous_thread_id == second


def _make_app() -> DeepAgentsApp:
    app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-1")
    app._mount_message = AsyncMock()  # ty: ignore
    app._show_thread_selector = AsyncMock()  # ty: ignore
    app._resume_thread = AsyncMock()  # ty: ignore
    return app


class TestHandleThreadsCommand:
    """`/threads` dispatch: bare opens the selector, `-r` resumes in place."""

    async def test_bare_opens_selector(self) -> None:
        app = _make_app()
        await app._handle_threads_command("/threads")
        app._show_thread_selector.assert_awaited_once()  # ty: ignore
        app._resume_thread.assert_not_awaited()  # ty: ignore

    async def test_resume_flag_resolves_and_resumes(self) -> None:
        app = _make_app()
        target = _ThreadsResumeTarget("thread-x", "agent")
        app._resolve_threads_resume_target = AsyncMock(return_value=target)  # ty: ignore
        await app._handle_threads_command("/threads -r")
        app._resolve_threads_resume_target.assert_awaited_once_with(None)  # ty: ignore
        app._resume_thread.assert_awaited_once_with("thread-x")  # ty: ignore
        app._show_thread_selector.assert_not_awaited()  # ty: ignore

    async def test_resume_specific_id(self) -> None:
        app = _make_app()
        target = _ThreadsResumeTarget("abc", "agent")
        app._resolve_threads_resume_target = AsyncMock(return_value=target)  # ty: ignore
        await app._handle_threads_command("/threads -r abc")
        app._resolve_threads_resume_target.assert_awaited_once_with("abc")  # ty: ignore
        app._resume_thread.assert_awaited_once_with("abc")  # ty: ignore

    async def test_resume_long_form_flag(self) -> None:
        app = _make_app()
        target = _ThreadsResumeTarget("abc", "agent")
        app._resolve_threads_resume_target = AsyncMock(return_value=target)  # ty: ignore
        await app._handle_threads_command("/threads --resume abc")
        app._resolve_threads_resume_target.assert_awaited_once_with("abc")  # ty: ignore

    async def test_cross_agent_resume_schedules_confirmation(self) -> None:
        """A cross-agent target is confirmed off Textual's message pump."""
        app = _make_app()
        target = _ThreadsResumeTarget("abc", "researcher")
        app._resolve_threads_resume_target = AsyncMock(return_value=target)  # ty: ignore

        with patch.object(app, "_schedule_off_message_pump") as schedule:
            await app._handle_threads_command("/threads -r")

        app._resume_thread.assert_not_awaited()  # ty: ignore
        schedule.assert_called_once()
        assert schedule.call_args.kwargs == {"context": "threads-resume:abc"}
        continuation = schedule.call_args.args[0]
        continuation.close()

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

    async def test_specific_id_exists(self) -> None:
        app = _make_app()
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
            target = await app._resolve_threads_resume_target("abc")
        assert target == _ThreadsResumeTarget("abc", "agent")

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

    async def test_bare_preserves_previous_thread_from_another_agent(self) -> None:
        app = _make_app()
        app._assistant_id = "coder"
        state = TextualSessionState(thread_id="cur")
        state.previous_thread_id = "research-thread"
        app._session_state = state
        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=True),
            ),
            patch(
                "deepagents_code.sessions.get_thread_agent",
                AsyncMock(return_value="researcher"),
            ),
            patch(
                "deepagents_code.sessions.get_most_recent",
                AsyncMock(return_value="coder-thread"),
            ) as most_recent,
        ):
            target = await app._resolve_threads_resume_target(None)
        assert target == _ThreadsResumeTarget("research-thread", "researcher")
        most_recent.assert_not_awaited()

    async def test_bare_falls_back_to_most_recent(self) -> None:
        app = _make_app()
        app._session_state = TextualSessionState(thread_id="cur")
        app._assistant_id = "coder"
        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=False),
            ),
            patch(
                "deepagents_code.sessions.get_most_recent",
                AsyncMock(return_value="recent"),
            ) as most_recent,
        ):
            target = await app._resolve_threads_resume_target(None)
        assert target == _ThreadsResumeTarget("recent", "coder")
        most_recent.assert_awaited_once_with(
            "coder",
            exclude_thread_id="cur",
        )

    async def test_bare_previous_deleted_falls_back(self) -> None:
        """A `previous_thread_id` pruned since `/clear` falls through to recent."""
        app = _make_app()
        app._assistant_id = "coder"
        state = TextualSessionState(thread_id="cur")
        state.previous_thread_id = "prev"
        app._session_state = state
        with (
            # previous exists no more; the fallback thread does.
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=False),
            ),
            patch(
                "deepagents_code.sessions.get_thread_agent",
                AsyncMock(return_value="coder"),
            ) as thread_agent,
            patch(
                "deepagents_code.sessions.get_most_recent",
                AsyncMock(return_value="recent"),
            ) as most_recent,
        ):
            target = await app._resolve_threads_resume_target(None)
        assert target == _ThreadsResumeTarget("recent", "coder")
        # The deleted previous never reaches the ownership check.
        thread_agent.assert_not_awaited()
        most_recent.assert_awaited_once_with("coder", exclude_thread_id="cur")

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

    async def test_bare_default_agent_fallback_is_filtered(self) -> None:
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
            ) as most_recent,
        ):
            await app._resolve_threads_resume_target(None)
        most_recent.assert_awaited_once_with(
            "agent",
            exclude_thread_id="cur",
        )

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

    async def test_confirmation_switches_agent_and_exact_thread(
        self,
        tmp_path: Path,
    ) -> None:
        """Accepting performs one combined, session-only transition."""
        app = _make_app()
        app._server_kwargs = {"assistant_id": "agent"}
        app._server_proc = MagicMock()
        (tmp_path / "researcher").mkdir()
        payload = MagicMock()
        app._push_screen_wait = AsyncMock(return_value="switch")  # ty: ignore
        app._fetch_thread_history_data = AsyncMock(return_value=payload)  # ty: ignore
        app._offer_thread_cwd_switch = AsyncMock(return_value="continue")  # ty: ignore
        app._restart_server_for_agent_swap = AsyncMock(return_value=True)  # ty: ignore

        with patch("deepagents_code.config.settings") as settings:
            settings.user_deepagents_dir = tmp_path
            await app._confirm_then_resume_cross_agent_thread(
                _ThreadsResumeTarget("research-thread", "researcher")
            )

        app._fetch_thread_history_data.assert_awaited_once_with("research-thread")  # ty: ignore
        app._offer_thread_cwd_switch.assert_awaited_once_with(  # ty: ignore
            "research-thread",
            restart_server=False,
            abort="thread_switch",
        )
        app._restart_server_for_agent_swap.assert_awaited_once_with(  # ty: ignore
            "researcher",
            resume_thread_id="research-thread",
            preloaded_payload=payload,
            persist_default_agent=False,
        )

    async def test_cancel_keeps_current_session_untouched(self, tmp_path: Path) -> None:
        """Esc exits before history, cwd, or server state is mutated."""
        app = _make_app()
        app._server_kwargs = {"assistant_id": "agent"}
        app._server_proc = MagicMock()
        (tmp_path / "researcher").mkdir()
        app._push_screen_wait = AsyncMock(return_value="cancel")  # ty: ignore
        fetch = AsyncMock()
        app._fetch_thread_history_data = fetch  # ty: ignore

        with patch("deepagents_code.config.settings") as settings:
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

    async def test_local_cross_agent_hint_is_shown(self) -> None:
        """Owned local servers can honor the hint through confirmation."""
        app = _make_app()
        app._assistant_id = "coder"
        app._server_kwargs = {"assistant_id": "coder"}

        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=True),
            ),
            patch(
                "deepagents_code.sessions.get_thread_agent",
                AsyncMock(return_value="researcher"),
            ),
            patch.object(app, "_schedule_thread_message_link") as schedule,
        ):
            hinted = await app._mount_previous_thread_hint("research-thread")

        app._mount_message.assert_awaited_once()  # ty: ignore
        schedule.assert_called_once()
        # The agent swap keys its relaunch fallback off this, so a hint that
        # mounts must report `True` or the user gets both hints at once.
        assert hinted is True

    async def test_remote_cross_agent_hint_is_suppressed(self) -> None:
        """Remote sessions do not promise an agent restart they cannot do."""
        app = _make_app()
        app._assistant_id = "coder"
        app._server_kwargs = None

        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=True),
            ),
            patch(
                "deepagents_code.sessions.get_thread_agent",
                AsyncMock(return_value="researcher"),
            ),
            patch.object(app, "_schedule_thread_message_link") as schedule,
        ):
            hinted = await app._mount_previous_thread_hint("research-thread")

        app._mount_message.assert_not_awaited()  # ty: ignore
        schedule.assert_not_called()
        # Reporting `False` is what lets the agent swap fall back to the
        # relaunch command, so a remote session is not left with no way back.
        assert hinted is False

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
            hinted = await app._mount_previous_thread_hint("research-thread")

        assert hinted is False
