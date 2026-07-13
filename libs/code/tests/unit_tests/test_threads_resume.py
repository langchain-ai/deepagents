"""Unit tests for in-TUI `/threads -r` resume and previous-thread tracking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from deepagents_code.app import DeepAgentsApp, TextualSessionState


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
        app._resolve_threads_resume_target = AsyncMock(return_value="thread-x")  # ty: ignore
        await app._handle_threads_command("/threads -r")
        app._resolve_threads_resume_target.assert_awaited_once_with(None)  # ty: ignore
        app._resume_thread.assert_awaited_once_with("thread-x")  # ty: ignore
        app._show_thread_selector.assert_not_awaited()  # ty: ignore

    async def test_resume_specific_id(self) -> None:
        app = _make_app()
        app._resolve_threads_resume_target = AsyncMock(return_value="abc")  # ty: ignore
        await app._handle_threads_command("/threads -r abc")
        app._resolve_threads_resume_target.assert_awaited_once_with("abc")  # ty: ignore
        app._resume_thread.assert_awaited_once_with("abc")  # ty: ignore

    async def test_resume_long_form_flag(self) -> None:
        app = _make_app()
        app._resolve_threads_resume_target = AsyncMock(return_value="abc")  # ty: ignore
        await app._handle_threads_command("/threads --resume abc")
        app._resolve_threads_resume_target.assert_awaited_once_with("abc")  # ty: ignore

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
        app._mount_message.assert_awaited()  # ty: ignore

    async def test_too_many_args_shows_usage(self) -> None:
        app = _make_app()
        app._resolve_threads_resume_target = AsyncMock()  # ty: ignore
        await app._handle_threads_command("/threads -r a b")
        app._resolve_threads_resume_target.assert_not_awaited()  # ty: ignore
        app._resume_thread.assert_not_awaited()  # ty: ignore


class TestResolveResumeTarget:
    """`-r` argument resolution against the checkpoint store and session state."""

    async def test_specific_id_exists(self) -> None:
        app = _make_app()
        with patch(
            "deepagents_code.sessions.thread_exists",
            AsyncMock(return_value=True),
        ):
            target = await app._resolve_threads_resume_target("abc")
        assert target == "abc"

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
        app._mount_message.assert_awaited()  # ty: ignore

    async def test_specific_id_database_failure_notifies(self) -> None:
        app = _make_app()
        with (
            patch(
                "deepagents_code.sessions.thread_exists",
                AsyncMock(return_value=False),
            ),
            patch(
                "deepagents_code.sessions.find_similar_threads",
                AsyncMock(side_effect=RuntimeError("db unavailable")),
            ),
        ):
            target = await app._resolve_threads_resume_target("abc")
        assert target is None
        app._mount_message.assert_awaited_once()  # ty: ignore

    async def test_bare_prefers_previous_thread(self) -> None:
        app = _make_app()
        state = TextualSessionState(thread_id="cur")
        state.previous_thread_id = "prev"
        app._session_state = state
        with patch(
            "deepagents_code.sessions.thread_exists",
            AsyncMock(return_value=True),
        ):
            target = await app._resolve_threads_resume_target(None)
        assert target == "prev"

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
        assert target == "recent"
        most_recent.assert_awaited_once_with(
            "coder",
            exclude_thread_id="cur",
        )

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
        app._mount_message.assert_awaited()  # ty: ignore

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
