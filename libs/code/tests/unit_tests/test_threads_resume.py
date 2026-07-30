"""Unit tests for in-TUI `/threads -r` resume and previous-thread tracking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from deepagents_code.app import (
    DeepAgentsApp,
    TextualSessionState,
    _ThreadHistoryPayload,
)


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


class TestResumeCompactionChoice:
    """Large saved contexts should gate both resume paths."""

    async def test_below_threshold_continues_without_modal(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-1")
        push_wait = AsyncMock(return_value="compact")
        app._push_screen_wait = push_wait

        with patch(
            "deepagents_code.model_config.load_resume_compaction_threshold",
            return_value=300_000,
        ):
            choice = await app._offer_resume_compaction(300_000)

        assert choice == "continue"
        push_wait.assert_not_awaited()

    async def test_above_threshold_uses_modal_choice(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-1")
        push_wait = AsyncMock(return_value="compact")
        app._push_screen_wait = push_wait

        with patch(
            "deepagents_code.model_config.load_resume_compaction_threshold",
            return_value=300_000,
        ):
            choice = await app._offer_resume_compaction(300_001)

        assert choice == "compact"
        assert push_wait.await_args is not None
        screen = push_wait.await_args.args[0]
        assert screen._context_tokens == 300_001
        assert screen._threshold == 300_000

    async def test_modal_dismissal_cancels_resume(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-1")
        app._push_screen_wait = AsyncMock(return_value=None)

        with patch(
            "deepagents_code.model_config.load_resume_compaction_threshold",
            return_value=300_000,
        ):
            choice = await app._offer_resume_compaction(300_001)

        assert choice == "cancel"

    async def test_initial_resume_compacts_before_startup_finishes(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-1")
        app._session_state = TextualSessionState(thread_id="thread-1")
        app._initial_resume_requested = True
        payload = _ThreadHistoryPayload(
            messages=[],
            context_tokens=350_000,
            model_spec="",
        )
        app._fetch_thread_history_data = AsyncMock(return_value=payload)
        app._offer_resume_compaction = AsyncMock(return_value="compact")
        load_history = AsyncMock()
        app._load_thread_history = load_history
        app._run_session_start_hook = AsyncMock(return_value=True)
        compact = AsyncMock()
        app._handle_offload = compact
        app._remount_pending_goal_rubric_review = AsyncMock()
        app._drain_startup_backlog = AsyncMock()

        await app._run_session_start_sequence()

        load_history.assert_awaited_once_with(
            preloaded_payload=payload,
            resolve_pending_goal=False,
        )
        compact.assert_awaited_once_with()
        assert app._initial_session_started is True

    async def test_initial_resume_cancel_starts_fresh_thread(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-1")
        app._session_state = TextualSessionState(thread_id="thread-1")
        app._initial_resume_requested = True
        payload = _ThreadHistoryPayload(
            messages=[],
            context_tokens=350_000,
            model_spec="",
        )
        app._fetch_thread_history_data = AsyncMock(return_value=payload)
        app._offer_resume_compaction = AsyncMock(return_value="cancel")
        load_history = AsyncMock()
        app._load_thread_history = load_history
        app._run_session_start_hook = AsyncMock(return_value=True)
        app._drain_startup_backlog = AsyncMock()
        app.notify = MagicMock()
        app._update_welcome_banner = MagicMock()

        with patch(
            "deepagents_code.sessions.generate_thread_id",
            return_value="fresh-thread",
        ):
            await app._run_session_start_sequence()

        assert app._lc_thread_id == "fresh-thread"
        assert app._session_state.thread_id == "fresh-thread"
        assert app._initial_resume_requested is False
        load_history.assert_awaited_once_with(
            preloaded_payload=None,
            resolve_pending_goal=False,
        )

    async def test_thread_switch_cancel_keeps_current_thread(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-1")
        app._session_state = TextualSessionState(thread_id="thread-1")
        app._lc_thread_id = "thread-1"
        payload = _ThreadHistoryPayload(
            messages=[],
            context_tokens=350_000,
            model_spec="",
        )
        app._offer_thread_cwd_switch = AsyncMock(return_value="continue")
        app._fetch_thread_history_data = AsyncMock(return_value=payload)
        app._offer_resume_compaction = AsyncMock(return_value="cancel")
        restore = AsyncMock()
        app._restore_cwd_after_failed_thread_switch = restore
        clear = AsyncMock()
        app._clear_messages = clear
        app._set_spinner = AsyncMock()
        app._update_status = MagicMock()

        await app._resume_thread("thread-2")

        assert app._session_state.thread_id == "thread-1"
        assert app._lc_thread_id == "thread-1"
        restore.assert_awaited_once()
        clear.assert_not_awaited()

    async def test_thread_switch_compacts_after_resume(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-1")
        app._session_state = TextualSessionState(thread_id="thread-1")
        app._lc_thread_id = "thread-1"
        payload = _ThreadHistoryPayload(
            messages=[],
            context_tokens=350_000,
            model_spec="",
        )
        app._offer_thread_cwd_switch = AsyncMock(return_value="continue")
        app._fetch_thread_history_data = AsyncMock(return_value=payload)
        app._offer_resume_compaction = AsyncMock(return_value="compact")
        app._clear_messages = AsyncMock()
        app._set_spinner = AsyncMock()
        load_history = AsyncMock()
        app._load_thread_history = load_history
        app._reload_hooks = AsyncMock()
        app._run_session_start_hook = AsyncMock(return_value=True)
        compact = AsyncMock()
        app._handle_offload = compact
        app._sync_status_queued = MagicMock()
        app._update_tokens = MagicMock()
        app._update_status = MagicMock()
        app._update_welcome_banner = MagicMock()

        with patch(
            "deepagents_code.hooks.manager.HooksManager.on_session_end",
            AsyncMock(),
        ):
            await app._resume_thread("thread-2")

        assert app._session_state.thread_id == "thread-2"
        assert app._lc_thread_id == "thread-2"
        load_history.assert_awaited_once_with(
            thread_id="thread-2",
            preloaded_payload=payload,
        )
        compact.assert_awaited_once_with()


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
        assert target == "abc"

    async def test_specific_id_for_another_agent_is_rejected(self) -> None:
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
        assert target is None
        message = app._mount_message.await_args.args[0]  # ty: ignore
        assert "belongs to agent 'researcher'" in str(message._content)

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
        assert target == "prev"

    async def test_bare_skips_previous_thread_from_another_agent(self) -> None:
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
        assert target == "coder-thread"
        most_recent.assert_awaited_once_with(
            "coder",
            exclude_thread_id="cur",
        )

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
        assert target == "recent"
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
