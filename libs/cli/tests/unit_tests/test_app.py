"""Unit tests for DeepAgentsApp."""

from __future__ import annotations

import asyncio
import inspect
import io
import os
import signal
import time
import webbrowser
from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, call, patch

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from pathlib import Path

    from deepagents_cli.sessions import ThreadInfo

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Checkbox, Input, Static

from deepagents_cli.app import (
    _ITERM_CURSOR_GUIDE_OFF,
    _ITERM_CURSOR_GUIDE_ON,
    _TYPING_IDLE_THRESHOLD_SECONDS,
    DeepAgentsApp,
    DeferredAction,
    QueuedMessage,
    TextualSessionState,
    _write_iterm_escape,
)
from deepagents_cli.widgets.chat_input import ChatInput
from deepagents_cli.widgets.messages import (
    AppMessage,
    ErrorMessage,
    QueuedUserMessage,
    UserMessage,
)


async def _wait_for_branch(app: DeepAgentsApp, branch: str) -> None:
    """Wait until the status bar reports the expected git branch."""
    for _ in range(100):
        if app._status_bar is not None and app._status_bar.branch == branch:
            return
        await asyncio.sleep(0.01)
    msg = f"Timed out waiting for branch {branch!r}"
    raise AssertionError(msg)


class TestInitialPromptOnMount:
    """Test that -m initial prompt is submitted on mount."""

    async def test_initial_prompt_triggers_handle_user_message(self) -> None:
        """When initial_prompt is set, the prompt should be auto-submitted."""
        mock_agent = MagicMock()
        app = DeepAgentsApp(
            agent=mock_agent,
            thread_id="new-thread-123",
            initial_prompt="hello world",
        )
        submitted: list[str] = []

        # Must be async to match _handle_user_message's signature
        async def capture(msg: str) -> None:  # noqa: RUF029
            submitted.append(msg)

        app._handle_user_message = capture  # type: ignore[assignment]

        async with app.run_test() as pilot:
            # Give call_after_refresh time to fire
            await pilot.pause()
            await pilot.pause()

        assert submitted == ["hello world"]

    async def test_initial_skill_triggers_invoke_skill(self) -> None:
        """When `--skill` is set, startup should invoke that skill."""
        mock_agent = MagicMock()
        app = DeepAgentsApp(
            agent=mock_agent,
            thread_id="new-thread-123",
            initial_prompt="  keep leading whitespace",
            initial_skill="code-review",
        )
        submitted: list[tuple[str, str, str | None]] = []

        async def capture(  # noqa: RUF029
            skill_name: str,
            args: str = "",
            *,
            command: str | None = None,
        ) -> None:
            submitted.append((skill_name, args, command))

        app._invoke_skill = capture  # type: ignore[assignment]

        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()

        assert submitted == [("code-review", "  keep leading whitespace", None)]

    async def test_initial_skill_runs_after_server_ready(self) -> None:
        """Deferred startup should invoke the requested skill after connect."""
        app = DeepAgentsApp(
            thread_id="new-thread-123",
            initial_prompt="review this diff",
            initial_skill="code-review",
        )
        app._connecting = True
        app.query_one = MagicMock(side_effect=NoMatches("welcome-banner"))  # type: ignore[assignment]
        app.call_after_refresh = lambda cb: cb()  # type: ignore[assignment]
        submitted: list[tuple[str, str, str | None]] = []

        async def capture(  # noqa: RUF029
            skill_name: str,
            args: str = "",
            *,
            command: str | None = None,
        ) -> None:
            submitted.append((skill_name, args, command))

        app._invoke_skill = capture  # type: ignore[assignment]

        app.on_deep_agents_app_server_ready(
            app.ServerReady(
                agent=MagicMock(),
                server_proc=None,
                mcp_server_info=[],
            )
        )
        # Server-ready schedules `_run_session_start_sequence` onto the loop.
        # A few yields keep the test stable across that async handoff.
        for _ in range(3):
            await asyncio.sleep(0)

        assert submitted == [("code-review", "review this diff", None)]


class TestStartupSequence:
    """Tests for post-connect startup sequencing."""

    async def test_resumed_history_loads_before_startup_command(self) -> None:
        """Resumed threads should mount prior history before startup output."""
        app = DeepAgentsApp(
            agent=MagicMock(),
            thread_id="thread-123",
            resume_thread="thread-123",
            startup_cmd="echo hi",
        )
        order: list[str] = []

        async def capture_history(  # noqa: RUF029
            *,
            thread_id: str | None = None,
            preloaded_payload: object | None = None,
        ) -> None:
            del thread_id, preloaded_payload
            order.append("history")

        async def capture_startup(command: str) -> None:  # noqa: RUF029
            assert command == "echo hi"
            order.append("startup")

        app._load_thread_history = capture_history  # type: ignore[assignment]
        app._run_startup_command = capture_startup  # type: ignore[assignment]

        await app._run_session_start_sequence()

        assert order == ["history", "startup"]
        assert app._startup_sequence_running is False

    async def test_startup_cleanup_defers_queue_until_initial_submission(self) -> None:
        """Queued input should wait until startup submission owns the agent slot."""
        app = DeepAgentsApp(
            agent=MagicMock(),
            thread_id="thread-123",
            initial_prompt="hello world",
            startup_cmd="echo hi",
        )
        order: list[str] = []

        async def capture_startup(command: str) -> None:
            assert command == "echo hi"
            order.append("startup")
            app._pending_messages.append(
                QueuedMessage(text="typed during startup", mode="normal")
            )
            await app._cleanup_shell_task()

        async def capture_initial_submission() -> None:  # noqa: RUF029
            order.append("initial")
            app._agent_running = True

        queue_mock = AsyncMock()
        app._run_startup_command = capture_startup  # type: ignore[assignment]
        app._submit_initial_submission = (  # type: ignore[assignment]
            capture_initial_submission
        )
        app._process_next_from_queue = queue_mock  # type: ignore[assignment]

        await app._run_session_start_sequence()

        assert order == ["startup", "initial"]
        queue_mock.assert_not_awaited()
        assert len(app._pending_messages) == 1
        assert app._pending_messages[0].text == "typed during startup"
        assert app._startup_sequence_running is False

    async def test_cleanup_shell_task_defers_queue_during_startup(self) -> None:
        """`_cleanup_shell_task` must not drain the queue while sequencing."""
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        app._startup_sequence_running = True
        app._pending_messages.append(QueuedMessage(text="queued", mode="normal"))
        drain_mock = AsyncMock()
        queue_mock = AsyncMock()
        app._process_next_from_queue = queue_mock  # type: ignore[assignment]
        app._maybe_drain_deferred = drain_mock  # type: ignore[assignment]

        await app._cleanup_shell_task()

        queue_mock.assert_not_awaited()
        assert app._shell_running is False

    async def test_cleanup_agent_task_defers_queue_during_startup(self) -> None:
        """`_cleanup_agent_task` must not drain the queue while sequencing."""
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        app._startup_sequence_running = True
        app._pending_messages.append(QueuedMessage(text="queued", mode="normal"))
        drain_mock = AsyncMock()
        queue_mock = AsyncMock()
        spinner_mock = AsyncMock()
        app._process_next_from_queue = queue_mock  # type: ignore[assignment]
        app._maybe_drain_deferred = drain_mock  # type: ignore[assignment]
        app._set_spinner = spinner_mock  # type: ignore[assignment]

        await app._cleanup_agent_task()

        queue_mock.assert_not_awaited()
        assert app._agent_running is False

    async def test_cleanup_agent_task_schedules_git_branch_refresh(self) -> None:
        """Agent cleanup should refresh repo state after a turn completes."""
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        drain_mock = AsyncMock()
        queue_mock = AsyncMock()
        spinner_mock = AsyncMock()
        refresh_mock = MagicMock()
        app._process_next_from_queue = queue_mock  # type: ignore[assignment]
        app._maybe_drain_deferred = drain_mock  # type: ignore[assignment]
        app._set_spinner = spinner_mock  # type: ignore[assignment]
        app._schedule_git_branch_refresh = refresh_mock  # type: ignore[assignment]

        await app._cleanup_agent_task()

        refresh_mock.assert_called_once_with()
        drain_mock.assert_awaited_once()
        queue_mock.assert_awaited_once()

    async def test_schedule_git_branch_refresh_noops_during_exit(self) -> None:
        """Shutdown should prevent new background git refresh tasks."""
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        app._exit = True

        with patch("deepagents_cli.app.asyncio.create_task") as mock_create_task:
            app._schedule_git_branch_refresh()

        assert app._git_branch_refresh_task is None
        mock_create_task.assert_not_called()

    async def test_schedule_git_branch_refresh_inline_fast_path(
        self, tmp_path: Path
    ) -> None:
        """Filesystem probe should update the footer without spawning a task."""
        repo = tmp_path / "repo"
        git_dir = repo / ".git"
        git_dir.mkdir(parents=True)
        (git_dir / "HEAD").write_text("ref: refs/heads/feature\n", encoding="utf-8")

        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        status_bar = MagicMock()
        app._status_bar = status_bar
        app._cwd = str(repo)

        with patch("deepagents_cli.app.asyncio.create_task") as mock_create_task:
            app._schedule_git_branch_refresh()

        assert status_bar.branch == "feature"
        mock_create_task.assert_not_called()
        assert app._git_branch_refresh_task is None

    async def test_schedule_git_branch_refresh_falls_back_to_subprocess(
        self,
    ) -> None:
        """Unusual repo layouts should spawn the off-thread subprocess fallback."""
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        status_bar = MagicMock()
        app._status_bar = status_bar

        fallback_mock = AsyncMock()
        app._refresh_git_branch_subprocess_fallback = (  # type: ignore[assignment]
            fallback_mock
        )

        with patch(
            "deepagents_cli.app.read_git_branch_from_filesystem",
            return_value=None,
        ):
            app._schedule_git_branch_refresh()

        refresh_task = app._git_branch_refresh_task
        assert refresh_task is not None
        await refresh_task
        fallback_mock.assert_awaited_once_with(app._cwd)

    def test_empty_startup_cmd_is_normalized_to_none(self) -> None:
        """Empty or whitespace-only `--startup-cmd` should be treated as unset."""
        for raw in ("", "   ", "\t\n"):
            app = DeepAgentsApp(
                agent=MagicMock(), thread_id="thread-123", startup_cmd=raw
            )
            assert app._startup_cmd is None, f"Expected {raw!r} to normalize to None"

    async def test_startup_cmd_cleared_after_execution(self) -> None:
        """`_startup_cmd` should be cleared before the command runs (one-shot)."""
        app = DeepAgentsApp(
            agent=MagicMock(), thread_id="thread-123", startup_cmd="echo hi"
        )
        observed_cmd: list[str] = []
        observed_attr_during_run: list[str | None] = []

        async def capture_startup(command: str) -> None:  # noqa: RUF029
            observed_cmd.append(command)
            observed_attr_during_run.append(app._startup_cmd)

        async def stub_history(  # noqa: RUF029
            *,
            thread_id: str | None = None,
            preloaded_payload: object | None = None,
        ) -> None:
            del thread_id, preloaded_payload

        app._run_startup_command = capture_startup  # type: ignore[assignment]
        app._load_thread_history = stub_history  # type: ignore[assignment]

        await app._run_session_start_sequence()

        assert observed_cmd == ["echo hi"]
        assert observed_attr_during_run == [None]
        assert app._startup_cmd is None


class TestAppCSSValidation:
    """Test that app CSS is valid and doesn't cause runtime errors."""

    async def test_app_css_validates_on_mount(self) -> None:
        """App should mount without CSS validation errors.

        This test catches invalid CSS properties like 'overflow: visible'
        which are only validated at runtime when styles are applied.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            # Give the app time to render and apply CSS
            await pilot.pause()
            # If we get here without exception, CSS is valid
            assert app.is_running


class TestThreadCachePrewarm:
    """Tests for startup thread-cache prewarming."""

    async def test_prewarm_uses_current_thread_limit(self) -> None:
        """Prewarm helper should pass the resolved thread limit through."""
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")

        with (
            patch("deepagents_cli.sessions.get_thread_limit", return_value=7),
            patch(
                "deepagents_cli.sessions.prewarm_thread_message_counts",
                new_callable=AsyncMock,
            ) as mock_prewarm,
        ):
            await app._prewarm_threads_cache()

        mock_prewarm.assert_awaited_once_with(limit=7)

    async def test_show_thread_selector_uses_cached_rows(self) -> None:
        """Thread selector should receive prefetched rows when available."""
        cached_threads = [
            {
                "thread_id": "thread-abc",
                "agent_name": "agent1",
                "updated_at": "2024-01-01T00:00:00+00:00",
                "message_count": 2,
            }
        ]
        app = DeepAgentsApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            with (
                patch("deepagents_cli.sessions.get_thread_limit", return_value=9),
                patch(
                    "deepagents_cli.sessions.get_cached_threads",
                    return_value=cached_threads,
                ),
                patch(
                    "deepagents_cli.widgets.thread_selector.ThreadSelectorScreen"
                ) as mock_screen_cls,
                patch.object(app, "push_screen") as mock_push_screen,
            ):
                mock_screen = MagicMock()
                mock_screen_cls.return_value = mock_screen
                await app._show_thread_selector()

                assert app._session_state is not None
                mock_screen_cls.assert_called_once_with(
                    current_thread=app._session_state.thread_id,
                    thread_limit=9,
                    initial_threads=cached_threads,
                )
                mock_push_screen.assert_called_once()


class TestAppBindings:
    """Test app keybindings."""

    def test_ctrl_c_binding_has_priority(self) -> None:
        """Ctrl+C should be priority-bound so focused modal inputs don't swallow it."""
        bindings = [b for b in DeepAgentsApp.BINDINGS if isinstance(b, Binding)]
        bindings_by_key = {b.key: b for b in bindings}
        ctrl_c = bindings_by_key.get("ctrl+c")

        assert ctrl_c is not None
        assert ctrl_c.action == "quit_or_interrupt"
        assert ctrl_c.priority is True

    def test_toggle_tool_output_has_ctrl_o_binding(self) -> None:
        """Ctrl+O should be bound to toggle_tool_output with priority."""
        bindings = [b for b in DeepAgentsApp.BINDINGS if isinstance(b, Binding)]
        bindings_by_key = {b.key: b for b in bindings}
        ctrl_o = bindings_by_key.get("ctrl+o")

        assert ctrl_o is not None
        assert ctrl_o.action == "toggle_tool_output"
        assert ctrl_o.priority is True

    def test_ctrl_e_not_bound(self) -> None:
        """Ctrl+E must not be bound — it shadows TextArea cursor_line_end."""
        bindings = [b for b in DeepAgentsApp.BINDINGS if isinstance(b, Binding)]
        bindings_by_key = {b.key: b for b in bindings}
        assert "ctrl+e" not in bindings_by_key


class TestITerm2CursorGuide:
    """Test iTerm2 cursor guide handling."""

    def test_escape_sequences_are_valid(self) -> None:
        """Escape sequences should be properly formatted OSC 1337 commands.

        Format: OSC (ESC ]) + "1337;" + command + ST (ESC backslash)
        """
        assert _ITERM_CURSOR_GUIDE_OFF.startswith("\x1b]1337;")
        assert _ITERM_CURSOR_GUIDE_OFF.endswith("\x1b\\")
        assert "HighlightCursorLine=no" in _ITERM_CURSOR_GUIDE_OFF

        assert _ITERM_CURSOR_GUIDE_ON.startswith("\x1b]1337;")
        assert _ITERM_CURSOR_GUIDE_ON.endswith("\x1b\\")
        assert "HighlightCursorLine=yes" in _ITERM_CURSOR_GUIDE_ON

    def test_write_iterm_escape_does_nothing_when_not_iterm(self) -> None:
        """_write_iterm_escape should no-op when _IS_ITERM is False."""
        mock_stderr = MagicMock()
        with (
            patch("deepagents_cli.app._IS_ITERM", False),
            patch("sys.__stderr__", mock_stderr),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)
            mock_stderr.write.assert_not_called()

    def test_write_iterm_escape_writes_sequence_when_iterm(self) -> None:
        """_write_iterm_escape should write sequence when in iTerm2."""
        mock_stderr = io.StringIO()
        with (
            patch("deepagents_cli.app._IS_ITERM", True),
            patch("sys.__stderr__", mock_stderr),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)
            assert mock_stderr.getvalue() == _ITERM_CURSOR_GUIDE_ON

    def test_write_iterm_escape_handles_oserror_gracefully(self) -> None:
        """_write_iterm_escape should not raise on OSError."""
        mock_stderr = MagicMock()
        mock_stderr.write.side_effect = OSError("Broken pipe")
        with (
            patch("deepagents_cli.app._IS_ITERM", True),
            patch("sys.__stderr__", mock_stderr),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)

    def test_write_iterm_escape_handles_none_stderr(self) -> None:
        """_write_iterm_escape should handle None __stderr__ gracefully."""
        with (
            patch("deepagents_cli.app._IS_ITERM", True),
            patch("sys.__stderr__", None),
        ):
            _write_iterm_escape(_ITERM_CURSOR_GUIDE_ON)


class TestITerm2Detection:
    """Test iTerm2 detection logic."""

    def test_detection_requires_tty(self) -> None:
        """_IS_ITERM should check that stderr is a TTY.

        Detection happens at module load, so we test the logic pattern directly.
        """
        with (
            patch.dict(os.environ, {"LC_TERMINAL": "iTerm2"}, clear=False),
            patch("os.isatty", return_value=False),
        ):
            result = (
                (
                    os.environ.get("LC_TERMINAL", "") == "iTerm2"
                    or os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
                )
                and hasattr(os, "isatty")
                and os.isatty(2)
            )
            assert result is False

    def test_detection_via_lc_terminal(self) -> None:
        """Detection should match LC_TERMINAL=iTerm2."""
        with (
            patch.dict(
                os.environ, {"LC_TERMINAL": "iTerm2", "TERM_PROGRAM": ""}, clear=False
            ),
            patch("os.isatty", return_value=True),
        ):
            result = (
                (
                    os.environ.get("LC_TERMINAL", "") == "iTerm2"
                    or os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
                )
                and hasattr(os, "isatty")
                and os.isatty(2)
            )
            assert result is True

    def test_detection_via_term_program(self) -> None:
        """Detection should match TERM_PROGRAM=iTerm.app."""
        env = {"LC_TERMINAL": "", "TERM_PROGRAM": "iTerm.app"}
        with (
            patch.dict(os.environ, env, clear=False),
            patch("os.isatty", return_value=True),
        ):
            result = (
                (
                    os.environ.get("LC_TERMINAL", "") == "iTerm2"
                    or os.environ.get("TERM_PROGRAM", "") == "iTerm.app"
                )
                and hasattr(os, "isatty")
                and os.isatty(2)
            )
            assert result is True


class TestModalScreenEscapeDismissal:
    """Test that escape key dismisses modal screens."""

    @staticmethod
    async def test_escape_dismisses_modal_screen() -> None:
        """Escape should dismiss any active ModalScreen.

        The app's action_interrupt binding intercepts escape with priority=True.
        When a modal screen is active, it should dismiss the modal rather than
        performing the default interrupt behavior.
        """

        class SimpleModal(ModalScreen[str | None]):
            """A simple test modal."""

            BINDINGS: ClassVar[list[BindingType]] = [("escape", "cancel", "Cancel")]

            def compose(self) -> ComposeResult:
                yield Static("Test Modal")

            def action_cancel(self) -> None:
                self.dismiss(None)

        class TestApp(App[None]):
            """Test app with escape -> action_interrupt binding."""

            BINDINGS: ClassVar[list[BindingType]] = [
                Binding("escape", "interrupt", "Interrupt", priority=True)
            ]

            def __init__(self) -> None:
                super().__init__()
                self.modal_dismissed = False
                self.interrupt_called = False

            def compose(self) -> ComposeResult:
                yield Container()

            def action_interrupt(self) -> None:
                if isinstance(self.screen, ModalScreen):
                    self.screen.dismiss(None)
                    return
                self.interrupt_called = True

            def show_modal(self) -> None:
                def on_dismiss(_result: str | None) -> None:
                    self.modal_dismissed = True

                self.push_screen(SimpleModal(), on_dismiss)

        app = TestApp()
        async with app.run_test() as pilot:
            app.show_modal()
            await pilot.pause()

            # Escape should dismiss the modal, not call interrupt
            await pilot.press("escape")
            await pilot.pause()

            assert app.modal_dismissed is True
            assert app.interrupt_called is False


class TestModalScreenCtrlDHandling:
    """Tests for app-level Ctrl+D behavior while modals are open."""

    async def test_ctrl_d_deletes_in_thread_selector_instead_of_quitting(self) -> None:
        """App-level quit binding should delegate to thread delete in the modal."""
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        mock_threads: list[ThreadInfo] = [
            {
                "thread_id": "thread-123",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
                "created_at": "2026-03-08T01:00:00+00:00",
                "initial_prompt": "prompt",
            }
        ]
        with patch(
            "deepagents_cli.sessions.list_threads",
            new_callable=AsyncMock,
            return_value=mock_threads,
        ):
            app = DeepAgentsApp()
            async with app.run_test() as pilot:
                await pilot.pause()

                screen = ThreadSelectorScreen(
                    current_thread=None,
                    initial_threads=mock_threads,
                )
                app.push_screen(screen)
                await pilot.pause()

                with patch.object(app, "exit") as mock_exit:
                    await pilot.press("ctrl+d")
                    await pilot.pause()
                    await pilot.pause()

                assert screen._confirming_delete is True
                mock_exit.assert_not_called()

    async def test_escape_closes_thread_delete_confirm_without_dismissing_modal(
        self,
    ) -> None:
        """Escape should close thread delete confirmation before dismissing modal."""
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        mock_threads: list[ThreadInfo] = [
            {
                "thread_id": "thread-123",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
                "created_at": "2026-03-08T01:00:00+00:00",
                "initial_prompt": "prompt",
            }
        ]
        with patch(
            "deepagents_cli.sessions.list_threads",
            new_callable=AsyncMock,
            return_value=mock_threads,
        ):
            app = DeepAgentsApp()
            async with app.run_test() as pilot:
                await pilot.pause()

                screen = ThreadSelectorScreen(
                    current_thread=None,
                    initial_threads=mock_threads,
                )
                app.push_screen(screen)
                await pilot.pause()

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()
                assert screen.is_delete_confirmation_open is True

                await pilot.press("escape")
                await pilot.pause()
                await pilot.pause()

                assert app.screen is screen
                assert screen.is_delete_confirmation_open is False

    async def test_ctrl_d_twice_quits_from_delete_confirmation(self) -> None:
        """Ctrl+D should use a double-press quit flow inside delete confirmation."""
        from deepagents_cli.widgets.thread_selector import (
            DeleteThreadConfirmScreen,
            ThreadSelectorScreen,
        )

        mock_threads: list[ThreadInfo] = [
            {
                "thread_id": "thread-123",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
                "created_at": "2026-03-08T01:00:00+00:00",
                "initial_prompt": "prompt",
            }
        ]
        with patch(
            "deepagents_cli.sessions.list_threads",
            new_callable=AsyncMock,
            return_value=mock_threads,
        ):
            app = DeepAgentsApp()
            async with app.run_test() as pilot:
                await pilot.pause()

                screen = ThreadSelectorScreen(
                    current_thread=None,
                    initial_threads=mock_threads,
                )
                app.push_screen(screen)
                await pilot.pause()

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()
                assert isinstance(app.screen, DeleteThreadConfirmScreen)

                with (
                    patch.object(app, "notify") as notify_mock,
                    patch.object(app, "exit") as exit_mock,
                ):
                    await pilot.press("ctrl+d")
                    await pilot.pause()
                    notify_mock.assert_called_once_with(
                        "Press Ctrl+D again to quit",
                        timeout=3,
                        markup=False,
                    )
                    assert app._quit_pending is True
                    exit_mock.assert_not_called()

                    await pilot.press("ctrl+d")
                    await pilot.pause()
                    exit_mock.assert_called_once()

    async def test_ctrl_c_still_works_from_delete_confirmation(self) -> None:
        """Ctrl+C should preserve the normal double-press quit flow in confirmation."""
        from deepagents_cli.widgets.thread_selector import (
            DeleteThreadConfirmScreen,
            ThreadSelectorScreen,
        )

        mock_threads: list[ThreadInfo] = [
            {
                "thread_id": "thread-123",
                "agent_name": "agent",
                "updated_at": "2026-03-08T02:00:00+00:00",
                "created_at": "2026-03-08T01:00:00+00:00",
                "initial_prompt": "prompt",
            }
        ]
        with patch(
            "deepagents_cli.sessions.list_threads",
            new_callable=AsyncMock,
            return_value=mock_threads,
        ):
            app = DeepAgentsApp()
            async with app.run_test() as pilot:
                await pilot.pause()

                screen = ThreadSelectorScreen(
                    current_thread=None,
                    initial_threads=mock_threads,
                )
                app.push_screen(screen)
                await pilot.pause()

                await pilot.press("ctrl+d")
                await pilot.pause()
                await pilot.pause()
                assert isinstance(app.screen, DeleteThreadConfirmScreen)

                with (
                    patch.object(app, "notify") as notify_mock,
                    patch.object(app, "exit") as exit_mock,
                ):
                    app.action_quit_or_interrupt()
                    notify_mock.assert_called_once_with(
                        "Press Ctrl+C again to quit",
                        timeout=3,
                        markup=False,
                    )
                    assert app._quit_pending is True
                    exit_mock.assert_not_called()

                    app.action_quit_or_interrupt()
                    exit_mock.assert_called_once()

    async def test_ctrl_d_quits_from_model_selector_with_input_focused(
        self,
    ) -> None:
        """Ctrl+D should not be swallowed or ignored in the model selector."""
        from deepagents_cli.widgets.model_selector import ModelSelectorScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = ModelSelectorScreen(
                current_model="claude-sonnet-4-5",
                current_provider="anthropic",
            )
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#model-filter", Input)
            assert filter_input.has_focus

            with patch.object(app, "exit") as exit_mock:
                await pilot.press("ctrl+d")
                await pilot.pause()

            exit_mock.assert_called_once()

    async def test_ctrl_d_quits_from_mcp_viewer(self) -> None:
        """Ctrl+D should still quit while the MCP viewer modal is open."""
        from deepagents_cli.mcp_tools import MCPServerInfo, MCPToolInfo
        from deepagents_cli.widgets.mcp_viewer import MCPViewerScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = MCPViewerScreen(
                server_info=[
                    MCPServerInfo(
                        name="filesystem",
                        transport="stdio",
                        tools=[
                            MCPToolInfo(
                                name="read_file",
                                description="Read a file",
                            )
                        ],
                    )
                ]
            )
            app.push_screen(screen)
            await pilot.pause()

            with patch.object(app, "exit") as exit_mock:
                await pilot.press("ctrl+d")
                await pilot.pause()

            exit_mock.assert_called_once()


class TestModalScreenShiftTabHandling:
    """Tests for app-level Shift+Tab behavior while modals are open."""

    async def test_shift_tab_moves_backward_in_thread_selector(self) -> None:
        """Shift+Tab should move backward in the thread selector controls."""
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = ThreadSelectorScreen(
                current_thread=None,
                initial_threads=[
                    {
                        "thread_id": "thread-123",
                        "agent_name": "agent",
                        "updated_at": "2026-03-08T02:00:00+00:00",
                        "created_at": "2026-03-08T01:00:00+00:00",
                        "initial_prompt": "prompt",
                    }
                ],
            )
            app.push_screen(screen)
            await pilot.pause()

            assert app._auto_approve is False
            filter_input = screen.query_one("#thread-filter", Input)
            sort_switch = screen.query_one("#thread-sort-toggle", Checkbox)

            await pilot.press("tab")
            await pilot.pause()
            assert sort_switch.has_focus

            await pilot.press("shift+tab")
            await pilot.pause()

            assert filter_input.has_focus
            assert app._auto_approve is False


class TestModalScreenCtrlCHandling:
    """Tests for app-level Ctrl+C behavior while modals are open."""

    async def test_ctrl_c_quits_from_thread_selector_with_input_focused(
        self,
    ) -> None:
        """Ctrl+C should reach the app even when the thread filter has focus."""
        from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = ThreadSelectorScreen(
                current_thread=None,
                initial_threads=[
                    {
                        "thread_id": "thread-123",
                        "agent_name": "agent",
                        "updated_at": "2026-03-08T02:00:00+00:00",
                        "created_at": "2026-03-08T01:00:00+00:00",
                        "initial_prompt": "prompt",
                    }
                ],
            )
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#thread-filter", Input)
            assert filter_input.has_focus

            with (
                patch.object(app, "notify") as notify_mock,
                patch.object(app, "exit") as exit_mock,
                patch.object(app, "set_timer"),
            ):
                await pilot.press("ctrl+c")
                await pilot.pause()
                notify_mock.assert_called_once_with(
                    "Press Ctrl+C again to quit",
                    timeout=3,
                    markup=False,
                )
                assert app._quit_pending is True
                exit_mock.assert_not_called()

                await pilot.press("ctrl+c")
                await pilot.pause()
                exit_mock.assert_called_once()

    async def test_ctrl_c_quits_from_model_selector_with_input_focused(
        self,
    ) -> None:
        """Ctrl+C should not be swallowed by the model filter input."""
        from deepagents_cli.widgets.model_selector import ModelSelectorScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = ModelSelectorScreen(
                current_model="claude-sonnet-4-5",
                current_provider="anthropic",
            )
            app.push_screen(screen)
            await pilot.pause()

            filter_input = screen.query_one("#model-filter", Input)
            assert filter_input.has_focus

            with (
                patch.object(app, "notify") as notify_mock,
                patch.object(app, "exit") as exit_mock,
                patch.object(app, "set_timer"),
            ):
                await pilot.press("ctrl+c")
                await pilot.pause()
                notify_mock.assert_called_once_with(
                    "Press Ctrl+C again to quit",
                    timeout=3,
                    markup=False,
                )
                assert app._quit_pending is True
                exit_mock.assert_not_called()

                await pilot.press("ctrl+c")
                await pilot.pause()
                exit_mock.assert_called_once()

    async def test_ctrl_c_quits_from_mcp_viewer(self) -> None:
        """Ctrl+C should still trigger app quit flow while the MCP modal is open."""
        from deepagents_cli.mcp_tools import MCPServerInfo, MCPToolInfo
        from deepagents_cli.widgets.mcp_viewer import MCPViewerScreen

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            screen = MCPViewerScreen(
                server_info=[
                    MCPServerInfo(
                        name="filesystem",
                        transport="stdio",
                        tools=[
                            MCPToolInfo(
                                name="read_file",
                                description="Read a file",
                            )
                        ],
                    )
                ]
            )
            app.push_screen(screen)
            await pilot.pause()

            with (
                patch.object(app, "notify") as notify_mock,
                patch.object(app, "exit") as exit_mock,
                patch.object(app, "set_timer"),
            ):
                await pilot.press("ctrl+c")
                await pilot.pause()
                notify_mock.assert_called_once_with(
                    "Press Ctrl+C again to quit",
                    timeout=3,
                    markup=False,
                )
                assert app._quit_pending is True
                exit_mock.assert_not_called()

                await pilot.press("ctrl+c")
                await pilot.pause()
                exit_mock.assert_called_once()


class TestMountMessageNoMatches:
    """Test _mount_message resilience when #messages container is missing.

    When a user interrupts a streaming response, the cancellation handler and
    error handler both call _mount_message. If the screen has been torn down
    (e.g. #messages container no longer exists), this should not crash.
    """

    async def test_mount_message_no_crash_when_messages_missing(self) -> None:
        """_mount_message should not raise NoMatches when #messages is absent."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Verify the #messages container exists initially
            messages_container = app.query_one("#messages", Container)
            assert messages_container is not None

            # Remove #messages to simulate a torn-down screen state
            await messages_container.remove()

            # Verify it's truly gone
            with pytest.raises(NoMatches):
                app.query_one("#messages", Container)

            # _mount_message should handle the missing container gracefully
            # Before the fix, this raises NoMatches
            await app._mount_message(AppMessage("Interrupted by user"))

    async def test_mount_error_message_no_crash_when_messages_missing(
        self,
    ) -> None:
        """ErrorMessage via _mount_message should not crash without #messages.

        This is the second crash in the cascade: after _mount_message fails
        in the CancelledError handler, _run_agent_task's except clause also
        calls _mount_message(ErrorMessage(...)), which fails the same way.
        """
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            messages_container = app.query_one("#messages", Container)
            await messages_container.remove()

            # Should not raise
            await app._mount_message(ErrorMessage("Agent error: something"))


class TestQueuedMessage:
    """Test QueuedMessage dataclass."""

    def test_frozen(self) -> None:
        """QueuedMessage should be immutable."""
        msg = QueuedMessage(text="hello", mode="normal")
        with pytest.raises(AttributeError):
            msg.text = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        """QueuedMessage should store text and mode."""
        msg = QueuedMessage(text="hello", mode="shell")
        assert msg.text == "hello"
        assert msg.mode == "shell"


class TestMessageQueue:
    """Test message queue behavior in DeepAgentsApp."""

    async def test_message_queued_when_agent_running(self) -> None:
        """Messages should be queued when agent is running."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            app.post_message(ChatInput.Submitted("queued msg", "normal"))
            await pilot.pause()

            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "queued msg"
            assert app._pending_messages[0].mode == "normal"

    async def test_message_queued_while_connecting(self) -> None:
        """Messages submitted during server startup should be queued."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._connecting = True

            app.post_message(ChatInput.Submitted("early msg", "normal"))
            await pilot.pause()

            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "early msg"
            widgets = app.query(QueuedUserMessage)
            assert len(widgets) == 1

    async def test_message_blocked_while_thread_switching(self) -> None:
        """Submissions should be ignored while thread switching is in-flight."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._thread_switching = True
            with patch.object(app, "notify") as notify_mock:
                app.post_message(ChatInput.Submitted("blocked msg", "normal"))
                await pilot.pause()

                assert len(app._pending_messages) == 0
                user_msgs = app.query(UserMessage)
                assert not any(w._content == "blocked msg" for w in user_msgs)
                notify_mock.assert_called_once_with(
                    "Thread switch in progress. Please wait.",
                    severity="warning",
                    timeout=3,
                )

    async def test_queued_widget_mounted(self) -> None:
        """Queued messages should produce a QueuedUserMessage widget."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            app.post_message(ChatInput.Submitted("test msg", "normal"))
            await pilot.pause()

            widgets = app.query(QueuedUserMessage)
            assert len(widgets) == 1
            assert len(app._queued_widgets) == 1

    async def test_immediate_processing_when_agent_idle(self) -> None:
        """Messages should process immediately when agent is not running."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert not app._agent_running

            app.post_message(ChatInput.Submitted("direct msg", "normal"))
            await pilot.pause()

            # Should not be queued
            assert len(app._pending_messages) == 0
            # Should be mounted as a regular UserMessage
            user_msgs = app.query(UserMessage)
            assert any(w._content == "direct msg" for w in user_msgs)

    async def test_fifo_order(self) -> None:
        """Queued messages should process in FIFO order."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            app.post_message(ChatInput.Submitted("first", "normal"))
            await pilot.pause()
            app.post_message(ChatInput.Submitted("second", "normal"))
            await pilot.pause()

            assert len(app._pending_messages) == 2
            assert app._pending_messages[0].text == "first"
            assert app._pending_messages[1].text == "second"

    async def test_escape_pops_last_queued_message(self) -> None:
        """Escape should pop the last queued message (LIFO), not nuke all."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            mock_worker = MagicMock()
            app._agent_worker = mock_worker

            app.post_message(ChatInput.Submitted("msg1", "normal"))
            await pilot.pause()
            app.post_message(ChatInput.Submitted("msg2", "normal"))
            await pilot.pause()

            assert len(app._pending_messages) == 2

            # First ESC pops the last queued message
            app.action_interrupt()
            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "msg1"
            mock_worker.cancel.assert_not_called()

            # Second ESC pops the remaining message
            app.action_interrupt()
            assert len(app._pending_messages) == 0
            mock_worker.cancel.assert_not_called()

            # Third ESC interrupts the agent
            app.action_interrupt()
            mock_worker.cancel.assert_called_once()

    async def test_escape_restores_text_to_empty_input(self) -> None:
        """Popped message text is restored to chat input when input is empty."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            app._agent_worker = MagicMock()

            app.post_message(ChatInput.Submitted("restore me", "normal"))
            await pilot.pause()
            assert len(app._pending_messages) == 1

            chat = app._chat_input
            assert chat is not None
            # Input is empty — text should be restored
            chat.value = ""
            app.action_interrupt()
            assert chat.value == "restore me"

    async def test_escape_preserves_existing_input_text(self) -> None:
        """Popped message text is discarded when chat input already has content."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            app._agent_worker = MagicMock()

            app.post_message(ChatInput.Submitted("queued msg", "normal"))
            await pilot.pause()
            assert len(app._pending_messages) == 1

            chat = app._chat_input
            assert chat is not None
            # Input has content — should NOT be overwritten
            chat.value = "draft text"
            app.action_interrupt()
            assert chat.value == "draft text"
            assert len(app._pending_messages) == 0

    async def test_escape_pop_shows_toast(self) -> None:
        """Popping a queued message shows a differentiated toast."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            app._agent_worker = MagicMock()

            # Queue a message and pop with empty input — "moved to input"
            app._pending_messages.append(QueuedMessage(text="a", mode="normal"))
            chat = app._chat_input
            assert chat is not None
            chat.value = ""
            with patch.object(app, "notify") as mock_notify:
                app.action_interrupt()
                mock_notify.assert_called_once_with(
                    "Queued message moved to input", timeout=2
                )

            # Queue another and pop with non-empty input — "discarded"
            app._pending_messages.append(QueuedMessage(text="b", mode="normal"))
            chat.value = "existing"
            with patch.object(app, "notify") as mock_notify:
                app.action_interrupt()
                mock_notify.assert_called_once_with(
                    "Queued message discarded (input not empty)", timeout=3
                )

    async def test_escape_pop_single_then_interrupt(self) -> None:
        """Single queued message is popped, then next ESC interrupts agent."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            mock_worker = MagicMock()
            app._agent_worker = mock_worker

            app._pending_messages.append(QueuedMessage(text="only", mode="normal"))
            app._queued_widgets.append(MagicMock())

            app.action_interrupt()
            assert len(app._pending_messages) == 0
            mock_worker.cancel.assert_not_called()

            app.action_interrupt()
            mock_worker.cancel.assert_called_once()

    async def test_escape_pop_handles_widget_desync(self) -> None:
        """Pop completes gracefully when _queued_widgets is empty but messages exist."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            app._agent_worker = MagicMock()

            # Messages without corresponding widgets (desync scenario)
            app._pending_messages.append(QueuedMessage(text="orphan", mode="normal"))
            assert len(app._queued_widgets) == 0

            app.action_interrupt()
            assert len(app._pending_messages) == 0
            # No crash — method handled the desync

    async def test_interrupt_dismisses_completion_without_stopping_agent(self) -> None:
        """Esc should dismiss completion popup without interrupting the agent."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            mock_worker = MagicMock()
            app._agent_worker = mock_worker

            # Activate completion by typing "/"
            chat = app._chat_input
            assert chat is not None
            assert chat._text_area is not None
            chat._text_area.text = "/"
            await pilot.pause()
            assert chat._current_suggestions  # completion is active

            # Esc should dismiss completion, NOT cancel the agent
            app.action_interrupt()

            assert chat._current_suggestions == []
            mock_worker.cancel.assert_not_called()
            assert app._agent_running is True

    async def test_interrupt_falls_through_when_no_completion(self) -> None:
        """Esc should interrupt the agent when completion is not active."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            mock_worker = MagicMock()
            app._agent_worker = mock_worker

            # No completion active — interrupt should reach the agent
            chat = app._chat_input
            assert chat is not None
            assert not chat._current_suggestions

            app.action_interrupt()

            mock_worker.cancel.assert_called_once()

    async def test_queue_cleared_on_ctrl_c(self) -> None:
        """Ctrl+C should clear the message queue."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True
            mock_worker = MagicMock()
            app._agent_worker = mock_worker

            app.post_message(ChatInput.Submitted("msg", "normal"))
            await pilot.pause()

            app.action_quit_or_interrupt()

            assert len(app._pending_messages) == 0
            assert len(app._queued_widgets) == 0

    async def test_process_next_from_queue_removes_widget(self) -> None:
        """Processing a queued message should remove its ephemeral widget."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Manually enqueue
            app._pending_messages.append(QueuedMessage(text="test", mode="normal"))
            widget = QueuedUserMessage("test")
            messages = app.query_one("#messages", Container)
            await messages.mount(widget)
            app._queued_widgets.append(widget)

            await app._process_next_from_queue()
            await pilot.pause()

            assert len(app._queued_widgets) == 0

    async def test_shell_command_continues_chain(self) -> None:
        """Shell/command messages should not break the queue processing chain."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Queue a shell command followed by a normal message
            app._pending_messages.append(QueuedMessage(text="!echo hi", mode="shell"))
            app._pending_messages.append(
                QueuedMessage(text="hello agent", mode="normal")
            )

            await app._process_next_from_queue()
            await pilot.pause()
            await pilot.pause()

            # The shell command should have been processed and the normal
            # message should also have been picked up (mounted as UserMessage)
            user_msgs = app.query(UserMessage)
            assert any(w._content == "hello agent" for w in user_msgs)


class TestAskUserLifecycle:
    """Tests for ask_user widget cleanup flows."""

    async def test_request_ask_user_timeout_cleans_old_widget(self) -> None:
        """Timeout cleanup should cancel then remove the previous widget."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            old_widget = MagicMock()
            old_widget.remove = AsyncMock()
            app._pending_ask_user_widget = old_widget

            with patch("deepagents_cli.app._monotonic", side_effect=[0.0, 31.0]):
                await app._request_ask_user([{"question": "Name?", "type": "text"}])

            old_widget.action_cancel.assert_called_once()
            old_widget.remove.assert_awaited_once()
            assert old_widget.mock_calls[:2] == [call.action_cancel(), call.remove()]
            assert app._pending_ask_user_widget is not old_widget

    async def test_on_ask_user_menu_answered_ignores_remove_errors(self) -> None:
        """Answered handler should swallow remove races and clear tracking."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            widget = MagicMock()
            widget.remove = AsyncMock(side_effect=RuntimeError("already removed"))
            app._pending_ask_user_widget = widget

            await app.on_ask_user_menu_answered(object())
            await pilot.pause()

            assert app._pending_ask_user_widget is None
            widget.remove.assert_awaited_once()

    async def test_on_ask_user_menu_cancelled_ignores_remove_errors(self) -> None:
        """Cancelled handler should swallow remove races and clear tracking."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            widget = MagicMock()
            widget.remove = AsyncMock(side_effect=RuntimeError("already removed"))
            app._pending_ask_user_widget = widget

            await app.on_ask_user_menu_cancelled(object())
            await pilot.pause()

            assert app._pending_ask_user_widget is None
            widget.remove.assert_awaited_once()


class TestLoadingSpinnerLifecycle:
    """Tests for loading spinner timer cleanup in app flows."""

    async def test_hide_stops_spinner_before_remove_completes(self) -> None:
        """Hiding the spinner should stop animation before DOM removal finishes."""
        app = DeepAgentsApp()
        original_remove = Widget.remove

        def delayed_remove(widget: Widget) -> Awaitable[None]:
            async def do_remove() -> None:
                await asyncio.sleep(0.3)
                await original_remove(widget)

            return do_remove()

        async with app.run_test() as pilot:
            await pilot.pause()
            await app._set_spinner("Thinking")
            await pilot.pause()

            widget = app._loading_widget
            assert widget is not None

            before_tick = widget._spinner._position
            await asyncio.sleep(0.25)
            assert widget._spinner._position != before_tick
            # Pre-condition: timer must be running before hide so the `is None`
            # assertion below isn't vacuously satisfied.
            assert widget._animation_timer is not None

            with patch.object(Widget, "remove", new=delayed_remove):
                hide_task = asyncio.create_task(app._set_spinner(None))
                # Sleep while delayed_remove is blocking (0.3s).  Check the
                # timer flag rather than a frozen position counter: the
                # Textual timer may fire one final tick before cancellation
                # on slow CI runners, making position equality racy.
                await asyncio.sleep(0.25)
                assert widget._animation_timer is None
                await hide_task

            assert app._loading_widget is None

    async def test_reposition_preserves_spinner_state(self) -> None:
        """Repositioning should reorder without disturbing widget state.

        Repositioning uses `move_child`, which keeps the same LoadingWidget
        instance mounted. Its animation timer and `_start_time` must carry
        through unchanged so the "(Ns, esc to interrupt)" hint doesn't jump
        back to 0s mid-stream.
        """
        app = DeepAgentsApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            await app._set_spinner("Thinking")
            await pilot.pause()

            widget = app._loading_widget
            assert widget is not None

            messages = app.query_one("#messages", Container)
            queued_widget = QueuedUserMessage("queued")
            await messages.mount(queued_widget, before=widget)
            app._queued_widgets.append(queued_widget)

            before_tick = widget._spinner._position
            original_timer = widget._animation_timer
            original_start_time = widget._start_time
            await asyncio.sleep(0.25)
            assert widget._spinner._position != before_tick
            assert original_timer is not None

            await app._set_spinner("Thinking")
            await pilot.pause()

            # Same instance, same timer, same start time — only DOM order changed.
            assert app._loading_widget is widget
            assert widget._animation_timer is original_timer
            assert widget._start_time == original_start_time

            children = list(messages.children)
            assert children.index(widget) == children.index(queued_widget) - 1

    async def test_reposition_moves_spinner_after_last_message_when_no_queue(
        self,
    ) -> None:
        """No queued widgets: spinner must move after the last non-spinner child.

        This is the common streaming case — an `AssistantMessage` mounts at
        the end of `#messages` (landing below the spinner), and the next
        `_set_spinner("Thinking")` call must re-anchor the spinner after it
        via `move_child(..., after=non_spinner[-1])`. Covers the no-queued
        branch of `_reposition_spinner` that the queued-widget test doesn't.
        """
        app = DeepAgentsApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            await app._set_spinner("Thinking")
            await pilot.pause()

            widget = app._loading_widget
            assert widget is not None
            assert not app._queued_widgets

            messages = app.query_one("#messages", Container)
            new_message = AppMessage("streamed")
            await messages.mount(new_message)
            await pilot.pause()

            # Sanity: mount appended at the end, so spinner is now above it.
            children = list(messages.children)
            assert children.index(widget) < children.index(new_message)

            await app._set_spinner("Thinking")
            await pilot.pause()

            # Same widget instance; spinner now sits at the end.
            assert app._loading_widget is widget
            children = list(messages.children)
            assert children[-1] is widget


class TestTraceCommand:
    """Test /trace slash command."""

    async def test_trace_opens_browser_when_configured(self) -> None:
        """Should open the LangSmith thread URL in the browser."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = TextualSessionState(thread_id="test-thread-123")

            with (
                patch(
                    "deepagents_cli.config.build_langsmith_thread_url",
                    return_value="https://smith.langchain.com/o/org/projects/p/proj/t/test-thread-123",
                ),
                patch("deepagents_cli.app.webbrowser.open") as mock_open,
            ):
                await app._handle_trace_command("/trace")
                await pilot.pause()

            mock_open.assert_called_once_with(
                "https://smith.langchain.com/o/org/projects/p/proj/t/test-thread-123"
            )
            app_msgs = app.query(AppMessage)
            assert any(  # not a URL check—just verifying the link was rendered
                "https://smith.langchain.com/o/org/projects/p/proj/t/test-thread-123"
                in str(w._content)
                for w in app_msgs
            )

    async def test_trace_shows_error_when_not_configured(self) -> None:
        """Should show configuration hint when LangSmith is not set up."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = TextualSessionState()

            with patch(
                "deepagents_cli.config.build_langsmith_thread_url",
                return_value=None,
            ):
                await app._handle_trace_command("/trace")
                await pilot.pause()

            app_msgs = app.query(AppMessage)
            assert any("LANGSMITH_API_KEY" in str(w._content) for w in app_msgs)

    async def test_trace_shows_error_when_no_session(self) -> None:
        """Should show error when there is no active session."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = None

            await app._handle_trace_command("/trace")
            await pilot.pause()

            app_msgs = app.query(AppMessage)
            assert any("No active session" in str(w._content) for w in app_msgs)

    async def test_trace_shows_link_when_browser_fails(self) -> None:
        """Should still display the URL link even if the browser cannot open."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = TextualSessionState(thread_id="test-thread-123")

            with (
                patch(
                    "deepagents_cli.config.build_langsmith_thread_url",
                    return_value="https://smith.langchain.com/t/test-thread-123",
                ),
                patch(
                    "deepagents_cli.app.webbrowser.open",
                    side_effect=webbrowser.Error("no browser"),
                ) as mock_open,
                patch("deepagents_cli.app.logger") as mock_logger,
            ):
                await app._handle_trace_command("/trace")
                # Give the executor thread time to run and fail
                await pilot.pause()
                await asyncio.sleep(0.1)

            # Browser was attempted
            mock_open.assert_called_once()
            # Exception was logged, not silently dropped
            mock_logger.debug.assert_called()
            calls = mock_logger.debug.call_args_list
            assert any("Could not open browser" in str(c) for c in calls)
            # Link still rendered despite browser failure
            app_msgs = app.query(AppMessage)
            assert any(
                "https://smith.langchain.com/t/test-thread-123" in str(w._content)
                for w in app_msgs
            )

    async def test_trace_defers_output_when_busy(self) -> None:
        """Should defer chat output when the agent is running."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = TextualSessionState(thread_id="test-thread-123")
            app._agent_running = True

            with (
                patch(
                    "deepagents_cli.config.build_langsmith_thread_url",
                    return_value="https://smith.langchain.com/t/test-thread-123",
                ),
                patch("deepagents_cli.app.webbrowser.open"),
            ):
                await app._handle_trace_command("/trace")
                await pilot.pause()

            # A QueuedUserMessage should be mounted as a placeholder
            queued = app.query(QueuedUserMessage)
            assert len(queued) == 1

            # A deferred action should be queued
            assert len(app._deferred_actions) == 1
            action = app._deferred_actions[0]
            assert action.kind == "chat_output"

            # Execute the deferred action (simulates drain after agent finishes)
            await action.execute()
            await pilot.pause()

            # Queued widget replaced by real UserMessage + AppMessage with link
            assert len(app.query(QueuedUserMessage)) == 0
            app_msgs = app.query(AppMessage)
            assert any(
                "https://smith.langchain.com/t/test-thread-123" in str(w._content)
                for w in app_msgs
            )

    async def test_trace_shows_error_when_url_build_raises(self) -> None:
        """Should show error message when build_langsmith_thread_url raises."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = TextualSessionState(thread_id="test-thread-123")

            with patch(
                "deepagents_cli.config.build_langsmith_thread_url",
                side_effect=RuntimeError("SDK error"),
            ):
                await app._handle_trace_command("/trace")
                await pilot.pause()

            app_msgs = app.query(AppMessage)
            assert any("Failed to resolve" in str(w._content) for w in app_msgs)

    async def test_trace_routed_from_handle_command(self) -> None:
        """'/trace' should be correctly routed through _handle_command."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._session_state = None

            await app._handle_command("/trace")
            await pilot.pause()

            app_msgs = app.query(AppMessage)
            assert any("No active session" in str(w._content) for w in app_msgs)


class TestRunAgentTaskMediaTracker:
    """Tests image tracker wiring from app into textual execution."""

    async def test_run_agent_task_passes_image_tracker(self) -> None:
        """`_run_agent_task` should forward the shared image tracker."""
        app = DeepAgentsApp(agent=MagicMock())
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._ui_adapter is not None

            with patch(
                "deepagents_cli.textual_adapter.execute_task_textual",
                new_callable=AsyncMock,
            ) as mock_execute:
                await app._run_agent_task("hello")

            mock_execute.assert_awaited_once()
            assert mock_execute.await_args is not None
            assert mock_execute.await_args.kwargs["image_tracker"] is app._image_tracker
            assert mock_execute.await_args.kwargs["sandbox_type"] is app._sandbox_type

    async def test_run_agent_task_finalizes_pending_tools_on_error(self) -> None:
        """Unexpected agent errors should stop/clear in-flight tool widgets."""
        app = DeepAgentsApp(agent=MagicMock())
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._ui_adapter is not None

            pending_tool = MagicMock()
            app._ui_adapter._current_tool_messages = {"tool-1": pending_tool}

            with patch(
                "deepagents_cli.textual_adapter.execute_task_textual",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ):
                await app._run_agent_task("hello")
                await pilot.pause()

            pending_tool.set_error.assert_called_once_with("Agent error: boom")
            assert app._ui_adapter._current_tool_messages == {}

            errors = app.query(ErrorMessage)
            assert any("Agent error: boom" in str(w._content) for w in errors)


class TestAppFocusRestoresChatInput:
    """Test `on_app_focus` restores chat input focus after terminal regains focus."""

    async def test_app_focus_restores_chat_input(self) -> None:
        """Regaining terminal focus should re-focus the chat input."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._chat_input is not None
            assert app._chat_input._text_area is not None

            # Blur the input to simulate focus loss from webbrowser.open
            app._chat_input._text_area.blur()
            await pilot.pause()

            app.on_app_focus()
            await pilot.pause()

            # chat_input.focus_input should have been called
            assert app._chat_input._text_area.has_focus

    async def test_app_focus_skips_when_modal_open(self) -> None:
        """Regaining focus should not steal focus from an open modal."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Push a modal screen
            from deepagents_cli.widgets.thread_selector import ThreadSelectorScreen

            screen = ThreadSelectorScreen(current_thread=None)
            app.push_screen(screen)
            await pilot.pause()

            assert isinstance(app.screen, ModalScreen)

            # on_app_focus should be a no-op with modal open
            with patch.object(app._chat_input, "focus_input") as mock_focus:
                app.on_app_focus()

            mock_focus.assert_not_called()

    async def test_app_focus_skips_when_approval_pending(self) -> None:
        """Regaining focus should not steal focus from the approval widget."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._chat_input is not None

            # Simulate a pending approval widget
            app._pending_approval_widget = MagicMock()

            with patch.object(app._chat_input, "focus_input") as mock_focus:
                app.on_app_focus()

            mock_focus.assert_not_called()


class TestPasteRouting:
    """Tests app-level paste routing when chat input focus lags."""

    async def test_on_paste_routes_unfocused_event_to_chat_input(self) -> None:
        """Unfocused paste events should be forwarded to chat input handler."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._chat_input is not None

            event = events.Paste("/tmp/photo.png")
            with (
                patch.object(app, "_is_input_focused", return_value=False),
                patch.object(
                    app._chat_input, "handle_external_paste", return_value=True
                ) as mock_handle,
                patch.object(event, "prevent_default") as mock_prevent,
                patch.object(event, "stop") as mock_stop,
            ):
                app.on_paste(event)

            mock_handle.assert_called_once_with("/tmp/photo.png")
            mock_prevent.assert_called_once()
            mock_stop.assert_called_once()

    async def test_on_paste_does_not_route_when_input_already_focused(self) -> None:
        """Focused input should keep normal TextArea paste handling path."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._chat_input is not None

            event = events.Paste("/tmp/photo.png")
            with (
                patch.object(app, "_is_input_focused", return_value=True),
                patch.object(
                    app._chat_input, "handle_external_paste", return_value=True
                ) as mock_handle,
                patch.object(event, "prevent_default") as mock_prevent,
                patch.object(event, "stop") as mock_stop,
            ):
                app.on_paste(event)

            mock_handle.assert_not_called()
            mock_prevent.assert_not_called()
            mock_stop.assert_not_called()


class TestShellCommandInterrupt:
    """Tests for interruptible shell commands (! prefix) using worker pattern."""

    async def test_escape_cancels_shell_worker(self) -> None:
        """Esc while shell command is running should cancel the worker."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            app._shell_running = True
            mock_worker = MagicMock()
            app._shell_worker = mock_worker

            app.action_interrupt()

            mock_worker.cancel.assert_called_once()
            assert len(app._pending_messages) == 0

    async def test_ctrl_c_cancels_shell_worker(self) -> None:
        """Ctrl+C while shell command is running should cancel the worker."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            app._shell_running = True
            mock_worker = MagicMock()
            app._shell_worker = mock_worker

            # Queue a message to verify it gets cleared
            app._pending_messages.append(QueuedMessage(text="queued", mode="normal"))

            app.action_quit_or_interrupt()

            mock_worker.cancel.assert_called_once()
            assert len(app._pending_messages) == 0
            assert app._quit_pending is False

    async def test_process_killed_on_cancelled_error(self) -> None:
        """CancelledError in _run_shell_task should kill the process."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.CancelledError)
            mock_proc.returncode = None
            mock_proc.pid = 12345
            mock_proc.wait = AsyncMock()

            with (
                patch(
                    "asyncio.create_subprocess_shell",
                    return_value=mock_proc,
                ),
                patch("os.killpg") as mock_killpg,
                patch("os.getpgid", return_value=12345),
                pytest.raises(asyncio.CancelledError),
            ):
                await app._run_shell_task("sleep 999")

            mock_killpg.assert_called()

    async def test_cleanup_clears_state(self) -> None:
        """_cleanup_shell_task should reset all shell state."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            app._shell_running = True
            app._shell_worker = MagicMock()
            app._shell_worker.is_cancelled = False
            app._shell_process = None

            await app._cleanup_shell_task()

            assert app._shell_process is None
            assert app._shell_running is False
            assert app._shell_worker is None

    async def test_cleanup_refreshes_git_branch(self, tmp_path: Path) -> None:
        """Verify branch refresh on shell cleanup.

        `_cleanup_shell_task` must re-resolve the branch so commands like
        `git checkout` are reflected in the footer.
        """
        import subprocess

        repo = tmp_path / "repo"
        repo.mkdir()

        def _init_repo_on_feature_branch() -> None:
            env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "t",
                "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "t",
                "GIT_COMMITTER_EMAIL": "t@t",
            }
            for args in (
                ["git", "init", "-q", "-b", "main"],
                ["git", "add", "f"],
                ["git", "commit", "-q", "-m", "init"],
                ["git", "checkout", "-q", "-b", "feature"],
            ):
                if args[1] == "add":
                    (repo / "f").write_text("x")
                subprocess.run(args, cwd=repo, env=env, check=True, capture_output=True)

        await asyncio.to_thread(_init_repo_on_feature_branch)

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            assert app._status_bar is not None
            app._cwd = str(repo)
            app._status_bar.branch = "stale"
            app._shell_running = True
            app._shell_worker = MagicMock()
            app._shell_worker.is_cancelled = False
            app._shell_process = None

            await app._cleanup_shell_task()
            await asyncio.wait_for(_wait_for_branch(app, "feature"), timeout=1)

            assert app._status_bar.branch == "feature"

    async def test_refresh_git_branch_reads_gitdir_pointer(
        self, tmp_path: Path
    ) -> None:
        """Worktree-style `.git` files should resolve to the pointed git dir."""
        repo = tmp_path / "repo"
        worktree = tmp_path / "worktree"
        nested = worktree / "src"
        git_dir = repo / ".git" / "worktrees" / "feature"

        nested.mkdir(parents=True)
        git_dir.mkdir(parents=True)
        (worktree / ".git").write_text(
            "gitdir: ../repo/.git/worktrees/feature\n",
            encoding="utf-8",
        )
        (git_dir / "HEAD").write_text(
            "ref: refs/heads/feature/nested\n",
            encoding="utf-8",
        )

        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            assert app._status_bar is not None
            app._cwd = str(nested)

            await app._refresh_git_branch()

            assert app._status_bar.branch == "feature/nested"

    async def test_refresh_git_branch_uses_inline_filesystem_fast_path(
        self, tmp_path: Path
    ) -> None:
        """Common branch reads should avoid the thread-offloaded fallback."""
        repo = tmp_path / "repo"
        git_dir = repo / ".git"
        git_dir.mkdir(parents=True)
        (git_dir / "HEAD").write_text("ref: refs/heads/feature\n", encoding="utf-8")

        app = DeepAgentsApp()
        status_bar = MagicMock()
        app._status_bar = status_bar
        app._cwd = str(repo)

        with patch(
            "deepagents_cli.app.asyncio.to_thread",
            new=AsyncMock(side_effect=AssertionError("unexpected thread hop")),
        ):
            await app._refresh_git_branch()

        assert status_bar.branch == "feature"

    async def test_cleanup_does_not_wait_for_git_branch_refresh(self) -> None:
        """Queue cleanup should not block on the subprocess fallback refresh."""
        app = DeepAgentsApp()
        refresh_started = asyncio.Event()
        release_refresh = asyncio.Event()
        drain_mock = AsyncMock()
        queue_mock = AsyncMock()

        async def block_refresh(_cwd: str) -> None:
            refresh_started.set()
            await release_refresh.wait()

        # Force the subprocess fallback path so the test can observe whether
        # cleanup awaits the background task.
        app._refresh_git_branch_subprocess_fallback = (  # type: ignore[assignment]
            block_refresh
        )
        app._maybe_drain_deferred = drain_mock  # type: ignore[assignment]
        app._process_next_from_queue = queue_mock  # type: ignore[assignment]
        app._shell_running = True
        app._shell_worker = MagicMock()
        app._shell_worker.is_cancelled = False
        app._shell_process = None

        with patch(
            "deepagents_cli.app.read_git_branch_from_filesystem",
            return_value=None,
        ):
            await app._cleanup_shell_task()
            await asyncio.wait_for(refresh_started.wait(), timeout=1)

        drain_mock.assert_awaited_once()
        queue_mock.assert_awaited_once()

        release_refresh.set()
        refresh_task = app._git_branch_refresh_task
        if refresh_task is not None:
            await refresh_task

    async def test_run_shell_task_starts_branch_refresh_before_render(self) -> None:
        """Successful shell runs should overlap branch refresh with rendering."""
        app = DeepAgentsApp()
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"hello\n", b""))
        mock_proc.returncode = 0
        mock_proc.pid = 12345
        refresh_mock = MagicMock()
        drain_mock = AsyncMock()
        queue_mock = AsyncMock()

        def assert_refresh_started(message: object) -> None:
            if type(message).__name__ == "AssistantMessage":
                assert refresh_mock.call_count == 1

        async with app.run_test() as pilot:
            await pilot.pause()

            app._schedule_git_branch_refresh = refresh_mock  # type: ignore[assignment]
            app._maybe_drain_deferred = drain_mock  # type: ignore[assignment]
            app._process_next_from_queue = queue_mock  # type: ignore[assignment]

            with (
                patch(
                    "asyncio.create_subprocess_shell",
                    return_value=mock_proc,
                ),
                patch(
                    "deepagents_cli.app.AssistantMessage.write_initial_content",
                    new=AsyncMock(),
                ),
                patch.object(
                    app,
                    "_mount_message",
                    AsyncMock(side_effect=assert_refresh_started),
                ),
            ):
                await app._run_shell_task("echo hi")

        refresh_mock.assert_called_once_with()
        drain_mock.assert_awaited_once()
        queue_mock.assert_awaited_once()

    async def test_messages_queued_during_shell(self) -> None:
        """Messages should be queued while shell command runs."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._shell_running = True

            app.post_message(ChatInput.Submitted("queued msg", "normal"))
            await pilot.pause()

            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "queued msg"

    async def test_queue_drains_after_shell_completes(self) -> None:
        """Pending messages should drain after _cleanup_shell_task."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            app._shell_running = True
            app._shell_worker = MagicMock()
            app._shell_worker.is_cancelled = False
            app._shell_process = None

            # Enqueue a message
            app._pending_messages.append(
                QueuedMessage(text="after shell", mode="normal")
            )

            await app._cleanup_shell_task()
            await pilot.pause()

            # Message should have been processed (mounted as UserMessage)
            user_msgs = app.query(UserMessage)
            assert any(w._content == "after shell" for w in user_msgs)

    async def test_interrupted_shows_message(self) -> None:
        """Cancelled worker should show 'Command interrupted'."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            app._shell_running = True
            mock_worker = MagicMock()
            mock_worker.is_cancelled = True
            app._shell_worker = mock_worker
            # Process still set means it was interrupted mid-flight
            mock_proc = MagicMock()
            mock_proc.returncode = None
            app._shell_process = mock_proc

            await app._cleanup_shell_task()
            await pilot.pause()

            app_msgs = app.query(AppMessage)
            assert any("Command interrupted" in str(w._content) for w in app_msgs)

    async def test_timeout_kills_and_shows_error(self) -> None:
        """Timeout in _run_shell_task should kill process and show error."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_proc.returncode = None
            mock_proc.pid = 12345
            mock_proc.wait = AsyncMock()

            with (
                patch(
                    "asyncio.create_subprocess_shell",
                    return_value=mock_proc,
                ),
                patch("os.killpg"),
                patch("os.getpgid", return_value=12345),
            ):
                await app._run_shell_task("sleep 999")
                await pilot.pause()

            assert app._shell_process is None
            error_msgs = app.query(ErrorMessage)
            assert any("timed out" in w._content for w in error_msgs)

    async def test_posix_killpg_called(self) -> None:
        """On POSIX, _kill_shell_process should use os.killpg with SIGTERM."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            mock_proc = AsyncMock()
            mock_proc.returncode = None
            mock_proc.pid = 42
            mock_proc.wait = AsyncMock()
            app._shell_process = mock_proc

            with (
                patch("deepagents_cli.app.sys") as mock_sys,
                patch("os.killpg") as mock_killpg,
                patch("os.getpgid", return_value=42) as mock_getpgid,
            ):
                mock_sys.platform = "linux"
                await app._kill_shell_process()

            mock_getpgid.assert_called_once_with(42)
            mock_killpg.assert_called_once_with(42, signal.SIGTERM)

    async def test_sigkill_escalation(self) -> None:
        """SIGKILL should be sent when SIGTERM times out."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            mock_proc = AsyncMock()
            mock_proc.returncode = None
            mock_proc.pid = 42
            mock_proc.wait = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_proc.kill = MagicMock()
            app._shell_process = mock_proc

            with (
                patch("deepagents_cli.app.sys") as mock_sys,
                patch("os.killpg") as mock_killpg,
                patch("os.getpgid", return_value=42),
            ):
                mock_sys.platform = "linux"
                await app._kill_shell_process()

            # First call: SIGTERM, second call: SIGKILL
            assert mock_killpg.call_count == 2
            mock_killpg.assert_any_call(42, signal.SIGTERM)
            mock_killpg.assert_any_call(42, signal.SIGKILL)

    async def test_no_op_when_no_shell_running(self) -> None:
        """Ctrl+C with no shell command running should fall through to quit hint."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            assert not app._shell_running
            app.action_quit_or_interrupt()

            assert app._quit_pending is True

    async def test_oserror_shows_error_message(self) -> None:
        """OSError from create_subprocess_shell should display error."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            with patch(
                "asyncio.create_subprocess_shell",
                side_effect=OSError("Permission denied"),
            ):
                await app._run_shell_task("forbidden")
                await pilot.pause()

            assert app._shell_process is None
            error_msgs = app.query(ErrorMessage)
            assert any("Permission denied" in w._content for w in error_msgs)

    async def test_handle_shell_command_sets_running_state(self) -> None:
        """_handle_shell_command should set _shell_running and spawn worker."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            with patch.object(app, "run_worker") as mock_rw:
                mock_rw.return_value = MagicMock()
                await app._handle_shell_command("echo hi")

            assert app._shell_running is True
            assert app._shell_worker is not None
            mock_rw.assert_called_once()
            # Close the unawaited coroutine to suppress RuntimeWarning
            coro = mock_rw.call_args[0][0]
            coro.close()

    async def test_kill_noop_when_already_exited(self) -> None:
        """_kill_shell_process should no-op if process already exited."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.pid = 42
            app._shell_process = mock_proc

            with patch("os.killpg") as mock_killpg:
                await app._kill_shell_process()

            mock_killpg.assert_not_called()
            mock_proc.terminate.assert_not_called()

    async def test_end_to_end_escape_during_shell(self) -> None:
        """Esc during a running shell worker should cancel execution."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Simulate a running shell state with a mock worker
            app._shell_running = True
            mock_worker = MagicMock()
            app._shell_worker = mock_worker

            await pilot.press("escape")
            await pilot.pause()

            mock_worker.cancel.assert_called_once()


class TestAppArgumentHints:
    """Full-app regressions for slash-command argument hints."""

    async def test_hint_clears_after_command_submission(self) -> None:
        """Submitting a slash command clears the inline argument hint."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("/")
            await pilot.pause()
            await pilot.pause()
            chat._text_area.insert("remember ")
            await pilot.pause()

            assert chat.mode == "command"
            assert chat._text_area.argument_hint == "[context]"
            assert chat._text_area.render_line(0).text.rstrip() == "remember [context]"

            await pilot.press("enter")
            await pilot.pause()

            assert chat.mode == "normal"
            assert chat._text_area.text == ""
            assert chat._text_area.argument_hint == ""

    async def test_hint_clears_after_backspace_mode_exit(self) -> None:
        """Backspace mode exit clears the hint in the mounted app."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat = app.query_one(ChatInput)
            assert chat._text_area is not None

            chat._text_area.insert("/")
            await pilot.pause()
            await pilot.pause()
            chat._text_area.insert("remember ")
            await pilot.pause()

            assert chat.mode == "command"
            assert chat._text_area.argument_hint == "[context]"
            assert chat._text_area.render_line(0).text.rstrip() == "remember [context]"

            for _ in "remember ":
                await pilot.press("left")
            await pilot.pause()
            assert chat._text_area.cursor_location == (0, 0)
            assert chat._text_area.render_line(0).text.rstrip() == "remember [context]"

            await pilot.press("backspace")
            await pilot.pause()

            assert chat.mode == "normal"
            assert chat._text_area.text == "remember "
            assert chat._text_area.argument_hint == ""


class TestInterruptApprovalPriority:
    """Tests for escape interrupt priority when HITL approval is pending."""

    async def test_escape_rejects_approval_before_canceling_worker(self) -> None:
        """When both HITL approval and worker are active, reject approval first."""
        app = DeepAgentsApp()
        approval = MagicMock()
        worker = MagicMock()

        async with app.run_test() as pilot:
            await pilot.pause()

            app._pending_approval_widget = approval
            app._agent_running = True
            app._agent_worker = worker

            app.action_interrupt()

        approval.action_select_reject.assert_called_once()
        worker.cancel.assert_not_called()

    async def test_escape_pops_queue_before_cancelling_worker(self) -> None:
        """Escape pops queued messages (LIFO) before cancelling the worker."""
        app = DeepAgentsApp()
        worker = MagicMock()
        queued_w1 = MagicMock()
        queued_w2 = MagicMock()

        async with app.run_test() as pilot:
            await pilot.pause()

            app._pending_approval_widget = None
            app._agent_running = True
            app._agent_worker = worker
            app._pending_messages.append(QueuedMessage(text="q1", mode="normal"))
            app._pending_messages.append(QueuedMessage(text="q2", mode="normal"))
            app._queued_widgets.append(queued_w1)
            app._queued_widgets.append(queued_w2)

            # First ESC pops last queued message, does not cancel worker
            app.action_interrupt()
            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "q1"
            queued_w2.remove.assert_called_once()
            queued_w1.remove.assert_not_called()
            worker.cancel.assert_not_called()

            # Second ESC pops remaining message
            app.action_interrupt()
            assert len(app._pending_messages) == 0
            queued_w1.remove.assert_called_once()
            worker.cancel.assert_not_called()

            # Third ESC finally cancels the worker
            app.action_interrupt()
            worker.cancel.assert_called_once()

    async def test_escape_rejects_approval_when_no_worker(self) -> None:
        """Approval rejection works even without an active agent worker."""
        app = DeepAgentsApp()
        approval = MagicMock()

        async with app.run_test() as pilot:
            await pilot.pause()

            app._pending_approval_widget = approval
            app._agent_running = False
            app._agent_worker = None

            app.action_interrupt()

        approval.action_select_reject.assert_called_once()

    async def test_ctrl_c_rejects_approval_before_canceling_worker(self) -> None:
        """Ctrl+C should also reject approval before canceling worker."""
        app = DeepAgentsApp()
        approval = MagicMock()
        worker = MagicMock()

        async with app.run_test() as pilot:
            await pilot.pause()

            app._pending_approval_widget = approval
            app._agent_running = True
            app._agent_worker = worker

            app.action_quit_or_interrupt()

        approval.action_select_reject.assert_called_once()
        worker.cancel.assert_not_called()
        assert app._quit_pending is False


class TestIsUserTyping:
    """Unit tests for `_is_user_typing()` threshold logic."""

    def test_returns_false_when_never_typed(self) -> None:
        """Should return False if _last_typed_at is None."""
        app = DeepAgentsApp()
        assert app._is_user_typing() is False

    def test_returns_true_within_threshold(self) -> None:
        """Should return True right after a keystroke."""
        app = DeepAgentsApp()
        app._last_typed_at = time.monotonic()
        assert app._is_user_typing() is True

    def test_returns_false_after_threshold(self) -> None:
        """Should return False once the idle threshold has elapsed."""
        app = DeepAgentsApp()
        app._last_typed_at = time.monotonic() - (_TYPING_IDLE_THRESHOLD_SECONDS + 0.1)
        assert app._is_user_typing() is False

    def test_boundary_just_within_threshold(self) -> None:
        """Should return True when just inside the threshold window."""
        app = DeepAgentsApp()
        app._last_typed_at = time.monotonic() - (_TYPING_IDLE_THRESHOLD_SECONDS - 0.1)
        assert app._is_user_typing() is True


class TestRequestApprovalBranching:
    """_request_approval should show a placeholder when the user is typing."""

    async def test_placeholder_mounted_when_typing(self) -> None:
        """If the user is typing, a Static placeholder is mounted instead of menu."""
        app = DeepAgentsApp(agent=MagicMock())
        # Simulate recent typing
        app._last_typed_at = time.monotonic()

        mounted_classes: list[str] = []

        async def fake_mount_before_queued(  # noqa: RUF029
            _container: object, widget: object
        ) -> None:
            if isinstance(widget, Static):
                mounted_classes.append(" ".join(widget.classes))

        app._mount_before_queued = fake_mount_before_queued  # type: ignore[assignment]

        # Prevent actual worker from running; we just want to check branching.
        run_worker_calls: list[object] = []

        def _stub_worker(coro: object, **_: object) -> MagicMock:
            # Consume the coroutine immediately to suppress RuntimeWarning.
            if inspect.iscoroutine(coro):
                coro.close()
            run_worker_calls.append(coro)
            return MagicMock()

        app.run_worker = _stub_worker  # type: ignore[method-assign]

        dummy_container = MagicMock()
        app.query_one = MagicMock(return_value=dummy_container)  # type: ignore[method-assign]

        action_requests = [
            {"name": "write_file", "args": {"path": "/tmp/x.txt", "content": "hi"}}
        ]
        future = asyncio.get_running_loop().create_future()

        with patch.object(asyncio, "get_running_loop") as mock_loop:
            mock_loop.return_value.create_future.return_value = future
            returned = await app._request_approval(action_requests, None)

        assert returned is future
        assert any("approval-placeholder" in cls for cls in mounted_classes), (
            f"Expected 'approval-placeholder' in mounted widget classes,"
            f" got {mounted_classes}"
        )
        assert len(run_worker_calls) == 1, (
            "run_worker should have been called once for the deferred swap"
        )

    async def test_placeholder_mount_failure_falls_back_to_menu(self) -> None:
        """If placeholder mount fails, the ApprovalMenu is shown directly."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        app = DeepAgentsApp(agent=MagicMock())
        app._last_typed_at = time.monotonic()

        mounted_types: list[type] = []

        call_count = 0

        async def failing_then_ok_mount(  # noqa: RUF029
            _container: object, widget: object
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "simulated mount failure"
                raise RuntimeError(msg)
            mounted_types.append(type(widget))

        app._mount_before_queued = failing_then_ok_mount  # type: ignore[assignment]
        app.call_after_refresh = MagicMock()  # type: ignore[method-assign]

        dummy_container = MagicMock()
        app.query_one = MagicMock(return_value=dummy_container)  # type: ignore[method-assign]

        action_requests = [
            {"name": "write_file", "args": {"path": "/tmp/z.txt", "content": "hi"}}
        ]
        future = asyncio.get_running_loop().create_future()

        with patch.object(asyncio, "get_running_loop") as mock_loop:
            mock_loop.return_value.create_future.return_value = future
            returned = await app._request_approval(action_requests, None)

        assert returned is future
        # Placeholder mount (1st call) fails, fallback menu mount (2nd call)
        # succeeds. The menu is now mounted and the future awaits user input.
        assert ApprovalMenu in mounted_types, (
            f"Expected ApprovalMenu fallback mount, got {mounted_types}"
        )

    async def test_menu_mounted_directly_when_not_typing(self) -> None:
        """If the user is NOT typing, the ApprovalMenu is mounted directly."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        app = DeepAgentsApp(agent=MagicMock())
        app._last_typed_at = None

        mounted_types: list[type] = []

        async def fake_mount_before_queued(  # noqa: RUF029
            _container: object, widget: object
        ) -> None:
            mounted_types.append(type(widget))

        app._mount_before_queued = fake_mount_before_queued  # type: ignore[assignment]
        app.call_after_refresh = MagicMock()  # type: ignore[method-assign]

        dummy_container = MagicMock()
        app.query_one = MagicMock(return_value=dummy_container)  # type: ignore[method-assign]

        action_requests = [
            {"name": "write_file", "args": {"path": "/tmp/y.txt", "content": "hi"}}
        ]
        future = asyncio.get_running_loop().create_future()

        with patch.object(asyncio, "get_running_loop") as mock_loop:
            mock_loop.return_value.create_future.return_value = future
            returned = await app._request_approval(action_requests, None)

        assert returned is future
        assert ApprovalMenu in mounted_types, (
            f"Expected ApprovalMenu to be mounted, got {mounted_types}"
        )


class TestDeferredShowApproval:
    """_deferred_show_approval should swap placeholder once idle."""

    async def test_swaps_placeholder_for_menu_after_idle(self) -> None:
        """Once typing stops, placeholder is removed and menu is mounted."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        app = DeepAgentsApp(agent=MagicMock())
        app._last_typed_at = time.monotonic()

        placeholder = MagicMock(spec=Static)
        placeholder.is_attached = True
        remove_called = False

        async def fake_remove() -> None:  # noqa: RUF029
            nonlocal remove_called
            remove_called = True

        placeholder.remove = fake_remove

        action_requests = [{"name": "write_file", "args": {}}]
        future = asyncio.get_running_loop().create_future()
        menu = ApprovalMenu(action_requests[0])
        menu.set_future(future)

        mount_called = False

        async def fake_mount_approval(  # noqa: RUF029
            m: ApprovalMenu,  # noqa: ARG001
            f: asyncio.Future[dict[str, str]],  # noqa: ARG001
        ) -> None:
            nonlocal mount_called
            mount_called = True

        app._mount_approval_widget = fake_mount_approval  # type: ignore[method-assign]

        async def stop_typing() -> None:
            await asyncio.sleep(0.05)
            app._last_typed_at = None

        typing_task = asyncio.create_task(stop_typing())
        await app._deferred_show_approval(placeholder, menu, future)
        await typing_task

        assert remove_called, "placeholder.remove() should have been called"
        assert mount_called, "_mount_approval_widget should have been called"

    async def test_bails_if_placeholder_detached_and_cancels_future(self) -> None:
        """If placeholder is detached, worker cancels the future and exits."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        app = DeepAgentsApp(agent=MagicMock())
        app._last_typed_at = None

        placeholder = MagicMock(spec=Static)
        placeholder.is_attached = False

        mount_called = False

        async def fake_mount_approval(  # noqa: RUF029
            m: ApprovalMenu,  # noqa: ARG001
            f: asyncio.Future[dict[str, str]],  # noqa: ARG001
        ) -> None:
            nonlocal mount_called
            mount_called = True

        app._mount_approval_widget = fake_mount_approval  # type: ignore[method-assign]

        action_requests = [{"name": "shell", "args": {"command": "ls"}}]
        future = asyncio.get_running_loop().create_future()
        menu = ApprovalMenu(action_requests[0])
        menu.set_future(future)

        await app._deferred_show_approval(placeholder, menu, future)

        assert not mount_called, "_mount_approval_widget should NOT have been called"
        assert future.cancelled(), "future should have been cancelled"
        assert app._pending_approval_widget is None
        assert app._approval_placeholder is None

    async def test_timeout_shows_approval_after_deadline(self) -> None:
        """If the user types continuously past the deadline, menu is shown anyway."""
        from deepagents_cli.widgets.approval import ApprovalMenu

        app = DeepAgentsApp(agent=MagicMock())
        # Simulate user typing *forever* by keeping _last_typed_at fresh
        app._last_typed_at = time.monotonic()

        placeholder = MagicMock(spec=Static)
        placeholder.is_attached = True

        remove_called = False

        async def fake_remove() -> None:  # noqa: RUF029
            nonlocal remove_called
            remove_called = True

        placeholder.remove = fake_remove

        mount_called = False

        async def fake_mount_approval(  # noqa: RUF029
            m: ApprovalMenu,  # noqa: ARG001
            f: asyncio.Future[dict[str, str]],  # noqa: ARG001
        ) -> None:
            nonlocal mount_called
            mount_called = True

        app._mount_approval_widget = fake_mount_approval  # type: ignore[method-assign]

        action_requests = [{"name": "write_file", "args": {}}]
        future = asyncio.get_running_loop().create_future()
        menu = ApprovalMenu(action_requests[0])
        menu.set_future(future)

        # Patch the timeout to be tiny so the test doesn't actually wait 30s
        with patch("deepagents_cli.app._DEFERRED_APPROVAL_TIMEOUT_SECONDS", 0.05):
            await app._deferred_show_approval(placeholder, menu, future)

        assert remove_called, "placeholder.remove() should have been called"
        assert mount_called, (
            "_mount_approval_widget should have been called after timeout"
        )


class TestOnChatInputTyping:
    """on_chat_input_typing should set _last_typed_at."""

    def test_sets_last_typed_at(self) -> None:
        """Calling on_chat_input_typing records a recent monotonic time."""
        app = DeepAgentsApp()
        assert app._last_typed_at is None

        event = MagicMock()
        before = time.monotonic()
        app.on_chat_input_typing(event)
        after = time.monotonic()

        assert app._last_typed_at is not None
        assert before <= app._last_typed_at <= after

    def test_updates_on_subsequent_calls(self) -> None:
        """Each call should update _last_typed_at to a newer timestamp."""
        app = DeepAgentsApp()
        event = MagicMock()

        app.on_chat_input_typing(event)
        first = app._last_typed_at

        app.on_chat_input_typing(event)
        second = app._last_typed_at

        assert second is not None
        assert first is not None
        assert second >= first


class TestOnApprovalMenuDecidedCleanup:
    """on_approval_menu_decided should defensively clean up placeholders."""

    async def test_removes_attached_placeholder(self) -> None:
        """An attached placeholder should be removed and nulled."""
        app = DeepAgentsApp(agent=MagicMock())

        placeholder = MagicMock(spec=Static)
        placeholder.is_attached = True
        remove_called = False

        async def fake_remove() -> None:  # noqa: RUF029
            nonlocal remove_called
            remove_called = True

        placeholder.remove = fake_remove
        app._approval_placeholder = placeholder
        app._pending_approval_widget = None

        event = MagicMock()
        app._chat_input = None
        await app.on_approval_menu_decided(event)

        assert remove_called
        assert app._approval_placeholder is None

    async def test_nulls_detached_placeholder(self) -> None:
        """A detached placeholder should be nulled without calling remove."""
        app = DeepAgentsApp(agent=MagicMock())

        placeholder = MagicMock(spec=Static)
        placeholder.is_attached = False
        app._approval_placeholder = placeholder
        app._pending_approval_widget = None

        event = MagicMock()
        app._chat_input = None
        await app.on_approval_menu_decided(event)

        assert app._approval_placeholder is None
        placeholder.remove.assert_not_called()

    async def test_no_placeholder_works_normally(self) -> None:
        """When no placeholder exists, handler proceeds without error."""
        app = DeepAgentsApp(agent=MagicMock())
        app._approval_placeholder = None
        app._pending_approval_widget = None

        event = MagicMock()
        app._chat_input = None
        await app.on_approval_menu_decided(event)

        assert app._approval_placeholder is None


class TestActionOpenEditor:
    """Tests for the external editor action."""

    async def test_updates_text_on_successful_edit(self) -> None:
        app = DeepAgentsApp(agent=MagicMock())
        text_area = MagicMock()
        text_area.text = "original"
        chat_input = MagicMock()
        chat_input._text_area = text_area
        app._chat_input = chat_input

        with (
            patch.object(app, "suspend"),
            patch("deepagents_cli.editor.open_in_editor", return_value="edited"),
        ):
            await app.action_open_editor()

        assert text_area.text == "edited"
        chat_input.focus_input.assert_called_once()

    async def test_no_update_when_editor_returns_none(self) -> None:
        app = DeepAgentsApp(agent=MagicMock())
        text_area = MagicMock()
        text_area.text = "original"
        chat_input = MagicMock()
        chat_input._text_area = text_area
        app._chat_input = chat_input

        with (
            patch.object(app, "suspend"),
            patch("deepagents_cli.editor.open_in_editor", return_value=None),
        ):
            await app.action_open_editor()

        assert text_area.text == "original"
        chat_input.focus_input.assert_called_once()

    async def test_early_return_when_chat_input_is_none(self) -> None:
        app = DeepAgentsApp(agent=MagicMock())
        app._chat_input = None

        # Should not raise
        await app.action_open_editor()

    async def test_early_return_when_text_area_is_none(self) -> None:
        app = DeepAgentsApp(agent=MagicMock())
        chat_input = MagicMock()
        chat_input._text_area = None
        app._chat_input = chat_input

        await app.action_open_editor()

    async def test_notifies_on_exception(self) -> None:
        app = DeepAgentsApp(agent=MagicMock())
        text_area = MagicMock()
        text_area.text = ""
        chat_input = MagicMock()
        chat_input._text_area = text_area
        app._chat_input = chat_input

        with (
            patch.object(app, "suspend"),
            patch(
                "deepagents_cli.editor.open_in_editor",
                side_effect=RuntimeError("boom"),
            ),
            patch.object(app, "notify") as mock_notify,
        ):
            await app.action_open_editor()

        mock_notify.assert_called_once()
        assert "failed" in mock_notify.call_args[0][0].lower()
        chat_input.focus_input.assert_called_once()


class TestEditorSlashCommand:
    """Test that /editor dispatches to action_open_editor."""

    async def test_editor_command_calls_action(self) -> None:
        app = DeepAgentsApp(agent=MagicMock())
        with patch.object(app, "action_open_editor", new_callable=AsyncMock) as mock:
            app._chat_input = MagicMock()
            await app._handle_command("/editor")
        mock.assert_awaited_once()


class TestFetchThreadHistoryData:
    """Verify _fetch_thread_history_data handles server-mode resume scenarios."""

    async def test_dict_messages_converted_to_message_objects(self) -> None:
        """Dict-based messages from server mode are deserialized before conversion."""
        from deepagents_cli.widgets.message_store import MessageData, MessageType

        state = MagicMock()
        state.values = {
            "messages": [
                {"type": "human", "content": "hello", "id": "h1"},
                {
                    "type": "ai",
                    "content": "Hi there!",
                    "id": "a1",
                    "tool_calls": [],
                },
            ],
        }

        mock_agent = AsyncMock()
        mock_agent.aget_state.return_value = state

        app = DeepAgentsApp(agent=mock_agent, thread_id="t-1")
        payload = await app._fetch_thread_history_data("t-1")

        assert len(payload.messages) == 2
        assert isinstance(payload.messages[0], MessageData)
        assert payload.messages[0].type == MessageType.USER
        assert payload.messages[0].content == "hello"
        assert isinstance(payload.messages[1], MessageData)
        assert payload.messages[1].type == MessageType.ASSISTANT
        assert payload.messages[1].content == "Hi there!"

    async def test_server_mode_falls_back_to_checkpointer(self) -> None:
        """When the server returns empty state, read SQLite checkpointer directly."""
        from langchain_core.messages import AIMessage, HumanMessage

        from deepagents_cli.remote_client import RemoteAgent
        from deepagents_cli.widgets.message_store import MessageData, MessageType

        # Server returns empty state (fresh restart, thread not loaded)
        empty_state = MagicMock()
        empty_state.values = {}

        # spec=RemoteAgent so _remote_agent() isinstance check passes
        mock_agent = MagicMock(spec=RemoteAgent)
        mock_agent.aget_state = AsyncMock(return_value=empty_state)

        app = DeepAgentsApp(agent=mock_agent, thread_id="t-1")

        # Patch the checkpointer fallback to return messages
        checkpointer_msgs = [
            HumanMessage(content="hello", id="h1"),
            AIMessage(content="world", id="a1"),
        ]
        with patch.object(
            DeepAgentsApp,
            "_read_channel_values_from_checkpointer",
            return_value={"messages": checkpointer_msgs},
        ):
            payload = await app._fetch_thread_history_data("t-1")

        assert len(payload.messages) == 2
        assert payload.messages[0].type == MessageType.USER
        assert payload.messages[0].content == "hello"
        assert payload.messages[1].type == MessageType.ASSISTANT
        assert payload.messages[1].content == "world"

    async def test_server_mode_fallback_includes_context_tokens(self) -> None:
        """Server-mode fallback should merge `_context_tokens` from the checkpointer."""
        from langchain_core.messages import HumanMessage

        from deepagents_cli.remote_client import RemoteAgent
        from deepagents_cli.widgets.message_store import MessageType

        empty_state = MagicMock()
        empty_state.values = {}

        mock_agent = MagicMock(spec=RemoteAgent)
        mock_agent.aget_state = AsyncMock(return_value=empty_state)

        app = DeepAgentsApp(agent=mock_agent, thread_id="t-1")

        checkpointer_data = {
            "messages": [HumanMessage(content="hi", id="h1")],
            "_context_tokens": 5000,
        }
        with patch.object(
            DeepAgentsApp,
            "_read_channel_values_from_checkpointer",
            return_value=checkpointer_data,
        ):
            payload = await app._fetch_thread_history_data("t-1")

        assert payload.context_tokens == 5000
        assert len(payload.messages) == 1
        assert payload.messages[0].type == MessageType.USER


class TestRemoteAgent:
    """Tests for DeepAgentsApp._remote_agent()."""

    def test_returns_instance_with_remote_agent(self) -> None:
        from deepagents_cli.remote_client import RemoteAgent

        app = DeepAgentsApp()
        agent = RemoteAgent("http://test:0")
        app._agent = agent
        assert app._remote_agent() is agent

    def test_none_when_agent_is_none(self) -> None:
        app = DeepAgentsApp()
        assert app._remote_agent() is None

    def test_none_with_non_remote_agent(self) -> None:
        """Local Pregel-like agent returns None."""
        app = DeepAgentsApp()
        app._agent = MagicMock()
        assert app._remote_agent() is None

    def test_none_with_mock_spec_pregel(self) -> None:
        """MagicMock without RemoteAgent spec returns None."""
        app = DeepAgentsApp()
        app._agent = MagicMock(spec=[])
        assert app._remote_agent() is None


class TestSlashCommandBypass:
    """Test that certain slash commands bypass the queue gate."""

    async def test_quit_bypasses_queue_when_agent_running(self) -> None:
        """/quit should exit immediately even when agent is running."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            with patch.object(app, "exit") as exit_mock:
                app.post_message(ChatInput.Submitted("/quit", "command"))
                await pilot.pause()

            exit_mock.assert_called_once()
            assert len(app._pending_messages) == 0

    async def test_quit_bypasses_queue_when_connecting(self) -> None:
        """/quit should exit immediately even when connecting."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._connecting = True

            with patch.object(app, "exit") as exit_mock:
                app.post_message(ChatInput.Submitted("/quit", "command"))
                await pilot.pause()

            exit_mock.assert_called_once()
            assert len(app._pending_messages) == 0

    async def test_quit_bypasses_thread_switching(self) -> None:
        """/quit should exit even during a thread switch."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._thread_switching = True

            with patch.object(app, "exit") as exit_mock:
                app.post_message(ChatInput.Submitted("/quit", "command"))
                await pilot.pause()

            exit_mock.assert_called_once()

    async def test_q_alias_bypasses_queue(self) -> None:
        """/q alias should also bypass the queue."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            with patch.object(app, "exit") as exit_mock:
                app.post_message(ChatInput.Submitted("/q", "command"))
                await pilot.pause()

            exit_mock.assert_called_once()
            assert len(app._pending_messages) == 0

    async def test_version_executes_during_connecting(self) -> None:
        """/version should process immediately when only connecting."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._connecting = True

            with patch.object(app, "_process_message", new_callable=AsyncMock) as pm:
                app.post_message(ChatInput.Submitted("/version", "command"))
                await pilot.pause()

            pm.assert_called_once_with("/version", "command")
            assert len(app._pending_messages) == 0

    async def test_version_queues_during_agent_running(self) -> None:
        """/version should still queue when agent is actively running."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            app.post_message(ChatInput.Submitted("/version", "command"))
            await pilot.pause()

            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "/version"

    async def test_model_no_args_opens_selector_during_agent_running(self) -> None:
        """/model (no args) should process immediately during agent run."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            with patch.object(app, "_process_message", new_callable=AsyncMock) as pm:
                app.post_message(ChatInput.Submitted("/model", "command"))
                await pilot.pause()

            pm.assert_called_once_with("/model", "command")
            assert len(app._pending_messages) == 0

    async def test_model_no_args_opens_selector_during_connecting(self) -> None:
        """/model (no args) should process immediately during connecting."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._connecting = True

            with patch.object(app, "_process_message", new_callable=AsyncMock) as pm:
                app.post_message(ChatInput.Submitted("/model", "command"))
                await pilot.pause()

            pm.assert_called_once_with("/model", "command")

    async def test_model_with_args_still_queues(self) -> None:
        """/model <name> (with args) should still queue normally."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            app.post_message(ChatInput.Submitted("/model gpt-4", "command"))
            await pilot.pause()

            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "/model gpt-4"

    async def test_threads_opens_selector_during_agent_running(self) -> None:
        """/threads should process immediately during agent run."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            with patch.object(app, "_process_message", new_callable=AsyncMock) as pm:
                app.post_message(ChatInput.Submitted("/threads", "command"))
                await pilot.pause()

            pm.assert_called_once_with("/threads", "command")
            assert len(app._pending_messages) == 0

    async def test_threads_opens_selector_during_connecting(self) -> None:
        """/threads should process immediately during connecting."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._connecting = True

            with patch.object(app, "_process_message", new_callable=AsyncMock) as pm:
                app.post_message(ChatInput.Submitted("/threads", "command"))
                await pilot.pause()

            pm.assert_called_once_with("/threads", "command")

    async def test_threads_blocked_during_thread_switching(self) -> None:
        """/threads should NOT bypass the thread-switching guard."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._thread_switching = True

            with patch.object(app, "_process_message", new_callable=AsyncMock) as pm:
                app.post_message(ChatInput.Submitted("/threads", "command"))
                await pilot.pause()

            pm.assert_not_called()
            assert len(app._pending_messages) == 0

    async def test_model_blocked_during_thread_switching(self) -> None:
        """/model should NOT bypass the thread-switching guard."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._thread_switching = True

            with patch.object(app, "_process_message", new_callable=AsyncMock) as pm:
                app.post_message(ChatInput.Submitted("/model", "command"))
                await pilot.pause()

            pm.assert_not_called()
            assert len(app._pending_messages) == 0


class TestBypassFrozensetDrift:
    """Ensure bypass frozensets stay in sync with _handle_command dispatch.

    Every slash command must appear in exactly one of the five policy
    frozensets (derived from `command_registry.COMMANDS`) AND in
    `_handle_command`. Adding a command to one without the other will fail
    these tests.
    """

    # Dynamic namespace prefixes handled via startswith() rather than
    # static command dispatch.  These are not registered in COMMANDS and
    # should be excluded from the drift check.
    _DYNAMIC_PREFIXES = frozenset({"/skill:"})

    @classmethod
    def _handled_commands(cls) -> set[str]:
        """Extract slash-command literals from `_handle_command` source."""
        import ast
        import inspect
        import textwrap

        source = textwrap.dedent(inspect.getsource(DeepAgentsApp._handle_command))
        tree = ast.parse(source)

        handled: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.strip()
                if val.startswith("/") and len(val) > 1:
                    handled.add(val)
        # Exclude dynamic namespace prefixes (e.g. /skill:*) and their
        # derivatives (e.g. /skill:<name> from help text).
        return {
            cmd
            for cmd in handled
            if not any(cmd.startswith(p) for p in cls._DYNAMIC_PREFIXES)
        }

    def test_all_bypass_commands_are_handled(self) -> None:
        """Every command in a bypass frozenset must appear in _handle_command."""
        from deepagents_cli.command_registry import (
            ALWAYS_IMMEDIATE,
            BYPASS_WHEN_CONNECTING,
            IMMEDIATE_UI,
            SIDE_EFFECT_FREE,
        )

        handled = self._handled_commands()
        bypass = (
            ALWAYS_IMMEDIATE | BYPASS_WHEN_CONNECTING | IMMEDIATE_UI | SIDE_EFFECT_FREE
        )
        missing = bypass - handled
        assert not missing, (
            f"Bypass commands {missing} are not handled in _handle_command. "
            "Add a handler or remove from the bypass frozenset."
        )

    def test_all_handled_commands_are_classified(self) -> None:
        """Every command in _handle_command must be in a policy frozenset."""
        from deepagents_cli.command_registry import ALL_CLASSIFIED

        handled = self._handled_commands()
        missing = handled - ALL_CLASSIFIED
        assert not missing, (
            f"Commands {missing} in _handle_command are not in any bypass "
            "or QUEUE_BOUND frozenset. Classify them explicitly."
        )


class TestDeferredActions:
    """Test deferred action queueing and draining."""

    async def test_deferred_actions_drain_after_agent_cleanup(self) -> None:
        """Deferred actions should execute when agent task completes."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            executed: list[str] = []

            async def action() -> None:  # noqa: RUF029
                executed.append("ran")

            app._deferred_actions.append(
                DeferredAction(kind="model_switch", execute=action)
            )
            app._agent_running = True

            # Simulate agent finishing
            await app._cleanup_agent_task()

            assert executed == ["ran"]
            assert len(app._deferred_actions) == 0

    async def test_deferred_actions_drain_after_shell_cleanup(self) -> None:
        """Deferred actions should execute when shell task completes."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            executed: list[str] = []

            async def action() -> None:  # noqa: RUF029
                executed.append("ran")

            app._deferred_actions.append(
                DeferredAction(kind="model_switch", execute=action)
            )
            app._shell_running = True

            await app._cleanup_shell_task()

            assert executed == ["ran"]
            assert len(app._deferred_actions) == 0

    async def test_deferred_actions_not_drained_while_connecting(self) -> None:
        """Deferred actions should NOT drain if still connecting."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            executed: list[str] = []

            async def action() -> None:  # noqa: RUF029
                executed.append("ran")

            app._deferred_actions.append(
                DeferredAction(kind="model_switch", execute=action)
            )
            app._agent_running = True
            app._connecting = True

            await app._cleanup_agent_task()

            assert executed == []
            assert len(app._deferred_actions) == 1

    async def test_deferred_actions_cleared_on_interrupt(self) -> None:
        """Deferred actions should be cleared when queue is discarded."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            async def action() -> None:
                pass

            app._deferred_actions.append(
                DeferredAction(kind="model_switch", execute=action)
            )
            app._discard_queue()

            assert len(app._deferred_actions) == 0

    async def test_deferred_actions_cleared_on_server_failure(self) -> None:
        """Deferred actions should be cleared when server startup fails."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            async def action() -> None:
                pass

            app._deferred_actions.append(
                DeferredAction(kind="model_switch", execute=action)
            )
            app._connecting = True

            app.on_deep_agents_app_server_start_failed(
                DeepAgentsApp.ServerStartFailed(error=RuntimeError("test"))
            )

            assert len(app._deferred_actions) == 0

    async def test_server_failure_stores_error(self) -> None:
        """Server startup error should be stored for _send_to_agent fallback."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._connecting = True

            app.on_deep_agents_app_server_start_failed(
                DeepAgentsApp.ServerStartFailed(error=RuntimeError("exit code 3"))
            )

            assert app._server_startup_error == "RuntimeError: exit code 3"
            assert app._connecting is False

    async def test_failing_deferred_action_does_not_block_others(self) -> None:
        """A failing deferred action should not prevent subsequent ones."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            executed: list[str] = []

            async def bad_action() -> None:  # noqa: RUF029
                msg = "boom"
                raise RuntimeError(msg)

            async def good_action() -> None:  # noqa: RUF029
                executed.append("ok")

            app._deferred_actions.append(
                DeferredAction(kind="model_switch", execute=bad_action)
            )
            app._deferred_actions.append(
                DeferredAction(kind="thread_switch", execute=good_action)
            )

            await app._drain_deferred_actions()

            assert executed == ["ok"]
            assert len(app._deferred_actions) == 0

    async def test_defer_action_deduplicates_by_kind(self) -> None:
        """Deferring two actions of the same kind keeps only the last."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            executed: list[str] = []

            async def first() -> None:  # noqa: RUF029
                executed.append("first")

            async def second() -> None:  # noqa: RUF029
                executed.append("second")

            app._defer_action(DeferredAction(kind="model_switch", execute=first))
            app._defer_action(DeferredAction(kind="model_switch", execute=second))

            assert len(app._deferred_actions) == 1
            await app._drain_deferred_actions()
            assert executed == ["second"]

    async def test_can_bypass_queue_version_only_connecting(self) -> None:
        """/version bypasses only during connection, not agent/shell."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            # Connecting only → bypass
            app._connecting = True
            app._agent_running = False
            app._shell_running = False
            assert app._can_bypass_queue("/version") is True

            # Agent running (even if connecting) → no bypass
            app._agent_running = True
            assert app._can_bypass_queue("/version") is False

            # Shell running (even if connecting) → no bypass
            app._agent_running = False
            app._shell_running = True
            assert app._can_bypass_queue("/version") is False

            # Not connecting → no bypass
            app._connecting = False
            app._shell_running = False
            assert app._can_bypass_queue("/version") is False

    async def test_can_bypass_queue_bare_model_bypasses(self) -> None:
        """Bare /model should bypass the queue."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._can_bypass_queue("/model") is True
            assert app._can_bypass_queue("/threads") is True

    async def test_can_bypass_queue_model_with_args_no_bypass(self) -> None:
        """/model with args should NOT bypass (direct switch must queue)."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._can_bypass_queue("/model gpt-4") is False
            assert app._can_bypass_queue("/model --default foo") is False

    async def test_model_with_args_still_queues(self) -> None:
        """/model gpt-4 should be queued when busy, not bypass."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._agent_running = True

            app.post_message(ChatInput.Submitted("/model gpt-4", "command"))
            await pilot.pause()

            assert len(app._pending_messages) == 1
            assert app._pending_messages[0].text == "/model gpt-4"

    async def test_side_effect_free_bypasses_queue(self) -> None:
        """SIDE_EFFECT_FREE commands bypass the queue."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            for cmd in ("/changelog", "/docs", "/feedback", "/mcp"):
                assert app._can_bypass_queue(cmd) is True

    async def test_queued_commands_do_not_bypass(self) -> None:
        """QUEUED commands must not bypass the queue."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            for cmd in ("/help", "/clear", "/tokens"):
                assert app._can_bypass_queue(cmd) is False

    async def test_can_bypass_queue_empty_string(self) -> None:
        """Empty string should not bypass the queue."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._can_bypass_queue("") is False

    async def test_defer_action_mixed_kinds_preserves_ordering(self) -> None:
        """Deferring mixed kinds keeps ordering; same-kind replaces in place."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            executed: list[str] = []

            async def first_model() -> None:  # noqa: RUF029
                executed.append("first_model")

            async def thread_fn() -> None:  # noqa: RUF029
                executed.append("thread")

            async def second_model() -> None:  # noqa: RUF029
                executed.append("second_model")

            app._defer_action(DeferredAction(kind="model_switch", execute=first_model))
            app._defer_action(DeferredAction(kind="thread_switch", execute=thread_fn))
            app._defer_action(DeferredAction(kind="model_switch", execute=second_model))

            assert len(app._deferred_actions) == 2
            assert app._deferred_actions[0].kind == "thread_switch"
            assert app._deferred_actions[1].kind == "model_switch"

            await app._drain_deferred_actions()
            assert executed == ["thread", "second_model"]


class TestServerStartupError:
    """Test error messages when the server fails to start."""

    async def test_send_to_agent_shows_server_error(self) -> None:
        """_send_to_agent should show the server startup error as an ErrorMessage."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._server_startup_error = (
                "RuntimeError: Server process exited with code 3"
            )

            await app._send_to_agent("hello")
            await pilot.pause()

            msgs = app.query(ErrorMessage)
            assert len(msgs) == 1
            assert "Server failed to start" in str(msgs[0]._content)
            assert "exited with code 3" in str(msgs[0]._content)

    async def test_send_to_agent_shows_generic_when_no_server_error(self) -> None:
        """_send_to_agent should show the generic AppMessage when no server error."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            await app._send_to_agent("hello")
            await pilot.pause()

            msgs = app.query(AppMessage)
            assert len(msgs) == 1
            assert msgs[0]._content == "Agent not configured for this session."


class TestHasConversationMessages:
    """Tests for _has_conversation_messages guard."""

    async def test_returns_false_when_no_agent(self) -> None:
        """Should return False when the agent is not initialised."""
        app = DeepAgentsApp()
        async with app.run_test():
            assert app._agent is None
            assert await app._has_conversation_messages() is False

    async def test_returns_false_when_state_empty(self) -> None:
        """Should return False when state has no values."""
        app = DeepAgentsApp()
        async with app.run_test():
            state = MagicMock()
            state.values = {}
            agent = AsyncMock()
            agent.aget_state = AsyncMock(return_value=state)
            app._agent = agent

            assert await app._has_conversation_messages() is False

    async def test_returns_false_when_only_system_messages(self) -> None:
        """Should return False when messages list has no HumanMessage."""
        from langchain_core.messages import SystemMessage

        app = DeepAgentsApp()
        async with app.run_test():
            state = MagicMock()
            state.values = {"messages": [SystemMessage(content="sys")]}
            agent = AsyncMock()
            agent.aget_state = AsyncMock(return_value=state)
            app._agent = agent

            assert await app._has_conversation_messages() is False

    async def test_returns_true_when_human_message_present(self) -> None:
        """Should return True when at least one HumanMessage exists."""
        from langchain_core.messages import HumanMessage

        app = DeepAgentsApp()
        async with app.run_test():
            state = MagicMock()
            state.values = {"messages": [HumanMessage(content="hi")]}
            agent = AsyncMock()
            agent.aget_state = AsyncMock(return_value=state)
            app._agent = agent

            assert await app._has_conversation_messages() is True

    async def test_returns_true_on_aget_state_exception(self) -> None:
        """Should return True on transient errors so /remember is not blocked."""
        app = DeepAgentsApp()
        async with app.run_test():
            agent = AsyncMock()
            agent.aget_state = AsyncMock(side_effect=RuntimeError("connection lost"))
            app._agent = agent

            assert await app._has_conversation_messages() is True

    async def test_returns_false_when_state_values_is_none(self) -> None:
        """Should return False when state.values is None."""
        app = DeepAgentsApp()
        async with app.run_test():
            state = MagicMock()
            state.values = None
            agent = AsyncMock()
            agent.aget_state = AsyncMock(return_value=state)
            app._agent = agent

            assert await app._has_conversation_messages() is False


class TestRememberRequiresMessages:
    """Ensure /remember early-returns when no conversation exists."""

    async def test_remember_no_messages_shows_early_return(self) -> None:
        """/remember should mount an AppMessage and skip the skill."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            with patch.object(app, "_has_conversation_messages", return_value=False):
                await app._handle_command("/remember")
                await pilot.pause()

            msgs = app.query(AppMessage)
            assert len(msgs) == 1
            assert "Nothing to remember yet" in str(msgs[0]._content)

    async def test_remember_with_messages_delegates_to_skill(self) -> None:
        """/remember should delegate to _handle_skill_command when messages exist."""
        app = DeepAgentsApp()
        async with app.run_test() as pilot:
            await pilot.pause()

            with (
                patch.object(app, "_has_conversation_messages", return_value=True),
                patch.object(app, "_handle_skill_command") as mock_skill,
            ):
                await app._handle_command("/remember")
                await pilot.pause()

            mock_skill.assert_called_once_with("/skill:remember")


class TestSwitchAgentGuards:
    """Guards in `_switch_agent` before the restart worker is launched."""

    async def test_noop_when_same_agent(self) -> None:
        """Switching to the already-active agent should do nothing."""
        app = DeepAgentsApp(
            agent=MagicMock(),
            assistant_id="coder",
            server_kwargs={"assistant_id": "coder"},
            server_proc=MagicMock(),
        )
        async with app.run_test():
            with patch.object(app, "run_worker") as worker:
                app._switch_agent("coder")
            worker.assert_not_called()
            assert app._agent_switching is False

    async def test_rejects_remote_server_mode(self) -> None:
        """Without a local `server_proc` the CLI can't restart the agent."""
        app = DeepAgentsApp(
            agent=MagicMock(),
            assistant_id="coder",
            server_kwargs=None,
            server_proc=None,
        )
        async with app.run_test():
            notifications: list[str] = []
            with (
                patch.object(
                    app, "notify", side_effect=lambda m, **_: notifications.append(m)
                ),
                patch.object(app, "run_worker") as worker,
            ):
                app._switch_agent("researcher")
            worker.assert_not_called()
            assert any("remote server" in m for m in notifications)
            assert app._assistant_id == "coder"
            assert app._agent_switching is False

    async def test_defers_while_local_server_is_connecting(self) -> None:
        """Local startup should queue the swap instead of warning as remote."""
        app = DeepAgentsApp(
            agent=MagicMock(),
            assistant_id="coder",
            server_kwargs={"assistant_id": "coder"},
            server_proc=None,
        )

        async with app.run_test():
            notifications: list[str] = []
            with (
                patch.object(
                    app, "notify", side_effect=lambda m, **_: notifications.append(m)
                ),
                patch.object(app, "run_worker") as worker,
            ):
                app._switch_agent("researcher")

            worker.assert_not_called()
            assert len(app._deferred_actions) == 1
            action = app._deferred_actions[0]
            assert action.kind == "agent_switch"
            assert any("connection completes" in m for m in notifications)
            assert all("remote server" not in m for m in notifications)
            assert app._assistant_id == "coder"
            assert app._agent_switching is False

            with patch.object(app, "_switch_agent") as switch:
                await action.execute()
            switch.assert_called_once_with("researcher")

    async def test_rejects_while_agent_running(self) -> None:
        """Mid-run swaps are rejected so in-flight streams aren't orphaned."""
        app = DeepAgentsApp(
            agent=MagicMock(),
            assistant_id="coder",
            server_kwargs={"assistant_id": "coder"},
            server_proc=MagicMock(),
        )
        app._agent_running = True
        async with app.run_test():
            notifications: list[str] = []
            with (
                patch.object(
                    app, "notify", side_effect=lambda m, **_: notifications.append(m)
                ),
                patch.object(app, "run_worker") as worker,
            ):
                app._switch_agent("researcher")
            worker.assert_not_called()
            assert any("task is running" in m for m in notifications)

    async def test_rejects_reentry_while_switching(self) -> None:
        """A second switch while the first is in flight is a no-op."""
        app = DeepAgentsApp(
            agent=MagicMock(),
            assistant_id="coder",
            server_kwargs={"assistant_id": "coder"},
            server_proc=MagicMock(),
        )
        app._agent_switching = True
        async with app.run_test():
            with patch.object(app, "run_worker") as worker:
                app._switch_agent("researcher")
            worker.assert_not_called()
        # Reset the flag so the test app can tear down cleanly.
        app._agent_switching = False

    async def test_rejects_missing_agent_dir(self, tmp_path: Path) -> None:
        """Missing `~/.deepagents/<name>/` rejects before launching a worker."""
        app = DeepAgentsApp(
            agent=MagicMock(),
            assistant_id="coder",
            server_kwargs={"assistant_id": "coder"},
            server_proc=MagicMock(),
        )
        (tmp_path / "coder").mkdir()
        async with app.run_test():
            with (
                patch("deepagents_cli.config.settings") as mock_settings,
                patch.object(app, "run_worker") as worker,
            ):
                mock_settings.user_deepagents_dir = tmp_path
                app._switch_agent("ghost")
            worker.assert_not_called()
            assert app._assistant_id == "coder"

    async def test_launches_worker_on_valid_switch(self, tmp_path: Path) -> None:
        """A valid switch enqueues the restart worker and sets the guard flag."""
        app = DeepAgentsApp(
            agent=MagicMock(),
            assistant_id="coder",
            server_kwargs={"assistant_id": "coder"},
            server_proc=MagicMock(),
        )
        (tmp_path / "researcher").mkdir()
        async with app.run_test():
            with (
                patch("deepagents_cli.config.settings") as mock_settings,
                patch.object(app, "run_worker") as worker,
            ):
                mock_settings.user_deepagents_dir = tmp_path
                app._switch_agent("researcher")
            worker.assert_called_once()
            assert app._agent_switching is True
            # Close the coroutine that the mocked run_worker never awaited,
            # otherwise pytest emits "coroutine was never awaited".
            coro = worker.call_args.args[0]
            coro.close()
        app._agent_switching = False


class TestRestartServerForAgentSwap:
    """End-to-end wiring for the agent-swap worker."""

    @staticmethod
    def _make_app() -> tuple[DeepAgentsApp, MagicMock]:
        """Return `(app, server_proc_mock)`.

        Returning the mock directly keeps its `.url` / `.restart` /
        `.update_env` mutable attributes typed as `Any` (via `MagicMock`)
        so tests can reassign them without fighting `ServerProcess`'s
        static type.
        """
        server_proc = MagicMock()
        server_proc.update_env = MagicMock()
        server_proc.restart = AsyncMock()
        server_proc.url = "http://127.0.0.1:54321"
        app = DeepAgentsApp(
            agent=MagicMock(),
            assistant_id="coder",
            thread_id="old-thread",
            server_kwargs={"assistant_id": "coder"},
            server_proc=server_proc,
        )
        return app, server_proc

    async def test_happy_path_rebuilds_agent_and_updates_identity(
        self,
    ) -> None:
        """Successful restart stages env, calls restart, and rewires client."""
        from deepagents_cli.widgets.message_store import MessageData, MessageType

        app, server_proc = self._make_app()
        # Seed an assistant message so the previous thread looks like it
        # produced agent-side output — the resume hint is gated on the
        # presence of a `ASSISTANT`/`TOOL`/`SKILL` entry, since those only
        # land in the store after a server round-trip wrote a checkpoint.
        app._message_store.append(
            MessageData(type=MessageType.ASSISTANT, content="hi there")
        )
        async with app.run_test() as pilot:
            await pilot.pause()

            # Simulate port rebind during restart (TIME_WAIT) so the test
            # catches any regression that reuses the old URL.
            def _rebind_port() -> None:
                server_proc.url = "http://127.0.0.1:60000"

            server_proc.restart = AsyncMock(side_effect=_rebind_port)

            mounted: list[object] = []
            spawned_workers: list[tuple[object, dict]] = []
            real_run_worker = app.run_worker

            def _run_worker_spy(coro_or_fn: Any, **kwargs: Any) -> Any:  # noqa: ANN401  # test spy — accepts whatever run_worker does
                spawned_workers.append((coro_or_fn, kwargs))
                # Close the coroutine so pytest doesn't warn about it being
                # unawaited — we only care that the worker was scheduled.
                if hasattr(coro_or_fn, "close"):
                    coro_or_fn.close()
                return real_run_worker(asyncio.sleep(0))

            with (
                patch(
                    "deepagents_cli.model_config.save_recent_agent",
                    return_value=True,
                ) as save_mock,
                patch.object(app, "_mount_message", side_effect=mounted.append),
                patch.object(app, "run_worker", side_effect=_run_worker_spy),
            ):
                await app._restart_server_for_agent_swap("researcher")

            server_proc.update_env.assert_called_once_with(
                DEEPAGENTS_CLI_SERVER_ASSISTANT_ID="researcher"
            )
            server_proc.restart.assert_awaited_once()
            assert app._assistant_id == "researcher"
            assert app._server_kwargs is not None
            assert app._server_kwargs["assistant_id"] == "researcher"
            assert app._agent is not None
            # RemoteAgent must target the URL that `server_proc.url` holds
            # *after* restart, not the original.
            assert app._agent._url == "http://127.0.0.1:60000"  # type: ignore[union-attr]
            assert app._connecting is False
            assert app._agent_switching is False
            assert app._lc_thread_id != "old-thread"

            save_mock.assert_called_once_with("researcher")

            # Skill-discovery worker launched with the dedicated group.
            groups = {kw.get("group") for _, kw in spawned_workers}
            assert "agent-switch-skill-discovery" in groups

            # Confirmation + resume-hint messages reached the user.
            plain = [str(getattr(m, "_content", m)) for m in mounted]
            assert any("Switched to researcher" in s for s in plain)
            assert any(
                "deepagents -r old-thread" in s and "to resume" in s for s in plain
            )

    async def test_no_resume_hint_when_previous_thread_has_no_agent_output(
        self,
    ) -> None:
        """Untouched thread (no agent output) skips the resume hint.

        An agent switch immediately after launch has a thread ID but no
        server-side checkpoint, so `-r <thread>` would fail to resume
        anything. Don't surface a hint that points at an empty thread.
        """
        app, _server_proc = self._make_app()
        assert app._message_store.total_count == 0  # sanity check

        mounted: list[object] = []
        async with app.run_test() as pilot:
            await pilot.pause()
            with (
                patch(
                    "deepagents_cli.model_config.save_recent_agent",
                    return_value=True,
                ),
                patch.object(app, "_mount_message", side_effect=mounted.append),
                patch.object(app, "run_worker"),
            ):
                await app._restart_server_for_agent_swap("researcher")

        plain = [str(getattr(m, "_content", m)) for m in mounted]
        assert any("Switched to researcher" in s for s in plain)
        assert not any("to resume" in s for s in plain)

    async def test_no_resume_hint_when_only_local_user_messages(self) -> None:
        """Local-only slash commands don't count as agent-side activity.

        Flows like `/update` and `!shell` mount a `UserMessage` widget but
        never call the server, so no checkpoint exists. A `USER`-only store
        must not trigger the resume hint.
        """
        from deepagents_cli.widgets.message_store import MessageData, MessageType

        app, _server_proc = self._make_app()
        app._message_store.append(MessageData(type=MessageType.USER, content="/update"))

        mounted: list[object] = []
        async with app.run_test() as pilot:
            await pilot.pause()
            with (
                patch(
                    "deepagents_cli.model_config.save_recent_agent",
                    return_value=True,
                ),
                patch.object(app, "_mount_message", side_effect=mounted.append),
                patch.object(app, "run_worker"),
            ):
                await app._restart_server_for_agent_swap("researcher")

        plain = [str(getattr(m, "_content", m)) for m in mounted]
        assert any("Switched to researcher" in s for s in plain)
        assert not any("to resume" in s for s in plain)

    async def test_no_resume_hint_when_no_previous_thread(self) -> None:
        """Fresh session (no previous thread) skips the resume hint."""
        server_proc = MagicMock()
        server_proc.update_env = MagicMock()
        server_proc.restart = AsyncMock()
        server_proc.url = "http://127.0.0.1:54321"
        app = DeepAgentsApp(
            agent=MagicMock(),
            assistant_id="coder",
            thread_id=None,
            server_kwargs={"assistant_id": "coder"},
            server_proc=server_proc,
        )
        mounted: list[object] = []
        async with app.run_test() as pilot:
            await pilot.pause()
            with (
                patch(
                    "deepagents_cli.model_config.save_recent_agent",
                    return_value=True,
                ),
                patch.object(app, "_mount_message", side_effect=mounted.append),
                patch.object(app, "run_worker"),
            ):
                await app._restart_server_for_agent_swap("researcher")

        plain = [str(getattr(m, "_content", m)) for m in mounted]
        assert any("Switched to researcher" in s for s in plain)
        assert not any("to resume" in s for s in plain)

    async def test_failure_rolls_back_identity_and_posts_failed(
        self,
    ) -> None:
        """If restart raises, identity reverts and the banner failure fires."""
        app, server_proc = self._make_app()
        boom = RuntimeError("langgraph dev exited")
        server_proc.restart = AsyncMock(side_effect=boom)
        posted: list[object] = []
        with patch.object(app, "post_message", side_effect=lambda m: posted.append(m)):
            async with app.run_test() as pilot:
                await pilot.pause()
                await app._restart_server_for_agent_swap("researcher")

        assert app._assistant_id == "coder"
        assert app._server_kwargs is not None
        assert app._server_kwargs["assistant_id"] == "coder"
        assert app._agent is None
        assert app._agent_switching is False
        # The ServerStartFailed handler sets _connecting=False on message
        # dispatch, but the worker itself must also clear it synchronously
        # before posting so any code reading the flag in between sees the
        # correct value.
        assert app._connecting is False
        failures = [m for m in posted if isinstance(m, DeepAgentsApp.ServerStartFailed)]
        assert len(failures) == 1
        assert failures[0].error is boom
