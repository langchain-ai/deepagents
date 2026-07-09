r"""Tests for the Debug Console modal and its `Ctrl+\` / `/debug` toggle."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import RichLog, Static

import deepagents_code.widgets.debug_console as debug_console_mod
from deepagents_code._debug_buffer import get_log_buffer
from deepagents_code.app import DeepAgentsApp
from deepagents_code.widgets.debug_console import DebugConsoleScreen

if TYPE_CHECKING:
    import pytest

logger = logging.getLogger("deepagents_code._test_console")


def _widget_text(widget: Static) -> str:
    return str(widget.render())


class _Harness(App[None]):
    """Minimal app wrapper for testing `DebugConsoleScreen` in isolation."""

    def compose(self) -> ComposeResult:
        yield Static("base")


def _snapshot() -> list[tuple[str, str]]:
    return [
        ("Version", "9.9.9"),
        ("Model", "openai:gpt-test"),
        ("CWD", "/tmp/[brackets]/work"),
    ]


class TestDebugConsoleScreen:
    async def test_renders_snapshot_fields(self) -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()

            text = _widget_text(screen.query_one(".debug-console-snapshot", Static))
            assert "Version" in text
            assert "9.9.9" in text
            assert "openai:gpt-test" in text
            assert "/tmp/[brackets]/work" in text

    async def test_live_tail_writes_buffered_records(self) -> None:
        logger.info("debug-console-tail-marker")
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", RichLog)
            assert len(log.lines) > 0

    async def test_live_tail_appends_new_records_incrementally(self) -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", RichLog)
            before = len(log.lines)

            logger.info("debug-console-incremental-marker")
            screen._poll_logs()
            await pilot.pause()

            # Exactly one new line appended; already-consumed lines not re-written.
            assert len(log.lines) == before + 1

    async def test_empty_snapshot_renders_placeholder(self) -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen([])
            app.push_screen(screen)
            await pilot.pause()
            text = _widget_text(screen.query_one(".debug-console-snapshot", Static))
            assert "no session data" in text

    async def test_poll_logs_handles_missing_buffer_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(debug_console_mod, "get_log_buffer", lambda: None)
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", RichLog)
            assert len(log.lines) == 1  # on_mount poll writes the notice

            screen._poll_logs()
            await pilot.pause()
            assert len(log.lines) == 1  # one-shot guard: notice not repeated

    async def test_clear_view_key_empties_log_and_advances_pointer(self) -> None:
        logger.info("debug-console-clear-marker")
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            buffer = get_log_buffer()
            assert buffer is not None

            await pilot.press("ctrl+l")
            await pilot.pause()
            log = screen.query_one("#debug-log", RichLog)
            assert len(log.lines) == 0
            assert screen._rendered_upto == buffer.total_emitted
            assert screen._view_floor == buffer.total_emitted

    async def test_copy_key_invokes_clipboard_with_retained_lines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, str] = {}

        def fake_copy(_app: App, text: str) -> tuple[bool, str | None]:
            captured["text"] = text
            return True, None

        monkeypatch.setattr(debug_console_mod, "copy_text_to_clipboard", fake_copy)

        logger.info("debug-console-copy-marker")
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()

        assert "debug-console-copy-marker" in captured["text"]

    async def test_copy_failure_notifies_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_copy(_app: App, _text: str) -> tuple[bool, str | None]:
            return False, "clipboard unavailable"

        monkeypatch.setattr(debug_console_mod, "copy_text_to_clipboard", fake_copy)

        logger.info("debug-console-copy-fail-marker")
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()

            latest = list(app._notifications)[-1]
            assert latest.severity == "warning"
            assert "clipboard unavailable" in latest.message

    async def test_escape_dismisses(self) -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            app.push_screen(DebugConsoleScreen(_snapshot()))
            await pilot.pause()
            assert isinstance(app.screen, DebugConsoleScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, DebugConsoleScreen)


class TestDebugConsoleToggle:
    async def test_ctrl_backslash_opens_and_closes(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+backslash")
            await pilot.pause()
            assert isinstance(app.screen, DebugConsoleScreen)
            await pilot.press("ctrl+backslash")
            await pilot.pause()
            assert not isinstance(app.screen, DebugConsoleScreen)

    async def test_toggle_action_closes_open_console(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_toggle_debug_console()
            await pilot.pause()
            assert isinstance(app.screen, DebugConsoleScreen)

            # Call the app-level toggle directly to exercise its pop branch: a
            # `ctrl+backslash` keypress would be intercepted by the modal's own
            # priority binding and closed via the screen action instead.
            app.action_toggle_debug_console()
            await pilot.pause()
            assert not isinstance(app.screen, DebugConsoleScreen)

    async def test_debug_command_opens_console(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._handle_command("/debug")
            await pilot.pause()
            assert isinstance(app.screen, DebugConsoleScreen)

    async def test_does_not_stack_on_open_modal(self) -> None:
        class _OtherModal(ModalScreen[None]):
            def compose(self) -> ComposeResult:
                yield Static("other")

        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(_OtherModal())
            await pilot.pause()

            await pilot.press("ctrl+backslash")
            await pilot.pause()

            assert isinstance(app.screen, _OtherModal)
            latest = list(app._notifications)[-1]
            assert "debug console" in latest.message.lower()

    async def test_build_snapshot_contains_core_fields(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-xyz", cwd="/tmp/work")
        async with app.run_test():
            snapshot = dict(app._build_debug_snapshot())
            assert snapshot["Thread"] == "thread-xyz"
            assert snapshot["CWD"] == "/tmp/work"
            assert "Version" in snapshot
            assert "Auto-approve" in snapshot
            assert snapshot["MCP servers"] == "none"

    async def test_build_snapshot_formats_mcp_servers(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
        async with app.run_test():
            servers = [
                SimpleNamespace(name="fs", status="connected"),
                SimpleNamespace(name="web", status="error"),
            ]
            app._mcp_server_info = servers  # ty: ignore[invalid-assignment]  # stub servers expose .name/.status
            snapshot = dict(app._build_debug_snapshot())
            assert snapshot["MCP servers"] == "fs (connected), web (error)"

    async def test_build_snapshot_degrades_on_field_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
        async with app.run_test():

            def boom() -> str:
                msg = "kaboom"
                raise RuntimeError(msg)

            monkeypatch.setattr(app, "_effective_model_spec", boom)
            snapshot = dict(app._build_debug_snapshot())
            # The failing field degrades; the rest of the snapshot still builds.
            assert snapshot["Model"].startswith("(unavailable")
            assert "Version" in snapshot
            assert snapshot["Thread"] == "t"
