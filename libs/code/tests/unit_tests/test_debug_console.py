r"""Tests for the Debug Console modal and its `Ctrl+\` / `/debug` toggle."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Select, Static
from textual.widgets._select import SelectOverlay

import deepagents_code.widgets.debug_console as debug_console_mod
from deepagents_code._debug_buffer import get_log_buffer
from deepagents_code.app import DeepAgentsApp
from deepagents_code.widgets.debug_console import DebugConsoleScreen, _DebugLogView

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
            log = screen.query_one("#debug-log", _DebugLogView)
            assert log.line_count > 0

    async def test_live_tail_appends_new_records_incrementally(self) -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            before = len(log.records)

            logger.info("debug-console-incremental-marker")
            screen._poll_logs()
            await pilot.pause()

            # Exactly one new record appended; already-consumed records not re-written.
            assert len(log.records) == before + 1

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
            log = screen.query_one("#debug-log", _DebugLogView)
            assert log.line_count == 1  # on_mount poll writes the notice

            screen._poll_logs()
            await pilot.pause()
            assert log.line_count == 1  # one-shot guard: notice not repeated

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
            log = screen.query_one("#debug-log", _DebugLogView)
            assert log.line_count == 0
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

    async def test_level_filter_supports_threshold_and_exact_modes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(debug_console_mod, "_debug_records_enabled", lambda: True)
        logger.info("debug-console-filter-info-marker")
        logger.warning("debug-console-filter-warning-marker")
        logger.error("debug-console-filter-error-marker")
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            select = screen.query_one("#debug-level-filter", Select)

            select.value = "min:WARNING"
            await pilot.pause()

            assert not any(
                "debug-console-filter-info-marker" in record.message
                for record in log.records
            )
            assert any(
                "debug-console-filter-warning-marker" in record.message
                for record in log.records
            )
            assert any(
                "debug-console-filter-error-marker" in record.message
                for record in log.records
            )

            select.value = "min:DEBUG"
            await pilot.pause()

            assert any(
                "debug-console-filter-info-marker" in record.message
                for record in log.records
            )
            assert any(
                "debug-console-filter-error-marker" in record.message
                for record in log.records
            )

            select.value = "only:WARNING"
            await pilot.pause()

            assert any(
                "debug-console-filter-warning-marker" in record.message
                for record in log.records
            )
            assert not any(
                "debug-console-filter-error-marker" in record.message
                for record in log.records
            )

    async def test_level_filter_hides_debug_when_runtime_level_excludes_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(debug_console_mod, "_debug_records_enabled", lambda: False)
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            select = screen.query_one("#debug-level-filter", Select)
            values = {value for _prompt, value in select._options}

        assert "min:DEBUG" not in values
        assert "only:DEBUG" not in values
        assert "min:INFO" in values

    async def test_arrow_keys_move_between_logical_records(self) -> None:
        logger.info("debug-console-arrow-first-marker")
        logger.info("debug-console-arrow-second-marker")
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            log.focus()
            second_index = next(
                index
                for index, record in enumerate(log.records)
                if "debug-console-arrow-second-marker" in record.message
            )
            log._select_record(second_index)

            await pilot.press("up")
            await pilot.pause()
            assert log._selected_index == second_index - 1

            await pilot.press("down")
            await pilot.pause()
            assert log._selected_index == second_index

    async def test_selected_row_highlight_extends_to_full_width(self) -> None:
        logger.info("debug-console-full-width-marker")
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            index = next(
                index
                for index, record in enumerate(log.records)
                if "debug-console-full-width-marker" in record.message
            )
            log._select_record(index)
            strip = log.render_line(int(log._wrap_prefix[index] - log.scroll_y))

        first_segment = strip._segments[0]
        trailing_segment = strip._segments[-1]
        assert trailing_segment.text
        assert first_segment.style is not None
        assert trailing_segment.style is not None
        assert first_segment.style.bgcolor is not None
        assert trailing_segment.style.bgcolor == first_segment.style.bgcolor

    async def test_enter_copies_selected_logical_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, str] = {}

        def fake_copy(_app: App, text: str) -> tuple[bool, str | None]:
            captured["text"] = text
            return True, None

        monkeypatch.setattr(debug_console_mod, "copy_text_to_clipboard", fake_copy)

        logger.info("debug-console-enter-copy-marker")
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            log.focus()
            index = next(
                index
                for index, record in enumerate(log.records)
                if "debug-console-enter-copy-marker" in record.message
            )
            log._select_record(index)

            await pilot.press("enter")
            await pilot.pause()

        assert "debug-console-enter-copy-marker" in captured["text"]

    async def test_tab_cycles_between_level_filter_and_log_lines(self) -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            select = screen.query_one("#debug-level-filter", Select)
            log.focus()

            await pilot.press("tab")
            await pilot.pause()
            assert screen.focused is select

            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen.focused is log

    async def test_tab_moves_open_level_dropdown_highlight(self) -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            select = screen.query_one("#debug-level-filter", Select)
            select.action_show_overlay()
            await pilot.pause()
            overlay = select.query_one(SelectOverlay)
            assert overlay.highlighted is not None
            before = overlay.highlighted

            await pilot.press("tab")
            await pilot.pause()
            assert overlay.highlighted == before + 1

            await pilot.press("shift+tab")
            await pilot.pause()
            assert overlay.highlighted == before
            assert select.expanded

    async def test_escape_collapses_level_dropdown_before_dismissing(self) -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            select = screen.query_one("#debug-level-filter", Select)
            select.action_show_overlay()
            await pilot.pause()
            assert select.expanded

            await pilot.press("escape")
            await pilot.pause()
            assert not select.expanded
            assert isinstance(app.screen, DebugConsoleScreen)

            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, DebugConsoleScreen)

    async def test_click_copy_invokes_clipboard_with_logical_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, str] = {}

        def fake_copy(_app: App, text: str) -> tuple[bool, str | None]:
            captured["text"] = text
            return True, None

        monkeypatch.setattr(debug_console_mod, "copy_text_to_clipboard", fake_copy)

        logger.info("debug-console-click-copy-marker")
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            record = log._record_at_visual_y(log.line_count - 1)
            assert record is not None
            assert "debug-console-click-copy-marker" in record.message
            screen._copy_record(record)
            await pilot.pause()

        assert "debug-console-click-copy-marker" in captured["text"]

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
