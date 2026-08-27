r"""Tests for the Debug Console modal and its `Ctrl+\` / `/debug` toggle."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Checkbox, Select, Static
from textual.widgets._select import SelectOverlay

import deepagents_code.tui.widgets.debug_console as debug_console_mod
from deepagents_code._debug_buffer import InMemoryLogRecord, get_log_buffer
from deepagents_code.app import DeepAgentsApp
from deepagents_code.tui.widgets._copy_spans import COPY_LABEL_META, COPY_TEXT_META
from deepagents_code.tui.widgets.debug_console import (
    DebugConsoleScreen,
    SnapshotField,
    _DebugLogView,
    _record_matches_filter,
)
from deepagents_code.tui.widgets.message_store import (
    MessageData,
    MessageStore,
    MessageType,
)

if TYPE_CHECKING:
    import pytest

    from deepagents_code.tui.widgets.debug_console import FilterValue

logger = logging.getLogger("deepagents_code._test_console")


def _widget_text(widget: Static) -> str:
    return str(widget.render())


def _snapshot_dict(fields: list[SnapshotField]) -> dict[str, str]:
    return {field.label: field.value for field in fields}


def _log_record(
    message: str, *, level: str = "INFO", levelno: int = logging.INFO
) -> InMemoryLogRecord:
    return InMemoryLogRecord(
        timestamp="12:00:00",
        level=level,
        levelno=levelno,
        logger="deepagents_code._test_console",
        message=message,
    )


class _Harness(App[None]):
    """Minimal app wrapper for testing `DebugConsoleScreen` in isolation."""

    def compose(self) -> ComposeResult:
        yield Static("base")


def _snapshot() -> list[SnapshotField]:
    return [
        SnapshotField("Version", "9.9.9"),
        SnapshotField("Model", "openai:gpt-test"),
        SnapshotField("CWD", "/tmp/[brackets]/work"),
    ]


class TestDebugConsoleScreen:
    async def test_wrapped_snapshot_values_align_to_value_column(self) -> None:
        fields = [
            SnapshotField(
                "MCP servers",
                "notion (ok), slack (ok), langsmith (ok), onepassword (ok)",
            ),
            SnapshotField(
                "Debug log",
                "/tmp/deepagents_debug/a/very/long/path/to/the/log/file.log",
            ),
        ]
        app = _Harness()
        async with app.run_test(size=(50, 40)) as pilot:
            screen = DebugConsoleScreen(fields)
            app.push_screen(screen)
            await pilot.pause()

            view = screen.query_one(".debug-console-snapshot", Static)
            lines = _widget_text(view).splitlines()

        indent = max(len(field.label) for field in fields) + 2
        mcp_row = next(
            index for index, line in enumerate(lines) if "MCP servers" in line
        )
        log_row = next(index for index, line in enumerate(lines) if "Debug log" in line)
        assert log_row > mcp_row + 1
        assert len(lines) > log_row + 1
        for line in (*lines[mcp_row + 1 : log_row], *lines[log_row + 1 :]):
            assert line[:indent] == " " * indent
            assert line[indent:].strip()

    async def test_cramped_value_column_wraps_snapshot_rows_flat(self) -> None:
        fields = [SnapshotField("Approval mode", "auto-edit")]
        app = _Harness()
        async with app.run_test(size=(30, 40)) as pilot:
            screen = DebugConsoleScreen(fields)
            app.push_screen(screen)
            await pilot.pause()

            view = screen.query_one(".debug-console-snapshot", Static)
            lines = _widget_text(view).splitlines()
            content_width = view.content_size.width

        indent = len("Approval mode") + 2
        # The label fits, but the remaining seven cells are too narrow for a
        # useful hanging value column and would make the snapshot very tall.
        assert content_width - indent == 7
        assert len(lines) > 1
        for line in lines:
            assert line[:indent] != " " * indent
        assert any("auto" in line for line in lines[1:])

    def test_footer_omits_click_to_copy_hint(self) -> None:
        footer = str(DebugConsoleScreen._render_help())

        assert "Enter copy line" in footer
        assert "check 'Click to copy'" not in footer

    def test_repeated_snapshot_provider_failures_warn_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A stuck provider must not flood the buffer the console is tailing."""
        failing = True

        def provider() -> list[SnapshotField]:
            if failing:
                msg = "snapshot provider boom"
                raise RuntimeError(msg)
            return [SnapshotField("Messages", "1")]

        screen = DebugConsoleScreen(
            [SnapshotField("Messages", "0")], snapshot_provider=provider
        )

        with caplog.at_level(
            logging.DEBUG, logger="deepagents_code.tui.widgets.debug_console"
        ):
            for _ in range(3):
                screen._poll_snapshot()

            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1
            assert len(caplog.records) == 3

            # Recovering re-arms the WARNING so a later failure is still loud.
            caplog.clear()
            failing = False
            screen._poll_snapshot()
            failing = True
            screen._poll_snapshot()

        rearmed = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(rearmed) == 1

    def test_custom_levels_share_fallback_retention_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom levels must collectively obey the buffer's fallback bound."""
        monkeypatch.setattr(debug_console_mod, "_RECORD_LIMIT", 3)
        screen = DebugConsoleScreen(_snapshot())
        screen._records = [
            _log_record(
                f"custom-{index}",
                level=f"Level {25 + index}",
                levelno=25 + index,
            )
            for index in range(5)
        ]

        assert screen._prune_records() is True
        assert [record.message for record in screen._records] == [
            "custom-2",
            "custom-3",
            "custom-4",
        ]

    def test_prune_keeps_newest_per_standard_level_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the oldest records of an over-capacity level are dropped.

        Interleaves a DEBUG flood with sparse INFO/WARNING: the under-capacity
        levels survive untouched, DEBUG is trimmed to its newest `_RECORD_LIMIT`,
        and the surviving records stay in chronological order.
        """
        monkeypatch.setattr(debug_console_mod, "_RECORD_LIMIT", 2)
        screen = DebugConsoleScreen(_snapshot())
        info = _log_record("info", level="INFO", levelno=logging.INFO)
        warning = _log_record("warning", level="WARNING", levelno=logging.WARNING)
        debugs = [
            _log_record(f"debug{index}", level="DEBUG", levelno=logging.DEBUG)
            for index in range(4)
        ]
        # Chronological: info, debug0, debug1, warning, debug2, debug3.
        screen._records = [info, debugs[0], debugs[1], warning, debugs[2], debugs[3]]

        assert screen._prune_records() is True
        assert [record.message for record in screen._records] == [
            "info",
            "warning",
            "debug2",
            "debug3",
        ]

    async def test_notice_replaced_by_incoming_records(self) -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)

            log.show_notice("(log buffer unavailable)")
            await pilot.pause()
            assert log._notice is not None

            log.append_records([_log_record("debug-console-recovery-marker")])
            await pilot.pause()

            assert log._notice is None
            assert any(
                "debug-console-recovery-marker" in record.message
                for record in log.records
            )

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

    async def test_toggling_checkbox_invokes_persist_callback(self) -> None:
        changes: list[bool] = []
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(
                _snapshot(), on_click_to_copy_change=changes.append
            )
            app.push_screen(screen)
            await pilot.pause()

            screen.query_one("#debug-click-to-copy", Checkbox).value = True
            await pilot.pause()
            screen.query_one("#debug-click-to-copy", Checkbox).value = False
            await pilot.pause()

        assert changes == [True, False]

    async def test_clicking_langsmith_link_opens_it_without_copying(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[object] = []
        monkeypatch.setattr(
            debug_console_mod, "open_style_link", lambda event: opened.append(event)
        )
        copied: list[str] = []

        def fake_copy(_app: App, text: str) -> tuple[bool, str | None]:
            copied.append(text)
            return True, None

        monkeypatch.setattr(debug_console_mod, "copy_text_to_clipboard", fake_copy)

        url = "https://smith.langchain.com/o/org/projects/p/proj/t/thread-abc"
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(
                [
                    SnapshotField(
                        "Thread", "thread-abc", copyable=True, thread_id="thread-abc"
                    )
                ]
            )
            app.push_screen(screen)
            await pilot.pause()

            screen._langsmith_urls["thread-abc"] = url
            screen._refresh_snapshot()
            await pilot.pause()

            snapshot_widget = screen.query_one(".debug-console-snapshot", Static)
            # Row renders "Thread  thread-abc  (open in langsmith)": label (6)
            # + 2-space gutter = col 8, value (10 chars) spans 8-17, 2-space gap,
            # then "(open in langsmith)" starts at col 20. An x offset of 22 is
            # inside it.
            await pilot.click(snapshot_widget, offset=(22, 0))
            await pilot.pause()

        # The link branch wins and returns early: the trace opens, no copy fires.
        assert len(opened) == 1
        assert copied == []

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
    async def test_shift_tab_reverses_focus_despite_app_toggle_binding(
        self,
    ) -> None:
        """Shift+Tab reverses console focus instead of toggling auto-approve.

        Must drive the real `DeepAgentsApp` (not `_Harness`): the App defines a
        priority `shift+tab -> toggle_auto_approve` binding that would otherwise
        consume the key App-first. This guards the `check_action` step-aside that
        lets the console's own reverse-focus traversal run; a `_Harness`-based
        test has no such binding and would pass regardless of that logic.
        """
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+backslash")
            await pilot.pause()
            screen = cast("DebugConsoleScreen", app.screen)
            log = screen.query_one("#debug-log", _DebugLogView)
            select = screen.query_one("#debug-level-filter", Select)
            assert screen.focused is log
            assert app._auto_approve is False

            await pilot.press("tab")
            await pilot.pause()
            assert screen.focused is select

            await pilot.press("shift+tab")
            await pilot.pause()
            # This focus move is the discriminating assertion: without the
            # `check_action` step-aside, shift+tab is swallowed and focus stays
            # on `select`. The `_auto_approve` check below is defense-in-depth
            # only -- the toggle already no-ops under any modal, so it reads
            # `False` in both the fixed and broken cases.
            assert screen.focused is log
            assert app._auto_approve is False

    async def test_check_action_gates_toggle_binding_by_screen(self) -> None:
        """`check_action` steps aside the toggle binding only under the console.

        Guards the enabled path the reverse-focus fix depends on: on the main
        screen `check_action` must leave `toggle_auto_approve` enabled (return
        `True`) so Shift+Tab still toggles auto-approve; the
        `test_shift_tab_reverses_focus_*` test only exercises the disabled path.
        """
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.check_action("toggle_auto_approve", ()) is True

            await pilot.press("ctrl+backslash")
            await pilot.pause()
            assert isinstance(app.screen, DebugConsoleScreen)
            assert app.check_action("toggle_auto_approve", ()) is False

    async def test_clear_persists_across_reopen(self) -> None:
        logger.info("debug-console-persist-marker")
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+backslash")
            await pilot.pause()
            screen = cast("DebugConsoleScreen", app.screen)
            log = screen.query_one("#debug-log", _DebugLogView)
            assert any(
                "debug-console-persist-marker" in record.message
                for record in log.records
            )

            buffer = get_log_buffer()
            assert buffer is not None
            expected = buffer.total_emitted
            await pilot.press("ctrl+l")
            await pilot.pause()
            assert app._debug_console_cleared_upto == expected

            # A record emitted after the clear must survive the reopen; only the
            # pre-clear tail is suppressed.
            logger.info("debug-console-post-clear-marker")

            # Close and reopen: the cleared records must not come back, but the
            # post-clear record must appear.
            await pilot.press("ctrl+backslash")
            await pilot.pause()
            await pilot.press("ctrl+backslash")
            await pilot.pause()
            reopened = cast("DebugConsoleScreen", app.screen)
            reopened_log = reopened.query_one("#debug-log", _DebugLogView)
            assert not any(
                "debug-console-persist-marker" in record.message
                for record in reopened_log.records
            )
            assert any(
                "debug-console-post-clear-marker" in record.message
                for record in reopened_log.records
            )

    async def test_opens_over_existing_modal(self) -> None:
        class _OtherModal(ModalScreen[None]):
            def compose(self) -> ComposeResult:
                yield Static("other")

        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(_OtherModal())
            await pilot.pause()
            modal = app.screen

            await pilot.press("ctrl+backslash")
            await pilot.pause()

            assert isinstance(app.screen, DebugConsoleScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is modal

    async def test_build_snapshot_contains_core_fields(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-xyz", cwd="/tmp/work")
        async with app.run_test():
            snapshot = _snapshot_dict(app._build_debug_snapshot())
            assert snapshot["Thread"] == "thread-xyz"
            assert snapshot["CWD"] == "/tmp/work"
            assert "Version" in snapshot
            assert snapshot["Approval mode"] == "manual"
            assert snapshot["MCP servers"] == "none"

    async def test_build_snapshot_experimental_off_when_env_falsy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A present-but-falsy `DEEPAGENTS_CODE_EXPERIMENTAL` reads as `off`.

        Locks the truthy gate (`is_env_truthy`) against a regression to a bare
        presence check (`EXPERIMENTAL in os.environ`), which the unset and
        truthy cases would both pass.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_EXPERIMENTAL", "0")
        app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
        async with app.run_test():
            snapshot = _snapshot_dict(app._build_debug_snapshot())
            assert snapshot["Experimental"] == "off"

    async def test_build_snapshot_editable_install_path_is_copyable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deepagents_code.config as config_mod

        monkeypatch.setattr(
            config_mod,
            "_get_editable_install_path",
            lambda: "~/oss/deepagents/libs/code",
        )
        app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
        async with app.run_test():
            field = next(
                field
                for field in app._build_debug_snapshot()
                if field.label == "Install path"
            )
            assert field.value == "~/oss/deepagents/libs/code"
            assert field.copyable is True

    async def test_build_snapshot_omits_non_editable_install_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deepagents_code.config as config_mod

        monkeypatch.setattr(config_mod, "_get_editable_install_path", lambda: None)
        app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
        async with app.run_test():
            fields = {field.label: field for field in app._build_debug_snapshot()}
            assert "Install path" not in fields

    async def test_build_snapshot_debug_log_path_is_copyable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deepagents_code._debug as debug_mod

        monkeypatch.setattr(
            debug_mod, "installed_debug_log_path", lambda: "/tmp/custom-debug.log"
        )
        app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
        async with app.run_test():
            field = next(
                field
                for field in app._build_debug_snapshot()
                if field.label == "Debug log"
            )
            assert field.value == "/tmp/custom-debug.log"
            assert field.copyable is True

    async def test_build_snapshot_in_memory_log_is_not_copyable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deepagents_code._debug as debug_mod

        monkeypatch.setattr(debug_mod, "installed_debug_log_path", lambda: None)
        monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG", raising=False)
        app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
        async with app.run_test():
            field = next(
                field
                for field in app._build_debug_snapshot()
                if field.label == "Debug log"
            )
            assert field.value == "in-memory only"
            assert field.copyable is False
