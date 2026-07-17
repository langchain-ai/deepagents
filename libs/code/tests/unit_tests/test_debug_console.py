r"""Tests for the Debug Console modal and its `Ctrl+\` / `/debug` toggle."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Select, Static
from textual.widgets._select import SelectOverlay

import deepagents_code.tui.widgets.debug_console as debug_console_mod
from deepagents_code._debug_buffer import InMemoryLogRecord, get_log_buffer
from deepagents_code.app import DeepAgentsApp
from deepagents_code.tui.widgets.debug_console import (
    DebugConsoleScreen,
    SnapshotField,
    _DebugLogView,
    _record_matches_filter,
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


def test_snapshot_field_tuple_contract_includes_interaction_metadata() -> None:
    field = SnapshotField("Thread", "thread-abc", copyable=True, thread_id="thread-abc")

    assert tuple(field) == ("Thread", "thread-abc", True, "thread-abc")


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

    async def test_no_color_renders_every_segment_with_a_style(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Monochrome filtering must never receive an unstyled Rich segment."""
        monkeypatch.setenv("NO_COLOR", "1")
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            log.set_records([_log_record("unstyled message")])
            await pilot.pause()

            strip = log.render_line(0)

        assert strip._segments
        assert all(segment.style is not None for segment in strip._segments)

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

    async def test_live_tail_stays_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = [_log_record(f"debug-console-bounded-{index}") for index in range(5)]

        class FakeBuffer:
            total_emitted = len(records)

            def snapshot_records_since(
                self, index: int
            ) -> tuple[list[InMemoryLogRecord], int]:
                if index >= self.total_emitted:
                    return [], self.total_emitted
                return records[index:], self.total_emitted

        monkeypatch.setattr(debug_console_mod, "_RECORD_LIMIT", 3)
        monkeypatch.setattr(debug_console_mod, "get_log_buffer", lambda: FakeBuffer())

        class FakeLog:
            is_vertical_scroll_end = True

            def __init__(self) -> None:
                self.records: list[InMemoryLogRecord] = []

            def append_records(self, records: list[InMemoryLogRecord]) -> None:
                self.records.extend(records)

            def set_records(
                self,
                records: list[InMemoryLogRecord],
                *,
                scroll_end: bool,
            ) -> None:
                _ = scroll_end
                self.records = list(records)

        def fake_query_one(*args: object) -> FakeLog:
            _ = args
            return log

        log = FakeLog()
        screen = DebugConsoleScreen(_snapshot())
        monkeypatch.setattr(screen, "query_one", fake_query_one)

        screen._poll_logs()

        messages = [record.message for record in screen._records]
        visible_messages = [record.message for record in log.records]

        assert messages == [
            "debug-console-bounded-2",
            "debug-console-bounded-3",
            "debug-console-bounded-4",
        ]
        assert visible_messages == messages

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

    async def test_poll_degrades_on_buffer_failure_without_crashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)

            class BoomBuffer:
                total_emitted = 0

                def snapshot_records_since(
                    self, _index: int
                ) -> tuple[list[InMemoryLogRecord], int]:
                    msg = "poll boom"
                    raise RuntimeError(msg)

            monkeypatch.setattr(
                debug_console_mod, "get_log_buffer", lambda: BoomBuffer()
            )

            # Must not raise: the repeating timer would otherwise crash the app.
            screen._poll_logs()
            await pilot.pause()

            assert log._notice is not None
            assert isinstance(app.screen, DebugConsoleScreen)  # app still alive

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

    async def test_poll_applies_active_filter_to_new_records(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(debug_console_mod, "_debug_records_enabled", lambda: True)
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            select = screen.query_one("#debug-level-filter", Select)
            select.value = "min:WARNING"
            await pilot.pause()
            before = len(log.records)

            logger.info("debug-console-poll-filter-info")
            logger.error("debug-console-poll-filter-error")
            screen._poll_logs()
            await pilot.pause()

            new_messages = [record.message for record in log.records[before:]]
            assert any("poll-filter-error" in message for message in new_messages)
            assert not any("poll-filter-info" in message for message in new_messages)

    async def test_empty_filtered_copy_notifies_information(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_copy(_app: App, _text: str) -> tuple[bool, str | None]:
            msg = "copy should not be attempted for an empty selection"
            raise AssertionError(msg)

        monkeypatch.setattr(debug_console_mod, "copy_text_to_clipboard", fail_copy)

        logger.info("debug-console-empty-copy-marker")
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            # No CRITICAL records exist, so the filtered view is empty.
            screen.query_one("#debug-level-filter", Select).value = "only:CRITICAL"
            await pilot.pause()

            await pilot.press("c")
            await pilot.pause()

            latest = list(app._notifications)[-1]
            assert latest.severity == "information"
            assert "No visible log lines" in latest.message

    async def test_wrapped_record_maps_clicks_and_arrows_as_one_unit(self) -> None:
        logger.info("W" * 240)  # long enough to wrap to multiple visual lines
        logger.info("debug-console-after-wrap-marker")
        app = _Harness()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            wrap_index = next(
                index
                for index, record in enumerate(log.records)
                if record.message.startswith("WWW")
            )
            start = log._wrap_prefix[wrap_index]
            span = log._wrap_prefix[wrap_index + 1] - start

            # The record occupies more visual lines than it is logical records.
            assert span > 1
            assert log.line_count > len(log.records)

            # A click on the continuation line maps back to the same record.
            assert log._record_at_visual_y(start + 1) is log.records[wrap_index]

            # Selecting it highlights every visual row it spans, not just the first.
            log._select_record(wrap_index)
            await pilot.pause()
            scroll_y = log.scroll_offset.y
            for visual_y in (start, start + 1):
                strip = log.render_line(visual_y - scroll_y)
                assert strip._segments[0].style is not None
                assert strip._segments[0].style.bgcolor is not None

            # One arrow press moves a whole logical record across the wrap.
            log.focus()
            log._select_record(wrap_index + 1)
            await pilot.press("up")
            await pilot.pause()
            assert log._selected_index == wrap_index

    async def test_multiline_record_spans_multiple_visual_lines(self) -> None:
        record = _log_record("first-line\nsecond-line\nthird-line")
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            log.set_records([record])
            await pilot.pause()

            # One logical record, three visual lines from the embedded newlines.
            assert len(log.records) == 1
            assert log.line_count >= 3

    async def test_selecting_offscreen_record_scrolls_into_view(self) -> None:
        for index in range(120):
            logger.info("debug-console-scroll-marker-%s", index)
        app = _Harness()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = DebugConsoleScreen(_snapshot())
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            assert log.line_count > log.size.height  # content overflows the viewport
            log.focus()

            log._select_record(0)  # jump to the top
            await pilot.pause()
            assert log.scroll_offset.y == 0

            log._select_record(len(log.records) - 1)  # jump to the bottom
            await pilot.pause()
            assert log.scroll_offset.y > 0

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

    async def test_clear_view_reports_cursor_via_on_clear(self) -> None:
        logger.info("debug-console-on-clear-marker")
        cleared: list[int] = []
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(_snapshot(), on_clear=cleared.append)
            app.push_screen(screen)
            await pilot.pause()
            buffer = get_log_buffer()
            assert buffer is not None

            # Capture the cursor at clear time; the shared process-wide buffer
            # may accrue records between the clear and a later re-read.
            expected = buffer.total_emitted
            await pilot.press("ctrl+l")
            await pilot.pause()
            assert cleared == [expected]

    async def test_cleared_upto_seeds_render_cursor(self) -> None:
        logger.info("debug-console-pre-clear-marker")
        app = _Harness()
        async with app.run_test() as pilot:
            buffer = get_log_buffer()
            assert buffer is not None
            # Simulate a prior clear by seeding past everything emitted so far.
            screen = DebugConsoleScreen(_snapshot(), cleared_upto=buffer.total_emitted)
            app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#debug-log", _DebugLogView)
            assert not any(
                "debug-console-pre-clear-marker" in record.message
                for record in log.records
            )

            logger.info("debug-console-post-clear-marker")
            screen._poll_logs()
            await pilot.pause()
            assert any(
                "debug-console-post-clear-marker" in record.message
                for record in log.records
            )

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

    async def test_level_filter_finds_info_after_debug_flood(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A DEBUG flood must not hide earlier INFO records from the filter."""
        monkeypatch.setattr(debug_console_mod, "_debug_records_enabled", lambda: True)
        # Small per-level cap so a modest flood exercises pruning quickly; a flat
        # cap would drop the INFO marker in favor of the newer DEBUG records.
        monkeypatch.setattr(debug_console_mod, "_RECORD_LIMIT", 5)
        package_logger = logging.getLogger("deepagents_code")
        original_level = package_logger.level
        package_logger.setLevel(logging.DEBUG)
        try:
            logger.info("debug-console-flood-info-marker")
            for index in range(50):  # far more DEBUG than the per-level cap
                logger.debug("debug-console-flood-debug-%d", index)
            app = _Harness()
            async with app.run_test() as pilot:
                screen = DebugConsoleScreen(_snapshot())
                app.push_screen(screen)
                await pilot.pause()
                log = screen.query_one("#debug-log", _DebugLogView)
                select = screen.query_one("#debug-level-filter", Select)

                select.value = "min:INFO"
                await pilot.pause()

                assert any(
                    "debug-console-flood-info-marker" in record.message
                    for record in log.records
                )
                # The filter hides DEBUG from the view, so assert on the
                # retained set that pruning actually bounded DEBUG to its own
                # per-level cap (newest kept) rather than dropping nothing or
                # evicting the rarer INFO marker.
                retained = [record.message for record in screen._records]
                debug_retained = [m for m in retained if "flood-debug" in m]
                assert debug_retained == [
                    f"debug-console-flood-debug-{index}" for index in range(45, 50)
                ]
                assert "debug-console-flood-info-marker" in retained
        finally:
            package_logger.setLevel(original_level)

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

    async def test_thread_field_click_copies_thread_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, str] = {}

        def fake_copy(_app: App, text: str) -> tuple[bool, str | None]:
            captured["text"] = text
            return True, None

        monkeypatch.setattr(debug_console_mod, "copy_text_to_clipboard", fake_copy)

        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(
                [SnapshotField("Thread", "thread-abc", copyable=True)]
            )
            app.push_screen(screen)
            await pilot.pause()

            screen._copy_snapshot_value("thread-abc")
            await pilot.pause()

        assert captured["text"] == "thread-abc"
        latest = list(app._notifications)[-1]
        assert latest.severity == "information"
        assert latest.message == "Thread ID copied"

    async def test_langsmith_link_renders_after_resolution(self) -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(
                [SnapshotField("Thread", "thread-abc", thread_id="thread-abc")]
            )
            app.push_screen(screen)
            await pilot.pause()

            snapshot_widget = screen.query_one(".debug-console-snapshot", Static)
            assert "(langsmith)" not in _widget_text(snapshot_widget)

            screen._langsmith_urls["thread-abc"] = (
                "https://smith.langchain.com/o/org/projects/p/proj/t/thread-abc"
            )
            screen._refresh_snapshot()
            await pilot.pause()

            assert "(langsmith)" in _widget_text(snapshot_widget)

    async def test_cached_langsmith_link_renders_on_first_frame_without_lookup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deepagents_code.config as config_mod

        url = "https://smith.langchain.com/o/org/projects/p/proj/t/thread-abc"
        monkeypatch.setattr(
            config_mod, "get_cached_langsmith_thread_url", lambda _thread_id: url
        )
        lookup = MagicMock()
        monkeypatch.setattr(config_mod, "build_langsmith_thread_url", lookup)

        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(
                [SnapshotField("Thread", "thread-abc", thread_id="thread-abc")]
            )
            app.push_screen(screen)
            await pilot.pause()

            snapshot_widget = screen.query_one(".debug-console-snapshot", Static)
            assert "(langsmith)" in _widget_text(snapshot_widget)

        lookup.assert_not_called()

    async def test_clicking_thread_value_copies_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, str] = {}

        def fake_copy(_app: App, text: str) -> tuple[bool, str | None]:
            captured["text"] = text
            return True, None

        monkeypatch.setattr(debug_console_mod, "copy_text_to_clipboard", fake_copy)

        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(
                [SnapshotField("Thread", "thread-abc", copyable=True)]
            )
            app.push_screen(screen)
            await pilot.pause()

            snapshot_widget = screen.query_one(".debug-console-snapshot", Static)
            # Single "Thread" field: label (6 chars) + 2-space gutter puts the
            # value at column 8, so an x offset of 10 lands inside the copyable
            # span. (Holds only while "Thread" is the widest label in this test.)
            await pilot.click(snapshot_widget, offset=(10, 0))
            await pilot.pause()

        assert captured["text"] == "thread-abc"

    async def test_copyable_value_carries_copy_meta_span(self) -> None:
        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(
                [SnapshotField("Thread", "thread-abc", copyable=True)]
            )
            app.push_screen(screen)
            await pilot.pause()

            content = screen._render_snapshot()
            copy_spans = [
                span
                for span in content.spans
                if isinstance(span.style, debug_console_mod.TStyle)
                and span.style.meta.get(debug_console_mod._SNAPSHOT_COPY_META)
                == "thread-abc"
            ]
            assert any(
                content.plain[span.start : span.end] == "thread-abc"
                for span in copy_spans
            )

    async def test_fetch_langsmith_link_stores_resolved_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deepagents_code.config as config_mod

        url = "https://smith.langchain.com/o/org/projects/p/proj/t/thread-abc"
        monkeypatch.setattr(
            config_mod, "build_langsmith_thread_url", lambda _thread_id: url
        )

        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(
                [SnapshotField("Thread", "thread-abc", thread_id="thread-abc")]
            )
            app.push_screen(screen)
            await pilot.pause()

            await screen._fetch_langsmith_link("thread-abc")
            await pilot.pause()

        assert screen._langsmith_urls["thread-abc"] == url

    async def test_fetch_langsmith_link_io_error_logs_debug_and_degrades(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import deepagents_code.config as config_mod

        def boom(_thread_id: str) -> str:
            msg = "network unavailable"
            raise OSError(msg)

        monkeypatch.setattr(config_mod, "build_langsmith_thread_url", boom)
        caplog.set_level(logging.DEBUG, logger=debug_console_mod.__name__)
        screen = DebugConsoleScreen(
            [SnapshotField("Thread", "thread-abc", thread_id="thread-abc")]
        )

        await screen._fetch_langsmith_link("thread-abc")

        assert screen._langsmith_urls == {}
        records = [
            record
            for record in caplog.records
            if record.name == debug_console_mod.__name__
            and "timed out/failed" in record.getMessage()
        ]
        assert len(records) == 1
        assert records[0].levelno == logging.DEBUG
        assert records[0].exc_info is not None

    async def test_fetch_langsmith_link_unexpected_error_logs_warning_and_degrades(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import deepagents_code.config as config_mod

        def boom(_thread_id: str) -> str:
            msg = "resolution bug"
            raise RuntimeError(msg)

        monkeypatch.setattr(config_mod, "build_langsmith_thread_url", boom)
        caplog.set_level(logging.WARNING, logger=debug_console_mod.__name__)
        screen = DebugConsoleScreen(
            [SnapshotField("Thread", "thread-abc", thread_id="thread-abc")]
        )

        await screen._fetch_langsmith_link("thread-abc")

        assert screen._langsmith_urls == {}
        records = [
            record
            for record in caplog.records
            if record.name == debug_console_mod.__name__
            and "errored unexpectedly" in record.getMessage()
        ]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert records[0].exc_info is not None

    async def test_fetch_langsmith_link_none_result_stores_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deepagents_code.config as config_mod

        monkeypatch.setattr(
            config_mod, "build_langsmith_thread_url", lambda _thread_id: None
        )

        app = _Harness()
        async with app.run_test() as pilot:
            screen = DebugConsoleScreen(
                [SnapshotField("Thread", "thread-abc", thread_id="thread-abc")]
            )
            app.push_screen(screen)
            await pilot.pause()

            refreshed: list[bool] = []
            monkeypatch.setattr(
                screen, "_refresh_snapshot", lambda: refreshed.append(True)
            )
            await screen._fetch_langsmith_link("thread-abc")
            await pilot.pause()

        # An unconfigured LangSmith (the common case) resolves to None: nothing
        # is stored and no needless re-render is triggered.
        assert screen._langsmith_urls == {}
        assert refreshed == []

    async def test_refresh_snapshot_without_widget_is_noop(self) -> None:
        app = _Harness()
        async with app.run_test():
            screen = DebugConsoleScreen(
                [SnapshotField("Thread", "thread-abc", thread_id="thread-abc")]
            )
            # Never pushed/composed, so the snapshot widget doesn't exist -- the
            # same state as a worker resolving after the console was dismissed.
            # The NoMatches guard must swallow this rather than raise.
            screen._refresh_snapshot()

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
            # Row renders "Thread  thread-abc  (langsmith)": label (6) + 2-space
            # gutter = col 8, value (10 chars) spans 8-17, 2-space gap, then
            # "(langsmith)" starts at col 20. An x offset of 22 is inside it.
            await pilot.click(snapshot_widget, offset=(22, 0))
            await pilot.pause()

        # The link branch wins and returns early: the trace opens, no copy fires.
        assert len(opened) == 1
        assert copied == []

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


class TestRecordMatchesFilter:
    def test_all_matches_everything(self) -> None:
        record = _log_record("x", level="DEBUG", levelno=logging.DEBUG)
        assert _record_matches_filter(record, "all")

    def test_min_and_only_modes(self) -> None:
        info = _log_record("i", level="INFO", levelno=logging.INFO)
        warning = _log_record("w", level="WARNING", levelno=logging.WARNING)
        assert not _record_matches_filter(info, "min:WARNING")
        assert _record_matches_filter(warning, "min:WARNING")
        assert _record_matches_filter(warning, "only:WARNING")
        assert not _record_matches_filter(info, "only:WARNING")

    def test_custom_numeric_level_sorts_by_levelno(self) -> None:
        # A custom level between INFO and WARNING. The old name-based lookup
        # mapped unknown names to 0 and mis-sorted them below DEBUG.
        notice = _log_record("n", level="NOTICE", levelno=25)
        assert _record_matches_filter(notice, "min:INFO")
        assert not _record_matches_filter(notice, "min:WARNING")

    def test_unknown_threshold_level_shows_record(self) -> None:
        # Unreachable via FilterValue, but a diagnostic must not hide records
        # (or raise KeyError on the poll timer) if a bad filter ever slips
        # through: an unrecognized threshold level matches everything.
        record = _log_record("x", level="INFO", levelno=logging.INFO)
        assert _record_matches_filter(record, cast("FilterValue", "min:BOGUS"))


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
        `True`) so shift+tab and ctrl+t still toggle auto-approve; the
        `test_shift_tab_reverses_focus_*` test only exercises the disabled path.
        Branching on the action name (not the key) means the same gate covers
        the `ctrl+t` binding, which also maps to `toggle_auto_approve`.
        """
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.check_action("toggle_auto_approve", ()) is True

            await pilot.press("ctrl+backslash")
            await pilot.pause()
            assert isinstance(app.screen, DebugConsoleScreen)
            assert app.check_action("toggle_auto_approve", ()) is False

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

    async def test_debug_command_opens_console(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-123")
        async with app.run_test() as pilot:
            await pilot.pause()
            await app._handle_command("/debug")
            await pilot.pause()
            assert isinstance(app.screen, DebugConsoleScreen)

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
            assert "Auto-approve" in snapshot
            assert snapshot["MCP servers"] == "none"

    async def test_build_snapshot_thread_field_is_copyable_and_linkable(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="thread-xyz")
        async with app.run_test():
            thread_field = next(
                field
                for field in app._build_debug_snapshot()
                if field.label == "Thread"
            )
            assert thread_field.value == "thread-xyz"
            assert thread_field.copyable is True
            assert thread_field.thread_id == "thread-xyz"

    async def test_build_snapshot_thread_field_degrades_when_lookup_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
        async with app.run_test():

            def boom(_self: object) -> str:
                msg = "bad thread"
                raise RuntimeError(msg)

            # A class-level data descriptor shadows the instance attribute, so
            # reading `self._lc_thread_id` inside `_thread_field` raises.
            # (`raising=False`: it's normally only an instance attribute.)
            monkeypatch.setattr(
                type(app), "_lc_thread_id", property(boom), raising=False
            )
            thread_field = next(
                field
                for field in app._build_debug_snapshot()
                if field.label == "Thread"
            )
            # The Thread row degrades to a safe, non-interactive placeholder.
            assert thread_field.value.startswith("(unavailable:")
            assert thread_field.copyable is False
            assert thread_field.thread_id is None

    async def test_build_snapshot_formats_mcp_servers(self) -> None:
        app = DeepAgentsApp(agent=MagicMock(), thread_id="t")
        async with app.run_test():
            servers = [
                SimpleNamespace(name="fs", status="connected"),
                SimpleNamespace(name="web", status="error"),
            ]
            app._mcp_server_info = servers  # ty: ignore[invalid-assignment]  # stub servers expose .name/.status
            snapshot = _snapshot_dict(app._build_debug_snapshot())
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
            snapshot = _snapshot_dict(app._build_debug_snapshot())
            # The failing field degrades; the rest of the snapshot still builds.
            assert snapshot["Model"].startswith("(unavailable")
            assert "Version" in snapshot
            assert snapshot["Thread"] == "t"
