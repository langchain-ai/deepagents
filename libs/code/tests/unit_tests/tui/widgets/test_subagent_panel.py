"""Behavioral tests for the SubagentPanel widget.

Each test mounts the panel in a minimal App, feeds it realistic subagent
lifecycle events, and asserts on rendered content / observable state — not on
types. Uses the Textual `run_test()` pilot harness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.widgets import Static

from deepagents_code.tui.widgets.subagent_panel import (
    SubagentPanel,
    _Phase,
    _SubagentRecord,
)

if TYPE_CHECKING:
    from typing import Any


class PanelApp(App):
    """Minimal app that mounts a SubagentPanel for testing."""

    def compose(self) -> ComposeResult:
        yield SubagentPanel(id="panel")


class _FakeClick:
    """Stand-in for a Textual Click that reports an offset for one target id."""

    def __init__(self, *, row_y: int, target_id: str) -> None:
        self._y = row_y
        self._target = target_id
        self.stopped = False

    def get_content_offset(self, widget: object) -> Offset | None:
        if getattr(widget, "id", None) == self._target:
            return Offset(0, self._y)
        return None

    def stop(self) -> None:
        self.stopped = True


def _start(
    sub_id: str, eval_id: str, desc: str = "task", label: str | None = "work"
) -> dict:
    event = {
        "type": "subagent",
        "phase": "start",
        "id": sub_id,
        "eval_id": eval_id,
        "subagent_type": "research",
        "description": desc,
    }
    if label is not None:
        event["label"] = label
    return event


def _complete(sub_id: str, eval_id: str, duration_ms: int = 100) -> dict:
    return {
        "type": "subagent",
        "phase": "complete",
        "id": sub_id,
        "eval_id": eval_id,
        "duration_ms": duration_ms,
    }


def _error(sub_id: str, eval_id: str, message: str = "boom") -> dict:
    return {
        "type": "subagent",
        "phase": "error",
        "id": sub_id,
        "eval_id": eval_id,
        "duration_ms": 50,
        "error": message,
    }


def _render(widget: Static) -> str:
    content = widget.render()
    plain = getattr(content, "plain", None)
    return plain if isinstance(plain, str) else str(content)


def _displayed_id(panel: SubagentPanel) -> str:
    phase = panel._displayed_phase()
    assert phase is not None
    return phase.eval_id


class TestSelection:
    async def test_selection_follows_active_then_locks_on_navigation(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1", label="phase one work"))
            panel.on_subagent_event(_complete("a", "E1"))
            panel.on_subagent_event(_start("b", "E2", label="phase two work"))
            await pilot.pause()
            assert _displayed_id(panel) == "E2"
            panel._move_selection(-1)
            await pilot.pause()
            assert _displayed_id(panel) == "E1"
            rows = _render(pilot.app.query_one("#subagent-agents", Static))
            assert "phase one work" in rows


class TestHeaderToggle:
    async def test_user_collapse_persists_across_turn_reset(self) -> None:
        async with PanelApp().run_test(size=(160, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            await pilot.pause()
            panel.toggle()  # user closes it
            panel.reset()  # new user turn
            panel.on_subagent_event(_start("b", "E2"))
            await pilot.pause()
            assert panel.expanded is False  # preference persists

    # 34 columns is the supported floor: the hint plus its 2-cell margin claims
    # 29 of the 30 content columns, leaving the summary at its 1-cell minimum.
    # Narrower than that and the hint is pushed past the panel edge.
    @pytest.mark.parametrize("width", [80, 40, 34])
    async def test_narrow_header_preserves_full_toggle_hint(self, width: int) -> None:
        async with PanelApp().run_test(size=(width, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            # A second phase and a failure inflate the summary past any of these
            # widths; finalize_running() cancels the still-running "b", which
            # both widens the summary further and stops the spinner timer so the
            # header stops re-rendering under us.
            panel.on_subagent_event(_start("a", "E1"))
            panel.on_subagent_event(_error("a", "E1"))
            panel.on_subagent_event(_start("b", "E2"))
            panel.finalize_running()
            await pilot.pause()

            summary = pilot.app.query_one("#subagent-header-summary", Static)
            hint = pilot.app.query_one("#subagent-header-hint", Static)
            text = "click or Ctrl+T to collapse"
            # `_render` returns the unclipped content, so this is the "summary
            # really is overflowing" precondition; the ellipsis assertion below
            # reads the painted strip, which is where truncation happens.
            assert summary.size.width < len(_render(summary))
            assert "\u2026" in summary.render_line(0).text
            assert _render(hint) == text
            assert hint.size.width == len(text)
            assert hint.region.right <= panel.content_region.right


class TestReset:
    async def test_panel_clears_on_next_turn(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            panel.on_subagent_event(_complete("a", "E1"))
            await pilot.pause()
            assert panel.has_class("-visible")
            panel.prepare_turn()
            await pilot.pause()
            assert not panel.has_class("-visible")
            assert panel._phase_order == []
            panel.on_subagent_event(_start("b", "E2"))
            await pilot.pause()
            assert panel.has_class("-visible")
            assert panel._phase_order == ["E2"]
            assert panel._find_record("a") is None

    async def test_finalize_running_marks_cancelled(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            panel.on_subagent_event(_start("b", "E1"))
            await pilot.pause()
            assert panel._any_running() is True
            # The turn is interrupted — finalize the in-flight rows.
            panel.finalize_running()
            await pilot.pause()
            assert panel._any_running() is False
            rec_a = panel._find_record("a")
            rec_b = panel._find_record("b")
            assert rec_a is not None
            assert rec_b is not None
            assert rec_a.status == "cancelled"
            assert rec_b.status == "cancelled"
            header = _render(pilot.app.query_one("#subagent-header-summary", Static))
            assert "2 cancelled" in header

    async def test_finalize_running_preserves_finished_rows(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            panel.on_subagent_event(_complete("a", "E1"))
            panel.on_subagent_event(_start("b", "E1"))  # still running
            await pilot.pause()
            panel.finalize_running()
            await pilot.pause()
            rec_a = panel._find_record("a")
            rec_b = panel._find_record("b")
            assert rec_a is not None
            assert rec_b is not None
            assert rec_a.status == "done"  # already finished — untouched
            assert rec_b.status == "cancelled"  # in-flight — cancelled

    async def test_prepare_turn_clears_stuck_running_rows(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            # A subagent starts but never finishes (e.g. the turn was cancelled
            # before a terminal event arrived — CancelledError bypasses the
            # bridge's terminal-event emission).
            panel.on_subagent_event(_start("a", "E1"))
            await pilot.pause()
            assert panel._any_running() is True
            # The next turn must not persist a stale, still-running fan-out.
            panel.prepare_turn()
            await pilot.pause()
            assert not panel.has_class("-visible")
            assert panel._phase_order == []


class TestSafety:
    async def test_strips_escapes_and_bounds_length(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            nasty = "evil\x1b[31m\nsecond line"
            panel.on_subagent_event(_start("a", "E1", label=nasty))
            await pilot.pause()
            rows = _render(pilot.app.query_one("#subagent-agents", Static))
            assert "\x1b" not in rows  # escape stripped
            # The data row is a single line (newline flattened to a space).
            data_row = rows.split("\n")[1]
            assert "\n" not in data_row
            assert "second line" in data_row


class TestPhaseTiming:
    def test_phase_elapsed_is_wall_clock_not_longest_subagent(self) -> None:
        # Two subagents with staggered starts, each running 3s:
        #   A: starts 100.0, ends 103.0
        #   B: starts 102.0, ends 105.0
        # Wall-clock span is 5.0s (100.0 -> 105.0), not the 3.0s longest run.
        phase = _Phase(eval_id="E1", index=1)
        phase.add(
            _SubagentRecord(
                id="a",
                label="a",
                status="done",
                started_monotonic=100.0,
                duration_ms=3000,
            )
        )
        phase.add(
            _SubagentRecord(
                id="b",
                label="b",
                status="done",
                started_monotonic=102.0,
                duration_ms=3000,
            )
        )
        assert phase.elapsed_seconds() == pytest.approx(5.0)
