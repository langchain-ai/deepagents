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


class TestLifecycle:
    async def test_hidden_until_first_spawn(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            assert not panel.has_class("-visible")

    async def test_visible_and_expanded_after_spawn(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            await pilot.pause()
            assert panel.has_class("-visible")
            assert panel.expanded is True
            assert panel._counts() == (0, 1)

    async def test_any_running_tracks_terminal_state(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            panel.on_subagent_event(_start("b", "E1"))
            await pilot.pause()
            assert panel._any_running() is True
            panel.on_subagent_event(_complete("a", "E1"))
            panel.on_subagent_event(_complete("b", "E1"))
            await pilot.pause()
            assert panel._any_running() is False
            assert panel._counts() == (2, 2)

    async def test_missing_label_falls_back_to_short_description(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            description = "Review\n" + "a" * 100
            panel.on_subagent_event(_start("a", "E1", desc=description, label=None))
            await pilot.pause()
            rows = _render(pilot.app.query_one("#subagent-agents", Static))
            assert ("Review " + "a" * 100)[:60] in rows

    async def test_error_shows_full_reason_in_task_column(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1", label="db.ts"))
            panel.on_subagent_event(_error("a", "E1", message="rate limit exceeded"))
            await pilot.pause()
            assert panel._any_running() is False
            record = panel._find_record("a")
            assert record is not None
            assert record.status == "error"
            assert record.error == "rate limit exceeded"
            # The full reason appears in the wide task column; it would be cut to
            # ~6 chars if it were rendered in the (narrow) TIME column.
            rows = _render(pilot.app.query_one("#subagent-agents", Static))
            assert "rate limit exceeded" in rows

    async def test_orphan_error_surfaces_without_start(self) -> None:
        # An error whose `start` was dropped on the wire must still surface as a
        # failed row rather than vanishing silently.
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_error("orphan", "E1", message="dropped boom"))
            await pilot.pause()
            assert panel.has_class("-visible")
            record = panel._find_record("orphan")
            assert record is not None
            assert record.status == "error"
            assert record.error == "dropped boom"
            rows = _render(pilot.app.query_one("#subagent-agents", Static))
            assert "dropped boom" in rows

    async def test_orphan_error_after_prepare_turn_replaces_prior_turn(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1", label="old work"))
            panel.on_subagent_event(_complete("a", "E1"))
            panel.prepare_turn()
            panel.on_subagent_event(_error("orphan", "E2", message="dropped boom"))
            panel.on_subagent_event(_start("b", "E3", label="later work"))
            await pilot.pause()
            assert panel._phase_order == ["E2", "E3"]
            assert panel._find_record("a") is None
            assert panel._find_record("orphan") is not None
            assert panel._find_record("b") is not None

    async def test_orphan_error_becomes_active_phase(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1", label="old work"))
            panel.on_subagent_event(_complete("a", "E1"))
            panel.on_subagent_event(_error("orphan", "E2", message="dropped boom"))
            await pilot.pause()
            assert _displayed_id(panel) == "E2"
            rows = _render(pilot.app.query_one("#subagent-agents", Static))
            assert "dropped boom" in rows

    async def test_orphan_error_without_duration_still_renders(self) -> None:
        # The realistic dropped-wire case: a partial error event missing
        # `duration_ms` must still surface, leaving the duration unset rather
        # than crashing on the missing/non-numeric field.
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(
                {
                    "type": "subagent",
                    "phase": "error",
                    "id": "orphan",
                    "eval_id": "E1",
                    "error": "dropped boom",
                }
            )
            await pilot.pause()
            record = panel._find_record("orphan")
            assert record is not None
            assert record.status == "error"
            assert record.duration_ms is None
            rows = _render(pilot.app.query_one("#subagent-agents", Static))
            assert "dropped boom" in rows


class TestPhases:
    async def test_phases_accumulate_and_track_active(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            panel.on_subagent_event(_complete("a", "E1"))
            panel.on_subagent_event(_start("b", "E2"))
            await pilot.pause()
            assert panel._phase_order == ["E1", "E2"]
            assert panel._active_eval_id == "E2"
            # Earlier phase retained; active table shows only the new phase.
            assert set(panel._phases["E1"].records) == {"a"}
            assert _displayed_id(panel) == "E2"

    async def test_eval_without_subagents_creates_no_phase(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            # A complete with no prior start is a no-op (no phantom phase).
            panel.on_subagent_event(_complete("ghost", "E9"))
            await pilot.pause()
            assert panel._phase_order == []
            assert not panel.has_class("-visible")

    async def test_missing_eval_id_groups_into_single_phase(self) -> None:
        # When the runtime exposes no tool_call_id the producer omits `eval_id`;
        # such events share the empty-string phase key. Document that collapse so
        # a future change that needs to distinguish them is forced to notice.
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            for sub_id in ("a", "b"):
                panel.on_subagent_event(
                    {
                        "type": "subagent",
                        "phase": "start",
                        "id": sub_id,
                        "subagent_type": "research",
                        "description": "task",
                        "label": "work",
                    }
                )
            await pilot.pause()
            assert panel._phase_order == [""]
            assert set(panel._phases[""].records) == {"a", "b"}


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

    async def test_selection_clamped_at_bounds(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            panel.on_subagent_event(_start("b", "E2"))
            await pilot.pause()
            panel._move_selection(-5)  # past the top
            assert _displayed_id(panel) == "E1"
            panel._move_selection(5)  # past the bottom
            assert _displayed_id(panel) == "E2"

    async def test_click_selects_phase_row(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            panel.on_subagent_event(_start("b", "E2"))
            await pilot.pause()
            # Row 1 is the first phase (row 0 is the "Phases" title).
            panel.on_click(
                cast("Any", _FakeClick(row_y=1, target_id="subagent-phases"))
            )
            await pilot.pause()
            assert _displayed_id(panel) == "E1"


class TestAgentsTable:
    async def test_row_shows_label_not_description(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(
                _start("a", "E1", desc="a long boilerplate prompt", label="R16: #3")
            )
            await pilot.pause()
            rows = _render(pilot.app.query_one("#subagent-agents", Static))
            assert "R16: #3" in rows  # the label is what's rendered
            assert "boilerplate" not in rows  # the description never reaches the row

    async def test_rows_show_session_model_label(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.reset(model_label="opus")
            panel.on_subagent_event(_start("a", "E1", label="x"))
            await pilot.pause()
            rows = _render(pilot.app.query_one("#subagent-agents", Static))
            assert "opus" in rows

    async def test_rows_show_headings(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1", label="x"))
            await pilot.pause()
            rows = _render(pilot.app.query_one("#subagent-agents", Static))
            assert "TASK" in rows
            assert "MODEL" in rows
            assert "TIME" in rows


class TestHeaderToggle:
    async def test_toggle_flips_expanded(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            await pilot.pause()
            assert panel.expanded is True
            panel.toggle()
            await pilot.pause()
            assert panel.expanded is False

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

    async def test_header_shows_turn_totals_and_failed(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            panel.on_subagent_event(_start("b", "E1"))
            panel.on_subagent_event(_complete("a", "E1"))
            panel.on_subagent_event(_error("b", "E1"))
            await pilot.pause()
            header = _render(pilot.app.query_one("#subagent-header", Static))
            assert "2/2 done" in header
            assert "1 phase" in header  # singular for a single phase
            assert "1 failed" in header

    async def test_header_phase_count_pluralizes(self) -> None:
        async with PanelApp().run_test(size=(200, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            await pilot.pause()
            header = _render(pilot.app.query_one("#subagent-header", Static))
            assert "1 phase" in header
            assert "1 phases" not in header  # singular, not "1 phases"
            # A second eval batch makes it plural.
            panel.on_subagent_event(_start("b", "E2"))
            await pilot.pause()
            header = _render(pilot.app.query_one("#subagent-header", Static))
            assert "2 phases" in header


class TestReset:
    async def test_reset_hides_and_clears(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            await pilot.pause()
            panel.reset()
            await pilot.pause()
            assert not panel.has_class("-visible")
            assert panel._phase_order == []
            assert panel._counts() == (0, 0)

    async def test_panel_persists_until_next_workflow(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1"))
            panel.on_subagent_event(_complete("a", "E1"))
            await pilot.pause()
            assert panel.has_class("-visible")
            # A new turn begins but spawns no subagents — results persist.
            panel.prepare_turn()
            await pilot.pause()
            assert panel.has_class("-visible")
            assert panel._phase_order == ["E1"]
            # The next workflow's first subagent clears the prior fan-out.
            panel.on_subagent_event(_start("b", "E2"))
            await pilot.pause()
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
            header = _render(pilot.app.query_one("#subagent-header", Static))
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


class TestStability:
    async def test_body_height_stable_across_phase_switch(self) -> None:
        async with PanelApp().run_test(size=(160, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            # Phase 1 has 3 subagents; phase 2 has 1.
            for sid in ("a", "b", "c"):
                panel.on_subagent_event(_start(sid, "E1"))
                panel.on_subagent_event(_complete(sid, "E1"))
            panel.on_subagent_event(_start("d", "E2"))
            await pilot.pause()
            height_active = panel._body_height()
            panel._move_selection(-1)  # show the smaller phase 1
            await pilot.pause()
            assert panel._body_height() == height_active  # sized to largest phase

    async def test_idle_refresh_skips_redundant_agent_update(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async with PanelApp().run_test(size=(160, 24)) as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event(_start("a", "E1", label="x"))
            panel.on_subagent_event(_complete("a", "E1"))
            await pilot.pause()
            agents = pilot.app.query_one("#subagent-agents", Static)
            calls = {"n": 0}

            def _counting(*_args: object, **_kwargs: object) -> None:
                calls["n"] += 1

            monkeypatch.setattr(agents, "update", _counting)
            panel._refresh()  # nothing changed since last render
            assert calls["n"] == 0


class TestSafety:
    async def test_ignored_events_are_noops(self) -> None:
        async with PanelApp().run_test() as pilot:
            panel = pilot.app.query_one("#panel", SubagentPanel)
            panel.on_subagent_event({"phase": "start"})  # no id
            panel.on_subagent_event({"phase": "weird", "id": "a"})  # bad phase
            await pilot.pause()
            assert panel._phase_order == []
            assert not panel.has_class("-visible")

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
