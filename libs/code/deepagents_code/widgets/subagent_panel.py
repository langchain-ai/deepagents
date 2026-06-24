"""Live panel showing subagents fanned out from within `js_eval` calls.

When the agent writes code that calls the top-level `task()` global, each
dispatch runs as a subagent *inside* a single `js_eval` tool call which is
invisible to the normal message stream. The QuickJS task bridge emits
lifecycle events on the custom stream. This widget consumes them and renders
a docked, live-updating fan-out panel.

Trust note: `description`/`subagent_type` and `error` strings originate
from LLM-authored JavaScript executed in the sandbox, so they are untrusted.
We route every rendered string through `sanitize_control_chars` which strips
control/escape/bidi characters and only ever render via `Content.styled` /
`markup=False` `Static` updates, so embedded Textual markup and terminal
escapes cannot influence rendering or panel state.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.reactive import reactive
from textual.widgets import Static

from deepagents_code.config import get_glyphs
from deepagents_code.formatting import format_duration
from deepagents_code.theme import get_theme_colors
from deepagents_code.unicode_security import sanitize_control_chars
from deepagents_code.widgets.loading import Spinner

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult
    from textual.timer import Timer

SubagentStatus = Literal["running", "done", "error"]

_MODEL_COL = 16
_TIMING_COL = 6
_STATUS_COL = 5
_MIN_TASK_COL = 16
_SCROLLBAR_RESERVE = 2
_FALLBACK_WIDTH = 100
_MIN_BODY_HEIGHT = 3
_MAX_BODY_HEIGHT = 12
_AGENTS_CHROME_LINES = 1
_TICK_INTERVAL = 0.2


def _right_block_width() -> int:
    """Total width of the right-aligned metadata block (model→time).

    Returns:
        The combined character width of the model and time columns.
    """
    gap = 2
    return _MODEL_COL + gap + _TIMING_COL


@dataclass
class _SubagentRecord:
    """One subagent's live state within a phase."""

    id: str
    label: str
    status: SubagentStatus = "running"
    started_monotonic: float = field(default_factory=time.monotonic)
    duration_ms: int | None = None
    error: str | None = None
    model: str | None = None

    def elapsed_seconds(self) -> float:
        """Seconds since this subagent started (live for running rows).

        Returns:
            The measured duration once finished, else the live elapsed time.
        """
        if self.duration_ms is not None:
            return self.duration_ms / 1000
        return max(0.0, time.monotonic() - self.started_monotonic)


@dataclass
class _Phase:
    """One `js_eval` fan-out batch, keyed by the eval's tool-call id."""

    eval_id: str
    index: int
    records: dict[str, _SubagentRecord] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def add(self, record: _SubagentRecord) -> None:
        """Insert or replace a subagent record, preserving arrival order."""
        if record.id not in self.records:
            self.order.append(record.id)
        self.records[record.id] = record

    def counts(self) -> tuple[int, int]:
        """Return (finished, total) subagent counts for this phase."""
        total = len(self.records)
        done = sum(1 for r in self.records.values() if r.status != "running")
        return done, total

    def any_running(self) -> bool:
        """Whether any subagent in this phase is still running.

        Returns:
            True if at least one subagent has not finished.
        """
        return any(r.status == "running" for r in self.records.values())

    def any_error(self) -> bool:
        """Whether any subagent in this phase ended in error.

        Returns:
            True if at least one subagent ended in error.
        """
        return any(r.status == "error" for r in self.records.values())

    def all_terminal(self) -> bool:
        """Whether the phase has records and none are still running.

        Returns:
            True if the phase has at least one record and all have finished.
        """
        return bool(self.records) and not self.any_running()

    def elapsed_seconds(self) -> float:
        """Wall-clock elapsed for the phase (frozen once all subagents end).

        Returns:
            Live elapsed time while running, else the longest measured duration.
        """
        if not self.records:
            return 0.0
        earliest = min(r.started_monotonic for r in self.records.values())
        if self.all_terminal():
            return max(r.elapsed_seconds() for r in self.records.values())
        return max(0.0, time.monotonic() - earliest)


def _format_timing(seconds: float) -> str:
    """Stable-width elapsed string for the table.

    `format_duration` drops the decimal on whole seconds (`4s` vs `4.2s`),
    which makes a live-ticking value jump left/right by a character each tick.
    Always keep one decimal under a minute so the width stays constant.

    Returns:
        e.g. `4.0s` or `4.2s` under a minute, else `format_duration`'s output.
    """
    if seconds < 60:  # noqa: PLR2004
        return f"{seconds:.1f}s"
    return format_duration(seconds)


def _sanitize(text: str, *, max_chars: int) -> str:
    """Neutralize control/escape/bidi chars and bound length for a one-line label.

    Inputs are LLM/JS-authored and untrusted. This flattens to a single line (newlines
    and ANSI escapes become spaces) so a crafted description cannot inject terminal
    escapes or extra rows.

    Returns:
        A single-line, length-bounded string safe to render as plain text.
    """
    return sanitize_control_chars(text, keep_newlines=False, max_length=max_chars)


class SubagentPanel(Vertical):
    """Docked two-pane panel visualizing `js_eval` subagent fan-out by phase.

    Hidden until the first spawn event. Phases (one per `js_eval`) list on the
    left and the selected phase's subagents render as a scrollable table on the
    right. Focus the panel and use up/down to revisit finished phases. Expands
    while any phase runs, collapses to the header when the turn goes idle, and
    re-expands when a new phase starts.
    """

    can_focus = True
    can_focus_children = False

    DEFAULT_CSS = """
    SubagentPanel {
        height: auto;
        background: $surface;
        border-top: solid $primary;
        display: none;
        padding: 1 2;
    }

    SubagentPanel.-collapsed {
        padding: 0 2;
    }

    SubagentPanel.-visible {
        display: block;
    }

    SubagentPanel:focus {
        border-top: solid $accent;
    }

    SubagentPanel #subagent-header {
        width: 1fr;
        height: 1;
        text-style: bold;
    }

    SubagentPanel #subagent-body {
        width: 1fr;
        height: auto;
        margin-top: 1;
    }

    SubagentPanel #subagent-body.-collapsed {
        display: none;
    }

    SubagentPanel #subagent-phases-scroll {
        width: 24;
        height: 100%;
        border-right: solid $primary-darken-2;
        padding-right: 2;
        margin-right: 2;
    }

    SubagentPanel #subagent-phases-scroll.-hidden {
        display: none;
    }

    SubagentPanel #subagent-agents-scroll {
        width: 1fr;
        height: 100%;
    }
    """

    expanded: reactive[bool] = reactive(default=True, init=False)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize an empty, hidden panel."""
        super().__init__(**kwargs)
        self._phases: dict[str, _Phase] = {}
        self._phase_order: list[str] = []
        self._active_eval_id: str | None = None
        self._selected_eval_id: str | None = None
        self._model_label: str | None = None
        self._applied_height: int | None = None
        self._last_render: dict[str, str] = {}
        self._spinner = Spinner()
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:  # noqa: PLR6301 — Textual widget method
        """Yield the header line and the two-pane body (phases | agents)."""
        yield Static("", id="subagent-header", markup=False)
        with Horizontal(id="subagent-body"):
            with VerticalScroll(id="subagent-phases-scroll"):
                yield Static("", id="subagent-phases", markup=False)
            with VerticalScroll(id="subagent-agents-scroll"):
                yield Static("", id="subagent-agents", markup=False)

    @property
    def _active_phase(self) -> _Phase | None:
        if self._active_eval_id is None:
            return self._phases.get("")
        return self._phases.get(self._active_eval_id)

    def _displayed_phase(self) -> _Phase | None:
        """The phase whose table is shown — the user's pick, else the active one.

        Returns:
            The selected phase if the user navigated to one, else the active
            (latest) phase, or None when no phase has started.
        """
        if self._selected_eval_id is not None:
            phase = self._phases.get(self._selected_eval_id)
            if phase is not None:
                return phase
        return self._active_phase

    @property
    def _batch_complete(self) -> bool:
        """Whether at least one phase exists and nothing is still running."""
        return bool(self._phases) and not self._any_running()

    def on_subagent_event(self, event: dict[str, Any]) -> None:
        """Apply one validated subagent lifecycle event.

        The caller (textual adapter) has already checked `type == "subagent"`
        and that this is the main-agent namespace. We defensively re-validate
        every field here so malformed payloads can never corrupt panel state.
        """
        phase = event.get("phase")
        sub_id = event.get("id")
        if not isinstance(sub_id, str) or not sub_id:
            return
        eval_id = event.get("eval_id")
        eval_key = eval_id if isinstance(eval_id, str) else ""

        if phase == "start":
            self._handle_start(sub_id, eval_key, event)
        elif phase in {"complete", "error"}:
            self._handle_finish(sub_id, phase, event)
        else:
            return

        self._refresh()

    def _handle_start(self, sub_id: str, eval_key: str, event: dict[str, Any]) -> None:
        """Create/replace a running record and (re-)show the panel."""
        phase = self._phases.get(eval_key)
        if phase is None:
            phase = _Phase(eval_id=eval_key, index=len(self._phase_order) + 1)
            self._phases[eval_key] = phase
            self._phase_order.append(eval_key)
        self._active_eval_id = eval_key

        record = _SubagentRecord(
            id=sub_id,
            label=_sanitize(self._row_label(event), max_chars=200),
            model=self._event_model(event),
        )
        phase.add(record)
        self._show()
        self._apply_body_height()
        self._ensure_timer()

    @staticmethod
    def _row_label(event: dict[str, Any]) -> str:
        """Build the row's task label: `"<type>: <label>"`.

        Returns:
            The combined `"<type>: <label>"` string for the task column.
        """
        sub_type = event.get("subagent_type", "subagent")
        return f"{sub_type}: {event.get('label', '')}"

    @staticmethod
    def _event_model(event: dict[str, Any]) -> str | None:
        """Sanitized per-subagent model override from the event, if any.

        Returns:
            The sanitized model string, or None when the event carries no model.
        """
        raw_model = event.get("model")
        if isinstance(raw_model, str):
            return _sanitize(raw_model, max_chars=_MODEL_COL)
        return None

    def _handle_finish(self, sub_id: str, phase: str, event: dict[str, Any]) -> None:
        """Mark a record done/error, record its duration, and stop the timer."""
        record = self._find_record(sub_id)
        if record is None:
            return
        record.status = "done" if phase == "complete" else "error"
        duration = event.get("duration_ms")
        if isinstance(duration, (int, float)):
            record.duration_ms = int(duration)
        if phase == "error":
            raw_err = event.get("error")
            record.error = (
                _sanitize(raw_err, max_chars=120) if isinstance(raw_err, str) else None
            )
        if not self._any_running():
            self._stop_timer()

    def _find_record(self, sub_id: str) -> _SubagentRecord | None:
        """Locate a record by id across all phases.

        Returns:
            The matching record, or None if no phase holds that id.
        """
        for phase in self._phases.values():
            record = phase.records.get(sub_id)
            if record is not None:
                return record
        return None

    def _any_running(self) -> bool:
        """Whether any phase still has a running subagent.

        Returns:
            True if any subagent in any phase is still running.
        """
        return any(phase.any_running() for phase in self._phases.values())

    def on_click(self, event: events.Click) -> None:
        """Click the header to toggle; click a phase row to select it."""
        if self._header_clicked(event):
            self.toggle()
            event.stop()
            return
        if len(self._phase_order) <= 1:
            return
        row = self._clicked_phase_row(event)
        if row is not None and 0 <= row < len(self._phase_order):
            self._selected_eval_id = self._phase_order[row]
            self._refresh()
            event.stop()

    def _header_clicked(self, event: events.Click) -> bool:
        """Whether the click landed on the header line.

        Returns:
            True if the click offset maps onto the header widget.
        """
        try:
            header = self.query_one("#subagent-header", Static)
        except Exception:  # noqa: BLE001 — not mounted yet
            return False
        return event.get_content_offset(header) is not None

    def _clicked_phase_row(self, event: events.Click) -> int | None:
        """Map a click in the phases pane to a phase index (0-based), or None.

        Row 0 is the "Phases" title; rows 1..N map to phases in order.

        Returns:
            The 0-based phase index for the clicked row, or None if the click
            was outside the phases pane.
        """
        try:
            phases = self.query_one("#subagent-phases", Static)
        except Exception:  # noqa: BLE001 — not mounted yet
            return None
        offset = event.get_content_offset(phases)
        if offset is None:
            return None
        return offset.y - 1

    def on_key(self, event: events.Key) -> None:
        """Navigate phases with up/down (or j/k) while the panel is focused."""
        if len(self._phase_order) <= 1:
            return
        if event.key in {"down", "j"}:
            self._move_selection(1)
            event.stop()
        elif event.key in {"up", "k"}:
            self._move_selection(-1)
            event.stop()

    def _move_selection(self, delta: int) -> None:
        """Move the selected phase by `delta`, clamped to the phase list."""
        if not self._phase_order:
            return
        current = self._displayed_phase()
        current_key = current.eval_id if current else self._phase_order[0]
        try:
            index = self._phase_order.index(current_key)
        except ValueError:
            index = 0
        new_index = max(0, min(len(self._phase_order) - 1, index + delta))
        self._selected_eval_id = self._phase_order[new_index]
        self._refresh()

    def reset(self, *, model_label: str | None = None, **_kwargs: Any) -> None:
        """Clear all phases and hide the panel."""
        self._phases.clear()
        self._phase_order.clear()
        self._active_eval_id = None
        self._selected_eval_id = None
        self._applied_height = None
        self._last_render.clear()
        self._model_label = (
            _sanitize(model_label, max_chars=_MODEL_COL) if model_label else None
        )
        self._stop_timer()
        self.remove_class("-visible")

    def _show(self) -> None:
        """Make the panel visible (idempotent)."""
        self.add_class("-visible")

    def toggle(self) -> None:
        """Toggle the body open/closed. This is the only thing that changes it."""
        self.expanded = not self.expanded

    def watch_expanded(self, expanded: bool) -> None:
        """Show/hide the body when the expanded state changes."""
        try:
            body = self.query_one("#subagent-body")
        except Exception:  # noqa: BLE001 — body not mounted yet
            return
        body.set_class(not expanded, "-collapsed")
        # Drop vertical padding when collapsed so the header bar is thin.
        self.set_class(not expanded, "-collapsed")
        if expanded:
            self._apply_body_height()
        self._refresh()
        # On re-expand the body was just shown (it had zero width while
        # collapsed), so the agents pane width isn't known yet. Re-render once
        # layout settles so the right-aligned columns use the real width.
        if expanded:
            self.call_after_refresh(self._refresh)

    def on_resize(self, _event: events.Resize) -> None:
        """Re-render so width-dependent column alignment tracks the new size."""
        self._refresh()

    def _ensure_timer(self) -> None:
        """Start the refresh timer if it isn't already running."""
        if self._timer is None:
            self._timer = self.set_interval(_TICK_INTERVAL, self._refresh)

    def _stop_timer(self) -> None:
        """Stop and drop the refresh timer if running."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _counts(self) -> tuple[int, int]:
        """(finished, total) for the displayed phase.

        Returns:
            A `(finished, total)` count for the displayed phase, or `(0, 0)`.
        """
        phase = self._displayed_phase()
        return phase.counts() if phase else (0, 0)

    def _turn_counts(self) -> tuple[int, int, int]:
        """Sum subagent counts across all phases.

        Returns:
            A `(finished, total, failed)` tuple over the whole turn.
        """
        total = done = failed = 0
        for phase in self._phases.values():
            for record in phase.records.values():
                total += 1
                if record.status != "running":
                    done += 1
                if record.status == "error":
                    failed += 1
        return done, total, failed

    def _body_height(self) -> int:
        """Constant body height for the turn — sized to the largest phase.

        Returns:
            A cell height clamped to the configured bounds, stable across phase
            switches.
        """
        if not self._phases:
            return _MIN_BODY_HEIGHT
        max_rows = max(len(p.records) for p in self._phases.values())
        agents_lines = _AGENTS_CHROME_LINES + max_rows
        phases_lines = 1 + len(self._phases)  # "Phases" title + one per phase
        return min(_MAX_BODY_HEIGHT, max(_MIN_BODY_HEIGHT, agents_lines, phases_lines))

    def _apply_body_height(self) -> None:
        """Lock the body to a constant height; only re-assign when it changes.

        Re-assigning `styles.height` every timer tick forces a relayout and
        causes visible flicker, so we cache the applied value and only set it
        when the phase composition actually changes the needed height.
        """
        if not self.expanded:
            return
        height = self._body_height()
        if height == self._applied_height:
            return
        with contextlib.suppress(Exception):
            self.query_one("#subagent-body").styles.height = height
            self._applied_height = height

    def _update_cached(self, widget_id: str, content: Content) -> None:
        """Update a Static only when its rendered text changed (anti-flicker)."""
        if self._last_render.get(widget_id) == content.plain:
            return
        try:
            self.query_one(f"#{widget_id}", Static).update(content)
        except Exception:  # noqa: BLE001 — not mounted yet
            return
        self._last_render[widget_id] = content.plain

    def _refresh(self) -> None:
        """Re-render all three regions (header, phases pane, agents pane)."""
        self._refresh_header()
        self._refresh_phases()
        self._refresh_agents()

    def _refresh_header(self) -> None:
        """Render the header: status icon, label, whole-turn totals, toggle hint."""
        colors = get_theme_colors(self)
        glyphs = get_glyphs()
        caret = "▾" if self.expanded else "▸"
        done, total, failed = self._turn_counts()
        if self._any_running() or not total:
            icon, tint = self._spinner.next_frame(), colors.warning
        elif failed:
            icon, tint = glyphs.error, colors.error
        else:
            icon, tint = glyphs.checkmark, colors.success
        lead_text = f"{caret} {icon}  dynamic subagents"
        parts: list[Content] = [Content.styled(lead_text, tint)]
        left_len = len(lead_text)

        if self.expanded and total:
            meta = self._header_meta_parts(done, total, failed, colors)
            parts.extend(meta)
            left_len += sum(len(p.plain) for p in meta)
        hint = (
            "click or Ctrl+G to collapse"
            if self.expanded
            else "click or Ctrl+G to expand"
        )
        spacer = max(2, self._header_width() - left_len - len(hint))
        parts.append(Content.styled(" " * spacer + hint, colors.muted))
        self._update_cached("subagent-header", Content.assemble(*parts))

    def _header_meta_parts(
        self,
        done: int,
        total: int,
        failed: int,
        colors: Any,  # noqa: ANN401 — ThemeColors
    ) -> list[Content]:
        """Whole-turn totals (+ phase count, + failures) shown when expanded.

        Returns:
            The styled `Content` pieces appended after the header label.
        """
        meta_text = f"   {done}/{total} done"
        if len(self._phase_order) > 1:
            meta_text += f"  ·  {len(self._phase_order)} phases"
        parts: list[Content] = [Content.styled(meta_text, colors.muted)]
        if failed:
            parts.append(Content.styled(f"  ·  {failed} failed", colors.error))
        return parts

    def _header_width(self) -> int:
        """Current cell width of the header line (fallback until laid out).

        Returns:
            The header width, or a fallback before first layout.
        """
        try:
            width = self.query_one("#subagent-header", Static).size.width
        except Exception:  # noqa: BLE001 — not mounted yet
            width = 0
        return width if width and width > 0 else _FALLBACK_WIDTH

    def _refresh_phases(self) -> None:
        """Render the left pane: one selectable row per phase (eval batch)."""
        try:
            scroll = self.query_one("#subagent-phases-scroll")
        except Exception:  # noqa: BLE001 — not mounted yet
            return
        # Hide only when there are no phases at all; otherwise always show the
        # list (even a single phase) for a consistent two-pane view.
        if not self._phase_order:
            scroll.add_class("-hidden")
            self._update_cached("subagent-phases", Content(""))
            return
        scroll.remove_class("-hidden")
        colors = get_theme_colors(self)
        displayed = self._displayed_phase()
        displayed_key = displayed.eval_id if displayed else None
        rows: list[Content] = [Content.styled("Phases", colors.muted)]
        rows.extend(
            self._phase_row(
                self._phases[key], selected=key == displayed_key, colors=colors
            )
            for key in self._phase_order
        )
        self._update_cached("subagent-phases", Content("\n").join(rows))

    def _phase_row(
        self,
        phase: _Phase,
        *,
        selected: bool,
        colors: Any,  # noqa: ANN401 — ThemeColors
    ) -> Content:
        """Render one phase row: caret, status glyph, index, counts, elapsed.

        Returns:
            The styled `Content` for the phase's row in the left pane.
        """
        glyphs = get_glyphs()
        done, total = phase.counts()
        if phase.all_terminal():
            mark = glyphs.error if phase.any_error() else glyphs.checkmark
        elif phase.eval_id == self._active_eval_id:
            mark = "▸"
        else:
            mark = glyphs.bullet
        caret = "›" if selected else " "  # noqa: RUF001 — selection marker
        tint = colors.primary if selected else colors.muted
        elapsed = _format_timing(phase.elapsed_seconds())
        return Content.styled(
            f"{caret} {mark} {phase.index} {done}/{total} · {elapsed}", tint
        )

    def _agents_width(self) -> int:
        """Usable width of the agents pane (fallback until laid out).

        Returns:
            The agents pane width minus a scrollbar reserve, floored at the
            minimum task width.
        """
        try:
            width = self.query_one("#subagent-agents", Static).size.width
        except Exception:  # noqa: BLE001 — not mounted yet
            width = 0
        if not width or width <= 0:
            width = _FALLBACK_WIDTH
        # Keep the flush-right column off the scroll bar.
        return max(_MIN_TASK_COL, width - _SCROLLBAR_RESERVE)

    def _task_col(self) -> int:
        """Width of the flexible task column so the right block sits flush-right.

        Returns:
            The task column width, clamped to a sensible minimum.
        """
        width = self._agents_width()
        return max(_MIN_TASK_COL, width - _STATUS_COL - _right_block_width())

    def _refresh_agents(self) -> None:
        """Render the right pane: heading + one row per subagent in the phase."""
        phase = self._displayed_phase()
        glyphs = get_glyphs()
        colors = get_theme_colors(self)
        rows: list[Content] = []
        if phase is not None and phase.order:
            task_col = self._task_col()
            rows.append(self._heading_row(task_col, colors))
            rows.extend(
                self._render_row(phase.records[sub_id], task_col, glyphs, colors)
                for sub_id in phase.order
            )
        self._update_cached(
            "subagent-agents", Content("\n").join(rows) if rows else Content("")
        )

    @staticmethod
    def _right_block(model: str, timing: str) -> str:
        """Format the fixed-width, flush-right metadata columns (model, time).

        Returns:
            The concatenated, column-aligned metadata string.
        """
        return (
            f"{model[:_MODEL_COL].ljust(_MODEL_COL)}  "
            f"{timing[:_TIMING_COL].rjust(_TIMING_COL)}"
        )

    def _render_row(
        self,
        record: _SubagentRecord,
        task_col: int,
        glyphs: Any,  # noqa: ANN401 — Glyphs
        colors: Any,  # noqa: ANN401 — ThemeColors
    ) -> Content:
        """Render one row: status | task (left) | model · time (right).

        Returns:
            The styled, width-filling row `Content`.
        """
        if record.status == "running":
            icon = self._spinner.current_frame()
            tint = colors.warning
            timing = _format_timing(record.elapsed_seconds())
        elif record.status == "done":
            icon = glyphs.checkmark
            tint = colors.success
            timing = _format_timing(record.elapsed_seconds())
        else:
            icon = glyphs.error
            tint = colors.error
            timing = record.error or "error"
        task = _sanitize(record.label, max_chars=task_col - 1).ljust(task_col)
        model = record.model or self._model_label or ""
        right = self._right_block(model, timing)
        return Content.assemble(
            Content.styled(f"  {icon}  ", tint),
            Content.styled(task, colors.foreground),
            Content.styled(right, colors.muted),
        )

    def _heading_row(
        self,
        task_col: int,
        colors: Any,  # noqa: ANN401 — ThemeColors
    ) -> Content:
        """Column heading row aligned to the data rows.

        Returns:
            A single dim heading line spanning the same columns as the rows.
        """
        prefix = " " * _STATUS_COL
        task = "TASK".ljust(task_col)
        right = self._right_block("MODEL", "TIME")
        return Content.styled(f"{prefix}{task}{right}", colors.muted)
