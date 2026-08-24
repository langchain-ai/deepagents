"""Inline prompt history search panel for the chat composer.

Opened by a first Ctrl+R; a second Ctrl+R escalates to the full
`PromptClipboardScreen` modal. Key handling lives on `ChatInput`
(`_prompt_search_active` / `_handle_prompt_search_key`), which the app action
routes to.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from textual import events
    from textual.app import ComposeResult

logger = logging.getLogger(__name__)

PROMPT_SEARCH_MAX_ROWS = 5
"""Result rows the inline panel shows before the list scrolls."""


def prompt_search_hint() -> str:
    """Build the footer line for the current charset mode.

    The Ctrl+R mention is what makes the modal tier discoverable. The line is
    kept short enough to wrap within `PROMPT_SEARCH_MAX_HINT_ROWS` at the
    narrow widths the composer supports.

    Returns:
        The hint text, using ASCII glyphs on terminals that need them.
    """
    from deepagents_code.config import get_glyphs

    glyphs = get_glyphs()
    sep = f"  {glyphs.bullet}  "
    return sep.join(
        (
            f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate",
            "Tab/Shift+Tab page",
            "Enter insert",
            "Ctrl+R full view",
            "Esc cancel",
        )
    )


PROMPT_SEARCH_PAGE_SIZE = PROMPT_SEARCH_MAX_ROWS
"""Rows Tab/Shift+Tab jump through the filtered list."""

PROMPT_SEARCH_WINDOW = 50
"""Most rows rendered in the DOM at once, windowed around the selection.

Mounting one `PromptSearchOption` per filtered prompt costs real time at the
100-entry history cap and throws most of them away above the 5-row viewport, so
the panel renders a sliding window instead. Navigation re-windows via
`update_selection`, so any reachable row is mounted before it can be selected;
unmounted rows are simply never the selection target.
"""

PROMPT_SEARCH_MAX_HINT_ROWS = 3
"""Hint rows the panel is willing to render; it wraps on narrow windows."""

PROMPT_SEARCH_PANEL_ROWS = 1 + PROMPT_SEARCH_MAX_ROWS + PROMPT_SEARCH_MAX_HINT_ROWS
"""Rows the panel reports at its tallest: query, results, and wrapped hint.

This is the ceiling on what `show()` can report, so it is what `ChatInputBox`
reserves when fitting a manual composer height and what the panel's own
`max-height` clamps to. `show()` clamps its hint estimate to
`PROMPT_SEARCH_MAX_HINT_ROWS` so the two can never disagree.
"""


def prompt_title(prompt: str) -> str:
    """Return a prompt's first non-empty line for a one-row summary."""
    nonempty = [line.strip() for line in prompt.splitlines() if line.strip()]
    return (nonempty[0] if nonempty else prompt.strip()) or "(empty prompt)"


def _window_bounds(total: int, selected_index: int) -> tuple[int, int]:
    """Return the [start, stop) window of rows to render around a selection."""
    if total <= PROMPT_SEARCH_WINDOW:
        return 0, total
    # Keep the selection comfortably inside the window so single-step moves
    # stay mounted. It sits near the window top, so most mounted rows are the
    # older ones the user is scanning toward.
    start = max(
        0,
        min(selected_index - PROMPT_SEARCH_WINDOW // 5, total - PROMPT_SEARCH_WINDOW),
    )
    return start, start + PROMPT_SEARCH_WINDOW


def filter_prompts(prompts: tuple[str, ...], query: str) -> list[str]:
    """Return prompts containing the query, case-insensitively."""
    needle = query.strip().casefold()
    if not needle:
        return list(prompts)
    return [prompt for prompt in prompts if needle in prompt.casefold()]


class PromptSearchInput(Input):
    """Query field for the inline prompt search panel.

    Plain `Input` apart from the class name, which lets `ChatInput` filter
    `Input.Changed` / `Input.Submitted` messages down to this field, plus
    bindings for the keys the panel owns while focused.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        # The app binds these chords with priority, so they never reach a
        # focused widget unless `check_action` steps them aside; it does while
        # the panel is open (see `DeepAgentsApp.check_action`).
        Binding("escape", "abandon_search", "Cancel", show=False, priority=True),
        Binding("tab", "page(True)", "Page Down", show=False, priority=True),
        Binding("shift+tab", "page(False)", "Page Up", show=False, priority=True),
    ]

    class AbandonSearch(Message):
        """Posted on Escape or empty-query Backspace, to close the search."""

    class PageRequested(Message):
        """Posted on Tab (older) or Shift+Tab (newer) to page the results."""

        def __init__(self, *, older: bool) -> None:
            """Initialize with the page direction."""
            super().__init__()
            self.older = older

    def action_abandon_search(self) -> None:
        """Forward Escape as an abandon request."""
        self.post_message(self.AbandonSearch())

    def action_page(self, older: bool) -> None:
        """Forward Tab/Shift+Tab as a page request."""
        self.post_message(self.PageRequested(older=older))

    async def _on_key(self, event: events.Key) -> None:
        """Forward the empty-query Backspace chord; defer the rest to Input."""
        if event.key == "backspace" and not self.value:
            event.prevent_default()
            event.stop()
            self.post_message(self.AbandonSearch())
            return
        await super()._on_key(event)


class PromptSearchOption(Static):
    """One clickable prompt row in the inline search panel."""

    DEFAULT_CSS = """
    PromptSearchOption {
        height: 1;
        padding: 0 1;
        /* Rows are one cell high, so overlong titles clip visually. Textual
           only ellipsizes truncated (nowrap) lines, so nowrap must accompany
           text-overflow: ellipsis for the clipped end to be marked. */
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }

    PromptSearchOption.prompt-search-selected {
        background: $primary;
        color: $background;
    }
    """

    class Clicked(Message):
        """Message sent when a prompt row is clicked."""

        def __init__(self, index: int) -> None:
            """Initialize with the clicked row index."""
            super().__init__()
            self.index = index

    def __init__(self, title: str, index: int, *, is_selected: bool) -> None:
        """Initialize the row."""
        super().__init__(Content.styled(title, "bold"))
        self.index = index
        self._is_selected = is_selected
        if is_selected:
            self.add_class("prompt-search-selected")

    def set_content(self, title: str, index: int, *, is_selected: bool) -> None:
        """Update the row in place during a panel rebuild."""
        self.update(Content.styled(title, "bold"))
        self.index = index
        self.set_selected(is_selected=is_selected)

    def set_selected(self, *, is_selected: bool) -> None:
        """Toggle the selected styling."""
        self._is_selected = is_selected
        self.set_class(is_selected, "prompt-search-selected")

    def on_click(self) -> None:
        """Post the click up to `ChatInput`."""
        self.post_message(self.Clicked(self.index))


class PromptSearchPanel(Vertical):
    """Inline prompt history search rendered above the input row.

    Display-only: `ChatInput` owns the query, filtered snapshot, and
    selection, drives key handling, and re-opens the modal tier. The panel
    mirrors `CompletionPopup`: hidden by default, rebuilt via generation
    counters so a stale async rebuild cannot re-show a dismissed panel, and
    reports its rendered row count so `ChatInputBox` can reserve space.
    """

    DEFAULT_CSS = f"""
    PromptSearchPanel {{
        display: none;
        height: auto;
        max-height: {PROMPT_SEARCH_PANEL_ROWS};
    }}

    PromptSearchPanel #prompt-search-input {{
        height: 1;
        padding: 0 1;
        border: none;
        background: transparent;
    }}

    PromptSearchPanel #prompt-search-input:focus {{
        border: none;
    }}

    PromptSearchPanel #prompt-search-results {{
        height: auto;
        max-height: {PROMPT_SEARCH_MAX_ROWS};
        scrollbar-gutter: stable;
    }}

    PromptSearchPanel .prompt-search-empty {{
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }}

    PromptSearchPanel .prompt-search-hint {{
        /* auto height + wrapping so a narrow window wraps the hint onto a
           second line instead of dropping its tail. */
        height: auto;
        padding: 0 1;
        color: $text-muted;
        text-style: italic;
    }}
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize display state."""
        super().__init__(**kwargs)
        self.can_focus = False
        self._query_input: PromptSearchInput | None = None
        self._results: VerticalScroll | None = None
        self._hint_static: Static | None = None
        self._options: list[PromptSearchOption] = []
        self._empty_widget: Static | None = None
        self._selected_index = 0
        self._pending_titles: list[str] = []
        self._pending_selected: int = 0
        self._pending_empty: str | None = None
        self._rebuild_generation: int = 0
        self._reported_rows = 0
        self._result_rows = 0

    class RowsChanged(Message):
        """Message sent when the panel's rendered row count changes."""

        def __init__(self, rows: int) -> None:
            """Initialize with the row count (0 when hidden)."""
            super().__init__()
            self.rows = rows

    class OptionSelected(Message):
        """Message sent when a prompt row is clicked in the panel."""

        def __init__(self, index: int) -> None:
            """Initialize with the clicked row index."""
            super().__init__()
            self.index = index

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual convention
        """Compose the query line, results list, and hint line.

        Yields:
            Widgets for the prompt search panel.
        """
        yield PromptSearchInput(
            placeholder="Search submitted prompts", id="prompt-search-input"
        )
        yield VerticalScroll(id="prompt-search-results")
        yield Static(prompt_search_hint(), classes="prompt-search-hint")

    def on_mount(self) -> None:
        """Grab the composed children."""
        self._query_input = self.query_one("#prompt-search-input", PromptSearchInput)
        self._results = self.query_one("#prompt-search-results", VerticalScroll)
        self._hint_static = self.query_one(".prompt-search-hint", Static)

    def update_state(
        self, query: str, titles: list[str], selected_index: int, empty: str | None
    ) -> None:
        """Show the panel for a new query/result set.

        Args:
            query: Current filter text, rendered on the query line.
            titles: All filtered row titles; the panel renders a window of
                them around `selected_index`.
            selected_index: Index of the selected row.
            empty: Empty-state message to show instead of rows, if any.
        """
        self._selected_index = selected_index
        self._pending_titles = titles
        self._pending_selected = selected_index
        self._pending_empty = empty
        # Increment generation so stale callbacks from prior calls are skipped.
        self._rebuild_generation += 1
        gen = self._rebuild_generation
        if self._query_input is not None and self._query_input.value != query:
            self._query_input.value = query
        self.call_next(lambda: self._rebuild_options(gen))

    async def _rebuild_options(self, generation: int) -> None:
        """Rebuild row widgets from pending state.

        Reuses existing DOM nodes where possible to avoid flicker from a full
        teardown/mount cycle while the panel is visible.

        Args:
            generation: Caller's generation counter; skipped if superseded.
        """
        if generation != self._rebuild_generation or self._results is None:
            return

        titles = self._pending_titles
        selected_index = self._pending_selected

        start, stop = _window_bounds(len(titles), selected_index)
        window = titles[start:stop]

        existing = len(self._options)
        needed = len(window)

        for i in range(min(existing, needed)):
            self._options[i].set_content(
                window[i], start + i, is_selected=(start + i == selected_index)
            )

        try:
            # The empty-state message is a single tracked widget, not a
            # per-rebuild mount: the previous one must come down before
            # anything new goes up, or every no-match keystroke adds a row.
            if self._empty_widget is not None:
                await self._empty_widget.remove()
                self._empty_widget = None

            if existing > needed:
                for option in self._options[needed:]:
                    await option.remove()
                del self._options[needed:]

            if needed > existing:
                new_widgets = [
                    PromptSearchOption(
                        title=window[idx],
                        index=start + idx,
                        is_selected=(start + idx == selected_index),
                    )
                    for idx in range(existing, needed)
                ]
                self._options.extend(new_widgets)
                await self._results.mount(*new_widgets)
        except Exception:
            logger.exception("Failed to rebuild prompt search panel; abandoning search")
            # Hiding alone would leave `ChatInput` still holding a draft
            # snapshot, so `_prompt_search_active` stays true and the composer
            # swallows every key into an invisible panel. `AbandonSearch` ends
            # the session and restores the draft, and the rows have to come out
            # of the DOM here: `hide()` only sets `display: none`, so dropping
            # the list without unmounting orphans them beside the next rebuild.
            try:
                await self._results.remove_children()
            except Exception:  # Recovery must not raise in turn
                logger.exception("Failed to clear prompt search rows")
            self._options = []
            self._empty_widget = None
            self.hide()
            self.post_message(PromptSearchInput.AbandonSearch())
            self.notify("Prompt search could not be displayed", severity="warning")
            return

        # The DOM mutations above can await, during which a hide() (or a newer
        # rebuild) bumps the generation to cancel this one. Re-check so a stale
        # rebuild cannot re-show a dismissed panel.
        if generation != self._rebuild_generation:
            return

        if self._pending_empty is not None:
            self._empty_widget = Static(
                self._pending_empty, classes="prompt-search-empty"
            )
            await self._results.mount(self._empty_widget)
        self.show(len(self._options) if not self._pending_empty else 1)

        if start <= selected_index < start + len(self._options):
            self._options[selected_index - start].scroll_visible(animate=False)

    def update_selection(self, selected_index: int) -> None:
        """Update which row is selected, re-windowing the DOM when needed.

        Rows are windowed around the selection, so a move beyond the mounted
        range (a Tab page, or arrows past the window edge) rebuilds the window
        instead of just re-styling a row.
        """
        self._pending_selected = selected_index
        if self._selected_index == selected_index:
            return
        start, _stop = _window_bounds(len(self._pending_titles), selected_index)
        if (
            not self._options
            or not (start <= selected_index < start + len(self._options))
            or self._options[0].index != start
        ):
            # Selection left the mounted window (or the window is stale); a
            # rebuild is the only way to show it. `_rebuild_options` applies
            # the selection as it mounts, so nothing else is needed here.
            self._selected_index = selected_index
            self._rebuild_generation += 1
            gen = self._rebuild_generation
            self.call_next(lambda: self._rebuild_options(gen))
            return
        old_local = self._selected_index - self._options[0].index
        if 0 <= old_local < len(self._options):
            self._options[old_local].set_selected(is_selected=False)
        self._selected_index = selected_index
        new_local = selected_index - self._options[0].index
        if 0 <= new_local < len(self._options):
            self._options[new_local].set_selected(is_selected=True)
            self._options[new_local].scroll_visible(animate=False)

    def on_prompt_search_option_clicked(
        self, event: PromptSearchOption.Clicked
    ) -> None:
        """Forward row clicks as this panel's own message for `ChatInput`."""
        event.stop()
        self.post_message(self.OptionSelected(event.index))

    def _report_rows(self, rows: int) -> None:
        """Announce the rendered row count when it changes."""
        if self._reported_rows != rows:
            self._reported_rows = rows
            self.post_message(self.RowsChanged(rows))

    def hide(self) -> None:
        """Hide the panel and cancel any in-flight rebuild."""
        self._pending_titles = []
        self._pending_empty = None
        self._rebuild_generation += 1
        # Keep the `_empty_widget` reference: hide() only sets display: none,
        # so a mounted empty-state row survives the hide, and the next rebuild
        # is the only code path that removes it. Dropping the reference here
        # would orphan the row, leaving a stale "No matching prompts." beside
        # the options when the panel reopens with matches.
        self.styles.display = "none"  # ty: ignore[invalid-assignment]  # Textual accepts string display values
        self._report_rows(0)

    def _hint_rows(self) -> int:
        """Measure the hint's wrapped height.

        Textual wraps the hint on word boundaries, so a cell-count estimate
        undercounts. Before the first layout the width is 0 and there is
        nothing to measure; reserving the maximum then is the safe direction to
        be wrong, because under-reporting clips the hint's last row.

        Returns:
            Rows the hint needs, clamped to `PROMPT_SEARCH_MAX_HINT_ROWS`.
        """
        if self._hint_static is None:
            return PROMPT_SEARCH_MAX_HINT_ROWS
        width = self._hint_static.content_region.width
        if width <= 0:
            return PROMPT_SEARCH_MAX_HINT_ROWS
        measured = self._hint_static.get_content_height(self.size, self.size, width)
        return max(1, min(measured, PROMPT_SEARCH_MAX_HINT_ROWS))

    def show(self, result_rows: int) -> None:
        """Show the panel.

        Args:
            result_rows: Number of result rows about to render, so the reported
                height tracks the caller's list rather than a stale widget
                count. Capped at `PROMPT_SEARCH_MAX_ROWS`.
        """
        self.styles.display = "block"
        self._result_rows = min(result_rows, PROMPT_SEARCH_MAX_ROWS)
        self._report_rows(1 + self._result_rows + self._hint_rows())

    def on_resize(self) -> None:
        """Re-report the height once a width is known, or when it changes.

        The first `show()` of a session runs before layout, so it reserves the
        maximum hint height. This corrects the reservation to the measured one
        instead of leaving the composer over-shrunk until the next keystroke.
        """
        if self.styles.display == "none":
            return
        self._report_rows(1 + self._result_rows + self._hint_rows())
