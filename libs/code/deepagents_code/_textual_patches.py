r"""Runtime patches over Textual internals, imported for side effect.

This module hosts six independent best-effort patches over private Textual
APIs. Each guards its own import/assignment and degrades to stock Textual
behavior (logging a warning) if the targeted internals move, so they have
separate lifecycles — do not delete the whole file when only one lands
upstream.

1. Alt-modifier preservation on legacy `ESC + <byte>` sequences. Upstream
    `XTermParser._sequence_to_key_events` drops the `alt` flag on the
    tuple-branch fast path, so VSCode's `sendSequence` shift+enter binding
    (which writes `\x1b\r` to the PTY) arrives as bare `enter` instead of
    `alt+enter`. Tracked in Textualize/textual#6378. Remove this patch and
    the Textual pin comment in `pyproject.toml` when that lands.

2. Kitty lock-key and unsupported sub-field handling. Two related problems
    remain with the pinned Textual parser:

    a. Lock keys (Caps Lock / Num Lock / Scroll Lock) must never produce
        text, but terminals encode them inconsistently. kitty/Ghostty/VS Code
        send the functional key code (`CSI 57358 ... u`) with associated text
        set to the letter the *next* key would have produced. iTerm2 instead
        reports the Caps Lock toggle as a bare upper-case ASCII letter (`CSI
        65 u` → 'A') with no modifier or associated-text field — not a valid
        encoding for a real key press per the kitty spec. Either way the chat
        input would type a stray capital. The patch collapses both forms to a
        single character-less `caps_lock` event, regardless of the modifier,
        associated-text, or event-type sub-fields the terminal includes.

    b. Textual 8.2.8 handles `:`-separated code points in the third,
        associated-text field, but not alternate keys in the first field
        (`unicode:shifted:base`) or an event type in the second field
        (`modifiers:event`). The patch strips only those unsupported sub-fields
        before Textual parses the sequence and preserves every associated-text
        code point.

    Remove when Textual neutralizes lock keys and handles key-code and modifier
    sub-fields natively.

3. Double-click word selection. Stock Textual selects the entire widget on
    a click chain; these patches narrow a double-click (and double-click
    drag) to word boundaries. No upstream issue tracks this yet, so it has
    no removal criterion — it stays until Textual grows native word select.

4. Shift-click selection extension. Stock Textual replaces an existing text
    selection on every mouse press, so Shift+click cannot move the active end
    of a drag-selected range. The patch preserves the original selection anchor
    and applies the click as its new end. No upstream issue tracks this yet.

    Known terminal limitation: this only works when the terminal delivers the
    modified click to the app. Ghostty binds Shift+click to its own selection
    and never forwards the event while mouse reporting is active, so
    Shift+click is a no-op there (users can `keybind = shift+click=unbind` to
    opt out). iTerm2, kitty, and WezTerm forward it with the shift modifier
    bit set, which is what the patch keys on. If Shift+click "does nothing"
    for a user, check terminal delivery first — e.g. run
    `printf '\e[?1003h\e[?1006h'; cat -v` and confirm Shift+click prints a
    `^[[<;...;4M`-style sequence (the `4` is the shift bit) — before digging
    into this patch.

5. Detached-widget hit filtering. The compositor keeps reporting a widget as
    visible for a few event-loop iterations after it leaves the DOM, which
    `Markdown.update` (and therefore the `MarkdownStream` that drives every
    streaming assistant message) does constantly. `Screen._forward_event`
    starts a text selection from `content_widget.parent`, which is `None` for
    such a widget, so a mouse press landing on freshly replaced markdown
    crashes the app with `AttributeError: 'NoneType' object has no attribute
    'region'`. Tracked in Textualize/textual#6643; remove when that lands.

6. Diff-gutter exclusion from selections. Textual paints the selection
    highlight from the geometry stored in `Screen.selections`, so excluding
    a diff row's decorative gutter (line number, `+`/`-` marker) only in
    `Widget.get_selection` would copy the right text while still highlighting
    the gutter. The patch rewrites each selection whenever that reactive
    changes, shifting any endpoint inside a row's gutter past it via
    `diff.clamp_selection`. No upstream issue tracks per-widget selection
    masking; it stays until Textual grows one.

Imported for side effect from `app.py` before any `App()` is created.

Not every Textual-internals workaround lives here: subclasses that shadow a
private base method carry theirs inline. When auditing a Textual bump, grep for
`Textual's private` and `Validated against Textual` alongside this module. Those
two markers are a starting point rather than a complete inventory — a workaround
can always land without one, so treat an unflagged subclass of a Textual widget
as unaudited until you have read it.
"""

from __future__ import annotations

import logging
import re
from inspect import isawaitable
from typing import TYPE_CHECKING

from rich.text import Text
from textual import __version__ as _textual_version
from textual.content import Content
from textual.geometry import Offset
from textual.selection import Selection

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from textual.events import Click, Event
    from textual.screen import Screen
    from textual.selection import SelectState
    from textual.widget import Widget

logger = logging.getLogger(__name__)

_ESC_PREFIX_LEN = 2
_DOUBLE_CLICK_CHAIN = 2
_TRIPLE_CLICK_CHAIN = 3
_DEEPAGENTS_WORD_SELECT_ACTIVE = "_deepagents_word_select_active"

try:
    from textual import events
    from textual._ansi_sequences import (  # noqa: PLC2701
        ANSI_SEQUENCES_KEYS,
        IGNORE_SEQUENCE,
    )
    from textual._xterm_parser import XTermParser  # noqa: PLC2701

    _original = XTermParser._sequence_to_key_events
except (ImportError, AttributeError) as exc:  # pragma: no cover - defensive
    logger.warning("Textual keyboard parser patch skipped: %s", exc)
else:
    # Kitty functional key codes for the lock keys (Caps Lock, Scroll Lock,
    # Num Lock). The kitty protocol assigns these Private Use Area codepoints;
    # they appear as the leading key-code field of a `CSI ... u` sequence.
    _KITTY_LOCK_KEY_CODES = frozenset({"57358", "57359", "57360"})
    _KITTY_LOCK_KEY_NAMES = {
        "57358": "caps_lock",
        "57359": "scroll_lock",
        "57360": "num_lock",
    }

    # Any `CSI <code>[:...][;...] u` sequence. Group 1 is the leading key-code
    # field (before any `:` alternate-key sub-field); `_lock_key_event` checks
    # it against the lock-key set. The match is deliberately broad so the code
    # is extracted regardless of the modifier / associated-text / event-type
    # sub-fields that follow, which iTerm2 and other terminals encode in
    # varying shapes.
    _KITTY_KEY_SEQUENCE = re.compile(r"\x1b\[(\d+)[\d;:]*u")

    # Kitty extended-key sequence carrying `:` sub-fields. Textual 8.2.8
    # handles them in the associated-text field, but not the key-code or
    # modifier fields; normalize only those first two fields below.
    _KITTY_SUBFIELD_KEY = re.compile(r"\x1b\[[\d;:]*:[\d;:]*[u~ABCDEFHPQRS]")

    # iTerm2 reports the Caps Lock toggle as a `CSI u` sequence whose primary
    # key code is the *uppercase* ASCII letter that would be produced next
    # (e.g. `CSI 65 u` → 'A'), with no real modifier bits and no associated
    # text. The kitty spec requires the primary code to be the unshifted
    # (lower-case) code point, so a bare upper-case letter here is iTerm2's
    # Caps Lock artifact rather than a real key press. Group 1 is the code
    # point; group 2 the optional modifier field; group 3 the optional text.
    _KITTY_CSI_U = re.compile(
        r"\x1b\[(\d+)(?::\d+)*(?:;(\d+)[\d:]*)?(?:;(\d+)[\d:]*)?u"
    )
    _ASCII_UPPER_A = 65
    _ASCII_UPPER_Z = 90
    # Modifier mask for the "real" modifiers (shift|alt|ctrl|super|hyper|meta);
    # excludes the caps_lock (64) and num_lock (128) lock bits.
    _REAL_MODIFIER_MASK = 0b111111

    def _spurious_caps_lock(sequence: str) -> bool:
        """Whether `sequence` is iTerm2's bare Caps Lock toggle report.

        Matches a `CSI u` key whose primary code point is an upper-case ASCII
        letter with no real modifiers and no associated-text field — which the
        kitty spec never produces for a genuine key press.

        Returns:
            `True` if `sequence` is the spurious Caps Lock toggle report.
        """
        match = _KITTY_CSI_U.fullmatch(sequence)
        if match is None:
            return False
        code = int(match.group(1))
        if not _ASCII_UPPER_A <= code <= _ASCII_UPPER_Z:
            return False
        modifier_bits = (int(match.group(2)) - 1) if match.group(2) else 0
        has_text = match.group(3) is not None
        return modifier_bits & _REAL_MODIFIER_MASK == 0 and not has_text

    def _strip_kitty_subfields(sequence: str) -> str:
        """Drop unsupported `:` sub-fields from a kitty key sequence.

        Keeps the primary key code and modifier while preserving every
        colon-separated code point in the associated-text field, which
        Textual 8.2.8 handles natively.

        Returns:
            The sequence with only key-code and modifier sub-fields removed.
        """
        body, terminator = sequence[2:-1], sequence[-1]
        fields = body.split(";")
        fields[:2] = [field.split(":", 1)[0] for field in fields[:2]]
        return f"\x1b[{';'.join(fields)}{terminator}"

    def _lock_key_event(sequence: str) -> events.Key | None:
        """Return a text-free lock-key event for a kitty lock-key sequence.

        Lock keys must never produce text. Under the kitty protocol with
        associated-text reporting, terminals (notably iTerm2) encode Caps
        Lock as a `CSI 57358 ... u` sequence whose associated-text field is
        the letter the *next* key would have produced — Textual then either
        types that letter or, when `:` sub-fields are present, leaks the raw
        sequence byte by byte. Collapsing any lock-key sequence to a single
        character-less event stops both failure modes at the source, for
        every widget.

        Returns:
            A `Key` event for the lock key, or `None` if `sequence` is not a
            kitty lock-key sequence.
        """
        match = _KITTY_KEY_SEQUENCE.fullmatch(sequence)
        if match is None or match.group(1) not in _KITTY_LOCK_KEY_CODES:
            return None
        return events.Key(_KITTY_LOCK_KEY_NAMES[match.group(1)], None)

    def _emit_alt(keys: tuple, character: str | None) -> Iterable[events.Key]:
        for key in keys:
            yield events.Key(f"alt+{key.value}", character)

    def _sequence_to_key_events_with_alt(
        self: XTermParser, sequence: str, alt: bool = False
    ) -> Iterable[events.Key]:
        # Lock keys (Caps Lock / Num Lock / Scroll Lock) must never type. Emit
        # a single character-less event regardless of how the terminal encoded
        # the modifiers, associated text, or event-type sub-fields.
        if (lock_event := _lock_key_event(sequence)) is not None:
            yield lock_event
            return
        # iTerm2 reports the Caps Lock toggle as a bare upper-case letter (e.g.
        # `CSI 65 u` → 'A') rather than the kitty `57358` functional code. Drop
        # it so the toggle never types a stray capital into the input.
        if _spurious_caps_lock(sequence):
            yield events.Key("caps_lock", None)
            return
        # Normalize unsupported key-code and modifier sub-fields while leaving
        # Textual 8.2.8's colon-separated associated text intact.
        if _KITTY_SUBFIELD_KEY.fullmatch(sequence):
            sequence = _strip_kitty_subfields(sequence)
        # Fast path: \x1b<byte> on first pass. Short-circuits the ~100 ms
        # escape-delay wait when both bytes arrive together. Semantic side
        # effect: \x1b\x1b dispatches as `alt+escape` with no delay, matching
        # crossterm and Node TTY.
        if not alt and len(sequence) == _ESC_PREFIX_LEN and sequence[0] == "\x1b":
            inner = ANSI_SEQUENCES_KEYS.get(sequence[1])
            if inner is not IGNORE_SEQUENCE and isinstance(inner, tuple):
                yield from _emit_alt(inner, None)
                return
        # Correctness fix (Textualize/textual#6378): preserve `alt` on the
        # reissue path for single-byte tuple mappings.
        if alt:
            keys = ANSI_SEQUENCES_KEYS.get(sequence)
            if keys is not IGNORE_SEQUENCE and isinstance(keys, tuple):
                character = sequence if len(sequence) == 1 else None
                yield from _emit_alt(keys, character)
                return
        yield from _original(self, sequence, alt=alt)

    try:
        XTermParser._sequence_to_key_events = _sequence_to_key_events_with_alt  # ty: ignore[invalid-assignment]
    except (AttributeError, TypeError) as exc:  # pragma: no cover - defensive
        logger.warning("Textual keyboard parser patch assignment rejected: %s", exc)


def _rendered_text(widget: Widget) -> str | None:
    visual = widget._render()  # match Textual's get_selection path
    if isinstance(visual, (Content, Text)):
        return str(visual)
    return None


def _word_bounds(text: str, offset: Offset) -> tuple[Offset, Offset] | None:
    lines = text.splitlines()
    if not lines:
        return None

    y = min(max(offset.y, 0), len(lines) - 1)
    line = lines[y]
    if not line:
        return None

    x = min(max(offset.x, 0), len(line))
    index = min(x, len(line) - 1)
    if line[index].isspace():
        # A click just past the final character (x == len(line)) lands on the
        # virtual end-of-line position; snap back onto the trailing word so
        # double-clicking after a word still selects it. Genuine whitespace
        # clicks fall through and select nothing.
        if x == len(line) and x > 0 and not line[x - 1].isspace():
            index = x - 1
        else:
            return None

    start = index
    while start > 0 and not line[start - 1].isspace():
        start -= 1

    end = index + 1
    while end < len(line) and not line[end].isspace():
        end += 1

    return Offset(start, y), Offset(end, y)


def _word_selection(widget: Widget, selection: Selection) -> Selection | None:
    if selection.start is None or selection.end is None:
        return None

    text = _rendered_text(widget)
    if text is None:
        return None

    start, end = selection.start, selection.end
    # `Offset.transpose` is (y, x) — Textual's reading-order key. A backward
    # drag leaves end before start in reading order; normalize so the word
    # bounds below extend outward from the correct endpoints.
    if end.transpose < start.transpose:
        start, end = end, start

    start_bounds = _word_bounds(text, start)
    end_bounds = _word_bounds(text, end)
    if start_bounds is None and end_bounds is None:
        return None

    return Selection(
        start_bounds[0] if start_bounds is not None else start,
        end_bounds[1] if end_bounds is not None else end,
    )


def _select_word_at_click(widget: Widget, event: Click) -> bool:
    offset = event.get_content_offset(widget)
    if offset is None:
        return False

    text = _rendered_text(widget)
    if text is None:
        return False

    bounds = _word_bounds(text, offset)
    if bounds is None:
        widget.screen.clear_selection()
        return True

    widget.screen.selections = {widget: Selection(*bounds)}
    return True


try:
    from textual import events as _events
    from textual.screen import Screen as _Screen
    from textual.widget import Widget as _Widget

    _original_forward_event = _Screen._forward_event
    _original_watch_select_state = _Screen._watch__select_state
    _original_widget_on_click = _Widget._on_click
except (ImportError, AttributeError) as exc:  # pragma: no cover - defensive
    logger.warning(
        "Textual word-selection patch skipped (textual %s): %s",
        _textual_version,
        exc,
    )
else:

    def _is_word_select_start(screen: Screen, event: Event) -> bool:
        # Mirrors Textual's own click-chain detection (App._on_mouse_down),
        # reading its private `_click_chain_last_*` bookkeeping to recognize
        # the second press of a double-click before Textual increments the
        # chain count. Re-verify these attribute names on every Textual bump.
        if not isinstance(event, _events.MouseDown) or screen.app.mouse_captured:
            return False

        last_offset = getattr(screen.app, "_click_chain_last_offset", None)
        last_time = getattr(screen.app, "_click_chain_last_time", None)
        if last_offset != event.screen_offset or last_time is None:
            return False

        if event.time - last_time > screen.app.CLICK_CHAIN_TIME_THRESHOLD:
            return False

        select_widget, select_offset = screen.get_widget_and_offset_at(event.x, event.y)
        return (
            select_widget is not None
            and select_widget.allow_select
            and screen.allow_select
            and screen.app.ALLOW_SELECT
            and select_offset is not None
        )

    def _forward_event_with_word_select(self: Screen, event: Event) -> None:
        if isinstance(event, _events.MouseDown):
            setattr(
                self,
                _DEEPAGENTS_WORD_SELECT_ACTIVE,
                _is_word_select_start(self, event),
            )
        try:
            _original_forward_event(self, event)
        finally:
            if isinstance(event, _events.MouseUp):
                setattr(self, _DEEPAGENTS_WORD_SELECT_ACTIVE, False)

    async def _watch_select_state_with_word_select(
        self: Screen,
        select_state: SelectState | None,
    ) -> None:
        result = _original_watch_select_state(self, select_state)
        # `_watch__select_state` is synchronous in the pinned Textual; the
        # isawaitable guard tolerates a future release making it a coroutine
        # without forcing a same-day patch update.
        if isawaitable(result):
            await result
        if not getattr(self, _DEEPAGENTS_WORD_SELECT_ACTIVE, False):
            return

        selections = dict(self.selections)
        changed = False
        for widget, selection in selections.items():
            word_selection = _word_selection(widget, selection)
            if word_selection is None or word_selection == selection:
                continue
            selections[widget] = word_selection
            changed = True

        if changed:
            self.selections = selections

    async def _on_click_with_word_select(self: Widget, event: Click) -> None:
        if (
            event.widget is self
            and self.allow_select
            and self.screen.allow_select
            and self.app.ALLOW_SELECT
        ):
            if event.chain == _DOUBLE_CLICK_CHAIN and _select_word_at_click(
                self, event
            ):
                await self.broker_event("click", event)
                return
            if event.chain == _TRIPLE_CLICK_CHAIN:
                self.text_select_all()
                await self.broker_event("click", event)
                return

        await _original_widget_on_click(self, event)

    try:
        _Screen._forward_event = _forward_event_with_word_select  # ty: ignore[invalid-assignment]
        _Screen._watch__select_state = _watch_select_state_with_word_select  # ty: ignore[invalid-assignment]
        _Widget._on_click = _on_click_with_word_select  # ty: ignore[invalid-assignment]
    except (AttributeError, TypeError) as exc:  # pragma: no cover - defensive
        logger.warning(
            "Textual word-selection patch assignment rejected (textual %s): %s",
            _textual_version,
            exc,
        )


try:
    from textual import events as _shift_events
    from textual.screen import Screen as _ShiftScreen
    from textual.selection import SelectEnd, SelectStart, SelectState

    _original_forward_event_with_shift = _ShiftScreen._forward_event
except (ImportError, AttributeError) as exc:  # pragma: no cover - defensive
    logger.warning(
        "Textual Shift+click selection patch skipped (textual %s): %s",
        _textual_version,
        exc,
    )
else:

    def _shift_click_anchor(screen: Screen, event: Event) -> SelectState | None:
        # When this returns None the press falls through to stock Textual,
        # which starts a fresh selection — indistinguishable from the event
        # never arriving. Before debugging the branches below, confirm the
        # terminal forwarded a shift-modified click at all (see the module
        # docstring's patch 4 note); terminals like Ghostty consume Shift+click
        # locally, so `event.shift` never becomes True here.
        if (
            not isinstance(event, _shift_events.MouseDown)
            or not event.shift
            or screen.app.mouse_captured
            or not screen.selections
        ):
            return None
        select_state = screen._select_state
        if select_state is None or select_state.end is None:
            return None
        content_widget = select_state.start.content_widget
        if content_widget is not None and not content_widget.is_attached:
            return None
        return select_state if select_state.is_attached_to_dom() else None

    def _rebase_anchor_scroll(anchor_start: SelectStart) -> SelectStart:
        # `SelectStart.pointer_start_offset` adds the scroll travelled since
        # the drag began, which tracks the viewport rather than the anchored
        # text. That is harmless mid-drag, but a Shift+click can arrive many
        # rows after the container scrolled — the transcript auto-scrolls while
        # a message streams — leaving the anchor pointing at whatever text now
        # occupies its old screen row. Fold the drift into the pointer delta
        # and re-base against the current scroll offset, so the anchor stays on
        # its original text.
        container = anchor_start.container
        drift = container.scroll_offset - anchor_start.container_initial_scroll_offset
        return SelectStart(
            container,
            anchor_start.container_pointer_delta - drift,
            anchor_start.container_initial_offset,
            container.scroll_offset,
            content_widget=anchor_start.content_widget,
            content_offset=anchor_start.content_offset,
        )

    def _extend_selection_to_click(
        screen: Screen,
        anchor: SelectState | None,
        event: _shift_events.MouseDown,
    ) -> None:
        click_state = screen._select_state
        if anchor is None or click_state is None:
            return
        click = click_state.start
        # Stock Textual clears the selection when a MouseUp lands on the
        # offset of its MouseDown. Forget the offset so the Shift+click's own
        # MouseUp leaves the extension we install below alone.
        screen._mouse_down_offset = None
        screen._select_state = SelectState(
            event.screen_offset,
            _rebase_anchor_scroll(anchor.start),
            SelectEnd(click.container, click.content_widget, click.content_offset),
        )

    def _forward_event_with_shift_select(self: Screen, event: Event) -> None:
        shift_anchor = _shift_click_anchor(self, event)
        _original_forward_event_with_shift(self, event)
        if isinstance(event, _shift_events.MouseDown):
            _extend_selection_to_click(self, shift_anchor, event)

    try:
        _ShiftScreen._forward_event = _forward_event_with_shift_select  # ty: ignore[invalid-assignment]
    except (AttributeError, TypeError) as exc:  # pragma: no cover - defensive
        logger.warning(
            "Textual Shift+click selection patch assignment rejected (textual %s): %s",
            _textual_version,
            exc,
        )


try:
    from textual.screen import Screen as _HitScreen

    _original_get_widget_and_offset_at = _HitScreen.get_widget_and_offset_at
except (ImportError, AttributeError) as exc:  # pragma: no cover - defensive
    logger.warning(
        "Textual detached-hit patch skipped (textual %s): %s",
        _textual_version,
        exc,
    )
else:

    def _get_widget_and_offset_at_attached(
        self: Screen,
        x: int,
        y: int,
    ) -> tuple[Widget | None, Offset | None]:
        """Ignore compositor hits on widgets that already left the DOM.

        Returns:
            The stock result, or `(None, None)` when the hit widget is
            detached, which sends Textual down its existing "nothing
            selectable here" branch instead of dereferencing a `None` parent.
        """
        widget, offset = _original_get_widget_and_offset_at(self, x, y)
        if (
            widget is not None
            and not isinstance(widget, _HitScreen)
            and (widget.parent is None or not widget.is_attached)
        ):
            return None, None
        return widget, offset

    try:
        _HitScreen.get_widget_and_offset_at = _get_widget_and_offset_at_attached  # ty: ignore[invalid-assignment]
    except (AttributeError, TypeError) as exc:  # pragma: no cover - defensive
        logger.warning(
            "Textual detached-hit patch assignment rejected (textual %s): %s",
            _textual_version,
            exc,
        )


try:
    from textual.screen import Screen as _ClampScreen

    from deepagents_code.tui.widgets.diff import clamp_selection

    _original_watch_selections_for_clamp = _ClampScreen._watch_selections
except (ImportError, AttributeError) as exc:  # pragma: no cover - defensive
    logger.warning(
        "Textual gutter-selection patch skipped (textual %s): %s",
        _textual_version,
        exc,
    )
else:

    def _watch_selections_with_gutter_clamp(
        self: Screen,
        old_selections: dict[Widget, Selection],
        selections: dict[Widget, Selection],
    ) -> Awaitable[None] | None:
        # Textual stores the new dict before invoking this watcher. Clamp that
        # object synchronously before returning the original watcher's awaitable;
        # assigning `self.selections` here would schedule this watcher again.
        for widget, selection in list(selections.items()):
            clamped = clamp_selection(widget, selection)
            if clamped is None:
                del selections[widget]
            elif clamped != selection:
                selections[widget] = clamped

        result = _original_watch_selections_for_clamp(self, old_selections, selections)
        # `_watch_selections` is async in the pinned Textual; the isawaitable
        # guard tolerates a future synchronous implementation.
        return result if isawaitable(result) else None

    try:
        _ClampScreen._watch_selections = _watch_selections_with_gutter_clamp  # ty: ignore[invalid-assignment]
    except (AttributeError, TypeError) as exc:  # pragma: no cover - defensive
        logger.warning(
            "Textual gutter-selection patch assignment rejected (textual %s): %s",
            _textual_version,
            exc,
        )
