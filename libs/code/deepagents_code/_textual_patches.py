r"""Runtime patches over Textual internals, imported for side effect.

This module hosts two independent best-effort patches over private Textual
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

2. Kitty lock-key and sub-field handling. Two related problems with the
    pinned Textual parser:

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

    b. `_re_extended_key` only accepts `;`-separated numeric fields, so any
        *non-lock* kitty sequence carrying `:`-separated sub-fields — alternate
        keys (`unicode:shifted:base`) or an event-type (`modifiers:event`) —
        fails to match and is re-emitted one byte at a time as literal text.
        The patch strips the `:` sub-fields before Textual parses the sequence
        so it resolves to a single key event.

    Remove when the pinned Textual neutralizes lock keys and widens its parser.

3. Double-click word selection. Stock Textual selects the entire widget on
    a click chain; these patches narrow a double-click (and double-click
    drag) to word boundaries. No upstream issue tracks this yet, so it has
    no removal criterion — it stays until Textual grows native word select.

Imported for side effect from `app.py` before any `App()` is created.
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
    from collections.abc import Iterable

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

    # Kitty extended-key sequence carrying `:` sub-fields (alternate keys or an
    # event-type sub-field). The pinned Textual's `_re_extended_key` rejects the
    # colons, so non-lock keys with these sub-fields would otherwise leak as
    # literal text — strip the sub-fields so they parse to a single key event.
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
        """Drop `:` sub-fields from a kitty extended-key sequence.

        Keeps the primary value of each `;`-separated field (the unicode key
        code, modifier mask, and associated text), which is all Textual reads.

        Returns:
            The sequence with every `:` sub-field removed.
        """
        body, terminator = sequence[2:-1], sequence[-1]
        primary = ";".join(field.split(":", 1)[0] for field in body.split(";"))
        return f"\x1b[{primary}{terminator}"

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
        # Normalize any other kitty sequence with `:` sub-fields so it resolves
        # to a single key event instead of leaking raw bytes.
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
