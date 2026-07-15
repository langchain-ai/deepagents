"""Shared paste handling for text-area inputs.

Terminals deliver a paste in one of two shapes: a single bracketed `Paste`
event, or — when bracketed paste is unavailable — a rapid stream of individual
key events. Both the primary chat input and the inline free-text prompts need
to (a) keep a multi-line paste grouped instead of submitting on the first
embedded newline, and (b) collapse a large paste into a compact
`[Pasted text #N]` placeholder that expands back to the full text on submit.

`PasteBurstTextArea` owns the burst detection and Enter-suppression state
machine, leaving policy (slash-command context, whether collapsing is enabled,
how a flushed payload is handled) to overridable hooks.
`CollapsingPasteTextArea` layers the large-paste collapse + placeholder storage
on top, keeping the full content off-screen until submission.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from textual.widgets import TextArea

from deepagents_code.paste_collapse import (
    PASTE_PLACEHOLDER_PATTERN,
    PastedContent,
    count_lines,
    expand_paste_refs,
    format_paste_ref,
    should_collapse_paste,
)

if TYPE_CHECKING:
    from textual import events
    from textual.timer import Timer

logger = logging.getLogger(__name__)

PASTE_BURST_CHAR_GAP_SECONDS = 0.03
"""Maximum time between chars to treat input as a paste-like burst."""

PASTE_BURST_FLUSH_DELAY_SECONDS = 0.08
"""Idle timeout before flushing buffered burst text."""

PASTE_BURST_START_CHARS = {"'", '"'}
"""Characters that can start dropped-path payloads."""

PASTE_BURST_MIN_CHARS = 3
"""Consecutive fast keystrokes before a stream is treated as a paste burst.

Terminals that lack bracketed paste replay a paste as individual key events.
Counting a short run of rapid chars distinguishes that from human typing,
which has much larger inter-key gaps.
"""

PASTE_ENTER_SUPPRESS_WINDOW_SECONDS = 0.12
"""Window after recent burst activity during which `enter` inserts a newline.

Keeps multi-line pastes grouped as one input even when newlines arrive as
`enter` key events slightly after the surrounding characters (e.g. across
terminal read boundaries), instead of submitting mid-paste.
"""


class PasteBurstTextArea(TextArea):
    """`TextArea` that detects paste-like keystroke bursts.

    Subclasses drive the state machine from their own `_on_key` by calling the
    helper methods here, and override the policy hooks
    (`_in_slash_command_context`, `_dispatch_burst_payload`) as needed. The base
    inserts a flushed burst verbatim; collapsing into placeholders is layered on
    by `CollapsingPasteTextArea`.
    """

    _paste_burst_buffer: str
    _paste_burst_last_char_time: float | None
    _paste_burst_timer: Timer | None
    _paste_burst_run: int
    _paste_burst_run_text: str
    _paste_burst_last_key_time: float | None
    _paste_burst_last_suppressed_enter_time: float | None
    _paste_burst_window_until: float | None

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the text area and its paste-burst state."""
        super().__init__(**kwargs)
        self._init_paste_burst_state()

    def _init_paste_burst_state(self) -> None:
        """Reset all paste-burst tracking fields to their initial values."""
        # Buffer high-frequency key bursts from terminals that emulate paste via
        # rapid key events instead of dispatching a paste event.
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time = None
        self._paste_burst_timer = None
        # Counts consecutive rapid keystrokes so a paste-like stream can be
        # detected even when it doesn't begin with a quote.
        self._paste_burst_run = 0
        self._paste_burst_run_text = ""
        self._paste_burst_last_key_time = None
        self._paste_burst_last_suppressed_enter_time = None
        # Deadline until which `enter` inserts a newline rather than submitting,
        # keeping multi-line pastes grouped across read boundaries.
        self._paste_burst_window_until = None

    # -- Policy hooks (override in subclasses) --------------------------------

    def _in_slash_command_context(self) -> bool:  # noqa: PLR6301  # overridable hook
        """Return whether Enter should keep submit/dispatch semantics.

        Base text areas have no slash-command surface, so the Enter-suppression
        window always applies. Override to opt keystrokes out of grouping.
        """
        return False

    async def _dispatch_burst_payload(self, payload: str) -> None:
        """Handle a flushed burst payload. Base behavior inserts it verbatim."""
        self.insert(payload)

    # -- Burst state machine --------------------------------------------------

    def _cancel_paste_burst_timer(self) -> None:
        """Cancel any scheduled paste-burst flush timer."""
        if self._paste_burst_timer is None:
            return
        self._paste_burst_timer.stop()
        self._paste_burst_timer = None

    def _schedule_paste_burst_flush(self) -> None:
        """Schedule idle-time flush for buffered paste-burst text."""
        self._cancel_paste_burst_timer()
        self._paste_burst_timer = self.set_timer(
            PASTE_BURST_FLUSH_DELAY_SECONDS, self._flush_paste_burst
        )

    def _start_paste_burst(self, char: str, now: float) -> None:
        """Start buffering a paste-like keystroke burst."""
        self._paste_burst_buffer = char
        self._paste_burst_last_char_time = now
        self._paste_burst_window_until = now + PASTE_ENTER_SUPPRESS_WINDOW_SECONDS
        self._schedule_paste_burst_flush()

    def _append_paste_burst(self, text: str, now: float) -> None:
        """Append text to an active paste-burst buffer."""
        if not self._paste_burst_buffer:
            self._start_paste_burst(text, now)
            return
        self._paste_burst_buffer += text
        self._paste_burst_last_char_time = now
        self._paste_burst_window_until = now + PASTE_ENTER_SUPPRESS_WINDOW_SECONDS
        self._schedule_paste_burst_flush()

    def _note_paste_burst_keystroke(self, char: str, now: float) -> None:
        """Track text and timing for consecutive rapid keystrokes."""
        last = self._paste_burst_last_key_time
        if last is not None and (now - last) <= PASTE_BURST_CHAR_GAP_SECONDS:
            self._paste_burst_run += 1
            self._paste_burst_run_text += char
        else:
            self._paste_burst_run = 1
            self._paste_burst_run_text = char
        self._paste_burst_last_key_time = now

    def _reset_paste_burst_run(self) -> None:
        """Clear consecutive-keystroke tracking after non-burst input."""
        self._paste_burst_run = 0
        self._paste_burst_run_text = ""
        self._paste_burst_last_key_time = None
        self._paste_burst_last_suppressed_enter_time = None

    def _reset_paste_burst_state(self) -> None:
        """Reset all paste-burst tracking to a clean slate.

        Used by text-replacing entry points so a wholesale text swap never
        leaves stale burst timing that would misclassify the next keystroke.
        """
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time = None
        self._paste_burst_window_until = None
        self._reset_paste_burst_run()
        self._cancel_paste_burst_timer()

    def _enter_inserts_newline_during_burst(self, now: float) -> bool:
        """Return whether `enter` should insert a newline rather than submit.

        True when the preceding keystroke was part of a rapid run or the
        previous `enter` was already suppressed, and the suppression window is
        still open. The char-gap check keeps a deliberate `enter` pressed after
        a burst settles from being swallowed; the window bounds how long a
        replayed paste's newlines stay grouped. Returns `False` immediately in
        slash-command context (see `_in_slash_command_context`).
        """
        if self._in_slash_command_context():
            return False
        # Defensive: the shipped `_on_key`s absorb (via `_absorb_key_into_burst`)
        # or flush any active buffer before Enter reaches this helper, so this
        # branch is unreachable today. It keeps the helper's contract
        # self-contained for future callers.
        if self._paste_burst_buffer:
            return True
        until = self._paste_burst_window_until
        if until is None or now > until:
            return False
        last_enter = self._paste_burst_last_suppressed_enter_time
        if last_enter is not None:
            return True
        last_key = self._paste_burst_last_key_time
        return last_key is not None and (now - last_key) <= PASTE_BURST_CHAR_GAP_SECONDS

    def _should_start_paste_burst(self, char: str) -> bool:
        """Return whether a keypress should start paste-burst buffering.

        Quote-prefixed input at an empty cursor is buffered immediately for
        dropped-path parsing. Other printable runs are promoted into the same
        buffer once they reach `PASTE_BURST_MIN_CHARS` rapid keystrokes.
        """
        if char not in PASTE_BURST_START_CHARS:
            return False
        if self.text or not self.selection.is_empty:
            return False
        row, col = self.cursor_location
        return row == 0 and col == 0

    async def _flush_paste_burst(self) -> None:
        """Flush buffered burst text through the payload dispatch hook.

        When the buffer is empty this is a no-op, so it is safe to call
        defensively before handling a bracketed paste.
        """
        payload = self._paste_burst_buffer
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time = None
        self._cancel_paste_burst_timer()
        if not payload:
            return
        await self._dispatch_burst_payload(payload)

    def _promote_paste_burst_run(self, char: str, now: float) -> bool:
        """Move a detected rapid run from the document into the burst buffer.

        The first keys in an unquoted run are inserted normally while the run is
        still indistinguishable from typing. Once the threshold is reached, this
        removes those keys and buffers the complete run so its eventual flush can
        apply dropped-path and paste-collapse policy.

        Args:
            char: Current character, which has not yet been inserted.
            now: Monotonic timestamp for the current key event.

        Returns:
            `True` when the run was promoted and the current key was buffered.
        """
        if not char or not self.selection.is_empty:
            return False
        prefix = self._paste_burst_run_text[: -len(char)]
        cursor = self.cursor_location
        cursor_offset = self.document.get_index_from_location(cursor)  # ty: ignore[unresolved-attribute]  # Document has this method; DocumentBase stub is narrower
        start_offset = cursor_offset - len(prefix)
        if start_offset < 0 or self.text[start_offset:cursor_offset] != prefix:
            return False
        start = self.document.get_location_from_index(start_offset)  # ty: ignore[unresolved-attribute]
        self.delete(start, cursor)
        self._start_paste_burst(self._paste_burst_run_text, now)
        return True

    def action_insert_newline(self) -> None:
        """Insert a newline at the cursor."""
        self.insert("\n")

    # -- `_on_key` building blocks (shared by concrete text areas) ------------

    async def _absorb_key_into_burst(self, event: events.Key, now: float) -> bool:
        """Absorb a key into an active burst buffer, flushing if it breaks it.

        Returns:
            `True` when the key was buffered and the caller should stop handling
            it; `False` when there is no active burst (or it was just flushed)
            and the caller should continue normal key handling.
        """
        if not self._paste_burst_buffer:
            return False
        if event.key == "enter":
            self._append_paste_burst("\n", now)
            return True
        if event.is_printable and event.character is not None:
            last_time = self._paste_burst_last_char_time
            if (
                last_time is not None
                and (now - last_time) <= PASTE_BURST_CHAR_GAP_SECONDS
            ):
                self._append_paste_burst(event.character, now)
                return True
        await self._flush_paste_burst()
        return False

    def _maybe_start_burst(self, event: events.Key, now: float) -> bool:
        """Start buffering when a keypress looks like the head of a paste.

        Returns:
            `True` when a burst was started and the caller should stop handling
            the key.
        """
        if (
            event.is_printable
            and event.character is not None
            and self._should_start_paste_burst(event.character)
        ):
            self._start_paste_burst(event.character, now)
            return True
        return False

    def _track_burst_run(self, event: events.Key, now: float) -> bool:
        """Track a rapid run and promote it into the paste buffer once detected.

        Returns:
            `True` when the current key was buffered and should not be handled by
            the caller.
        """
        if event.is_printable and event.character is not None:
            self._paste_burst_last_suppressed_enter_time = None
            self._note_paste_burst_keystroke(event.character, now)
            if (
                self._paste_burst_run >= PASTE_BURST_MIN_CHARS
                and not self._in_slash_command_context()
            ):
                self._paste_burst_window_until = (
                    now + PASTE_ENTER_SUPPRESS_WINDOW_SECONDS
                )
                return self._promote_paste_burst_run(event.character, now)
        elif event.key != "enter":
            self._reset_paste_burst_run()
        return False

    def _consume_enter_as_burst_newline(self, now: float) -> bool:
        """Insert a newline instead of submitting when inside a paste burst.

        Returns:
            `True` when Enter was consumed as a newline (part of a paste);
            `False` when Enter should fall through to its submit handling.
        """
        if not self._enter_inserts_newline_during_burst(now):
            self._paste_burst_last_suppressed_enter_time = None
            return False
        self._paste_burst_window_until = now + PASTE_ENTER_SUPPRESS_WINDOW_SECONDS
        self._paste_burst_last_suppressed_enter_time = now
        self.action_insert_newline()
        return True


class CollapsingPasteTextArea(PasteBurstTextArea):
    """Paste-aware text area that collapses large pastes into placeholders.

    The full pasted text is stored off-screen in `_pasted_contents` and a
    compact `[Pasted text #N]` placeholder is shown in its place. Read
    `submitted_value` to get the text with all placeholders expanded back.
    """

    _pasted_contents: dict[int, PastedContent]
    _next_paste_id: int
    _collapse_pastes: bool

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the text area and its collapsed-paste storage."""
        super().__init__(**kwargs)
        self._pasted_contents = {}
        self._next_paste_id = 1
        # Resolve the preference once, mirroring how `ChatInput` caches it at
        # construction, so paste handling never re-reads config from disk and
        # stays consistent with the chat input for the widget's lifetime.
        self._collapse_pastes = _collapse_pastes_enabled()

    @property
    def submitted_value(self) -> str:
        """The current text with collapsed-paste placeholders expanded."""
        return expand_paste_refs(self.text, self._pasted_contents)

    def reset_paste_state(self) -> None:
        """Drop burst timing and collapsed-paste storage after a text swap.

        Call after a wholesale programmatic `text` swap (e.g. switching an
        inline editor between modes) so a stale flush timer can't fire against
        the new text and placeholders from the previous buffer don't linger in
        `_pasted_contents`.
        """
        self._reset_paste_burst_state()
        self._pasted_contents.clear()
        self._next_paste_id = 1

    def _paste_collapse_enabled(self) -> bool:
        """Return whether large pastes are collapsed into placeholders.

        Returns the preference resolved once at construction (see `__init__`).
        """
        return self._collapse_pastes

    async def _dispatch_burst_payload(self, payload: str) -> None:
        """Collapse a large flushed burst, otherwise insert it verbatim."""
        self._insert_paste_payload(payload)

    def _insert_paste_payload(self, payload: str) -> None:
        """Collapse `payload` into a placeholder when large, else insert it."""
        if self._paste_collapse_enabled() and should_collapse_paste(payload):
            self._collapse_and_insert_paste(payload)
        else:
            self.insert(payload)

    def _collapse_and_insert_paste(self, text: str) -> None:
        """Store full paste content and insert a compact placeholder.

        Pasting content identical to a visible already-collapsed placeholder
        expands that placeholder back to full text in place instead of adding a
        second placeholder — a repeat paste is treated as a request to see the
        content in full.

        Args:
            text: The full pasted text to collapse.
        """
        visible_ids = {
            int(match.group(1))
            for match in PASTE_PLACEHOLDER_PATTERN.finditer(self.text)
        }
        match_id = next(
            (
                pid
                for pid, stored in self._pasted_contents.items()
                if pid in visible_ids and stored.content == text
            ),
            None,
        )
        if match_id is not None and self._replace_placeholder_with_text(match_id, text):
            return
        paste_id = self._next_paste_id
        self._next_paste_id += 1
        self._pasted_contents[paste_id] = PastedContent(content=text)
        self.insert(format_paste_ref(paste_id, count_lines(text)))

    def _replace_placeholder_with_text(self, paste_id: int, content: str) -> bool:
        """Replace a `[Pasted text #id]` placeholder with its full text in place.

        Args:
            paste_id: The paste id whose placeholder should be expanded.
            content: The full text to insert where the placeholder was.

        Returns:
            `True` when a matching placeholder was found and replaced.
        """
        for match in PASTE_PLACEHOLDER_PATTERN.finditer(self.text):
            if int(match.group(1)) != paste_id:
                continue
            start, end = match.span()
            start_location = self.document.get_location_from_index(start)  # ty: ignore[unresolved-attribute]  # Document has this method; DocumentBase stub is narrower
            end_location = self.document.get_location_from_index(end)  # ty: ignore[unresolved-attribute]
            self.delete(start_location, end_location)
            self.insert(content, start_location)
            return True
        return False

    def _delete_placeholder_token(self, *, backwards: bool) -> bool:
        """Delete a full collapsed-paste placeholder in one keypress.

        Args:
            backwards: Whether the delete is backwards (`backspace`) or
                forwards (`delete`).

        Returns:
            `True` when a placeholder token was deleted.
        """
        if not self.text or not self.selection.is_empty:
            return False
        cursor_offset = self.document.get_index_from_location(self.cursor_location)  # ty: ignore[unresolved-attribute]  # Document has this method; DocumentBase stub is narrower
        span = self._find_placeholder_span(cursor_offset, backwards=backwards)
        if span is None:
            return False
        start, end = span
        start_location = self.document.get_location_from_index(start)  # ty: ignore[unresolved-attribute]
        end_location = self.document.get_location_from_index(end)  # ty: ignore[unresolved-attribute]
        self.delete(start_location, end_location)
        self.move_cursor(start_location)
        return True

    def _find_placeholder_span(
        self, cursor_offset: int, *, backwards: bool
    ) -> tuple[int, int] | None:
        """Return the collapsed-paste placeholder span to delete, if any.

        Only placeholders backed by an entry in `_pasted_contents` are treated
        as atomic tokens; placeholder-shaped text the user typed by hand edits
        character by character. The paste map is left untouched so an undo can
        restore the token with its content.

        Args:
            cursor_offset: Character offset of the cursor from the text start.
            backwards: Whether the delete is backwards (backspace) or forwards.

        Returns:
            The `(start, end)` span of the placeholder to delete, or `None`.
        """
        text = self.text
        pasted_ids = set(self._pasted_contents)
        for match in PASTE_PLACEHOLDER_PATTERN.finditer(text):
            if int(match.group(1)) not in pasted_ids:
                continue
            start, end = match.span()
            if backwards:
                if start < cursor_offset <= end:
                    return start, end
                if cursor_offset > 0:
                    previous_index = cursor_offset - 1
                    # Only a literal space (the separator auto-inserted after a
                    # token) is swallowed with it. A newline must not be — that
                    # would delete the whole placeholder when the user is only
                    # backspacing the line break to rejoin the lines.
                    if (
                        previous_index < len(text)
                        and previous_index == end
                        and text[previous_index] == " "
                    ):
                        return start, cursor_offset
            elif start <= cursor_offset < end:
                return start, end
        return None

    def action_delete_right(self) -> None:
        """Delete a bound paste placeholder atomically or the next character."""
        if not self._delete_placeholder_token(backwards=False):
            super().action_delete_right()

    def action_delete_word_left(self) -> None:
        """Delete a bound paste placeholder atomically or the previous word."""
        if not self._delete_placeholder_token(backwards=True):
            super().action_delete_word_left()

    async def _on_paste(self, event: events.Paste) -> None:
        """Collapse a large bracketed paste, else let the base area insert it."""
        if self._paste_burst_buffer:
            await self._flush_paste_burst()
        if self._paste_collapse_enabled() and should_collapse_paste(event.text):
            # Intercept so Textual's default paste handler doesn't also insert
            # the full text; store it and insert a compact placeholder instead.
            event.prevent_default()
            event.stop()
            self._collapse_and_insert_paste(event.text)
        # Otherwise fall through: Textual's TextArea._on_paste inserts the text.


def _collapse_pastes_enabled() -> bool:
    """Resolve whether large pastes should be collapsed into placeholders.

    Reads `DEEPAGENTS_CODE_COLLAPSE_PASTES`, then `[ui].collapse_pastes` in
    `~/.deepagents/config.toml`, defaulting to enabled. This is the single
    source of truth shared with the chat input (`ChatInput` calls it once at
    construction).

    Returns:
        The resolved preference (defaults to `True`).
    """
    from deepagents_code.config_manifest import (
        get_option,
        load_config_toml,
        resolve_scalar,
    )

    option = get_option("display.collapse_pastes")
    if option is None:
        # Unreachable unless the manifest key is renamed without updating this
        # literal; log so that mismatch surfaces instead of silently defaulting.
        logger.warning(
            "Unknown config option %r; defaulting to enabled", "display.collapse_pastes"
        )
        return True
    value, _ = resolve_scalar(option, toml_data=load_config_toml())
    return bool(value)
