"""Shared paste handling for text-area inputs.

Terminals deliver a paste in one of two shapes: a single bracketed `Paste`
event, or — when bracketed paste is unavailable — a rapid stream of individual
key events. Both the primary chat input and the inline free-text prompts keep a
multi-line paste grouped instead of submitting on the first embedded newline.
Bracketed and detected key-event pastes may additionally collapse into a compact
`[Pasted text #N]` placeholder that expands on submit, and dropped-path payloads
(quoted, or bare paths detected by shape) are routed to path/media handling.

Ordinary typing is never hidden: a rapid run stays in the document until
something confirms it is a paste — an embedded newline, a dropped-path shape, or
a length no human reaches at burst speed — at which point it is promoted into
the hidden buffer.

`PasteBurstTextArea` owns the burst detection and Enter-suppression state
machine, leaving policy (slash-command context, whether collapsing is enabled,
how a flushed payload is handled) to overridable hooks.
`CollapsingPasteTextArea` layers the large-paste collapse + placeholder storage
on top, keeping the full content off-screen until submission.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding
from textual.widgets import TextArea

from deepagents_code.input import looks_like_dropped_payload
from deepagents_code.paste_collapse import (
    PASTE_PLACEHOLDER_PATTERN,
    PASTE_THRESHOLD_CHARS,
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

PASTE_BURST_MIN_CHARS = 3
"""Consecutive fast keystrokes before a stream is treated as a paste burst.

Terminals that lack bracketed paste replay a paste as individual key events.
Counting a short run of rapid chars distinguishes that from human typing,
which has much larger inter-key gaps.

Reaching this count does not by itself hide the run: it arms the Enter
suppression window and makes the run eligible for promotion into
`_paste_burst_buffer`. Promotion itself needs further evidence of a paste — see
`_check_burst_run_for_promotion` and `_consume_enter_as_burst_newline`.
"""

PASTE_BURST_PROMOTE_CHARS = PASTE_THRESHOLD_CHARS
"""Rapid-run length that on its own confirms a key-event paste.

A run this long arriving at burst speed (each char within
`PASTE_BURST_CHAR_GAP_SECONDS`) is unreachable by human typing, so it is
promoted into the buffer even without an embedded newline. Derived from the
collapse threshold so a large single-line key-event paste reaches
`[Pasted text #N]` collapsing. The comparison here is `>=` while
`should_collapse_paste` uses `>`, so a run of exactly this length is promoted and
then re-inserted verbatim.
"""

PASTE_ENTER_SUPPRESS_WINDOW_SECONDS = 0.12
"""Window after recent burst activity during which `enter` inserts a newline.

Keeps multi-line pastes grouped as one input even when newlines arrive as
`enter` key events slightly after the surrounding characters (e.g. across
terminal read boundaries), instead of submitting mid-paste.
"""

_BACKSLASH_ENTER_GAP_SECONDS = 0.15
"""Maximum gap between a `\\` key and a following `enter` key to treat the
pair as a terminal-emitted shift+enter sequence.

Some terminals (e.g. VSCode's built-in terminal) send a literal backslash
followed by enter when the user presses shift+enter.  The gap is
generous (150 ms) because the terminal emits both characters nearly
simultaneously; a human deliberately typing `\\` then pressing Enter would
have a much larger gap."""


class PasteBurstTextArea(TextArea):
    """`TextArea` that detects paste-like keystroke bursts.

    Subclasses drive the state machine from their own `_on_key` by calling the
    helper methods here, and override the policy hooks
    (`_in_slash_command_context`, `_dispatch_burst_payload`) as needed. The base
    inserts a flushed burst verbatim; collapsing into placeholders is layered on
    by `CollapsingPasteTextArea`.
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding(
            "shift+enter,alt+enter,ctrl+enter,ctrl+j",
            "insert_newline",
            "New Line",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+backspace,alt+backspace",
            "delete_word_left",
            "Delete left to start of word",
            show=False,
        ),
    ]
    """Shared key bindings for every paste-aware text area.

    These are the single source of truth for shortcut keys, inherited by both
    the chat input and inline prompts, which no longer define their own (were a
    subclass to add its own BINDINGS, Textual would merge them across the MRO
    rather than replace these). `_NEWLINE_KEYS` is derived from this list so
    `_on_key` stays in sync.
    """

    _NEWLINE_KEYS: ClassVar[frozenset[str]] = frozenset(
        key
        for b in BINDINGS
        if b.action == "insert_newline"
        for key in b.key.split(",")
    )
    """Flattened set of keys that insert a newline, derived from `BINDINGS`."""

    _paste_burst_buffer: str
    _paste_burst_last_char_time: float | None
    _paste_burst_timer: Timer | None
    _paste_burst_run: int
    _paste_burst_run_text: str
    _paste_burst_last_key_time: float | None
    _paste_burst_last_suppressed_enter_time: float | None
    _paste_burst_window_until: float | None
    _backslash_pending_time: float | None

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the text area and its paste-burst state."""
        super().__init__(**kwargs)
        self._init_paste_burst_state()

    def _init_paste_burst_state(self) -> None:
        """Reset all paste-burst tracking fields to their initial values."""
        # Holds burst text only after promotion. A rapid run stays in the
        # document until something confirms it is a paste.
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time = None
        self._paste_burst_timer = None
        # Counts consecutive rapid keystrokes so a paste replayed as key events
        # can be recognized without a bracketed paste event.
        self._paste_burst_run = 0
        self._paste_burst_run_text = ""
        self._paste_burst_last_key_time = None
        self._paste_burst_last_suppressed_enter_time = None
        # Deadline until which `enter` inserts a newline rather than submitting,
        # keeping multi-line pastes grouped across read boundaries.
        self._paste_burst_window_until = None
        # Timestamp of a `\` keypress awaiting a fast `enter` to be treated as a
        # terminal-emitted shift+enter. See `_BACKSLASH_ENTER_GAP_SECONDS`.
        self._backslash_pending_time = None

    # -- Cursor blink ---------------------------------------------------------

    def _restart_blink(self) -> None:
        """Keep the cursor hidden while the text area is unfocused.

        Outside read-only mode Textual's `TextArea._draw_cursor` is
        `(has_focus and not cursor_blink) or (cursor_blink and _cursor_visible)`,
        so with blinking on (the default) it paints the cursor from
        `_cursor_visible` alone and never consults focus. Any `_restart_blink()`
        on an unfocused text area therefore shows a blinking cursor in a field
        that cannot accept keys.

        `TextArea._on_mouse_down` sets `_selecting` unconditionally, so the
        matching mouse-up reaches `_restart_blink()` through
        `_end_mouse_selection` unless a subclass gates the handler first (as
        `ChatTextArea` does for refocus clicks). That covers clicks landing
        while a focus-trapping widget is open: the `edit_file` approval menu
        re-grabs focus on blur between mouse-down and mouse-up, so the click
        left a phantom blinking cursor in the chat input. Programmatic
        multi-character `insert()` into an unfocused input has the same
        effect.

        `_pause_blink(visible=False)` also parks the blink timer. Textual's
        `_watch_has_focus` re-arms it when focus returns, so the cursor resumes
        blinking rather than coming back solid.

        Deliberately overrides Textual's private `_restart_blink`; verified
        against Textual 8.2.8. Re-verify these attribute names on every Textual
        bump — the `TestCursorHiddenWhileUnfocused` classes in
        `test_chat_input.py` and `test_inline_prompt.py` fail if this stops
        holding.
        """
        if not self.has_focus:
            self._pause_blink(visible=False)
            return
        super()._restart_blink()

    # -- Policy hooks (override in subclasses) --------------------------------

    def _in_slash_command_context(self) -> bool:  # noqa: PLR6301  # overridable hook
        """Return whether Enter should keep submit/dispatch semantics.

        Base text areas have no slash-command surface, so the Enter-suppression
        window always applies. Override to opt keystrokes out of grouping.
        """
        return False

    async def _dispatch_burst_payload(self, payload: str) -> None:
        """Handle a flushed burst payload. Base behavior inserts it verbatim.

        Implementations must apply the payload before returning. A payload
        applied later — from a posted message or a scheduled callback — is
        ordered behind any key event already waiting in this widget's queue, so
        the next keystroke would be inserted ahead of the paste.
        """
        self.insert(payload)

    def _burst_run_payload_for_dispatch(self, payload: str) -> str:  # noqa: PLR6301  # overridable hook
        """Return the payload represented by a visible rapid-key run.

        Most text areas display every character in the run, so the payload is
        unchanged. Subclasses with virtual prefixes may restore characters
        that were consumed before insertion.
        """
        return payload

    def _on_burst_run_promoted(
        self, visible_payload: str, dispatch_payload: str
    ) -> None:
        """React after a visible run has been promoted into the burst buffer."""

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

    def _append_recent_paste_burst_text(self, text: str, now: float) -> bool:
        """Append text only when it continues an active rapid burst.

        Returns:
            `True` when the text was appended, or `False` when there is no
            active burst or it has gone idle.
        """
        if not self._paste_burst_buffer:
            return False
        last_time = self._paste_burst_last_char_time
        if last_time is None or (now - last_time) > PASTE_BURST_CHAR_GAP_SECONDS:
            return False
        self._append_paste_burst(text, now)
        return True

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
        """Reset all paste-burst and backslash tracking to a clean slate.

        Used by text-replacing entry points so a wholesale text swap never
        leaves stale burst timing that would misclassify the next keystroke.
        """
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time = None
        self._paste_burst_window_until = None
        self._backslash_pending_time = None
        self._reset_paste_burst_run()
        self._cancel_paste_burst_timer()

    def _enter_inserts_newline_during_burst(self, now: float) -> bool:
        """Return whether `enter` should insert a newline rather than submit.

        True when the preceding keystroke was part of a rapid run or the
        previous `enter` was already suppressed, and the suppression window is
        still open. The window bounds how long a replayed paste's newlines stay
        grouped. Returns `False` immediately in slash-command context (see
        `_in_slash_command_context`), and an active burst buffer short-circuits to
        `True` regardless of the window (see below).

        The first suppressed `enter` must remain within the character gap. A
        completed single-line burst followed by a deliberate `enter` is
        otherwise indistinguishable from a delayed pasted newline, and submit
        behavior takes priority once the rapid stream has gone idle. After one
        `enter` is suppressed, the wider window keeps consecutive pasted blank
        lines grouped.
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
        if last_key is None:
            return False
        return (now - last_key) <= PASTE_BURST_CHAR_GAP_SECONDS

    async def _flush_paste_burst(self) -> None:
        """Flush buffered burst text through the payload dispatch hook.

        When the buffer is empty this is a no-op, so it is safe to call
        defensively before handling a bracketed paste. The payload is applied
        before this returns, so a caller may go on to handle the current key
        knowing it lands after the paste.

        The buffer is cleared before dispatch, and `_promote_paste_burst_run` has
        already deleted the run from the document, so a raising dispatch would
        leave the text nowhere at all — not on screen, not in the buffer, not in
        undo history. Media decoding, attachment tracking and notifications all
        run inside that call, so the payload is re-inserted verbatim before the
        error propagates.
        """
        payload = self._paste_burst_buffer
        self._paste_burst_buffer = ""
        self._paste_burst_last_char_time = None
        self._cancel_paste_burst_timer()
        if not payload:
            return
        try:
            await self._dispatch_burst_payload(payload)
        except Exception:
            logger.warning(
                "Burst dispatch failed (%d chars); inserting payload verbatim",
                len(payload),
                exc_info=True,
            )
            self.insert(payload)
            raise

    def _promote_paste_burst_run(self, now: float) -> bool:
        """Move an already-inserted rapid run out of the document into the buffer.

        Deletes the run's characters from the document — they are visible on
        screen at this point — and hands them to `_start_paste_burst` so the
        eventual flush can apply dropped-path and paste-collapse policy.

        No document mutation happens until every guard has passed, so a `False`
        return never leaves a partially-promoted document. A failed verification
        does still discard the tracked run, so `False` is not side-effect-free.

        Args:
            now: Monotonic timestamp for the current key event.

        Returns:
            `True` when the run was verified present immediately before the
            cursor and moved into the buffer. `False` when promotion is unsafe:
            an empty run, a non-empty selection (deleting would clobber the
            user's selected range), or a document whose text immediately before
            the cursor is no longer the tracked run — which means an intervening
            edit desynchronized the tracker. Callers must fall back to handling
            the key normally.
        """
        payload = self._paste_burst_run_text
        if not payload or not self.selection.is_empty:
            return False
        cursor = self.cursor_location
        cursor_offset = self.document.get_index_from_location(cursor)  # ty: ignore[unresolved-attribute]  # Document has this method; DocumentBase stub is narrower
        start_offset = cursor_offset - len(payload)
        if start_offset < 0 or self.text[start_offset:cursor_offset] != payload:
            # An untracked edit desynchronized the tracker, so drop the stale run
            # and let tracking restart on the next keystroke. Logged at warning
            # (not debug) so it survives the default INFO level — this is a state
            # -machine bug, not a user-reachable condition. The message carries
            # only sizes, never the payload, which is user content.
            logger.warning(
                "Burst run diverged from document (run=%d chars, start=%d, "
                "cursor=%d, doc=%d); skipping promotion",
                len(payload),
                start_offset,
                cursor_offset,
                len(self.text),
            )
            self._reset_paste_burst_run()
            return False
        start = self.document.get_location_from_index(start_offset)  # ty: ignore[unresolved-attribute]
        self.delete(start, cursor)
        self._start_paste_burst(payload, now)
        # The buffer now owns these characters; clearing the run keeps the
        # "run tracks visible text, buffer tracks hidden text" invariant true by
        # construction, so a later Enter cannot re-promote the same stale text.
        self._reset_paste_burst_run()
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
            and the caller should continue normal key handling. A flush applies
            its payload before returning, so handling the key afterwards orders
            it after the paste.
        """
        if not self._paste_burst_buffer:
            return False
        if event.key == "enter":
            self._append_paste_burst("\n", now)
            return True
        if (
            event.is_printable
            and event.character is not None
            and self._append_recent_paste_burst_text(event.character, now)
        ):
            return True
        await self._flush_paste_burst()
        return False

    def _track_burst_run(self, event: events.Key, now: float) -> None:
        """Track a rapid run, arming Enter suppression once it looks like a paste."""
        if event.is_printable and event.character is not None:
            self._note_printable_burst_keystroke(event.character, now)
        elif event.key != "enter":
            self._reset_paste_burst_run()

    def _note_printable_burst_keystroke(self, char: str, now: float) -> None:
        """Record a printable char in the rapid run and arm Enter suppression.

        Arming is conditional: the run must reach `PASTE_BURST_MIN_CHARS` and
        must not be a slash command, where Enter always submits.

        Any printable char also un-latches the suppressed-Enter state, so the next
        Enter must re-qualify through the character gap.

        Call this for any character that reaches the document without passing
        through `_track_burst_run` — a caller that inserts text itself and
        returns early must still keep the tracker in sync, or the run text will
        diverge from the document and the run is discarded, losing grouping for
        that stretch of the paste.

        Args:
            char: The character being inserted into the document.
            now: Monotonic timestamp for the current key event.
        """
        self._paste_burst_last_suppressed_enter_time = None
        self._note_paste_burst_keystroke(char, now)
        if (
            self._paste_burst_run >= PASTE_BURST_MIN_CHARS
            and not self._in_slash_command_context()
        ):
            self._paste_burst_window_until = now + PASTE_ENTER_SUPPRESS_WINDOW_SECONDS

    def _check_burst_run_for_promotion(self) -> None:
        """Promote a rapid run whose shape or size already confirms a paste.

        Callers must invoke this only once the current character is in the
        document, so the run is present and `_promote_paste_burst_run` can find
        it. Two confirmations do not need to wait for a newline:

        - A dropped-path shape (`/`, `~`, drive letter, `file://`, UNC). Without
          this, an unquoted single-line drop would never reach path parsing or
          media rejection.
        - A run that reaches `PASTE_BURST_PROMOTE_CHARS`, a length no human
          reaches at burst speed. Without this, a large single-line key-event
          paste would never collapse into a placeholder.

        Neither case flushes here. The paste is still streaming, so the run is
        only a prefix of the payload; once promoted, the remaining characters
        are absorbed straight into the buffer by `_absorb_key_into_burst` and
        the idle timer flushes the complete payload when the stream stops.
        Flushing per keystroke would instead hand each prefix to
        `_dispatch_burst_payload` — re-running path parsing, and its filesystem
        probes, once per character.

        Ordinary typing reaches neither confirmation: to qualify it would have to
        sustain `PASTE_BURST_MIN_CHARS` keystrokes inside
        `PASTE_BURST_CHAR_GAP_SECONDS` of each other (~400 WPM) *and* either open
        with a path shape or run past the length threshold.
        """
        if self._paste_burst_buffer or self._paste_burst_run < PASTE_BURST_MIN_CHARS:
            return
        payload = self._paste_burst_run_text
        dispatch_payload = self._burst_run_payload_for_dispatch(payload)
        # In slash-command context Enter always submits and the text is a command,
        # not content, so nothing may be hidden — with one exception: a payload the
        # dispatch hook rewrote is a path whose leading `/` the mode prefix
        # consumed (see `ChatTextArea._burst_run_payload_for_dispatch`), and that
        # must reach path handling precisely so it stops being read as a command.
        if self._in_slash_command_context() and dispatch_payload == payload:
            return
        if not (
            looks_like_dropped_payload(dispatch_payload)
            or len(payload) >= PASTE_BURST_PROMOTE_CHARS
        ):
            return
        last_key_time = self._paste_burst_last_key_time
        if last_key_time is None:
            # Unreachable: the run is at least `PASTE_BURST_MIN_CHARS`, and every
            # counted keystroke stamps this field. Refuse rather than substitute a
            # timestamp, which on a monotonic clock would land decades in the past
            # and make the burst flush per character.
            logger.warning(
                "Qualifying burst run (%d chars) has no key timestamp; "
                "skipping promotion",
                self._paste_burst_run,
            )
            return
        if not self._promote_paste_burst_run(last_key_time):
            # The shape or length already confirmed a paste, so failing here means
            # dropped-path routing, media rejection, and collapsing are all
            # silently skipped for this payload. Worth a breadcrumb.
            logger.warning(
                "Confirmed burst paste (%d chars) could not be promoted; "
                "path routing and collapsing are skipped for it",
                len(payload),
            )
            return
        self._paste_burst_buffer = dispatch_payload
        self._on_burst_run_promoted(payload, dispatch_payload)

    def _consume_enter_as_burst_newline(self, now: float) -> bool:
        """Insert a newline instead of submitting when inside a paste burst.

        Returns:
            `True` when Enter was consumed as a newline (part of a paste);
            `False` when Enter should fall through to its submit handling.
        """
        if not self._enter_inserts_newline_during_burst(now):
            self._paste_burst_last_suppressed_enter_time = None
            return False
        # This newline confirms a multi-line key-event paste, so pull the
        # still-visible run into the buffer and keep the newline with it. Both
        # shipped `_on_key`s absorb or flush an active buffer before Enter
        # reaches here, so the `_paste_burst_buffer` branch is unreachable. It
        # exists because falling through to `_promote_paste_burst_run` would call
        # `_start_paste_burst`, which *assigns* the buffer rather than appending
        # — silently dropping the text already in it. Loud rather than silently
        # wrong if a future caller gets here.
        if self._paste_burst_buffer:
            logger.warning(
                "Enter reached burst-newline handling with a live buffer "
                "(%d chars); keeping the newline with the buffer",
                len(self._paste_burst_buffer),
            )
            self._append_paste_burst("\n", now)
        elif self._promote_paste_burst_run(now):
            self._append_paste_burst("\n", now)
        else:
            self.action_insert_newline()
        # Set after promotion: `_promote_paste_burst_run` resets run tracking,
        # which clears `_paste_burst_last_suppressed_enter_time`.
        # `_paste_burst_window_until` is not cleared by that reset; it is
        # refreshed here to extend the grouping window.
        self._paste_burst_window_until = now + PASTE_ENTER_SUPPRESS_WINDOW_SECONDS
        self._paste_burst_last_suppressed_enter_time = now
        return True

    # -- Newline affordances (shared by concrete text areas) ------------------

    def _delete_preceding_backslash(self) -> bool:
        """Delete the backslash character immediately before the cursor.

        Caller must ensure a backslash is expected at this position. The
        method verifies the character before deleting it.

        Returns:
            `True` if a backslash was found and deleted, `False` otherwise.
        """
        row, col = self.cursor_location
        if col > 0:
            start = (row, col - 1)
            if self.document.get_text_range(start, self.cursor_location) == "\\":
                self.delete(start, self.cursor_location)
                return True
        elif row > 0:
            prev_line = self.document.get_line(row - 1)
            start = (row - 1, len(prev_line) - 1)
            end = (row - 1, len(prev_line))
            if self.document.get_text_range(start, end) == "\\":
                self.delete(start, self.cursor_location)
                return True
        return False

    def _consume_backslash_enter_newline(
        self, event: events.Key, now: float, *, enabled: bool = True
    ) -> bool:
        """Return whether a terminal-emitted backslash+Enter became a newline.

        Args:
            event: The key event being handled.
            now: Current monotonic timestamp, compared against the pending
                backslash time via `_BACKSLASH_ENTER_GAP_SECONDS`.
            enabled: When `False`, the fallback is skipped (still clearing any
                pending backslash). Callers pass `False` to suppress it while a
                competing affordance owns Enter (e.g. an open completion popup).
        """
        if (
            event.key == "enter"
            and enabled
            and self._backslash_pending_time is not None
            and (now - self._backslash_pending_time) <= _BACKSLASH_ENTER_GAP_SECONDS
            and not self._enter_inserts_newline_during_burst(now)
        ):
            self._backslash_pending_time = None
            if self._delete_preceding_backslash():
                event.prevent_default()
                event.stop()
                self.action_insert_newline()
                return True
        self._backslash_pending_time = None
        return False

    def _track_backslash_pending(self, event: events.Key, now: float) -> None:
        """Record a backslash keypress so a fast following Enter becomes a newline."""
        if event.key == "backslash" and event.character == "\\":
            self._backslash_pending_time = now

    def _consume_modifier_newline(self, event: events.Key) -> bool:
        """Return whether a modifier-Enter (or Ctrl+J) key inserted a newline."""
        if event.key in self._NEWLINE_KEYS:
            event.prevent_default()
            event.stop()
            self.action_insert_newline()
            return True
        return False


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
                    # Swallow trailing whitespace with the token, except for a
                    # newline: backspacing a line break should rejoin the lines
                    # without deleting the placeholder.
                    if (
                        previous_index < len(text)
                        and previous_index == end
                        and text[previous_index].isspace()
                        and text[previous_index] != "\n"
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
