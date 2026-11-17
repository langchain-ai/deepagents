"""Report agent status to the terminal with `OSC 7501` (Program Status Protocol).

Terminals and multiplexers that implement the protocol (for example Herdr)
show what each program is doing: working, waiting on the user, finished, or
failed. See https://www.superlogical.com/rex/docs/build/program-status.

The encoding functions are pure. `ProgramStatusReporter` is the only part that
writes, through `terminal_escape.write_terminal_escape`, so the
`DEEPAGENTS_CODE_NO_TERMINAL_ESCAPE` opt-out applies.

The reporter never sends the support query (`OSC 7501 ; ? ST`). The spec lets
programs report without asking, and waiting for a reply can silently disable
reporting on slow startups.
"""

from __future__ import annotations

import atexit
import base64
import logging
import re
import threading
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

APP_NAME = "deepagents-code"
"""Stable `app` identifier sent with every report."""

MAX_SEQUENCE_BYTES = 4096
"""Spec cap for a whole report, from `OSC` through `ST`."""

MAX_TITLE_BYTES = 192
"""Spec cap for the decoded `title` (256 bytes once base64-encoded)."""

MAX_MSG_BYTES = 2048
"""Spec cap for the decoded `msg` (2732 bytes once base64-encoded)."""

_SEGMENT = r"[A-Za-z0-9_.+-]{1,32}"
_APP_RE = re.compile(_SEGMENT)
_ID_RE = re.compile(rf"{_SEGMENT}(?:/{_SEGMENT}){{0,7}}")
_MAX_ID_BYTES = 128
_ST = "\x1b\\"


class ProgramState(StrEnum):
    """Values of the protocol's required `state` key."""

    IDLE = "idle"
    """At rest, waiting for the user's next instruction."""

    WORKING = "working"
    """Running. Terminals drop this record when the process exits."""

    DONE = "done"
    """Finished, result not yet seen. Survives process exit."""

    BLOCKED = "blocked"
    """Cannot continue until the user acts. `kind` says what kind of input."""

    ERROR = "error"
    """Failed and stopped. Survives process exit."""

    CLEAR = "clear"
    """Not a state: removes the addressed record and every record below it."""


class BlockedKind(StrEnum):
    """Values of the `kind` key, which terminals read only with `blocked`."""

    PERMISSION = "permission"
    """Waiting for the user to approve an action."""

    QUESTION = "question"
    """Waiting for the user to answer a question."""

    AUTH = "auth"
    """Waiting for the user to sign in or provide credentials."""


_PROGRESS_STATES = frozenset({ProgramState.WORKING, ProgramState.BLOCKED})


@dataclass(frozen=True, slots=True)
class ProgramStatus:
    """One report for one record.

    Each report replaces its record entirely on the terminal, so every field
    that should stay visible must be set on every report.

    Attributes:
        state: The record's state, or `CLEAR` to remove it.
        kind: What the user must do. Only valid with `BLOCKED`.
        progress: Percent complete, `0`-`100`. Only valid with `WORKING` or
            `BLOCKED`; `None` means indeterminate.
        record_id: Sub-record path such as `task/1`; `None` addresses the
            program's own (root) record.
        title: Short label. Control characters are replaced and the text is
            truncated to fit the spec when encoded.
        msg: One human-readable line, sanitized and truncated like `title`.
    """

    state: ProgramState
    kind: BlockedKind | None = None
    progress: int | None = None
    record_id: str | None = None
    title: str = ""
    msg: str = ""

    def __post_init__(self) -> None:
        """Reject combinations the protocol would silently ignore.

        Raises:
            ValueError: If `kind`, `progress`, `title`, `msg`, or `record_id`
                is invalid for this report.
        """
        if self.kind is not None and self.state is not ProgramState.BLOCKED:
            msg = f"kind is only valid with state=blocked, not {self.state}"
            raise ValueError(msg)
        if self.state is ProgramState.CLEAR and (self.title or self.msg):
            msg = "title and msg are not valid with state=clear"
            raise ValueError(msg)
        if self.progress is not None:
            _check_progress(self.state, self.progress)
        if self.record_id is not None and not is_valid_record_id(self.record_id):
            msg = f"invalid OSC 7501 record id: {self.record_id!r}"
            raise ValueError(msg)


def _check_progress(state: ProgramState, progress: int) -> None:
    """Raise unless `progress` is a percentage allowed for `state`.

    Raises:
        ValueError: If `state` takes no progress or `progress` is out of range.
    """
    if state not in _PROGRESS_STATES:
        msg = f"progress is only valid with working or blocked, not {state}"
        raise ValueError(msg)
    if not 0 <= progress <= 100:  # noqa: PLR2004  # the spec's percentage bounds
        msg = f"progress must be within 0-100, got {progress}"
        raise ValueError(msg)


def is_valid_record_id(record_id: str) -> bool:
    """Return whether `record_id` matches the spec's id grammar and limits.

    Args:
        record_id: Candidate path of `/`-separated segments.
    """
    return (
        len(record_id.encode()) <= _MAX_ID_BYTES
        and _ID_RE.fullmatch(record_id) is not None
    )


def is_valid_app(app: str) -> bool:
    """Return whether `app` matches the spec's `app` grammar.

    Args:
        app: Candidate program identifier.
    """
    return _APP_RE.fullmatch(app) is not None


def sanitize_text(text: str, max_bytes: int) -> str:
    """Return `text` as one line of at most `max_bytes` UTF-8 bytes.

    Terminals refuse a whole report whose decoded text contains a control
    character, so newlines, tabs, and other controls become spaces and runs of
    whitespace collapse. Truncation never splits a multi-byte character.

    Args:
        text: Arbitrary text, such as a thread name or an error message.
        max_bytes: Byte budget for the decoded text.
    """
    spaced = "".join(" " if unicodedata.category(ch) == "Cc" else ch for ch in text)
    line = " ".join(spaced.split())
    data = line.encode("utf-8", errors="replace")[:max_bytes]
    return data.decode("utf-8", errors="ignore").rstrip()


def _b64(text: str) -> str:
    """Return standard, padded base64 of `text` as UTF-8."""
    return base64.b64encode(text.encode("utf-8", errors="replace")).decode("ascii")


def _pairs(status: ProgramStatus, app: str) -> list[str]:
    """Return the `key=value` pairs for `status`, `state` first."""
    pairs = [f"state={status.state}"]
    if status.record_id is not None:
        pairs.append(f"id={status.record_id}")
    if status.kind is not None:
        pairs.append(f"kind={status.kind}")
    if status.progress is not None:
        pairs.append(f"progress={status.progress}")
    pairs.append(f"app={app}")
    title = sanitize_text(status.title, MAX_TITLE_BYTES)
    if title:
        pairs.append(f"title={_b64(title)}")
    msg = sanitize_text(status.msg, MAX_MSG_BYTES)
    if msg:
        pairs.append(f"msg={_b64(msg)}")
    return pairs


def encode_program_status(status: ProgramStatus, *, app: str = APP_NAME) -> str:
    r"""Return the full `OSC 7501` sequence for `status`, terminated with `ESC \`.

    Args:
        status: The report to encode.
        app: Program identifier; must match `[A-Za-z0-9_.+-]{1,32}`.

    Returns:
        The escape sequence, always within the spec's 4096-byte cap.

    Raises:
        ValueError: If `app` is invalid, or the sequence exceeds the
            4096-byte cap.
    """
    if not is_valid_app(app):
        msg = f"invalid OSC 7501 app: {app!r}"
        raise ValueError(msg)
    sequence = f"\x1b]7501;{':'.join(_pairs(status, app))}{_ST}"
    # Field caps keep the largest legal report near 3.3 KB; this guards edits.
    if len(sequence.encode("ascii")) > MAX_SEQUENCE_BYTES:
        msg = "OSC 7501 report exceeds the 4096-byte limit"
        raise ValueError(msg)
    return sequence


def _is_under(record_id: str, parent: str) -> bool:
    """Return whether `record_id` is `parent` or one of its descendants."""
    return record_id == parent or record_id.startswith(f"{parent}/")


class ProgramStatusReporter:
    """Send `OSC 7501` reports and clear them when the program exits.

    Skips a report identical to the last one sent for the same record, so
    callers can report on every UI update. Unless `clear_on_exit` is `False`,
    the first successful non-`clear` report registers an `atexit` hook that
    clears all records, so an unhandled exception or `sys.exit` does not leave
    a stale status behind. A hard kill skips the hook.

    Thread-safe: reports may come from the event loop and the `atexit` hook.
    """

    def __init__(
        self,
        *,
        write: Callable[[str], bool] | None = None,
        app: str = APP_NAME,
        enabled: bool = True,
        clear_on_exit: bool = True,
    ) -> None:
        """Create a reporter.

        Args:
            write: Writes one escape sequence and returns whether it was sent.
                Defaults to `terminal_escape.write_terminal_escape`.
            app: Program identifier sent with every report.
            enabled: When `False`, every method is a no-op.
            clear_on_exit: Clear every record at interpreter exit. Turn off for
                a one-shot run whose final `done` or `error` should outlive it.

        Raises:
            ValueError: If `app` is invalid.
        """
        if not is_valid_app(app):
            msg = f"invalid OSC 7501 app: {app!r}"
            raise ValueError(msg)
        self._write = write
        self._app = app
        self._enabled = enabled
        self._sent: dict[str | None, str] = {}
        self._lock = threading.Lock()
        self._atexit_registered = not clear_on_exit
        self._warned = False

    @property
    def enabled(self) -> bool:
        """Whether this reporter writes anything."""
        return self._enabled

    def report(self, status: ProgramStatus, *, force: bool = False) -> bool:
        """Send `status` unless it repeats the record's last report.

        Never raises: status output is cosmetic.

        Args:
            status: The report to send.
            force: Send even if it repeats the last report, for when another
                program may have replaced or dropped the record meanwhile.

        Returns:
            `True` if a sequence was written.
        """
        if not self._enabled:
            return False
        try:
            sequence = encode_program_status(status, app=self._app)
        except ValueError:
            logger.debug("Dropping invalid OSC 7501 report", exc_info=True)
            return False
        with self._lock:
            if not force and self._sent.get(status.record_id) == sequence:
                return False
            if not self._send(sequence):
                return False
            self._remember(status, sequence)
        return True

    def clear(self, record_id: str | None = None) -> bool:
        """Remove a record and everything below it; `None` removes all records.

        Does nothing when nothing in that subtree was reported.

        Args:
            record_id: The record to remove, or `None` for every record.

        Returns:
            `True` if a `clear` report was written.
        """
        with self._lock:
            if not any(
                record_id is None or (rid is not None and _is_under(rid, record_id))
                for rid in self._sent
            ):
                return False
        return self.report(ProgramStatus(ProgramState.CLEAR, record_id=record_id))

    def _send(self, sequence: str) -> bool:
        """Return whether `sequence` was written; never raises.

        The default writer handles its own I/O errors, so an exception here is
        a bug. Warn on the first one so it is visible, then log at debug to
        avoid one warning per UI update.
        """
        try:
            write = self._write or _default_write()
            return write(sequence)
        except Exception:  # noqa: BLE001  # status output must never break the app
            log = logger.debug if self._warned else logger.warning
            self._warned = True
            log("OSC 7501 write failed", exc_info=True)
            return False

    def _remember(self, status: ProgramStatus, sequence: str) -> None:
        """Record what the terminal now holds after `status` was sent."""
        if status.state is ProgramState.CLEAR:
            parent = status.record_id
            self._sent = {
                rid: seq
                for rid, seq in self._sent.items()
                if parent is not None and (rid is None or not _is_under(rid, parent))
            }
            return
        self._sent[status.record_id] = sequence
        if not self._atexit_registered:
            atexit.register(self._clear_at_exit)
            self._atexit_registered = True

    def _clear_at_exit(self) -> None:
        """Clear every record when the interpreter exits."""
        try:
            self.clear()
        except Exception:
            logger.debug("Failed to clear OSC 7501 status at exit", exc_info=True)


def _default_write() -> Callable[[str], bool]:
    """Return the shared best-effort terminal writer, imported lazily."""
    from deepagents_code.terminal_escape import write_terminal_escape

    return write_terminal_escape


class TurnOutcome(StrEnum):
    """How the most recent unit of work ended, shown while nothing is running."""

    IDLE = "idle"
    """Nothing to report: never ran, interrupted, rejected, or already seen."""

    DONE = "done"
    """The agent finished a turn the user has not acted on yet."""

    ERROR = "error"
    """The last turn, or startup, failed."""


_OUTCOME_STATES: dict[TurnOutcome, ProgramState] = {
    TurnOutcome.IDLE: ProgramState.IDLE,
    TurnOutcome.DONE: ProgramState.DONE,
    TurnOutcome.ERROR: ProgramState.ERROR,
}
"""The root-record state shown for each outcome while nothing is running."""


@dataclass(frozen=True, slots=True)
class Blocker:
    """One outstanding request for user input.

    Attributes:
        kind: What the user must do.
        msg: One line saying what is being asked.
        in_transcript: The prompt is a widget in the transcript, so clearing
            the transcript removes it.
    """

    kind: BlockedKind
    msg: str = ""
    in_transcript: bool = False


@dataclass(frozen=True, slots=True)
class AgentActivity:
    """Snapshot of everything that decides the program's root record.

    Attributes:
        working: Whether the agent or a long-running app task is busy.
        outcome: How the last unit of work ended.
        detail: Message for `outcome`, such as the error text.
        blockers: Outstanding user prompts as `(token, blocker)`, oldest first.
        title: Label for the record, such as the thread name.
    """

    working: bool = False
    outcome: TurnOutcome = TurnOutcome.IDLE
    detail: str = ""
    blockers: tuple[tuple[int, Blocker], ...] = ()
    title: str = ""


def activity_status(activity: AgentActivity) -> ProgramStatus:
    """Return the root record for `activity`.

    A pending prompt wins over everything because the user must act; the
    newest prompt is the one on screen. Work in progress wins over the last
    outcome.

    Args:
        activity: Current app state.
    """
    if activity.blockers:
        blocker = activity.blockers[-1][1]
        return ProgramStatus(
            ProgramState.BLOCKED,
            kind=blocker.kind,
            title=activity.title,
            msg=blocker.msg,
        )
    if activity.working:
        return ProgramStatus(ProgramState.WORKING, title=activity.title)
    return ProgramStatus(
        _OUTCOME_STATES[activity.outcome],
        title=activity.title,
        msg=activity.detail,
    )


def exit_status(exit_code: int | None) -> ProgramStatus:
    """Return the final report for a one-shot (headless) run.

    The spec asks a program that exits when it finishes to report `done` or
    `error` just before exiting. An interrupt or an unknown exit clears instead.

    Args:
        exit_code: The process exit code, or `None` when it is unknown.
    """
    if exit_code == 0:
        return ProgramStatus(ProgramState.DONE)
    if exit_code is None or exit_code == _INTERRUPTED_EXIT_CODE:
        return ProgramStatus(ProgramState.CLEAR)
    return ProgramStatus(ProgramState.ERROR, msg=f"Exited with code {exit_code}")


_INTERRUPTED_EXIT_CODE = 130


class AgentStatusTracker:
    """Keep `AgentActivity` current and report each change.

    The app calls the mutators from its event loop; `activity_status` decides
    what to send and the reporter drops repeats.
    """

    def __init__(self, reporter: ProgramStatusReporter) -> None:
        """Create a tracker that reports through `reporter`.

        Args:
            reporter: Destination for root-record reports.
        """
        self._reporter = reporter
        self._activity = AgentActivity()
        self._next_token = 0
        self._closed = False

    @property
    def activity(self) -> AgentActivity:
        """The current snapshot."""
        return self._activity

    def start(self) -> None:
        """Send the first report so the terminal knows the program is here."""
        self._sync()

    def set_working(self, working: bool) -> None:
        """Mark work as started or stopped.

        Starting new work forgets the previous outcome, since the user has
        moved on.

        Args:
            working: Whether work is in progress.
        """
        if working and not self._activity.working:
            self._apply(
                replace(
                    self._activity, working=True, outcome=TurnOutcome.IDLE, detail=""
                )
            )
        else:
            self._apply(replace(self._activity, working=working))

    def set_outcome(self, outcome: TurnOutcome, detail: str = "") -> None:
        """Record how the current or last unit of work ended.

        Args:
            outcome: The result to show once nothing is running.
            detail: Message to show with it, such as an error summary.
        """
        self._apply(replace(self._activity, outcome=outcome, detail=detail))

    def clear_error(self) -> None:
        """Return to `idle` if the last outcome was an error."""
        if self._activity.outcome is TurnOutcome.ERROR:
            self.set_outcome(TurnOutcome.IDLE)

    def set_title(self, title: str) -> None:
        """Set the label sent with every report.

        Args:
            title: Short label, or empty for none.
        """
        self._apply(replace(self._activity, title=title))

    def start_new_conversation(self) -> None:
        """Forget the last outcome and every prompt shown in the transcript.

        Call when the transcript is cleared for a new or different thread: its
        prompts are gone and the old thread's result no longer applies.
        """
        remaining = tuple(
            entry for entry in self._activity.blockers if not entry[1].in_transcript
        )
        self._apply(
            replace(
                self._activity,
                blockers=remaining,
                outcome=TurnOutcome.IDLE,
                detail="",
            )
        )

    def _block(self, blocker: Blocker) -> int:
        """Return a token for a newly recorded prompt waiting for the user."""
        self._next_token += 1
        token = self._next_token
        entry = (token, blocker)
        self._apply(replace(self._activity, blockers=(*self._activity.blockers, entry)))
        return token

    def _unblock(self, token: int) -> None:
        """Forget the prompt registered under `token`; unknown tokens are ignored.

        A user who closed a prompt has seen a `done` outcome, so it reverts to
        `idle`. An error stays: closing a prompt, such as cancelling a sign-in,
        does not fix it. `clear_error` or new work removes it.
        """
        blockers = self._activity.blockers
        remaining = tuple(entry for entry in blockers if entry[0] != token)
        if remaining == blockers:
            return
        activity = replace(self._activity, blockers=remaining)
        if activity.outcome is not TurnOutcome.ERROR:
            activity = replace(activity, outcome=TurnOutcome.IDLE, detail="")
        self._apply(activity)

    def track[T](
        self, future: asyncio.Future[T], kind: BlockedKind, msg: str = ""
    ) -> None:
        """Report blocked until `future` completes, fails, or is cancelled.

        For prompts mounted in the transcript; `start_new_conversation` drops
        them even if their future never completes.

        Args:
            future: Resolves when the user answers the prompt.
            kind: What the user must do.
            msg: One line saying what is being asked.
        """
        if future.done():
            return
        token = self._block(Blocker(kind, msg, in_transcript=True))
        future.add_done_callback(lambda _future: self._unblock(token))

    @contextmanager
    def blocked(self, kind: BlockedKind, msg: str = "") -> Iterator[None]:
        """Report blocked for the duration of the `with` block.

        Args:
            kind: What the user must do.
            msg: One line saying what is being asked.

        Yields:
            Nothing.
        """
        token = self._block(Blocker(kind, msg))
        try:
            yield
        finally:
            self._unblock(token)

    def refresh(self) -> None:
        """Resend the current report even if it was already sent.

        Use after the app resumes from suspension: the shell prompt shown
        meanwhile makes terminals drop `working` and `blocked` records.
        """
        if not self._closed:
            self._reporter.report(activity_status(self._activity), force=True)

    def close(self) -> None:
        """Clear every record and stop reporting; later calls do nothing."""
        if self._closed:
            return
        self._closed = True
        self._reporter.clear()

    def _apply(self, activity: AgentActivity) -> None:
        """Adopt `activity` and report it if anything changed."""
        if activity == self._activity:
            return
        self._activity = activity
        self._sync()

    def _sync(self) -> None:
        """Report the root record for the current snapshot."""
        if not self._closed:
            self._reporter.report(activity_status(self._activity))
