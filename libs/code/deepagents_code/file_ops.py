"""Helpers for tracking file operations and computing diffs for display."""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from deepagents_code._constants import FILE_NOT_FOUND
from deepagents_code.diff_utils import (
    DIFF_TRUNCATION_MARKER,
    DiffStats,
    count_diff_change_lines,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

FileOpStatus = Literal["pending", "success", "error"]

DiffOutcome = Literal[
    "shown", "untrusted_before", "unreadable_after", "terminators_only"
]
"""What a completed file operation can honestly say about what it changed.

One closed set rather than a set of independent booleans: the four states are
mutually exclusive, and every consumer needs to agree on which one holds. Split
across flags, a consumer that forgets one silently reports a change as fully
displayed — which is how a delete whose pre-image was lost came to render
identically to deleting an empty file.

- `shown`: `diff` and `diff_stats` describe the change. Also the state for a
  genuinely unchanged file and for operations that never compute a diff.
- `untrusted_before`: the pre-operation content could not be read, so
  `before_content` is a stand-in empty string. Any diff against it is fiction —
  an unchanged file looks like a no-op, a changed one like a whole-file
  insertion — so `diff_stats` is `None` and the body must not be rendered.
- `unreadable_after`: the operation succeeded but its result could not be read
  back, so there is nothing to diff. `after_read_error` carries the reason.
- `terminators_only`: the bytes changed but `diff` is `None`, because the change
  lives entirely in line terminators, which `splitlines()` discards.
"""


def display_caveat(
    outcome: DiffOutcome, tool_name: str, after_read_error: str | None = None
) -> str:
    """Return what a successful file operation could not show, if anything.

    Every case here leaves the caveat as the user's only account of a change the
    transcript cannot render, so each has to name what is missing. Silence would
    read as a complete report.

    Covers all of `DiffOutcome` rather than the subset that mounts a diff: a
    `delete` whose pre-image was lost produces no diff at all, so a caveat routed
    only through the diff would leave destroying a 5,000-line file rendering
    exactly like destroying an empty one.

    Lives here rather than on any one surface because three need it: the tool
    row, the `DiffMessage` that may replace it, and `non_interactive`'s printed
    output. A caveat produced in only one place is a caveat that goes missing on
    whichever surface the user happens to be using — which is how `-p` came to
    print an unqualified path for a change it could not verify.

    Keyed on the outcome rather than on a `FileOperationRecord` so a widget
    holding only the outcome produces the identical sentence, and the wording
    cannot drift between surfaces. That holds for every outcome whose text is
    self-contained; `unreadable_after` is the exception, degrading to a generic
    reason without `after_read_error`. No current caller reaches it — that path
    leaves `record.diff` unset, so no `DiffMessage` mounts — but a future one
    passing only the outcome would render a weaker sentence than the tool row.

    Args:
        outcome: What the operation can honestly say about what it changed.
        tool_name: Name of the tool that ran, quoted into the sentence.
        after_read_error: Reason the post-operation read failed. Required in
            substance for `unreadable_after`, where it is the only part of the
            sentence that says anything actionable.

    Returns:
        A caveat to show alongside the tool's own output, or empty when the
        operation's changes are fully displayable.
    """
    match outcome:
        case "unreadable_after":
            # Gate on the outcome, not on `record.status == "error"`: that is
            # also set when the tool's own output reports a failure, where
            # claiming the operation succeeded would be a false statement.
            # Reachable for `write_file` and `edit_file` only; `delete`
            # synthesizes an empty post-image instead of reading one back, so it
            # has no read to fail.
            #
            # `record.error` is deliberately not a fallback here: on that path it
            # is the fixed string "Could not read updated file content.", so
            # using it would restate the sentence it is meant to explain.
            detail = after_read_error or "the reason was not reported"
            return (
                f"The `{tool_name}` call succeeded, but its changes "
                f"could not be displayed: {detail}"
            )
        case "untrusted_before":
            return (
                f"The `{tool_name}` call succeeded, but the file's prior "
                "contents could not be read, so what changed cannot be shown."
            )
        case "terminators_only":
            return (
                f"The `{tool_name}` call succeeded. The change is confined to "
                "line terminators, so there is no line-level diff to show."
            )
        case "shown":
            return ""
        case _:
            # `diff_outcome` is a plain `str` at runtime, so a member added to
            # `DiffOutcome` without a case here would fall off the end and
            # return `None` against this signature — silently degrading to "the
            # change was fully displayed", the one answer that is never safe to
            # guess. Type-checking still flags the missing case; this only
            # decides what happens when it ships anyway.
            logger.warning("Unhandled diff outcome %r", outcome)
            return (
                f"The `{tool_name}` call succeeded, but its changes could not "
                "be fully displayed."
            )


def record_display_caveat(record: FileOperationRecord | None) -> str:
    """Return `display_caveat` for a successful completed record.

    An untrusted pre-image describes what a successful operation could not
    verify. It is not a property of a failed tool call: showing its success
    wording after the tool error would contradict the actual result. `status`
    cannot make that distinction because a successful write or edit whose
    post-image cannot be read is also recorded as an error.

    Returns:
        The caveat for a successful record's outcome, or empty otherwise.
    """
    if record is None or not record.tool_succeeded:
        return ""
    return display_caveat(
        record.diff_outcome, record.tool_name, record.after_read_error
    )


@dataclass
class ApprovalPreview:
    """Data used to render HITL previews."""

    title: str
    details: list[str]
    diff: str | None = None
    diff_title: str | None = None
    error: str | None = None
    stats: DiffStats | None = None
    """Change counts for `diff`, taken before it was clipped for display.

    The prompt's `+N -M` header must not be recounted from `diff`: the
    `write_file` and `delete` previews are built with `max_lines=100`, so a
    recount describes the preview rather than the change, and on the delete path
    that number gates destroying the file. The `edit_file` preview passes
    `max_lines=None` and so is safe to recount — do not read that as a cap
    covering all three.
    """
    before: str | None = None
    """Full pre-change file content, aligned to `diff`'s old line numbers.

    Syntax highlighting a diff needs the whole source, lexed from line 1, so
    changed rows can be located by hunk number — fragments of the change
    cannot serve (see `compose_diff_lines`' `before` contract). Only the
    `edit_file` preview populates this, and only when the replacement
    succeeded; it is `None` for the other producers, for an unreadable or
    oversized file, and for a replacement that does not apply. It is *not*
    clipped the way a `diff` may be.
    """
    after: str | None = None
    """Full post-change file content, aligned to `diff`'s new line numbers.

    Populated on the same terms as `before`.
    """


#: Reason reported when a backend response carries neither content nor an error.
#: Named rather than inlined so the call sites that narrow `_response_content`'s
#: optional reason cannot drift from the string this module actually produces.
NO_REASON_REPORTED = "no content and no error reported"


def _response_failure_reason(responses: list[Any]) -> str:
    """Return why a backend download response carries no usable content.

    Shared by the pre- and post-operation reads so the two stay byte-identical:
    `start_operation` compares the result against `FILE_NOT_FOUND` to tell an
    absent file from a lost one, which only works while both produce the same
    string for the same failure.

    Returns:
        A human-readable reason.
    """
    if not responses:
        return "no response"
    return getattr(responses[0], "error", None) or NO_REASON_REPORTED


def _response_content(responses: list[Any]) -> tuple[str | None, str | None]:
    """Decode a backend download response, keeping the reason a read failed.

    Every attribute here is reached through `getattr` and every type checked,
    so a backend returning a malformed response produces a *reason* rather than
    an `AttributeError`. That distinction matters: routing a contract violation
    through the same handler as a real read failure makes a local bug — a
    renamed field, a `None` where a response was expected — present as a broken
    workspace, silently degrading every file operation in the session.

    Returns:
        The decoded content and `None`, or `None` and a human-readable reason.
    """
    if not responses:
        return None, "no response"
    content = getattr(responses[0], "content", None)
    if content is None or getattr(responses[0], "error", None) is not None:
        return None, _response_failure_reason(responses)
    if not isinstance(content, (bytes, bytearray)):
        return None, (
            f"backend returned {type(content).__name__} content, expected bytes"
        )
    try:
        return bytes(content).decode("utf-8"), None
    except UnicodeDecodeError as e:
        return None, str(e)


def _read_with_reason(path: Path) -> tuple[str | None, str | None]:
    """Read file content, keeping the reason a failure happened.

    The reason is the point: collapsing every failure to `None` leaves the user
    with "content could not be read", which restates the problem. A permission
    error, a directory, and a binary file are all actionable, and only the
    exception says which one it was.

    Returns:
        The content and `None`, or `None` and a human-readable reason.
    """
    try:
        return path.read_text(encoding="utf-8"), None
    except (OSError, UnicodeDecodeError, ValueError) as e:
        # `ValueError` covers a path with an embedded NUL byte — malformed tool
        # args must not take down the approval prompt that surfaces them.
        return None, str(e)


def _safe_read(path: Path) -> str | None:
    """Read file content, returning None on failure.

    Returns:
        File content as string, or None if reading fails.
    """
    content, reason = _read_with_reason(path)
    if content is None:
        if path.exists():
            # The file is there and still could not be read — a permission
            # error, a directory, a binary file. That leaves an approval prompt
            # describing a change it cannot show, so it belongs at the same
            # level as the tracker's read failures rather than in debug-only
            # logs the user will never see.
            logger.warning("Failed to read file %s: %s", path, reason)
        else:
            logger.debug("Failed to read file %s: %s", path, reason)
    return content


def _count_lines(text: str) -> int:
    """Count lines in text, treating empty strings as zero lines.

    Returns:
        Number of lines in the text.
    """
    if not text:
        return 0
    return len(text.splitlines())


def compute_unified_diff(
    before: str,
    after: str,
    display_path: str,
    *,
    max_lines: int | None = 800,
    context_lines: int = 3,
) -> tuple[str | None, DiffStats]:
    """Compute a unified diff between before and after content.

    Args:
        before: Original content
        after: New content
        display_path: Path for display in diff headers
        max_lines: Maximum number of diff lines (None for unlimited)
        context_lines: Number of context lines around changes (default 3)

    Returns:
        The unified diff (None if no changes), and the change counts. The counts
        are taken before any truncation, so they stay true for a clipped body.
    """
    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"{display_path} (before)",
            tofile=f"{display_path} (after)",
            lineterm="",
            n=context_lines,
        )
    )
    if not diff_lines:
        return None, DiffStats(additions=0, deletions=0)
    stats = count_diff_change_lines(diff_lines)
    if max_lines is not None and len(diff_lines) > max_lines:
        diff_lines = [*diff_lines[: max_lines - 1], DIFF_TRUNCATION_MARKER]
    return "\n".join(diff_lines), stats


@dataclass
class FileOpMetrics:
    """Line and byte level metrics for a file operation."""

    lines_read: int = 0
    start_line: int | None = None
    end_line: int | None = None
    lines_written: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    bytes_written: int = 0


@dataclass
class FileOperationRecord:
    """Track a single filesystem tool call."""

    tool_name: str
    display_path: str
    physical_path: Path | None
    tool_call_id: str | None
    args: dict[str, Any] = field(default_factory=dict)
    status: FileOpStatus = "pending"
    tool_succeeded: bool = False
    """Whether the tool itself reported success.

    This is separate from `status`, which also becomes `error` when a
    successful write or edit cannot be read back for diffing.
    """
    error: str | None = None
    metrics: FileOpMetrics = field(default_factory=FileOpMetrics)
    diff: str | None = None
    diff_stats: DiffStats | None = None
    """Change counts for `diff`, taken before it was truncated for display.

    The single provenance for what a `DiffMessage` shows. `None` is the only
    way this says "unknown" — never `DiffStats(0, 0)`, which means a verified
    zero. Unset until a diff is computed, and left unset for both outcomes that
    have nothing to count: `untrusted_before`, where any count would be fiction,
    and `unreadable_after`, which returns before a diff is ever computed.

    Deliberately not `metrics.lines_added`/`lines_removed`, which are session
    accounting and do not always mean diff lines: a new-file `write_file` sets
    `lines_added` from the whole file rather than from a diff.
    """
    before_content: str | None = None
    after_content: str | None = None
    read_output: str | None = None
    hitl_approved: bool = False
    diff_outcome: DiffOutcome = "shown"
    """What this operation can honestly say about what it changed.

    See `DiffOutcome`. Consumers should branch on this rather than infer the
    state from `status`, `diff`, or `diff_stats`.
    """
    after_read_error: str | None = None
    """Why the post-operation read failed, as reported by the backend or OS.

    The payload for `diff_outcome == "unreadable_after"`, and set only with it.
    Carried separately from `error`, which on that path holds a fixed
    caller-facing summary (elsewhere it holds the tool's own output). Without
    this the user is told only that the content could not be read, which
    restates the problem instead of explaining it.
    """


def resolve_physical_path(
    path_str: str | None, assistant_id: str | None
) -> Path | None:
    """Convert a virtual/relative path to a physical filesystem path.

    Returns:
        Resolved physical Path, or None if path is empty or resolution fails.
    """
    if not path_str:
        return None
    try:
        if assistant_id and path_str.startswith("/memories/"):
            from deepagents_code.config import settings

            agent_dir = settings.get_agent_dir(assistant_id)
            suffix = path_str.removeprefix("/memories/").lstrip("/")
            return (agent_dir / suffix).resolve()
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()
    except (OSError, ValueError):
        # The exception is the only thing distinguishing an embedded NUL byte
        # from a name too long from a failing agent-dir lookup. `None` now
        # decides which pre-image branch runs and reaches the user as "no
        # physical path could be resolved", which explains nothing on its own.
        logger.warning(
            "Could not resolve physical path for %s", path_str, exc_info=True
        )
        return None


_SENSITIVE_FILE_NAMES = frozenset(
    {
        ".envrc",
        ".netrc",
        "_netrc",
        ".pgpass",
        ".npmrc",
        ".pypirc",
        ".htpasswd",
        ".git-credentials",
        "credentials",
        "credentials.json",
        "token.json",
        "auth.json",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
)
"""Basenames (lowercased) that commonly hold secrets and must not be rendered."""

_SENSITIVE_FILE_SUFFIXES = (
    ".pem",
    ".key",
    ".pfx",
    ".p12",
    ".keystore",
    ".jks",
)
"""File suffixes (lowercased) for private keys / keystores that hold secrets."""


def is_sensitive_file_path(path_str: str | None) -> bool:
    """Return whether a path points at a credential/secret file.

    Best-effort, filename-based, case-insensitive heuristic. It matches `.env`
    and its variants (e.g. `.env.local`), well-known credential filenames, and
    private-key/keystore suffixes, and is used to suppress diff/content
    rendering for those files so their contents are not shown in the terminal
    UI or scrollback. It classifies by name only, not content, so
    secret-bearing files with unrecognized names still render.

    Args:
        path_str: Filesystem path to classify (a display or absolute path).
            May be `None` or empty.

    Returns:
        `True` if the basename matches a known credential pattern. A falsy
        path returns `False` (nothing to classify). An unparseable path
        returns `True` and logs a warning, so the redaction gate fails
        closed on unexpected input rather than leaking.
    """
    if not path_str:
        return False
    try:
        name = Path(path_str).name.lower()
    except (OSError, ValueError, TypeError):
        logger.warning(
            "is_sensitive_file_path: could not parse %r; treating as sensitive",
            path_str,
        )
        return True
    if not name:
        return False
    if name == ".env" or name.startswith(".env."):
        return True
    if name in _SENSITIVE_FILE_NAMES:
        return True
    return name.endswith(_SENSITIVE_FILE_SUFFIXES)


def format_display_path(path_str: str | None) -> str:
    """Format a path for display.

    Returns:
        Formatted path string suitable for display.
    """
    if not path_str:
        return "(unknown)"
    try:
        path = Path(path_str)
        if path.is_absolute():
            return path.name or str(path)
        return str(path)
    except (OSError, ValueError):
        return str(path_str)


_EDIT_PREVIEW_MAX_LINES: Final = 100_000
"""Largest file, in lines, the `edit_file` approval preview reads and diffs.

`difflib` cost scales with line count, not bytes: it on this scale is tens of
milliseconds — comfortable on the Textual message pump, where the renderer
invokes the preview. Beyond it, whole-file diff time climbs super-linearly
and the prompt reads as hung.
"""


def build_approval_preview(
    tool_name: str,
    args: dict[str, Any],
    assistant_id: str | None,
) -> ApprovalPreview | None:
    """Collect summary info and diff for HITL approvals.

    Returns:
        ApprovalPreview with diff and details, or None if tool not supported.
    """
    path_str = str(args.get("file_path") or args.get("path") or "")
    display_path = format_display_path(path_str)
    physical_path = resolve_physical_path(path_str, assistant_id)

    if tool_name == "write_file":
        content = str(args.get("content", ""))
        existing = (
            _safe_read(physical_path)
            if physical_path and physical_path.exists()
            else ""
        )
        # `None` is a file that is there but could not be read, which is not the
        # same as no file at all: collapsing the two would drop the overwrite
        # warning and render the whole payload as a pure insertion, so the
        # prompt would describe creating a file when it is about to destroy one.
        before = existing or ""
        after = content
        diff, stats = compute_unified_diff(before, after, display_path, max_lines=100)
        additions = stats.additions
        total_lines = _count_lines(after)
        if existing is None:
            action_suffix = (
                " (existing contents could not be read — this may overwrite them)"
            )
        elif existing:
            action_suffix = " (overwrites existing content)"
        else:
            action_suffix = ""
        details = [
            f"File: {path_str}",
            "Action: Create new file" + action_suffix,
            f"Lines to write: {additions or total_lines}",
        ]
        return ApprovalPreview(
            title=f"Write {display_path}",
            details=details,
            diff=diff,
            diff_title=f"Diff {display_path}",
            stats=stats,
        )

    if tool_name == "delete":
        details = [f"File: {path_str}", "Action: Delete file or directory"]
        if physical_path is None:
            return ApprovalPreview(
                title=f"Delete {display_path}",
                details=details,
                error="Unable to resolve file path.",
            )
        before = _safe_read(physical_path)
        diff = None
        stats = None
        if before is not None:
            # The preview is clipped at 100 lines; `stats` is not. Carrying it
            # is what keeps the prompt's `-N` describing the file rather than
            # the excerpt — a 5,000-line delete otherwise asks the user to
            # approve what reads as a 96-line one.
            diff, stats = compute_unified_diff(before, "", display_path, max_lines=100)
            details.append(f"Lines to delete: {_count_lines(before)}")
        elif physical_path.exists():
            details.append("Contents: directory or unreadable file")
        else:
            # The pre-image is read straight from the local filesystem, so a
            # session whose files live on a backend (sandbox, store, LangSmith)
            # resolves to a path that is not there and lands here. Without this
            # the prompt carries no diff, no counts, and no explanation — the
            # user is asked to approve destroying a 5,000-line file on a screen
            # indistinguishable from destroying an empty one.
            details.append(
                "Contents: could not be read — this prompt cannot show what "
                "will be deleted"
            )
        return ApprovalPreview(
            title=f"Delete {display_path}",
            details=details,
            diff=diff,
            diff_title=f"Diff {display_path}",
            stats=stats,
        )

    if tool_name == "edit_file":
        if physical_path is None:
            # Not an `error` — same contract as the unreadable case below:
            # the edit may still apply, so the renderer falls back to the
            # fragment diff.
            return ApprovalPreview(
                title=f"Update {display_path}",
                details=[f"File: {path_str}", "Action: Replace text"],
            )
        before = _safe_read(physical_path)
        if before is None:
            # Not an `error`: unlike a replacement that cannot apply, an
            # unreadable file does not mean the edit is wrong — the renderer
            # falls back to diffing the replacement fragments, which is also
            # the only branch available to sandbox-backed sessions whose
            # files are not on the local filesystem at all.
            return ApprovalPreview(
                title=f"Update {display_path}",
                details=[f"File: {path_str}", "Action: Replace text"],
            )
        # Diffing runs on the Textual message pump (the renderer calls this
        # synchronously), so an unbounded `difflib` over a huge file would
        # freeze the approval prompt. Past the cap, return diffless and let
        # the renderer fall back to diffing the replacement fragments.
        # Credential files are exempt: their diff never renders, and the
        # fragment fallback would carry the file's secrets into widget data
        # where only the `diff_lines` branch is redacted.
        if before.count("\n") > _EDIT_PREVIEW_MAX_LINES and not is_sensitive_file_path(
            path_str
        ):
            return ApprovalPreview(
                title=f"Update {display_path}",
                details=[f"File: {path_str}", "Action: Replace text"],
            )
        old_string = str(args.get("old_string", ""))
        new_string = str(args.get("new_string", ""))
        replace_all = bool(args.get("replace_all"))
        from deepagents.backends.utils import perform_string_replacement

        replacement = perform_string_replacement(
            before, old_string, new_string, replace_all
        )
        if isinstance(replacement, str):
            return ApprovalPreview(
                title=f"Update {display_path}",
                details=[f"File: {path_str}", "Action: Replace text"],
                error=replacement,
            )
        after, occurrences = replacement
        diff, stats = compute_unified_diff(before, after, display_path, max_lines=None)
        action = "all occurrences" if replace_all else "single occurrence"
        details = [
            f"File: {path_str}",
            f"Action: Replace text ({action})",
            f"Occurrences matched: {occurrences}",
            f"Lines changed: +{stats.additions} / -{stats.deletions}",
        ]
        return ApprovalPreview(
            title=f"Update {display_path}",
            details=details,
            diff=diff,
            diff_title=f"Diff {display_path}",
            stats=stats,
            before=before,
            after=after,
        )

    return None


class FileOpTracker:
    """Collect file operation metrics during an interaction."""

    def __init__(
        self, *, assistant_id: str | None, backend: BackendProtocol | None = None
    ) -> None:
        """Initialize the tracker."""
        self.assistant_id = assistant_id
        self.backend = backend
        self.active: dict[str | None, FileOperationRecord] = {}
        self.completed: list[FileOperationRecord] = []

    def start_operation(
        self, tool_name: str, args: dict[str, Any], tool_call_id: str | None
    ) -> None:
        """Begin tracking a file operation.

        Creates a record for the operation and, for write/edit/delete
        operations, captures the file's content before the operation.
        """
        if tool_name not in {"read_file", "write_file", "edit_file", "delete"}:
            return
        path_str = str(args.get("file_path") or args.get("path") or "")
        display_path = format_display_path(path_str)
        record = FileOperationRecord(
            tool_name=tool_name,
            display_path=display_path,
            physical_path=resolve_physical_path(path_str, self.assistant_id),
            tool_call_id=tool_call_id,
            args=args,
        )

        def lost_pre_image(target: object, reason: str) -> None:
            """Record that the file's prior state could not be captured."""
            logger.warning("Could not read pre-edit content for %s: %s", target, reason)
            record.diff_outcome = "untrusted_before"

        if tool_name in {"write_file", "edit_file", "delete"}:
            if self.backend and path_str:
                try:
                    content, error = _response_content(
                        self.backend.download_files([path_str])
                    )
                    if content is not None:
                        record.before_content = content
                    else:
                        # A missing file is the normal create case only for
                        # `write_file`; for an edit or a delete that reports
                        # success it means we lost the pre-image. Anything else
                        # means the same, and a response carrying neither
                        # content nor an error violates the backend contract,
                        # so it never counts as an absent file either. Every
                        # such case leaves a diff that cannot be trusted.
                        # `_response_content` returns a reason on every failure
                        # path, so the fallback only narrows the optional away.
                        reason = error or NO_REASON_REPORTED
                        if reason != FILE_NOT_FOUND or tool_name != "write_file":
                            lost_pre_image(path_str, reason)
                        record.before_content = ""
                except OSError as e:
                    lost_pre_image(path_str, str(e))
                    record.before_content = ""
                except AttributeError as e:
                    # The backend object itself does not satisfy the protocol —
                    # a contract bug, not a file that could not be read. Kept
                    # out of the handler above so the two are distinguishable
                    # in the logs: routing both through one message makes a
                    # local defect present as a broken workspace, degrading
                    # every operation in the session with nothing to say which
                    # it was. Still caught, because `start_operation` runs
                    # unguarded on the turn loop and raising would abort the
                    # turn over a cosmetic read.
                    logger.exception("Backend violated the download contract")
                    lost_pre_image(path_str, f"backend contract violation: {e}")
                    record.before_content = ""
                except Exception as e:
                    # `OSError` does not cover what real backends raise: the
                    # store backend decodes base64 (`binascii.Error`, a
                    # `ValueError`) and the LangSmith backend catches only its
                    # own two error types, letting transport failures through.
                    # Narrower handling would let a transient sandbox blip abort
                    # the turn and drop every remaining tool's hooks — the exact
                    # outcome the handlers above exist to prevent.
                    logger.exception("Failed to read pre-edit content for %s", path_str)
                    lost_pre_image(path_str, str(e) or type(e).__name__)
                    record.before_content = ""
            elif record.physical_path:
                content, reason = _read_with_reason(record.physical_path)
                if content is None and (
                    tool_name != "write_file" or record.physical_path.exists()
                ):
                    # Same rule as the backend branch above: absence is the
                    # normal create case only for `write_file`. For an edit or a
                    # delete it means we lost the pre-image, whether the read
                    # raised or the path is simply not there (a broken symlink, a
                    # physical path that diverged from the backend's, a file
                    # replaced between the model emitting the call and this
                    # read). `exists()` alone is not sufficient: a read that
                    # raised must count as a lost pre-image too, or the diff
                    # renders as a confident whole-file insertion and, being
                    # eligible to supersede the row, becomes the only account of
                    # the edit.
                    lost_pre_image(record.physical_path, reason or "file not found")
                record.before_content = content or ""
            else:
                # No backend and no resolvable physical path: nothing was read,
                # so nothing about the prior state is known. Without this the
                # outcome stays at its `shown` default — the one answer that is
                # never safe to guess. `write_file` and `edit_file` are caught
                # downstream, where `_populate_after_content` hits the same
                # missing path and flips to `unreadable_after`; `delete` is not,
                # because it synthesizes an empty post-image instead of reading
                # one back, and empty-against-empty yields no diff. That path
                # would otherwise finish as a verified `+0 -0` about a file
                # whose contents were never seen.
                lost_pre_image(
                    record.display_path,
                    "no backend and no resolvable physical path",
                )
                record.before_content = ""
        self.active[tool_call_id] = record

    def complete_with_message(self, tool_message: Any) -> FileOperationRecord | None:  # noqa: ANN401  # Tool message type is dynamic
        """Complete a file operation with the tool message result.

        Returns:
            The completed FileOperationRecord, or None if no matching operation.
        """
        tool_call_id = getattr(tool_message, "tool_call_id", None)
        record = self.active.get(tool_call_id)
        if record is None:
            return None

        content = tool_message.content
        if isinstance(content, list):
            # Some tool messages may return list segments; join them for analysis.
            joined = []
            for item in content:
                if isinstance(item, str):
                    joined.append(item)
                else:
                    joined.append(str(item))
            content_text = "\n".join(joined)
        else:
            content_text = str(content) if content is not None else ""

        if getattr(
            tool_message, "status", "success"
        ) != "success" or content_text.lower().startswith("error"):
            record.status = "error"
            record.error = content_text
            self._finalize(record)
            return record

        record.status = "success"
        record.tool_succeeded = True

        if record.tool_name == "read_file":
            record.read_output = content_text
            lines = _count_lines(content_text)
            record.metrics.lines_read = lines
            offset = record.args.get("offset")
            limit = record.args.get("limit")
            if isinstance(offset, int):
                if offset > lines:
                    offset = 0
                record.metrics.start_line = offset + 1
                if lines:
                    record.metrics.end_line = offset + lines
            elif lines:
                record.metrics.start_line = 1
                record.metrics.end_line = lines
            if isinstance(limit, int) and lines > limit:
                record.metrics.end_line = (record.metrics.start_line or 1) + limit - 1
        else:
            if record.tool_name == "delete":
                # Reached only after the success-status check above, so the
                # tool reported the path removed. Model an empty "after" to
                # diff the removed content against; there is nothing to read
                # back from disk. This trusts the tool's success status and
                # is sound for backends where a successful delete means the
                # path is gone.
                record.after_content = ""
            else:
                # Write/edit: read the updated content back from backend/disk.
                self._populate_after_content(record)
                if record.after_content is None:
                    record.status = "error"
                    # Deliberately outranks a `untrusted_before` set by the
                    # pre-image read: when both reads fail there is one caveat
                    # to spend, and "the result could not be read back" is the
                    # one the user can act on — it says the file's current state
                    # is unverified, which subsumes not knowing what it was.
                    record.diff_outcome = "unreadable_after"
                    record.error = "Could not read updated file content."
                    # Record what the *request* knows before bailing. The write
                    # itself succeeded, so reporting zero lines and zero bytes to
                    # session accounting would understate real work with a
                    # plausible-looking number. Only `write_file` carries its
                    # full result in its args; an edit's does not, so its metrics
                    # stay unknown rather than guessed.
                    written = record.args.get("content")
                    if record.tool_name == "write_file" and isinstance(written, str):
                        record.metrics.lines_written = _count_lines(written)
                        record.metrics.bytes_written = len(written.encode("utf-8"))
                    self._finalize(record)
                    return record
            record.metrics.lines_written = _count_lines(record.after_content)
            before_lines = _count_lines(record.before_content or "")
            diff, stats = compute_unified_diff(
                record.before_content or "",
                record.after_content,
                record.display_path,
                max_lines=100,
            )
            record.diff = diff
            # Skipped entirely for a lost pre-image: `before_content` is a
            # stand-in empty string, so these counts describe a whole-file
            # insertion that never happened. Leaving `diff_stats` unset is what
            # makes "unknown" reach the header, and writing them into `metrics`
            # would feed the same fiction to session accounting, where nothing
            # marks it as unreliable.
            if record.diff_outcome != "untrusted_before":
                record.diff_stats = stats
                if diff:
                    record.metrics.lines_added = stats.additions
                    record.metrics.lines_removed = stats.deletions
                elif record.tool_name == "write_file" and not (
                    record.before_content or ""
                ):
                    record.metrics.lines_added = record.metrics.lines_written
                elif diff is None and before_lines != record.metrics.lines_written:
                    # Same guard, same reason: `before_lines` is counted from
                    # `before_content`, which under `untrusted_before` is the
                    # stand-in empty string. Outside the guard this would book a
                    # whole-file `lines_added` for an edit whose pre-image was
                    # lost. That is a no-op today only because a lost pre-image
                    # with no diff implies both sides are empty — a coincidence,
                    # not an invariant.
                    record.metrics.lines_added = max(
                        record.metrics.lines_written - before_lines, 0
                    )
            record.metrics.bytes_written = len(record.after_content.encode("utf-8"))
            if (
                record.diff_outcome == "shown"
                and record.diff is None
                and (record.before_content or "") != record.after_content
            ):
                # `compute_unified_diff` works on `splitlines()`, which erases
                # line terminators, so a change confined to them yields no diff
                # at all — a trailing newline added or removed, a CRLF
                # conversion, or a rewrite between any of the other boundaries
                # `splitlines()` recognizes. Recomputing cannot help; the inputs
                # are identical. Flag it so no caller claims the file is
                # unchanged.
                #
                # Gated on `shown` so a lost pre-image keeps the stronger
                # `untrusted_before`, which says the change cannot be shown at
                # all rather than merely that it has no line diff.
                record.diff_outcome = "terminators_only"

        self._finalize(record)
        return record

    def mark_hitl_approved(self, tool_name: str, args: dict[str, Any]) -> None:
        """Mark operations matching tool_name and file_path as HIL-approved."""
        file_path = args.get("file_path") or args.get("path")
        if not file_path:
            return

        # Mark all active records that match
        for record in self.active.values():
            if record.tool_name == tool_name:
                record_path = record.args.get("file_path") or record.args.get("path")
                if record_path == file_path:
                    record.hitl_approved = True

    def _populate_after_content(self, record: FileOperationRecord) -> None:
        def unreadable(target: object, reason: str) -> None:
            """Record that the operation's result could not be read back.

            One helper so the reason is never logged without also reaching the
            record: `after_read_error` is the only thing that can tell the user
            *why* a successful write cannot be shown, and the caller's own
            message is a tautology without it.
            """
            logger.warning(
                "Could not read post-edit content for %s: %s", target, reason
            )
            record.after_read_error = reason
            record.after_content = None

        # Use backend if available (works for any BackendProtocol implementation)
        if self.backend:
            file_path = record.args.get("file_path") or record.args.get("path")
            try:
                if not file_path:
                    unreadable(
                        record.display_path, "the tool call carried no file path"
                    )
                    return
                content, reason = _response_content(
                    self.backend.download_files([file_path])
                )
                if content is not None:
                    record.after_content = content
                else:
                    # As in the pre-image read: the fallback only narrows the
                    # optional, it is not a reachable message.
                    unreadable(file_path, reason or NO_REASON_REPORTED)
            except OSError as e:
                unreadable(file_path, str(e))
            except AttributeError as e:
                # Same split as the pre-image read: a backend that does not
                # satisfy the protocol is a contract bug, reported as one rather
                # than as an unreadable file, but still not allowed to abort the
                # turn.
                logger.exception("Backend violated the download contract")
                unreadable(file_path, f"backend contract violation: {e}")
            except Exception as e:
                # Same reasoning as the pre-image read: backends raise well
                # outside `OSError`, and a read-back failure must degrade to an
                # unreadable result rather than abort the turn.
                logger.exception("Failed to read post-edit content for %s", file_path)
                unreadable(file_path, str(e) or type(e).__name__)
        else:
            # Fallback: direct filesystem read when no backend provided. Reports
            # its reason at warning like the backend branch above — the same
            # failures (permission denied, a binary write, a file removed by
            # another process) are just as opaque to the user here, and the
            # earlier debug-only read left nothing in the logs to work from.
            if record.physical_path is None:
                unreadable(record.display_path, "no physical path could be resolved")
                return
            content, reason = _read_with_reason(record.physical_path)
            if content is None:
                unreadable(record.physical_path, reason or "file not found")
            else:
                record.after_content = content

    def _finalize(self, record: FileOperationRecord) -> None:
        self._enforce_outcome_invariants(record)
        self.completed.append(record)
        self.active.pop(record.tool_call_id, None)

    @staticmethod
    def _enforce_outcome_invariants(record: FileOperationRecord) -> None:
        """Correct payloads that contradict the record's `diff_outcome`.

        `diff_outcome` and the fields it speaks for (`diff_stats`,
        `after_read_error`) are set from three separate branches, so every
        "outcome implies payload" rule in `DiffOutcome` holds by convention
        only. This is the single funnel every completed record passes through,
        which makes it the one place the rules can be checked at all.

        Corrects rather than raises, matching this module's posture that a
        display defect must not abort the turn — but logs, because reaching
        here means a producer wrote a combination it documents as impossible.
        """
        if record.diff_outcome == "untrusted_before" and record.diff_stats is not None:
            # Counts taken against a stand-in empty pre-image describe a
            # whole-file insertion that never happened. The body is suppressed
            # for this outcome, but `diff_stats` is read directly by the
            # adapter and the message store, so a stale pair would still put
            # fabricated `+N` on screen.
            logger.error(
                "Record for %s carries diff_stats under an untrusted pre-image; "
                "dropping them",
                record.display_path,
            )
            record.diff_stats = None
        if (record.after_read_error is not None) != (
            record.diff_outcome == "unreadable_after"
        ):
            logger.warning(
                "Record for %s pairs after_read_error=%r with outcome %r",
                record.display_path,
                record.after_read_error,
                record.diff_outcome,
            )
