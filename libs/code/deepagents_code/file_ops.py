"""Helpers for tracking file operations and computing diffs for display."""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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


@dataclass
class ApprovalPreview:
    """Data used to render HITL previews."""

    title: str
    details: list[str]
    diff: str | None = None
    diff_title: str | None = None
    error: str | None = None


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
    except (OSError, UnicodeDecodeError) as e:
        return None, str(e)


def _safe_read(path: Path) -> str | None:
    """Read file content, returning None on failure.

    Returns:
        File content as string, or None if reading fails.
    """
    content, reason = _read_with_reason(path)
    if content is None:
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
        return None, DiffStats(0, 0)
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
    error: str | None = None
    metrics: FileOpMetrics = field(default_factory=FileOpMetrics)
    diff: str | None = None
    diff_stats: DiffStats = field(default_factory=lambda: DiffStats(0, 0))
    """Change counts for `diff`, taken before it was truncated for display.

    The single provenance for what a `DiffMessage` shows. Deliberately not
    `metrics.lines_added`/`lines_removed`, which are session accounting and do
    not always mean diff lines: a new-file `write_file` sets `lines_added` from
    the whole file rather than from a diff.
    """
    before_content: str | None = None
    after_content: str | None = None
    read_output: str | None = None
    hitl_approved: bool = False
    before_unreadable: bool = False
    """The pre-operation content could not be read, so `before_content` is a
    stand-in empty string rather than the file's real prior state.

    Any diff computed against it is unreliable: an unchanged file looks like a
    no-op, and a changed one looks like a whole-file insertion.
    """
    after_unreadable: bool = False
    """The post-operation content could not be read back.

    Distinct from a tool-reported failure: the operation itself succeeded, only
    its result could not be displayed.
    """
    after_read_error: str | None = None
    """Why the post-operation read failed, as reported by the backend or OS.

    Carried separately from `error`, which on this path holds a fixed
    caller-facing summary (elsewhere it holds the tool's own output). Without
    this the user is told only that the content could not be read, which
    restates the problem instead of explaining it. Always set alongside
    `after_unreadable`.
    """
    change_invisible_to_line_diff: bool = False
    """The file's bytes changed, but `diff` is empty because the change lives
    entirely in line terminators — a trailing newline added or removed, a CRLF
    conversion, or any of the other boundaries `splitlines()` recognizes — which
    `splitlines()` discards.

    A real change with nothing to render, so "no changes" would be a lie and the
    tool's own output must stay visible in its place.
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
        before = (
            _safe_read(physical_path)
            if physical_path and physical_path.exists()
            else ""
        )
        after = content
        diff, stats = compute_unified_diff(
            before or "", after, display_path, max_lines=100
        )
        additions = stats.additions
        total_lines = _count_lines(after)
        details = [
            f"File: {path_str}",
            "Action: Create new file"
            + (" (overwrites existing content)" if before else ""),
            f"Lines to write: {additions or total_lines}",
        ]
        return ApprovalPreview(
            title=f"Write {display_path}",
            details=details,
            diff=diff,
            diff_title=f"Diff {display_path}",
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
        if before is not None:
            diff, _ = compute_unified_diff(before, "", display_path, max_lines=100)
            details.append(f"Lines to delete: {_count_lines(before)}")
        elif physical_path.exists():
            details.append("Contents: directory or unreadable file")
        return ApprovalPreview(
            title=f"Delete {display_path}",
            details=details,
            diff=diff,
            diff_title=f"Diff {display_path}",
        )

    if tool_name == "edit_file":
        if physical_path is None:
            return ApprovalPreview(
                title=f"Update {display_path}",
                details=[f"File: {path_str}", "Action: Replace text"],
                error="Unable to resolve file path.",
            )
        before = _safe_read(physical_path)
        if before is None:
            return ApprovalPreview(
                title=f"Update {display_path}",
                details=[f"File: {path_str}", "Action: Replace text"],
                error="Unable to read current file contents.",
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
        if tool_name in {"write_file", "edit_file", "delete"}:
            if self.backend and path_str:
                try:
                    responses = self.backend.download_files([path_str])
                    if (
                        responses
                        and responses[0].content is not None
                        and responses[0].error is None
                    ):
                        record.before_content = responses[0].content.decode("utf-8")
                    else:
                        # A missing file is the normal create case only for
                        # `write_file`; for an edit or a delete that reports
                        # success it means we lost the pre-image. Anything else
                        # means the same, and a response carrying neither
                        # content nor an error violates the backend contract,
                        # so it never counts as an absent file either. Every
                        # such case leaves a diff that cannot be trusted.
                        if not responses:
                            error = "no response"
                        else:
                            error = (
                                responses[0].error or "no content and no error reported"
                            )
                        if error != FILE_NOT_FOUND or tool_name != "write_file":
                            logger.warning(
                                "Could not read pre-edit content for %s: %s",
                                path_str,
                                error,
                            )
                            record.before_unreadable = True
                        record.before_content = ""
                except (OSError, UnicodeDecodeError, AttributeError) as e:
                    # `AttributeError` covers a backend returning a malformed
                    # response. That is a contract bug, but this runs unguarded
                    # on the turn loop, so log it loudly rather than let it
                    # abort the turn.
                    logger.warning(
                        "Could not read pre-edit content for %s: %s", path_str, e
                    )
                    record.before_unreadable = True
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
                    # replaced since `start_operation`). Gating this on
                    # `exists()` alone let those render as a confident whole-file
                    # insertion — and, because the row then qualifies to be
                    # superseded, as the *only* account of the edit.
                    logger.warning(
                        "Could not read pre-edit content for %s: %s",
                        record.physical_path,
                        reason or "file not found",
                    )
                    record.before_unreadable = True
                record.before_content = content or ""
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
                    record.after_unreadable = True
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
            record.diff_stats = stats
            if diff:
                record.metrics.lines_added = stats.additions
                record.metrics.lines_removed = stats.deletions
            elif record.tool_name == "write_file" and not (record.before_content or ""):
                record.metrics.lines_added = record.metrics.lines_written
            record.metrics.bytes_written = len(record.after_content.encode("utf-8"))
            if (
                record.diff is None
                and (record.before_content or "") != record.after_content
            ):
                # `compute_unified_diff` works on `splitlines()`, which erases
                # line terminators, so a change confined to them yields no diff
                # at all — a trailing newline added or removed, a CRLF
                # conversion, or a rewrite between any of the other boundaries
                # `splitlines()` recognizes. Recomputing cannot help; the inputs
                # are identical. Flag it so no caller claims the file is
                # unchanged.
                record.change_invisible_to_line_diff = True
            if record.diff is None and before_lines != record.metrics.lines_written:
                record.metrics.lines_added = max(
                    record.metrics.lines_written - before_lines, 0
                )

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
        # Use backend if available (works for any BackendProtocol implementation)
        if self.backend:
            try:
                file_path = record.args.get("file_path") or record.args.get("path")
                if file_path:
                    responses = self.backend.download_files([file_path])
                    if (
                        responses
                        and responses[0].content is not None
                        and responses[0].error is None
                    ):
                        record.after_content = responses[0].content.decode("utf-8")
                    else:
                        # Keep the backend's reason: it is the only thing that
                        # can tell the user *why* a successful write cannot be
                        # shown, and the caller's own message is a tautology
                        # without it.
                        if not responses:
                            reason = "no response"
                        else:
                            reason = (
                                responses[0].error or "no content and no error reported"
                            )
                        logger.warning(
                            "Could not read post-edit content for %s: %s",
                            file_path,
                            reason,
                        )
                        record.after_read_error = reason
                        record.after_content = None
                else:
                    reason = "the tool call carried no file path"
                    logger.warning(
                        "Could not read post-edit content for %s: %s",
                        record.display_path,
                        reason,
                    )
                    record.after_read_error = reason
                    record.after_content = None
            except (OSError, UnicodeDecodeError, AttributeError) as e:
                logger.warning(
                    "Could not read post-edit content for %s: %s",
                    record.args.get("file_path") or record.args.get("path"),
                    e,
                )
                record.after_read_error = str(e)
                record.after_content = None
        else:
            # Fallback: direct filesystem read when no backend provided. Reports
            # its reason at warning like the backend branch above — the same
            # failures (permission denied, a binary write, a file removed by
            # another process) are just as opaque to the user here, and the
            # earlier debug-only read left nothing in the logs to work from.
            if record.physical_path is None:
                reason = "no physical path could be resolved"
                logger.warning(
                    "Could not read post-edit content for %s: %s",
                    record.display_path,
                    reason,
                )
                record.after_read_error = reason
                record.after_content = None
                return
            content, reason = _read_with_reason(record.physical_path)
            if content is None:
                logger.warning(
                    "Could not read post-edit content for %s: %s",
                    record.physical_path,
                    reason,
                )
                record.after_read_error = reason
            record.after_content = content

    def _finalize(self, record: FileOperationRecord) -> None:
        self.completed.append(record)
        self.active.pop(record.tool_call_id, None)
