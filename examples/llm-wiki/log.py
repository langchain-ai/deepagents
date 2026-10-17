"""Log-specific helpers for append-only wiki interaction timelines."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from collections.abc import Callable

_LOG_HEADER_MAX_LEN = 220
_LOG_SUMMARY_MAX_LEN = 320


def _normalize_log_text(text: str) -> str:
    """Normalize free-form log text to one line of compact whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def _truncate_log_text(text: str, *, max_len: int) -> str:
    """Clamp a normalized log text field to a deterministic length."""
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3].rstrip()}..."


def _format_log_value(value: object) -> str:
    """Format one metadata value as a compact key-value token."""
    normalized = _normalize_log_text(str(value))
    if not normalized:
        return '""'
    escaped = normalized.replace('"', "'")
    if re.search(r"[^a-zA-Z0-9_.:/+-]", escaped):
        return f'"{escaped}"'
    return escaped


def _build_log_entry(
    phase: str,
    outcome: str,
    metadata: dict[str, object] | None,
    summary: str | None,
) -> str:
    """Build one structured, parseable interaction entry."""
    now = datetime.now(UTC)
    date_text = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    detail_parts = [f"outcome={_format_log_value(outcome)}"]
    if metadata:
        for key in sorted(metadata):
            value = metadata[key]
            if value is None:
                continue
            detail_parts.append(f"{key}={_format_log_value(value)}")
    header_detail = _truncate_log_text(
        _normalize_log_text(" ".join(detail_parts)),
        max_len=_LOG_HEADER_MAX_LEN,
    )
    summary_text = _truncate_log_text(
        _normalize_log_text(summary or "No summary provided."),
        max_len=_LOG_SUMMARY_MAX_LEN,
    )
    return (
        f"\n## [{date_text}] {phase} | {header_detail}\n"
        f"- timestamp: {timestamp}\n"
        f"- summary: {summary_text}\n"
    )


def append_log_entry(
    workspace_dir: Path,
    phase: str,
    outcome: str,
    *,
    metadata: dict[str, object] | None = None,
    summary: str | None = None,
    ensure_file: Callable[[Path, str], None] | None = None,
    append_text: Callable[[Path, str], None] | None = None,
) -> None:
    """Append one structured entry to `/log.md`."""
    log_path = workspace_dir / "log.md"

    if ensure_file is None:
        if not log_path.exists():
            log_path.write_text("# Change Log\n", encoding="utf-8")
    else:
        ensure_file(log_path, "# Change Log\n")

    entry = _build_log_entry(phase, outcome, metadata, summary)
    if append_text is None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        return
    append_text(log_path, entry)


__all__ = ["append_log_entry"]
