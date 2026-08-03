"""Shared unified-diff helpers."""

from __future__ import annotations

import re

_HUNK_RE = re.compile(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))?")


def file_header_indexes(lines: list[str]) -> set[int]:
    """Locate paired file headers immediately preceding a hunk.

    Args:
        lines: Unified-diff lines.

    Returns:
        Indexes of file-header lines.
    """
    indexes: set[int] = set()
    old_remaining = new_remaining = 0
    inside_hunk = False
    for index, line in enumerate(lines):
        if match := _HUNK_RE.match(line):
            old_remaining = int(match.group(1) or 1)
            new_remaining = int(match.group(2) or 1)
            inside_hunk = bool(old_remaining or new_remaining)
            continue
        if inside_hunk:
            if line.startswith("-"):
                old_remaining -= 1
            elif line.startswith("+"):
                new_remaining -= 1
            elif line.startswith(" "):
                old_remaining -= 1
                new_remaining -= 1
            inside_hunk = old_remaining > 0 or new_remaining > 0
            continue
        if (
            index + 2 < len(lines)
            and line.startswith("--- ")
            and lines[index + 1].startswith("+++ ")
            and _HUNK_RE.match(lines[index + 2])
        ):
            indexes.update((index, index + 1))
    return indexes


def count_diff_changes(diff: str) -> tuple[int, int]:
    """Count added and removed lines in a unified diff.

    Args:
        diff: Unified diff string.

    Returns:
        Tuple of additions and deletions, excluding file headers.
    """
    lines = diff.splitlines()
    headers = file_header_indexes(lines)
    additions = sum(
        line.startswith("+") for index, line in enumerate(lines) if index not in headers
    )
    deletions = sum(
        line.startswith("-") for index, line in enumerate(lines) if index not in headers
    )
    return additions, deletions
