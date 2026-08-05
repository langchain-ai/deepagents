"""Shared unified-diff helpers."""

from __future__ import annotations

import re
from typing import NamedTuple

HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?")
"""Matches a hunk header.

Captures, in order: old start line, old line count, new start line, new line
count. Either count is absent for a single-line range, where it defaults to 1.
"""


DIFF_TRUNCATION_MARKER = "..."
"""Stand-in line marking where a diff body was clipped for display.

Written by `compute_unified_diff`, rendered as a `truncated` row, and the signal
that counts taken from the body would be short.
"""


class DiffStats(NamedTuple):
    """Line counts for a change, named so the pair cannot be swapped silently."""

    additions: int
    deletions: int


def split_diff_lines(diff: str) -> list[str]:
    r"""Split a unified diff back into the lines it was assembled from.

    Deliberately not `splitlines()`. Producers join their lines with `"\n"`
    (`compute_unified_diff`, `EditFileRenderer._generate_diff`), so `"\n"` is
    the exact inverse; `splitlines()` also breaks on `\r`, `\v`, `\f`, U+2028,
    U+2029 and U+0085, which splits a single diff line into fragments. The tail
    fragment carries no `+`/`-` marker, so it would render as an unmarked note —
    on the approval prompt that means changed content shown as neutral metadata.

    Args:
        diff: Unified diff string.

    Returns:
        The diff's lines, without a trailing empty entry for a terminating
        newline.
    """
    lines = diff.split("\n")
    if lines and not lines[-1]:
        lines.pop()
    return lines


def file_header_indexes(lines: list[str]) -> set[int]:
    """Locate paired file headers immediately preceding a hunk.

    A `---`/`+++` pair is only a file header when it appears *outside* a hunk
    body — a diff of a file that itself contains such lines would otherwise
    have its content mistaken for metadata. That is why this walks the hunks'
    declared old/new line budgets instead of just matching on the prefix.

    Args:
        lines: Unified-diff lines. Handles multi-file diffs, where headers
            recur between hunks.

    Returns:
        Indexes of file-header lines.
    """
    indexes: set[int] = set()
    old_remaining = new_remaining = 0
    inside_hunk = False
    for index, line in enumerate(lines):
        if match := HUNK_RE.match(line):
            old_remaining = int(match.group(2) or 1)
            new_remaining = int(match.group(4) or 1)
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
            and HUNK_RE.match(lines[index + 2])
        ):
            indexes.update((index, index + 1))
    return indexes


def count_diff_change_lines(lines: list[str]) -> DiffStats:
    """Count added and removed lines in unified-diff lines.

    Args:
        lines: Unified-diff lines.

    Returns:
        Additions and deletions, excluding file headers.
    """
    headers = file_header_indexes(lines)
    additions = deletions = 0
    for index, line in enumerate(lines):
        if index in headers:
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return DiffStats(additions, deletions)


def count_diff_changes(diff: str) -> DiffStats:
    """Count added and removed lines in a unified diff.

    Args:
        diff: Unified diff string.

    Returns:
        Additions and deletions, excluding file headers.
    """
    return count_diff_change_lines(split_diff_lines(diff))
