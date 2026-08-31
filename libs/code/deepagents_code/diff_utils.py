r"""Shared unified-diff helpers.

Every diff passing through this module is `"\n"`-joined from lines that came
from `splitlines()` or `split("\n")`, so no element can contain a line boundary.
That is what makes `split_diff_lines` the exact inverse and `splitlines()` wrong
here — see its docstring for what breaks. Check any helper added to this module,
and any new producer of a diff it reads, against that invariant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

HUNK_RE: Final[re.Pattern[str]] = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))?")
"""Matches a hunk header.

Captures, in order: old start line, old line count, new start line, new line
count. Either count is absent for a single-line range, where it defaults to 1.
"""


DIFF_TRUNCATION_MARKER: Final[str] = "..."
"""Stand-in line marking where a diff body was clipped for display.

Written by `compute_unified_diff` and rendered as a `truncated` row. It is also
the signal that any counts recomputed from the body would be short — see
`DiffMessage._recount`, which returns `None` rather than a known-low number.

Match it with `is_truncation_marker` rather than by hand: the renderer and the
recount must agree, or a body that renders "diff truncated" is also counted as
if it were complete.
"""


def is_truncation_marker(line: str) -> bool:
    """Return whether a diff line is the truncation marker.

    One predicate for both readers, which previously disagreed: the renderer
    stripped, the recount compared exactly.

    Exact, deliberately. `compute_unified_diff` writes the marker bare, while
    every real diff line carries a `+`, `-`, or space prefix — so an exact match
    cannot collide with file content, and a source line whose own text is `...`
    arrives here as `" ..."` and stays a context row. Stripping would classify
    that line as a clipped body and suppress the change counts for a diff that
    is complete.

    Args:
        line: A single unified-diff line.

    Returns:
        Whether the line marks a clipped body.
    """
    return line == DIFF_TRUNCATION_MARKER


@dataclass(frozen=True, kw_only=True)
class DiffStats:
    """Line counts for a change, named so the pair cannot be swapped silently.

    Keyword-only and frozen so that claim holds at construction as well as in
    transit: a positional pair is exactly the transposition this type exists to
    rule out, and the counts are read long after they are computed.

    Attributes:
        additions: Added change lines, excluding file headers, counted before
            any truncation of the body for display.
        deletions: Removed change lines, on the same terms.
    """

    additions: int
    deletions: int

    def __post_init__(self) -> None:
        """Reject negative counts.

        Both in-repo producers derive from `difflib`, so this only guards direct
        construction — but the type is public and reaches a delete prompt, where
        the number gates destroying a file.

        Raises:
            ValueError: If either count is negative.
        """
        if self.additions < 0 or self.deletions < 0:
            msg = (
                f"DiffStats counts cannot be negative, got additions="
                f"{self.additions}, deletions={self.deletions}"
            )
            raise ValueError(msg)


def split_diff_lines(diff: str) -> list[str]:
    r"""Split a unified diff back into the lines it was assembled from.

    Deliberately not `splitlines()`. Every diff reaching this function is
    `"\n"`-joined from lines that themselves came from `splitlines()` or
    `split("\n")`, so no element can contain a line boundary and `"\n"` is the
    exact inverse. `splitlines()` also breaks on `\r`, `\v`, `\f`, U+2028,
    U+2029 and U+0085, which splits a single diff line into fragments. The tail
    fragment carries no `+`/`-` marker, so it would render as an unmarked note —
    on the approval prompt that means changed content shown as neutral metadata.

    Check any new producer against that invariant rather than against a list of
    the current ones.

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
            inside_hunk = old_remaining > 0 or new_remaining > 0
            continue
        if inside_hunk:
            if _opens_file_header(lines, index):
                # The budget says this hunk is still running, but a full
                # `---`/`+++`/`@@` sequence starts here — so the budget
                # over-declared and the next file has begun. Without this the
                # `--- a/y.py` is consumed as a deletion, the following headers
                # render as source rows, and the change counts include them.
                # The three-line shape is what makes this safe: a removed line
                # can read `--- something`, but not while the two lines after
                # it also form a header pair and a hunk header.
                inside_hunk = False
            elif line.startswith("\\"):
                # "\ No newline at end of file" annotates the line before it and
                # belongs to neither budget.
                continue
            elif line.startswith("-"):
                old_remaining -= 1
            elif line.startswith("+"):
                new_remaining -= 1
            elif line.startswith(" "):
                old_remaining -= 1
                new_remaining -= 1
            else:
                # Not a hunk body line: the declared budget over-counts, or the
                # producer is not `difflib`. End the hunk here and re-read this
                # line as metadata below. Staying inside would consume the rest
                # of the diff — including a following file's `---`/`+++` pair —
                # as body, which counts those headers as a change and renders
                # them as source rows.
                inside_hunk = False
            if inside_hunk:
                # Only lines that consumed budget skip the header check. A
                # removed line may legitimately read `--- something`, and
                # treating it as metadata is the misreading this walk prevents.
                inside_hunk = old_remaining > 0 or new_remaining > 0
                continue
        if _opens_file_header(lines, index):
            indexes.update((index, index + 1))
    return indexes


def _opens_file_header(lines: list[str], index: int) -> bool:
    """Whether a file-header pair immediately preceding a hunk starts here.

    Args:
        lines: Unified-diff lines.
        index: Position of the candidate `---` line.

    Returns:
        Whether `lines[index:index + 3]` is `---`, `+++`, and a hunk header.
    """
    return (
        index + 2 < len(lines)
        and lines[index].startswith("--- ")
        and lines[index + 1].startswith("+++ ")
        and HUNK_RE.match(lines[index + 2]) is not None
    )


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
    return DiffStats(additions=additions, deletions=deletions)
