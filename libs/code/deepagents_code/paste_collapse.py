r"""Large paste collapsing for the chat input.

When the user pastes text exceeding a size or line threshold, the full text
is stored off-screen and a compact `[Pasted text #N +M lines]` placeholder
is inserted into the input box instead.  At submission time the placeholder
is expanded back to the original content so the agent receives the full text.

This mirrors the behavior of Claude Code's paste-collapsing system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PASTE_THRESHOLD_CHARS = 800
"""Minimum character count for a paste to be collapsed into a placeholder."""

PASTE_THRESHOLD_LINES = 2
"""Minimum line count (newline-separated) for a paste to be collapsed."""

_PASTE_REF_RE = re.compile(r"\[Pasted text #(\d+)(?: \+(\d+) lines)?\]")
"""Regex matching `[Pasted text #N]` or `[Pasted text #N +M lines]`."""


@dataclass(frozen=True)
class PastedContent:
    """Stored content for a collapsed paste.

    The paste's numeric identifier is the key under which this record is
    stored in the input's paste map, so it is not duplicated on the record.

    Attributes:
        content: The full pasted text.
    """

    content: str


def count_lines(text: str) -> int:
    r"""Return the number of newline characters in `text`.

    Args:
        text: The text to count newlines in.

    Returns:
        The number of `\n` occurrences (0 for single-line text).
    """
    return text.count("\n")


def should_collapse_paste(text: str) -> bool:
    """Return whether `text` should be collapsed into a placeholder.

    Collapses when the text exceeds the character threshold or the line
    threshold.

    Args:
        text: The pasted text to evaluate.

    Returns:
        `True` if the paste should be collapsed.
    """
    return (
        len(text) > PASTE_THRESHOLD_CHARS or count_lines(text) > PASTE_THRESHOLD_LINES
    )


def format_paste_ref(paste_id: int, num_lines: int) -> str:
    """Format a paste placeholder reference string.

    Args:
        paste_id: The numeric paste identifier.
        num_lines: The number of extra lines (newlines) in the pasted content.

    Returns:
        `[Pasted text #N]` when `num_lines` is 0, otherwise
        `[Pasted text #N +M lines]`.
    """
    if num_lines == 0:
        return f"[Pasted text #{paste_id}]"
    return f"[Pasted text #{paste_id} +{num_lines} lines]"


def expand_paste_refs(text: str, pasted_contents: dict[int, PastedContent]) -> str:
    """Replace all paste placeholders in `text` with their full content.

    Placeholders whose IDs are not in `pasted_contents` are left unchanged.

    Args:
        text: The text containing placeholders.
        pasted_contents: Mapping of paste IDs to stored content.

    Returns:
        The text with all known placeholders expanded.
    """

    def _replace(match: re.Match[str]) -> str:
        content = pasted_contents.get(int(match.group(1)))
        # Return the stored text literally; unknown IDs keep their placeholder.
        return content.content if content is not None else match.group(0)

    return _PASTE_REF_RE.sub(_replace, text)
