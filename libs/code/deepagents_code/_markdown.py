"""Escaping for external text that reaches Rich/Textual markdown source."""

from __future__ import annotations

MARKDOWN_ESCAPES = str.maketrans({char: f"\\{char}" for char in "\\&`*_[]<>|~"})
"""Translation table backing `escape_markdown`.

Covers the inline constructs Rich's markdown parser acts on (emphasis, code
spans, links, autolinks/HTML, HTML entities, strikethrough) plus the `|`
table-cell separator. Block-level punctuation (`#`, `-`, `>`) only has meaning
at the start of a line; line breaks are normalized before translation, and `>`
is escaped anyway because it is cheap.
"""


def escape_markdown(text: str) -> str:
    """Normalize line breaks and escape markdown syntax in external text.

    Args:
        text: Display string that may contain markdown punctuation.

    Returns:
        `text` on one line with markdown-significant characters escaped.
    """
    normalized = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return normalized.translate(MARKDOWN_ESCAPES)
