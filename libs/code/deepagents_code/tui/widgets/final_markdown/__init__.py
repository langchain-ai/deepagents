"""Render a finished assistant reply as one widget.

Textual's `Markdown` mounts one child widget per block, list item, bullet,
table cell, and fence label. A long transcript therefore holds thousands of
widgets, and every relayout walks all of them. A reply that no longer streams
does not need that structure, so `FinalMarkdown` lays out the same markdown
as rows of `Content` inside a single widget.

The layout copies the block styles, margins, and wrapping of Textual's
`Markdown` so a reply keeps its height when the streamed `Markdown` is
swapped for this widget. Validated against Textual 8.2.8: re-check
`MarkdownBlock._token_to_content`, the `Markdown` block CSS, `GridLayout`
table sizing, and `textual._resolve.resolve` on every Textual bump.
"""

from deepagents_code.tui.widgets.final_markdown._widget import FinalMarkdown

__all__ = ["FinalMarkdown"]
