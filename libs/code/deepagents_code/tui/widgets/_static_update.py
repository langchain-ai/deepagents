"""Cheap `Static` updates for animation hot paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.content import Content

if TYPE_CHECKING:
    from textual.widgets import Static


def update_in_place(widget: Static, content: str | Content) -> None:
    """Update a `Static`, and relayout only when the content width changes.

    `Static.update` relayouts the whole screen by default. That reflows every
    mounted transcript widget, so a 10 Hz spinner gets slower as the
    transcript grows. A spinner tick swaps one glyph at the same cell width,
    so a repaint is sufficient. A width change (for example, the elapsed time
    grows from `(59s)` to `(1m 0s)`) still relayouts, so `width: auto`
    widgets resize and wrapped lines reflow.

    Args:
        widget: Widget to update.
        content: New content. Plain strings are compared as unstyled text.
    """
    current = widget.content
    old = Content(current) if isinstance(current, str) else current
    if not isinstance(old, Content):
        widget.update(content)
        return
    new = Content(content) if isinstance(content, str) else content
    if new.is_same(old):
        return
    widget.update(content, layout=new.cell_length != old.cell_length)
