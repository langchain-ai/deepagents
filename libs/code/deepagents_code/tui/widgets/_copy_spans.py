"""Shared click-to-copy span metadata for Textual widgets.

Widgets that render `label: value` rows (the debug console snapshot, the welcome
banner) mark individual value spans as copyable by embedding the copy text and a
toast label in the span's style meta. Keeping the meta keys and the
build/extract pair here means the two ends of the protocol cannot drift apart.
"""

from __future__ import annotations

from textual.style import Style as TStyle

COPY_TEXT_META = "copy_text"
"""Meta key marking a span whose text is copied on click."""

COPY_LABEL_META = "copy_label"
"""Meta key carrying the field label used in the copy toast."""


def copy_span_style(text: str, label: str) -> TStyle:
    """Build the style that marks a span as click-to-copy.

    Args:
        text: The text copied to the clipboard when the span is clicked.
        label: The field label used to word the success toast.

    Returns:
        A style carrying only the copy metadata, so it can be combined with a
            visual style (e.g. `TStyle(dim=True) + copy_span_style(...)`).
    """
    return TStyle.from_meta({COPY_TEXT_META: text, COPY_LABEL_META: label})


def copy_span_target(style: object) -> tuple[str, str] | None:
    """Return the copy text and field label from a span style, if any.

    Args:
        style: The Textual event style under the pointer/click.

    Returns:
        `(text, label)` when the span carries a copy marker, else `None`.
    """
    meta = getattr(style, "meta", None)
    if not isinstance(meta, dict):
        return None
    text = meta.get(COPY_TEXT_META)
    if not isinstance(text, str) or not text:
        return None
    label = meta.get(COPY_LABEL_META)
    if not isinstance(label, str) or not label:
        return None
    return text, label
