"""Clipboard utilities for deepagents-cli."""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App

_PREVIEW_MAX_LENGTH = 40


def _copy_osc52(text: str) -> None:
    """Copy text using OSC 52 escape sequence (works over SSH/tmux)."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    osc52_seq = f"\033]52;c;{encoded}\a"
    if os.environ.get("TMUX"):
        osc52_seq = f"\033Ptmux;\033{osc52_seq}\033\\"

    with open("/dev/tty", "w") as tty:
        tty.write(osc52_seq)
        tty.flush()


def _shorten_preview(texts: list[str]) -> str:
    """Shorten text for notification preview."""
    dense_text = "⏎".join(texts).replace("\n", "⏎")
    if len(dense_text) > _PREVIEW_MAX_LENGTH:
        return f"{dense_text[: _PREVIEW_MAX_LENGTH - 1]}…"
    return dense_text


def copy_selection_to_clipboard(app: App) -> None:
    """Copy selected text from app widgets to clipboard.

    This queries all widgets for their text_selection and copies
    any selected text to the system clipboard.
    """
    selected_texts = []

    for widget in app.query("*"):
        if not hasattr(widget, "text_selection") or not widget.text_selection:
            continue

        selection = widget.text_selection

        try:
            result = widget.get_selection(selection)
        except Exception:
            continue

        if not result:
            continue

        selected_text, _ = result
        if selected_text.strip():
            selected_texts.append(selected_text)

    if not selected_texts:
        return

    combined_text = "\n".join(selected_texts)

    # Try multiple clipboard methods
    copy_methods = [_copy_osc52, app.copy_to_clipboard]

    # Try pyperclip if available
    try:
        import pyperclip

        copy_methods.insert(1, pyperclip.copy)
    except ImportError:
        pass

    for copy_fn in copy_methods:
        try:
            copy_fn(combined_text)
            # Use markup=False to prevent copied text from being parsed as Rich markup
            app.notify(
                f'"{_shorten_preview(selected_texts)}" copied',
                severity="information",
                timeout=2,
                markup=False,
            )
            return
        except Exception:
            continue

    # If all methods fail, still notify but warn
    app.notify(
        "Failed to copy - no clipboard method available",
        severity="warning",
        timeout=3,
    )
