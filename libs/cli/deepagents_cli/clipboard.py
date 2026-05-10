"""Clipboard utilities for deepagents-cli."""

from __future__ import annotations

import base64
import logging
import os
import pathlib
from typing import TYPE_CHECKING

from deepagents_cli.config import get_glyphs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.app import App

_PREVIEW_MAX_LENGTH = 40


def _copy_osc52(text: str) -> None:
    """Copy text using OSC 52 escape sequence (works over SSH/tmux)."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    osc52_seq = f"\033]52;c;{encoded}\a"
    if os.environ.get("TMUX"):
        osc52_seq = f"\033Ptmux;\033{osc52_seq}\033\\"

    with pathlib.Path("/dev/tty").open("w", encoding="utf-8") as tty:
        tty.write(osc52_seq)
        tty.flush()


def _shorten_preview(texts: list[str]) -> str:
    """Shorten text for notification preview.

    Returns:
        Shortened preview text suitable for notification display.
    """
    glyphs = get_glyphs()
    dense_text = glyphs.newline.join(texts).replace("\n", glyphs.newline)
    if len(dense_text) > _PREVIEW_MAX_LENGTH:
        return f"{dense_text[: _PREVIEW_MAX_LENGTH - 1]}{glyphs.ellipsis}"
    return dense_text


def copy_text_to_clipboard(app: App, text: str) -> tuple[bool, str | None]:
    """Copy text to the system clipboard.

    Args:
        app: The active Textual app, used for the app clipboard backend.
        text: Text to copy.

    Returns:
        Tuple of `(success, error_message)`.

            `success` is `True` when one backend completed without raising.
            `error_message` is `None` on success and the last backend's error
            string when every backend failed, suitable for surfacing to the
            user so they can self-diagnose missing clipboard support.
    """
    # Backend order: pyperclip first (most reliable when installed), then
    # Textual's app clipboard, then OSC 52 as a last resort for SSH/remote
    # sessions where no local clipboard is reachable.
    copy_methods: list[Callable[[str], object]] = [app.copy_to_clipboard]

    try:
        import pyperclip

        copy_methods.insert(0, pyperclip.copy)
    except ImportError:
        pass

    copy_methods.append(_copy_osc52)

    last_error: str | None = None
    for copy_fn in copy_methods:
        try:
            copy_fn(text)
        except (OSError, RuntimeError, TypeError) as e:
            last_error = str(e) or type(e).__name__
            logger.debug(
                "Clipboard copy method %s failed: %s",
                getattr(copy_fn, "__name__", repr(copy_fn)),
                e,
                exc_info=True,
            )
            continue
        else:
            return True, None

    return False, last_error


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

        if selection.end is None:
            continue

        try:
            result = widget.get_selection(selection)
        except (AttributeError, TypeError, ValueError, IndexError) as e:
            logger.debug(
                "Failed to get selection from widget %s: %s",
                type(widget).__name__,
                e,
                exc_info=True,
            )
            continue

        if not result:
            continue

        selected_text, _ = result
        if selected_text.strip():
            selected_texts.append(selected_text)

    if not selected_texts:
        return

    combined_text = "\n".join(selected_texts)

    success, _ = copy_text_to_clipboard(app, combined_text)
    if success:
        # Use markup=False to prevent copied text from being parsed as Rich markup
        app.notify(
            f'"{_shorten_preview(selected_texts)}" copied',
            severity="information",
            timeout=2,
            markup=False,
        )
        return

    # If all methods fail, still notify but warn. markup=False guards against
    # this string ever growing dynamic content (e.g., the backend error reason)
    # that could contain bracket characters.
    app.notify(
        "Failed to copy - no clipboard method available",
        severity="warning",
        timeout=3,
        markup=False,
    )
