"""Safe, reversible terminal tab titles independent of Textual's app header."""

from __future__ import annotations

import logging
import string
import sys
from typing import TextIO

from deepagents_code._env_vars import NO_TERMINAL_ESCAPE, is_env_truthy
from deepagents_code._invocation import invoked_name

logger = logging.getLogger(__name__)
_FIELDS = frozenset({"app_name", "thread_name", "cwd", "branch"})


def _valid_template(template: str) -> bool:
    """Return whether the template uses only documented, plain fields."""
    try:
        return all(
            field is None or (field in _FIELDS and not spec and not conversion)
            for _, field, spec, conversion in string.Formatter().parse(template)
        )
    except ValueError:
        return False


class TerminalTitle:
    """Manage one balanced terminal title save/restore lifecycle."""

    def __init__(self, template: str) -> None:
        """Set the template, falling back when its replacement fields are invalid.

        Args:
            template: Title with app_name, thread_name, cwd, and branch fields.
        """
        self._template = template if _valid_template(template) else "{app_name}"
        self._stream: TextIO | None = None
        self._last_title: str | None = None

    def _write(self, sequence: str) -> bool:
        """Return whether the sequence was written without a terminal error."""
        if self._stream is None:
            return False
        try:
            self._stream.write(sequence)
            self._stream.flush()
        except (OSError, ValueError):
            logger.debug("Could not update terminal title", exc_info=True)
            return False
        return True

    def start(self) -> None:
        """Save the original title once, honoring the terminal-escape opt-out."""
        if self._stream is not None or is_env_truthy(NO_TERMINAL_ESCAPE):
            return
        for stream in (sys.__stderr__, sys.__stdout__):
            if stream is not None and not stream.closed and stream.isatty():
                self._stream = stream
                break
        if self._stream is None:
            return
        if not self._write("\x1b[22;0t"):
            self._stream = None

    def update(self, *, thread_name: str = "", cwd: str = "", branch: str = "") -> None:
        """Render a sanitized title, avoiding duplicate terminal writes.

        Args:
            thread_name: Active thread's assigned name, or empty when unnamed.
            cwd: Active working directory.
            branch: Active Git branch.
        """
        title = self._template.format(
            app_name=invoked_name(), thread_name=thread_name, cwd=cwd, branch=branch
        )
        title = "".join(char for char in title if char.isprintable())[:512]
        if title != self._last_title and self._write(f"\x1b]0;{title}\x07"):
            self._last_title = title

    def restore(self) -> None:
        """Restore the saved title at most once, including on repeated cleanup."""
        self._write("\x1b[23;0t")
        self._stream = None
        self._last_title = None
