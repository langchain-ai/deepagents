"""Reference extension: a stateful tool.

State lives in the extension instance, created once when the factory runs, so the
tool accumulates notes for the life of the session. The registered shutdown hook
flushes them, showing the lifecycle contract: load → init → active → shutdown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents_code.extensions import ExtensionAPI


def extension(d: ExtensionAPI) -> None:
    """Register a session-scoped scratchpad tool.

    Args:
        d: The dcode extension API.
    """
    notes: list[str] = []

    def remember_note(note: str) -> str:
        """Save a short note for the rest of this session.

        Args:
            note: Text to remember.

        Returns:
            Confirmation including the current note count.
        """
        notes.append(note)
        return f"Saved. {len(notes)} note(s) in the scratchpad."

    def read_notes() -> str:
        """Return every note saved during this session.

        Returns:
            The saved notes, one per line.
        """
        return "\n".join(notes) if notes else "The scratchpad is empty."

    d.register_tool(remember_note)
    d.register_tool(read_notes)
    d.on_shutdown(notes.clear)
