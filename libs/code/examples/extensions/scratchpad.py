"""Reference extension: session-scoped tools with deterministic teardown."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents_code.extensions import ExtensionAPI


def extension(d: ExtensionAPI) -> None:
    """Register an in-memory scratchpad.

    Args:
        d: The dcode extension API.
    """
    notes: list[str] = []

    def remember_note(note: str) -> str:
        """Save a note for this server session.

        Args:
            note: Text to remember.

        Returns:
            Confirmation with the updated note count.
        """
        notes.append(note)
        return f"Saved {len(notes)} note(s)."

    def read_notes() -> str:
        """Return the session scratchpad."""
        return "\n".join(notes) if notes else "The scratchpad is empty."

    d.register_tool(remember_note)
    d.register_tool(read_notes)
    d.on_shutdown(notes.clear)
