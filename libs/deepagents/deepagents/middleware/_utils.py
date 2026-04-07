"""Utility functions for middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import ContentBlock, SystemMessage

from deepagents.backends import CompositeBackend

if TYPE_CHECKING:
    from deepagents.backends.protocol import BACKEND_TYPES


def resolve_artifacts_root(backend: BACKEND_TYPES | None) -> str:
    """Determine the artifacts root from the backend instance.

    Returns the `artifacts_root` attribute from a `CompositeBackend`, or
    `"/"` for all other backend types.

    Args:
        backend: Backend instance or factory, or None.

    Returns:
        The artifacts root path.
    """
    if isinstance(backend, CompositeBackend):
        return backend.artifacts_root
    return "/"


def append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """Append text to a system message.

    Args:
        system_message: Existing system message or None.
        text: Text to add to the system message.

    Returns:
        New SystemMessage with the text appended.
    """
    new_content: list[ContentBlock] = list(system_message.content_blocks) if system_message else []
    if new_content:
        text = f"\n\n{text}"
    new_content.append({"type": "text", "text": text})
    return SystemMessage(content_blocks=new_content)
