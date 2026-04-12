"""Utility functions for middleware."""

from langchain_core.messages import ContentBlock, SystemMessage


def append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """Append text to a system message.

    Preserves string content when the existing message uses plain string content,
    ensuring compatibility with chat models that do not support content blocks
    (e.g., `ChatLlamaCpp`). Falls back to content-block representation only when
    the existing message already contains non-text blocks (images, audio, etc.).

    Args:
        system_message: Existing system message or None.
        text: Text to add to the system message.

    Returns:
        New SystemMessage with the text appended.
    """
    if system_message is None:
        return SystemMessage(content=text)

    # Fast path: plain string content stays as a plain string so that models
    # which do not support content-block lists (e.g., ChatLlamaCpp) keep working.
    if isinstance(system_message.content, str):
        return SystemMessage(content=system_message.content + "\n\n" + text)

    # Multi-block content (e.g., images + text): use content_blocks to preserve
    # non-text blocks.
    new_content: list[ContentBlock] = list(system_message.content_blocks)
    if new_content:
        text = f"\n\n{text}"
    new_content.append({"type": "text", "text": text})
    return SystemMessage(content_blocks=new_content)
