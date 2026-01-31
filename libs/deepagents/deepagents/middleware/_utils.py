"""Utility functions for middleware."""

from langchain_core.messages import SystemMessage


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
    if system_message is None:
        return SystemMessage(content=text)

    # Preserve string format for OpenAI API compatibility
    if isinstance(system_message.content, str):
        new_content = f"{system_message.content}\n\n{text}"
        return SystemMessage(content=new_content)

    # Handle list format (for multimodal or other use cases)
    new_content_list: list[str | dict[str, str]] = list(system_message.content_blocks)
    if new_content_list:
        text = f"\n\n{text}"
    new_content_list.append({"type": "text", "text": text})
    return SystemMessage(content=new_content_list)
