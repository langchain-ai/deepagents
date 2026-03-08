"""Custom exceptions for the deepagents package."""


class EmptyContentError(ValueError):
    """Raised when subagent returns no extractable content.

    Common causes:
    - LLM failed to generate a response
    - LLM returned only empty messages
    - Response contains only non-text content (images, tool calls, etc.)
    """

    def __init__(self) -> None:
        """Initialize with standard error message."""
        super().__init__("No content found in subagent messages. This may indicate the LLM failed to respond properly.")
