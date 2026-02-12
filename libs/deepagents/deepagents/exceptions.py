"""Custom exceptions for the deepagents package."""


class EmptyContentError(ValueError):
    """Raised when subagent returns no extractable content.

    This error occurs when the subagent middleware cannot extract any valid text
    content from a subagent's response messages.

    Common causes:
    - LLM failed to generate a response
    - LLM returned only empty messages
    - Response contains only non-text content (images, tool calls, etc.)
    - New or unknown inference provider with incompatible message format
    - Message structure doesn't expose the `.text` property correctly

    The error is caught internally and converted to an error string that allows
    the conversation to continue gracefully.
    """
