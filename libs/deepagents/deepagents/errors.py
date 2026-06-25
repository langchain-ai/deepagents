"""Typed errors raised by the Deep Agents harness."""

from __future__ import annotations


class LLMGatewayError(RuntimeError):
    """Raised when the LLM gateway returns a non-2xx response that the chat-model client swallowed into an empty `AIMessage`."""

    def __init__(self, status_code: int, body: str) -> None:
        """Store the gateway `status_code` and `body` and format the error message."""
        self.status_code = status_code
        self.body = body
        super().__init__(f"LLM gateway returned {status_code}: {body.strip()}")
