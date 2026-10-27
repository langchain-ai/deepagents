"""Detect model replies that a provider cut off before they were complete.

LangChain has no normalized stop reason, so each provider's raw metadata is
read here. `output_truncation` classifies one message. `EmptyTruncatedReplyMiddleware`
turns an empty, truncated reply into a typed error, so the run fails visibly
instead of ending as a silent success with nothing to show.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from langchain.agents.middleware.types import (
    AgentMiddleware,
    TracePolicy,
    omit_payload,
)
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

TruncationReason = Literal[
    "max_output_tokens", "content_filter", "context_window_exceeded"
]
"""Provider-neutral reason a reply stopped early."""

MAX_OUTPUT_TOKENS: Final = "max_output_tokens"
CONTENT_FILTER: Final = "content_filter"
CONTEXT_WINDOW_EXCEEDED: Final = "context_window_exceeded"

_RESPONSES_INCOMPLETE_REASONS: Final[Mapping[str, TruncationReason]] = {
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "content_filter": CONTENT_FILTER,
}
"""OpenAI Responses API `incomplete_details.reason` values."""

_FINISH_REASONS: Final[Mapping[str, TruncationReason]] = {
    # Chat Completions (OpenAI and compatible providers such as Fireworks).
    "length": MAX_OUTPUT_TOKENS,
    "content_filter": CONTENT_FILTER,
    # Google GenAI `FinishReason` enum names.
    "MAX_TOKENS": MAX_OUTPUT_TOKENS,
    "SAFETY": CONTENT_FILTER,
    "RECITATION": CONTENT_FILTER,
    "BLOCKLIST": CONTENT_FILTER,
    "PROHIBITED_CONTENT": CONTENT_FILTER,
    "SPII": CONTENT_FILTER,
}
"""`finish_reason` values that mean the reply stopped early."""

_ANTHROPIC_STOP_REASONS: Final[Mapping[str, TruncationReason]] = {
    "max_tokens": MAX_OUTPUT_TOKENS,
    "refusal": CONTENT_FILTER,
    "model_context_window_exceeded": CONTEXT_WINDOW_EXCEEDED,
}
"""Anthropic `stop_reason` values that mean the reply stopped early."""

_BEDROCK_STOP_REASONS: Final[Mapping[str, TruncationReason]] = {
    "max_tokens": MAX_OUTPUT_TOKENS,
    "content_filtered": CONTENT_FILTER,
    "guardrail_intervened": CONTENT_FILTER,
    "model_context_window_exceeded": CONTEXT_WINDOW_EXCEEDED,
}
"""Bedrock Converse `stopReason` values that mean the reply stopped early."""

_METADATA_TABLES: Final[tuple[tuple[str, Mapping[str, TruncationReason]], ...]] = (
    ("finish_reason", _FINISH_REASONS),
    ("stop_reason", _ANTHROPIC_STOP_REASONS),
    ("stopReason", _BEDROCK_STOP_REASONS),
)
"""Top-level `response_metadata` keys and the values that signal truncation."""

_REASON_DESCRIPTIONS: Final[Mapping[TruncationReason, str]] = {
    MAX_OUTPUT_TOKENS: "it reached its output token limit",
    CONTENT_FILTER: "the provider blocked the reply",
    CONTEXT_WINDOW_EXCEEDED: "the conversation filled its context window",
}


@dataclass(frozen=True, slots=True)
class OutputTruncation:
    """Why a provider stopped a reply early."""

    reason: TruncationReason
    """Provider-neutral reason."""

    raw: str
    """The provider's own value, for example `length` or `MAX_TOKENS`."""


def output_truncation(message: BaseMessage) -> OutputTruncation | None:
    """Classify whether the provider stopped this message early.

    Reads the raw provider metadata on `response_metadata`, so it also works on
    the streamed chunk that carries the stop reason.

    Args:
        message: A model message or streamed message chunk.

    Returns:
        The truncation, or `None` when the reply ended normally or the
            metadata does not say.
    """
    metadata = message.response_metadata
    if not isinstance(metadata, dict):
        return None
    if metadata.get("status") == "incomplete":
        details = metadata.get("incomplete_details")
        raw = details.get("reason") if isinstance(details, dict) else None
        if isinstance(raw, str) and raw in _RESPONSES_INCOMPLETE_REASONS:
            return OutputTruncation(_RESPONSES_INCOMPLETE_REASONS[raw], raw)
    for key, table in _METADATA_TABLES:
        raw = metadata.get(key)
        if isinstance(raw, str) and raw in table:
            return OutputTruncation(table[raw], raw)
    return None


def has_visible_reply(message: AIMessage) -> bool:
    """Return whether a message has reply text or a tool call to act on.

    Reasoning alone does not count: the user still gets no answer.

    Args:
        message: A complete model message.

    Returns:
        `True` when the message has non-blank text or a parsed tool call.
    """
    return bool(message.tool_calls) or bool(message.text.strip())


def describe_truncation(truncation: OutputTruncation) -> str:
    """Return a short, user-facing explanation of a truncation.

    Args:
        truncation: The classified truncation.

    Returns:
        Text such as `it reached its output token limit (length)`.
    """
    return f"{_REASON_DESCRIPTIONS[truncation.reason]} ({truncation.raw})"


class ModelOutputTruncatedError(ValueError):
    """The model stopped before it gave any reply.

    Subclasses `ValueError` so that the LangGraph server keeps the message when
    it serializes the error for the client. Other exception types are redacted
    to "An internal error occurred".
    """

    def __init__(
        self,
        truncation: OutputTruncation,
        model: str | None = None,
        *,
        tool_count: int | None = None,
    ) -> None:
        """Build the error message.

        Args:
            truncation: Why the provider stopped the reply.
            model: Model identifier, when known.
            tool_count: Number of tools sent with the request, when the
                provider reported no token usage. Its presence means the stop
                reason is likely not the real cause.
        """
        self.truncation = truncation
        self.model = model
        self.tool_count = tool_count
        subject = f"Model {model!r}" if model else "The model"
        msg = f"{subject} stopped before replying: {describe_truncation(truncation)}."
        msg += _hint(truncation, tool_count)
        super().__init__(msg)

    def __reduce__(
        self,
    ) -> tuple[Callable[..., ModelOutputTruncatedError], tuple[object, ...]]:
        """Rebuild from the constructor arguments, not the formatted message.

        Returns:
            The class and its constructor arguments.
        """
        return _rebuild_truncated_error, (self.truncation, self.model, self.tool_count)


def _rebuild_truncated_error(
    truncation: OutputTruncation, model: str | None, tool_count: int | None
) -> ModelOutputTruncatedError:
    return ModelOutputTruncatedError(truncation, model, tool_count=tool_count)


def _hint(truncation: OutputTruncation, tool_count: int | None) -> str:
    """Return advice for the user, or an empty string.

    A reply with zero reported tokens did not run out of output budget, even
    when the provider says so. OpenAI returns that shape when a request has
    too many tools, so point at the tool count instead of the output limit.
    """
    if tool_count is not None:
        return (
            " The provider reported no token usage, so the stated reason is"
            f" likely not the cause. The request had {tool_count} tools; try"
            " disabling some MCP servers."
        )
    if truncation.reason == MAX_OUTPUT_TOKENS:
        return " Retry, or lower the reasoning effort."
    return ""


def _reported_no_usage(message: AIMessage) -> bool:
    """Whether the provider reported zero tokens for the whole request.

    Returns:
        `True` when usage is present and its total is zero.
    """
    usage = message.usage_metadata
    return usage is not None and not usage.get("total_tokens")


def _final_ai_message(response: ModelResponse) -> AIMessage | None:
    if response.structured_response is not None:
        return None
    messages = [item for item in response.result if isinstance(item, AIMessage)]
    return messages[-1] if messages else None


def _model_name(message: AIMessage, model: str | BaseChatModel) -> str | None:
    name = message.response_metadata.get("model_name")
    if isinstance(name, str) and name:
        return name
    if isinstance(model, str):
        return model
    # Private import: the SDK exposes no public helper for the identifier.
    from deepagents._models import (  # ruff:ignore[import-private-name]
        get_model_identifier,
    )

    return get_model_identifier(model)


class EmptyTruncatedReplyMiddleware(AgentMiddleware):
    """Fail the run when the provider cut off a reply before any output.

    Without this, a reply with no text, no tool calls, and a truncation stop
    reason ends the run as a success. The empty message is checkpointed and
    the UI shows nothing, so the app looks hung. Raising instead fails the
    model node, so the empty message is not saved and the client shows the
    error. A partial reply is left alone; the client warns about it.

    This middleware never retries. It must wrap the retry middleware, so the
    error is not retried, and any recovery middleware that may still repair
    the reply.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    @staticmethod
    def _check(response: ModelResponse, request: ModelRequest) -> None:
        message = _final_ai_message(response)
        if message is None or has_visible_reply(message):
            return
        truncation = output_truncation(message)
        if truncation is None:
            return
        tool_count = len(request.tools) if _reported_no_usage(message) else None
        error = ModelOutputTruncatedError(
            truncation, _model_name(message, request.model), tool_count=tool_count
        )
        logger.warning("%s", error)
        raise error

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Raise when the reply was cut off before any output.

        Returns:
            The unchanged model response.
        """
        response = handler(request)
        self._check(response, request)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Raise when the reply was cut off before any output.

        Returns:
            The unchanged model response.
        """
        response = await handler(request)
        self._check(response, request)
        return response
