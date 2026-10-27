"""Detect model replies that a provider cut off before they were complete.

LangChain has no normalized stop reason, so each provider's raw metadata is
read here. `output_truncation` classifies one message. `EmptyTruncatedReplyMiddleware`
turns an empty, truncated reply into a typed error, so the run fails visibly
instead of ending as a silent success with nothing to show. Subagent failures
become task-tool errors, while partial answers carry a warning to the parent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    TracePolicy,
    omit_payload,
)
from langchain_core.messages import AIMessage, ToolMessage

from deepagents_code.local_context import _FORKED_CONTEXT_KEY

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from langchain.agents.middleware.types import (
        ModelRequest,
        ModelResponse,
        ToolCallRequest,
    )
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langgraph.types import Command

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
        no_usage: bool = False,
        has_reasoning: bool = False,
        invalid_tool_call: bool = False,
    ) -> None:
        """Build the error message.

        Args:
            truncation: Why the provider stopped the reply.
            model: Model identifier, when known.
            no_usage: The provider reported zero tokens for the request, so
                the stated reason is likely not the real cause.
            has_reasoning: The reply contained reasoning content.
            invalid_tool_call: The reply contains an unparseable tool call.
        """
        self.truncation = truncation
        self.model = model
        self.no_usage = no_usage
        self.has_reasoning = has_reasoning
        self.invalid_tool_call = invalid_tool_call
        subject = f"Model {model!r}" if model else "The model"
        stopped = (
            "while writing a tool call" if invalid_tool_call else "before replying"
        )
        msg = f"{subject} stopped {stopped}: {describe_truncation(truncation)}."
        msg += _hint(truncation, no_usage=no_usage, has_reasoning=has_reasoning)
        super().__init__(msg)

    def __reduce__(
        self,
    ) -> tuple[Callable[..., ModelOutputTruncatedError], tuple[object, ...]]:
        """Rebuild from the constructor arguments, not the formatted message.

        Returns:
            The class and its constructor arguments.
        """
        return _rebuild_truncated_error, (
            self.truncation,
            self.model,
            self.no_usage,
            self.has_reasoning,
            self.invalid_tool_call,
        )


def _rebuild_truncated_error(
    truncation: OutputTruncation,
    model: str | None,
    no_usage: bool,
    has_reasoning: bool,
    invalid_tool_call: bool,
) -> ModelOutputTruncatedError:
    return ModelOutputTruncatedError(
        truncation,
        model,
        no_usage=no_usage,
        has_reasoning=has_reasoning,
        invalid_tool_call=invalid_tool_call,
    )


def _hint(truncation: OutputTruncation, *, no_usage: bool, has_reasoning: bool) -> str:
    """Return advice for the user, or an empty string.

    A reply with zero reported tokens did not run out of output budget, even
    when the provider says so, so do not suggest changing the output budget.
    """
    if no_usage:
        return (
            " The provider reported no token usage, so the request likely"
            " failed upstream rather than hitting the limit."
        )
    if truncation.reason == MAX_OUTPUT_TOKENS:
        if has_reasoning:
            return " Retry, or lower the reasoning effort."
        return " Retry, or raise the output token limit."
    if truncation.reason == CONTEXT_WINDOW_EXCEEDED:
        return " Run /compact to free context space, then retry."
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
    """Fail the main run when the provider cut off a reply before any output.

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
        if request.state.get(_FORKED_CONTEXT_KEY):
            return
        EmptyTruncatedReplyMiddleware._check_empty(response, request)

    @staticmethod
    def _check_empty(response: ModelResponse, request: ModelRequest) -> None:
        message = _final_ai_message(response)
        if message is None or has_visible_reply(message):
            return
        truncation = output_truncation(message)
        if truncation is None:
            return
        error = ModelOutputTruncatedError(
            truncation,
            _model_name(message, request.model),
            no_usage=_reported_no_usage(message),
            has_reasoning=any(
                block.get("type") == "reasoning" for block in message.content_blocks
            ),
            invalid_tool_call=bool(message.invalid_tool_calls),
        )
        logger.warning("%s", error)
        raise error

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Return truncated subagent failures to the parent.

        Raises:
            ModelOutputTruncatedError: If a tool other than `task` raised it.
        """
        try:
            return handler(request)
        except ModelOutputTruncatedError as error:
            if request.tool_call["name"] != "task":
                raise
            return ToolMessage(
                str(error), tool_call_id=request.tool_call["id"], status="error"
            )

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Return truncated subagent failures to the parent.

        Raises:
            ModelOutputTruncatedError: If a tool other than `task` raised it.
        """
        try:
            return await handler(request)
        except ModelOutputTruncatedError as error:
            if request.tool_call["name"] != "task":
                raise
            return ToolMessage(
                str(error), tool_call_id=request.tool_call["id"], status="error"
            )

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


class SubagentTruncatedReplyMiddleware(EmptyTruncatedReplyMiddleware):
    """Keep partial subagent answers and fail empty ones for the parent to handle."""

    @staticmethod
    def _check(response: ModelResponse, request: ModelRequest) -> None:
        EmptyTruncatedReplyMiddleware._check_empty(response, request)
        message = _final_ai_message(response)
        if message is None or message.tool_calls:
            return
        truncation = output_truncation(message)
        if truncation is None:
            return
        warning = f"Subagent reply was cut off: {describe_truncation(truncation)}."
        content = (
            f"{message.content}\n\n{warning}"
            if isinstance(message.content, str)
            else [*message.content, {"type": "text", "text": f"\n\n{warning}"}]
        )
        replacement = message.model_copy(update={"content": content})
        response.result = [
            replacement if item is message else item for item in response.result
        ]
