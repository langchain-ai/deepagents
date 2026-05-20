"""Middleware that tracks total context tokens in graph state.

Registers a `_context_tokens` channel (checkpointed, schema-private) and
writes it from `after_model` based on the latest `AIMessage.usage_metadata`.

Persisting from inside the graph (rather than via a separate client-side
`aupdate_state` call) keeps the write on the same checkpoint as the model
response and avoids creating a standalone `UpdateState` run in LangSmith.
It also works identically against local graphs and remote (HTTP) graphs.

The CLI reads `_context_tokens` back from `state_values` on thread resume
so `/tokens` and the status bar show accurate values immediately, without
having to replay or re-tokenize history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
)
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class TokenTrackingState(AgentState):
    """Extends agent state with a persisted context-token counter."""

    _context_tokens: Annotated[NotRequired[int], PrivateStateAttr]
    """Total context tokens reported by the model's last `usage_metadata`."""


def _extract_context_tokens(message: AIMessage) -> int | None:
    """Return the context-token count from an AI message, or `None` if absent.

    Prefers `input_tokens + output_tokens` when both are reported; falls back
    to `total_tokens` when the model only provides the aggregate.
    """
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return None
    input_toks = usage.get("input_tokens", 0) or 0
    output_toks = usage.get("output_tokens", 0) or 0
    if input_toks or output_toks:
        return input_toks + output_toks
    total = usage.get("total_tokens", 0) or 0
    return total or None


class TokenStateMiddleware(AgentMiddleware[TokenTrackingState, ContextT]):
    """Persists the latest context-token count after each model call.

    See the module docstring for why this rides the model node's checkpoint
    instead of a separate `aupdate_state` (avoids a standalone `UpdateState`
    run in LangSmith and works identically against remote graphs).
    """

    state_schema = TokenTrackingState

    def after_model(  # noqa: PLR6301  # AgentMiddleware hook must be an instance method.
        self,
        state: TokenTrackingState,
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Write `_context_tokens` from the most recent `AIMessage.usage_metadata`.

        Args:
            state: Current agent state; only `messages` is inspected.
            runtime: LangGraph runtime (unused; required by the hook signature).

        Returns:
            State update `{"_context_tokens": <int>}` when usage is reported on
            the latest `AIMessage`; otherwise `None`.
        """
        messages = state.get("messages") or []
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                tokens = _extract_context_tokens(msg)
                if tokens is not None:
                    return {"_context_tokens": tokens}
                return None
        return None
