"""Middleware that adds a `_context_tokens` channel to the graph state.

The field is checkpointed (survives across sessions) but not passed to the
model (`PrivateStateAttr`).  The CLI writes the latest total-context token
count here after every LLM response and context offload, and reads it back
when resuming a thread so that `/tokens` and the status bar show accurate
values immediately.
"""

from __future__ import annotations

from typing import Annotated, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
)


class TokenTrackingState(AgentState):
    """Extends agent state with a persisted context-token counter."""

    _context_tokens: Annotated[NotRequired[int], PrivateStateAttr]
    """Total context tokens reported by the model's last `usage_metadata`."""


class TokenStateMiddleware(AgentMiddleware):
    """Schema-only middleware that registers `_context_tokens` in the state schema."""

    state_schema = TokenTrackingState
