"""Schema and middleware for per-checkpoint state restored when resuming.

`ResumeState` declares several checkpointed, schema-private channels. They fall
into two groups with *different* write paths:

Written by `ResumeStateMiddleware.after_model`, from inside the graph:

- `_context_tokens` — total context tokens from the latest
    `AIMessage.usage_metadata`. Powers `/tokens` and the status bar.
- `_model_spec` — the `provider:model` spec that was effectively in use for
    the turn, read from `runtime.context["effective_model"]`. Lets `dcode -r`
    restore the model the resumed thread was actually using instead of falling
    back to the user's global default.

Written by the TUI client, via `aupdate_state` (see
`DeepAgentsApp._persist_goal_rubric_state`) — these are user/agent-owned and
have no model-node write site:

- `_goal_objective` / `_goal_status` / `_goal_rubric` / `_goal_status_note` —
    the accepted goal and its lifecycle status; `_goal_status`/`_goal_status_note`
    are also written from inside the graph by the agent's `update_goal` tool.
- `_pending_goal_objective` / `_pending_goal_rubric` — a proposed goal awaiting
    user acceptance of its criteria.

All of these are facts the CLI reads back from `state_values` on thread resume
so it can rehydrate the session without replaying or re-tokenizing history.

The `after_model` channels are persisted from inside the graph (rather than via
a separate client-side `aupdate_state` call) so the write rides the same
checkpoint as the model response and avoids creating a standalone `UpdateState`
run in LangSmith. Because they are versioned channel state, resuming a specific
checkpoint yields the values as of *that* checkpoint — not a thread-level
aggregate. The goal/rubric channels are necessarily client-written because the
user sets them outside any model turn. Both paths work identically against
local and remote (HTTP) graphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
)
from langchain_core.messages import AIMessage

from deepagents_code._cli_context import CLIContextSchema

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

GoalStatus = Literal["active", "blocked", "complete"]
"""Lifecycle status of a TUI-owned goal.

`active` and `blocked` are unfinished states; `complete` is terminal. A blocked
goal is still considered active (unfinished) by `get_goal`.
"""


class ResumeState(AgentState):
    """Extends agent state with per-checkpoint facts restored on resume."""

    _context_tokens: Annotated[NotRequired[int], PrivateStateAttr]
    """Total context tokens reported by the model's last `usage_metadata`."""

    _model_spec: Annotated[NotRequired[str], PrivateStateAttr]
    """`provider:model` spec effectively in use for the latest turn."""

    _goal_objective: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Accepted goal objective restored by the TUI on resume."""

    _goal_status: Annotated[NotRequired[GoalStatus | None], PrivateStateAttr]
    """Goal lifecycle status (`active`, `blocked`, `complete`, or `None`)."""

    _goal_rubric: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Accepted rubric associated with `_goal_objective`."""

    _goal_status_note: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Evidence or blocker note recorded by `update_goal`."""

    _pending_goal_objective: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Goal objective awaiting acceptance of proposed criteria."""

    _pending_goal_rubric: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Proposed criteria awaiting user acceptance."""


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


def _extract_model_spec(runtime: Runtime[ContextT]) -> str | None:
    """Return the effective `provider:model` spec from the runtime context.

    The CLI passes the resolved spec in `context["effective_model"]` on every
    invocation. Returns `None` when no context is present (e.g. non-CLI
    callers) or the field is unset/blank.
    """
    ctx = getattr(runtime, "context", None)
    if isinstance(ctx, CLIContextSchema):
        spec = ctx.effective_model
    elif isinstance(ctx, dict):
        spec = ctx.get("effective_model")
    else:
        return None
    if isinstance(spec, str) and spec:
        return spec
    return None


class ResumeStateMiddleware(AgentMiddleware[ResumeState, ContextT]):
    """Persists per-checkpoint resume facts after each model call.

    See the module docstring for why this rides the model node's checkpoint
    instead of a separate `aupdate_state` (avoids a standalone `UpdateState`
    run in LangSmith and works identically against remote graphs).
    """

    state_schema = ResumeState

    def after_model(  # noqa: PLR6301  # AgentMiddleware hook must be an instance method.
        self,
        state: ResumeState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Write `_context_tokens` and `_model_spec` for the latest turn.

        Token count comes from the most recent `AIMessage.usage_metadata`; the
        model spec comes from `runtime.context["effective_model"]`.

        Args:
            state: Current agent state; only `messages` is inspected.
            runtime: LangGraph runtime; `context["effective_model"]` is read.

        Returns:
            State update with whichever of `_context_tokens` / `_model_spec`
            could be resolved, or `None` when neither is available.
        """
        update: dict[str, Any] = {}

        for msg in reversed(state.get("messages") or []):
            if isinstance(msg, AIMessage):
                tokens = _extract_context_tokens(msg)
                if tokens is not None:
                    update["_context_tokens"] = tokens
                break

        spec = _extract_model_spec(runtime)
        if spec is not None:
            update["_model_spec"] = spec

        return update or None
