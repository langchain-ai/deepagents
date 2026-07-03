"""Schema and middleware for per-checkpoint state restored when resuming.

`ResumeState` declares several checkpointed, schema-private channels. They fall
into two groups with *different* write paths:

Written from inside the graph on successful model turns:

- `_context_tokens` — total context tokens from the latest
    `AIMessage.usage_metadata`, written by `ResumeStateMiddleware.after_model`.
    Powers `/tokens` and the status bar.
- `_model_spec` / `_model_params` — the model and invocation params effectively
    in use for the turn, written by `ConfigurableModelMiddleware` after a
    successful model call. Lets `dcode -r` restore the model the resumed thread
    was actually using instead of falling back to the user's global default.

Written primarily by the TUI client, via `aupdate_state` (see
`DeepAgentsApp._persist_goal_rubric_state`) — these are user/agent-owned. Most
have no model-node write site; the two exceptions are called out below:

- `_goal_objective` / `_goal_status` / `_goal_rubric` / `_goal_status_note` —
    the accepted goal and its lifecycle status. `_goal_objective`/`_goal_rubric`
    are client-only, but `_goal_status`/`_goal_status_note` are *also* written
    from inside the graph by the agent's `update_goal` tool.
- `_pending_goal_completion_note` — an agent-requested completion awaiting the
    post-turn rubric result and, when needed, user approval.
- `_sticky_rubric` — the TUI-owned persistent rubric. This is separate from
    the public `rubric` graph input so one-shot rubric turns can be checkpointed
    without being restored as sticky state.
- `_pending_goal_objective` / `_pending_goal_rubric` — a proposed goal awaiting
    user acceptance of its criteria.

All of these are facts the CLI reads back from `state_values` on thread resume
so it can rehydrate the session without replaying or re-tokenizing history.

The model-turn channels are persisted from inside the graph (rather than via a
separate client-side `aupdate_state` call) so the write rides the same checkpoint
as the model response and avoids creating a standalone `UpdateState` run in
LangSmith. Because they are versioned channel state, resuming a specific
checkpoint yields the values as of *that* checkpoint — not a thread-level
aggregate. The goal/rubric channels are client-written because the user sets
them outside any model turn (except `_goal_status`/`_goal_status_note`, which the
`update_goal` tool also writes from inside the graph). Both paths work
identically against local and remote (HTTP) graphs.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NotRequired,
    cast,
    get_args,
)

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
)
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

GoalStatus = Literal["active", "blocked", "complete"]
"""Lifecycle status of a TUI-owned goal.

`active` and `blocked` are unfinished states; `complete` is terminal. A blocked
goal is still considered active (unfinished) by `get_goal`.
"""

_GOAL_STATUS_VALUES: frozenset[str] = frozenset(get_args(GoalStatus))


def coerce_goal_status(value: object) -> GoalStatus | None:
    """Narrow a persisted goal-status value to a known `GoalStatus`.

    A corrupt or forward-version checkpoint can carry an unexpected status
    string (or a non-string). Coercing to `None` rather than passing the raw
    value through keeps the `GoalStatus` `Literal` load-bearing on the read
    path, so an unknown status is treated as "no goal status" instead of a
    silently active goal. Resume/restore callers should log the discard
    separately so it is surfaced rather than dropped; the model-read path
    (`_goal_snapshot`) intentionally treats an unknown status as `active`
    without logging.

    Args:
        value: Raw value read from checkpoint state.

    Returns:
        The value when it is a recognized `GoalStatus`, otherwise `None`.
    """
    if isinstance(value, str) and value in _GOAL_STATUS_VALUES:
        return cast("GoalStatus", value)
    return None


class GoalRubricChannels(AgentState):
    """Goal/rubric state channels shared by every schema that touches them.

    Declared once here so each schema that carries these channels —
    `ResumeState` and `goal_tools.GoalToolState` — inherits the *same*
    `PrivateStateAttr`-marked annotations. Middleware state schemas merge with
    later entries winning, so an independent re-declaration that dropped the
    `PrivateStateAttr` marker would override these and leak the field into the
    public graph input/output schema. Inheriting from a single base makes that
    drift unrepresentable.
    """

    _goal_objective: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Accepted goal objective restored by the TUI on resume."""

    _goal_status: Annotated[NotRequired[GoalStatus | None], PrivateStateAttr]
    """Goal lifecycle status (`active`, `blocked`, `complete`, or `None`)."""

    _goal_rubric: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Accepted rubric associated with `_goal_objective`."""

    _goal_status_note: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Evidence or blocker note recorded by `update_goal`."""

    _pending_goal_completion_note: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Completion evidence awaiting rubric and user approval."""

    _sticky_rubric: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Persistent rubric owned by the TUI, distinct from graph input `rubric`."""


class ResumeState(GoalRubricChannels):
    """Extends agent state with per-checkpoint facts restored on resume.

    Inherits the shared goal/rubric channels from `GoalRubricChannels` and adds
    the channels unique to resume: the after-model token/spec facts and the
    pending-goal proposal awaiting acceptance.
    """

    _context_tokens: Annotated[NotRequired[int], PrivateStateAttr]
    """Total context tokens reported by the model's last `usage_metadata`."""

    _session_cost: Annotated[NotRequired[float], PrivateStateAttr]
    """Cumulative estimated USD cost across every turn on this thread.

    Unlike `_context_tokens` (which reflects the *current* context size and is
    reset by `/compact`), this is a monotonically increasing session total. Each
    turn reads the prior value and adds the latest turn's estimated cost, so
    resuming a thread restores the accumulated spend.
    """

    _model_spec: Annotated[NotRequired[str], PrivateStateAttr]
    """`provider:model` spec effectively in use for the latest turn."""

    _model_params: Annotated[NotRequired[dict[str, Any] | None], PrivateStateAttr]
    """Invocation params effectively in use for the latest turn."""

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


def _split_provider_model(spec: object) -> tuple[str, str]:
    """Split a persisted `provider:model` spec into its parts.

    Args:
        spec: Value read from `_model_spec` state (may be absent or malformed).

    Returns:
        A `(provider, model)` pair. Either element may be `""` when the spec is
        missing or does not contain a `:` separator (in which case the whole
        value is treated as the model name).
    """
    if not isinstance(spec, str) or not spec:
        return "", ""
    provider, sep, model = spec.partition(":")
    if not sep:
        return "", provider
    return provider, model


def _estimate_turn_cost(message: AIMessage, spec: object) -> float | None:
    """Estimate the USD cost of the turn's latest AI message.

    Args:
        message: The most recent `AIMessage` carrying `usage_metadata`.
        spec: The persisted `_model_spec` used to recover provider/model.

    Returns:
        Estimated cost in USD, or `None` when usage or pricing is unavailable.
    """
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return None
    input_toks = usage.get("input_tokens", 0) or 0
    output_toks = usage.get("output_tokens", 0) or 0
    if not input_toks and not output_toks:
        # Only an aggregate total is available; attribute it all to input so the
        # request is still priced (input is the cheaper rate, so this is a
        # conservative lower bound rather than an overstatement).
        input_toks = usage.get("total_tokens", 0) or 0
    if not input_toks and not output_toks:
        return None

    provider, model = _split_provider_model(spec)
    if not model:
        from deepagents_code.config import settings

        provider = settings.model_provider or ""
        model = settings.model_name or ""

    from deepagents_code._cost import estimate_request_cost

    return estimate_request_cost(
        input_tokens=input_toks,
        output_tokens=output_toks,
        model_name=model,
        provider=provider,
    )


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
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Write `_context_tokens` and accumulate `_session_cost` for the turn.

        Model metadata is written by `ConfigurableModelMiddleware` from the
        actual request that completed successfully; this hook records token
        usage from the most recent `AIMessage.usage_metadata` and folds the
        turn's estimated cost into the running session total.

        Args:
            state: Current agent state; `messages`, `_model_spec`, and
                `_session_cost` are inspected.
            runtime: LangGraph runtime required by the middleware interface.

        Returns:
            State update with `_context_tokens` and/or `_session_cost`, or `None`
            when no token count is available.
        """
        update: dict[str, Any] = {}

        for msg in reversed(state.get("messages") or []):
            if isinstance(msg, AIMessage):
                tokens = _extract_context_tokens(msg)
                if tokens is not None:
                    update["_context_tokens"] = tokens
                turn_cost = _estimate_turn_cost(msg, state.get("_model_spec"))
                if turn_cost:
                    prior = state.get("_session_cost") or 0.0
                    update["_session_cost"] = float(prior) + turn_cost
                break

        return update or None
