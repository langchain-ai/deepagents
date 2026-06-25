"""Best-effort finalize middleware for the dcode agent.

As the run approaches the LangGraph ``recursion_limit``, nudge the agent once to
write and verify a deliverable, then end the run gracefully before a
``GraphRecursionError`` would crash it. The budget is counted in completed model
turns via a private state counter, which is task-independent and does not rely on
the ``RemainingSteps`` managed channel (not populated under this middleware stack).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
    hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

# Turn budget before the recursion_limit. One model turn costs ~18 graph
# supersteps under the dcode middleware stack, so recursion_limit=1000 is ~55
# turns; defaults stop with margin.
_DEFAULT_SOFT_TURNS = 36
_DEFAULT_HARD_TURNS = 42

_SOFT_ENV = "DEEPAGENTS_FINALIZE_SOFT_TURNS"
_HARD_ENV = "DEEPAGENTS_FINALIZE_HARD_TURNS"

_NUDGE_TEXT = (
    "You are running low on step budget — stop exploring and secure a deliverable now. "
    "Write your best-effort solution to the exact path the task requires, then confirm "
    "it exists with `ls`/`cat`. Re-read your notes file if you have lost track of the "
    "contract. Run the task's own check (or `verify_implementation`) against it. A "
    "submitted best-effort artifact can score; an unfinished one cannot."
)

_HARD_TEXT = (
    "Step budget exhausted — ending the run. Whatever is on disk is the final "
    "submission."
)


def _positive_int_env(name: str, default: int) -> int:
    """Return a positive int from env var ``name``, or ``default`` if unset/invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


class FinalizeState(AgentState):
    """State schema for ``FinalizeMiddleware``."""

    # `PrivateStateAttr` keeps these channels out of the input/output schema; they
    # still persist internally across turns (the default last-value reducer).
    finalize_turns: NotRequired[Annotated[int, PrivateStateAttr]]
    finalize_nudged: NotRequired[Annotated[bool, PrivateStateAttr]]


class FinalizeMiddleware(AgentMiddleware):
    """Nudge then gracefully end as the run nears the ``recursion_limit``.

    Counts completed model turns in private state. At the soft threshold it injects
    a one-time finalize nudge and lets the agent continue; at the hard threshold it
    jumps to the end so the run stops before ``GraphRecursionError``. Thresholds are
    in turns, default to module constants, and are overridable via
    ``DEEPAGENTS_FINALIZE_SOFT_TURNS`` / ``DEEPAGENTS_FINALIZE_HARD_TURNS``.
    """

    state_schema = FinalizeState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        soft_turns: int | None = None,
        hard_turns: int | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            soft_turns: Turn count at which to nudge. ``None`` reads env.
            hard_turns: Turn count at which to end. ``None`` reads env.
        """
        super().__init__()
        self._soft_override = soft_turns
        self._hard_override = hard_turns

    def _thresholds(self) -> tuple[int, int]:
        """Resolve (soft, hard) thresholds, falling back to defaults if misconfigured.

        Returns:
            A ``(soft, hard)`` pair with ``soft < hard`` guaranteed.
        """
        soft = (
            self._soft_override
            if self._soft_override is not None
            else _positive_int_env(_SOFT_ENV, _DEFAULT_SOFT_TURNS)
        )
        hard = (
            self._hard_override
            if self._hard_override is not None
            else _positive_int_env(_HARD_ENV, _DEFAULT_HARD_TURNS)
        )
        if not 0 < soft < hard:
            return _DEFAULT_SOFT_TURNS, _DEFAULT_HARD_TURNS
        return soft, hard

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: FinalizeState,
        runtime: Runtime[Any],  # noqa: ARG002  (part of the hook signature)
    ) -> dict[str, Any]:
        """Nudge or end based on completed model turns.

        Returns:
            ``{"jump_to": "end", ...}`` at the hard threshold; a one-time nudge at the
            soft threshold; otherwise just the incremented turn counter.
        """
        turns = state.get("finalize_turns", 0) + 1
        soft, hard = self._thresholds()
        if turns >= hard:
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=_HARD_TEXT)],
                "finalize_turns": turns,
            }
        if turns >= soft and not state.get("finalize_nudged"):
            return {
                "messages": [HumanMessage(content=_NUDGE_TEXT)],
                "finalize_nudged": True,
                "finalize_turns": turns,
            }
        return {"finalize_turns": turns}

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: FinalizeState, runtime: Runtime[Any]
    ) -> dict[str, Any]:
        """Async variant of ``before_model``.

        Returns:
            Same as ``before_model``.
        """
        return self.before_model(state, runtime)
