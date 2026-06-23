"""Best-effort finalize middleware for the dcode agent.

As the run approaches the LangGraph ``recursion_limit``, nudge the agent once to
write and verify a deliverable, then end the run gracefully before a
``GraphRecursionError`` would crash it. The trigger is ``RemainingSteps`` (supersteps
left), so the threshold is task-independent — no per-task calibration of turn counts.
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

# Runtime import: resolved at runtime when langgraph introspects the TypedDict
# annotation to register the managed channel; a type-checking-only import breaks that.
from langgraph.managed import RemainingSteps  # noqa: TC002

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

_DEFAULT_SOFT_STEPS_LEFT = 120
_DEFAULT_HARD_STEPS_LEFT = 40

_SOFT_ENV = "DEEPAGENTS_FINALIZE_SOFT_STEPS_LEFT"
_HARD_ENV = "DEEPAGENTS_FINALIZE_HARD_STEPS_LEFT"

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

    # `PrivateStateAttr` keeps the managed channel out of the input/output schema
    # (managed channels are rejected there); it still populates internal state.
    remaining_steps: NotRequired[Annotated[RemainingSteps, PrivateStateAttr]]
    finalize_nudged: NotRequired[Annotated[bool, PrivateStateAttr]]


class FinalizeMiddleware(AgentMiddleware):
    """Nudge then gracefully end as the run nears the ``recursion_limit``.

    Reads supersteps remaining from ``RemainingSteps``. At the soft threshold it
    injects a one-time finalize nudge and lets the agent continue; at the hard
    threshold it jumps to the end so the run stops before ``GraphRecursionError``.
    Thresholds default to module constants and are overridable via
    ``DEEPAGENTS_FINALIZE_SOFT_STEPS_LEFT`` / ``DEEPAGENTS_FINALIZE_HARD_STEPS_LEFT``.
    """

    state_schema = FinalizeState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        soft_steps_left: int | None = None,
        hard_steps_left: int | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            soft_steps_left: Steps-remaining at which to nudge. ``None`` reads env.
            hard_steps_left: Steps-remaining at which to end. ``None`` reads env.
        """
        super().__init__()
        self._soft_override = soft_steps_left
        self._hard_override = hard_steps_left

    def _thresholds(self) -> tuple[int, int]:
        """Resolve (soft, hard) thresholds, falling back to defaults if misconfigured.

        Returns:
            A ``(soft, hard)`` pair with ``hard < soft`` guaranteed.
        """
        soft = (
            self._soft_override
            if self._soft_override is not None
            else _positive_int_env(_SOFT_ENV, _DEFAULT_SOFT_STEPS_LEFT)
        )
        hard = (
            self._hard_override
            if self._hard_override is not None
            else _positive_int_env(_HARD_ENV, _DEFAULT_HARD_STEPS_LEFT)
        )
        if not 0 < hard < soft:
            return _DEFAULT_SOFT_STEPS_LEFT, _DEFAULT_HARD_STEPS_LEFT
        return soft, hard

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: FinalizeState,
        runtime: Runtime[Any],  # noqa: ARG002  (part of the hook signature)
    ) -> dict[str, Any] | None:
        """Nudge or end based on supersteps remaining.

        Returns:
            ``{"jump_to": "end", ...}`` at the hard threshold; a one-time nudge at the
            soft threshold; otherwise ``None``.
        """
        remaining = state.get("remaining_steps")
        if remaining is None:
            return None
        soft, hard = self._thresholds()
        if remaining <= hard:
            return {"jump_to": "end", "messages": [AIMessage(content=_HARD_TEXT)]}
        if remaining <= soft and not state.get("finalize_nudged"):
            return {
                "messages": [HumanMessage(content=_NUDGE_TEXT)],
                "finalize_nudged": True,
            }
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: FinalizeState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async variant of ``before_model``.

        Returns:
            Same as ``before_model``.
        """
        return self.before_model(state, runtime)
