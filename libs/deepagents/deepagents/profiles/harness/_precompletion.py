"""Pre-completion verification middleware (a lightweight "Ralph Wiggum" hook).

Today's models don't naturally enter a build-verify loop on exit, so a system
prompt that *asks* for verification is necessary but not sufficient. This
middleware intercepts the agent as it's about to finish a turn and, if that turn
made any tool calls (i.e. potentially changed state), forces exactly one more
model pass with a verification reminder before the response is returned.

It is deliberately light: no grader sub-agent and no separate model — it reuses
the agent's own model for the verification pass. It fires at most once per turn
(reset each turn via `before_agent`) and only when tool calls were made, so
pure-conversational turns are untouched.
"""

from __future__ import annotations

# ruff: noqa: E501
# The verification prompt is written as whole-sentence lines by design.
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

_VERIFICATION_SOURCE = "precompletion_verifier"
"""`name`/`lc_source` tag on the injected verification message."""

_VERIFICATION_PROMPT = """\
Before you reply, do a verification pass:
- For each action you told the user you would take, confirm the matching tool call succeeded and the resulting state matches your claim — if not, correct it now.
- Confirm you've addressed every part of the request, and that you've surfaced any rule or limit that changed what you could do.
Then write your response."""
"""Reminder injected as a `HumanMessage` to force one verification pass."""


class _PreCompletionState(AgentState):
    """State for the pre-completion hook."""

    _precompletion_verified: NotRequired[Annotated[bool, PrivateStateAttr]]
    """Set once the hook has fired this turn, so it fires at most once."""

    _precompletion_baseline: NotRequired[Annotated[int, PrivateStateAttr]]
    """Message count at turn start, to scope tool-call detection to this turn."""


class PreCompletionVerificationMiddleware(AgentMiddleware):
    """Force one verification pass before finishing a tool-using turn."""

    state_schema = _PreCompletionState

    def before_agent(self, state: _PreCompletionState, runtime: Runtime) -> dict[str, Any]:  # noqa: ARG002
        """Reset the per-turn guard and record the turn's start index."""
        return {
            "_precompletion_verified": False,
            "_precompletion_baseline": len(state["messages"]),
        }

    async def abefore_agent(self, state: _PreCompletionState, runtime: Runtime) -> dict[str, Any]:  # noqa: ARG002
        """Async variant of `before_agent`."""
        return {
            "_precompletion_verified": False,
            "_precompletion_baseline": len(state["messages"]),
        }

    def _decide(self, state: _PreCompletionState) -> dict[str, Any] | None:
        """Return the state update for `after_agent`, or `None` to finish."""
        if state.get("_precompletion_verified"):
            return None
        baseline = state.get("_precompletion_baseline", 0)
        recent = state["messages"][baseline:]
        made_tool_calls = any(
            isinstance(m, AIMessage) and getattr(m, "tool_calls", None) for m in recent
        )
        if not made_tool_calls:
            # Pure-conversational turn: nothing to verify, just finish.
            return {"_precompletion_verified": True}
        return {
            "_precompletion_verified": True,
            "messages": [
                HumanMessage(
                    content=_VERIFICATION_PROMPT,
                    name=_VERIFICATION_SOURCE,
                    additional_kwargs={"lc_source": _VERIFICATION_SOURCE},
                )
            ],
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: _PreCompletionState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
        """Force one verification pass when the finishing turn used tools."""
        return self._decide(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(self, state: _PreCompletionState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
        """Async variant of `after_agent`."""
        return self._decide(state)
