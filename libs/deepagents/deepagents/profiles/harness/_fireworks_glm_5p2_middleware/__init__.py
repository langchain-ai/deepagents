"""Behavioral middlewares for the GLM-5.2 (Fireworks) harness profile.

These are specific to the GLM-5.2 harness profile and are wired in via that
profile's ``extra_middleware`` (see ``_fireworks_glm_5p2.py``):

- ``FinalizeMiddleware`` — as the run nears the LangGraph ``recursion_limit``,
  nudge the agent once to write/verify a deliverable, then end the run gracefully
  before a ``GraphRecursionError`` would crash it. The budget is counted in
  completed model turns via a private state counter (task-independent; does not
  rely on the ``RemainingSteps`` managed channel).
- ``RambleMiddleware`` — when the model emits a long message with no tool call
  (rambling prose instead of acting), nudge it once to act and loop back so the
  turn is not wasted.
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


# --- Finalize -----------------------------------------------------------------

# Turn budget before the recursion_limit. Empirically one model turn is ~6 graph
# supersteps under the dcode middleware stack (observed langgraph_step≈244 at 42
# turns), so recursion_limit=2000 leaves ample headroom for the hard cap below.
# Soft nudge fires at ~75% of budget — a genuine "running low" warning — but now
# leaves ~18 turns of runway (was 6) so an over-exploring agent that pivots to a
# deliverable has room to finish; hard cap raised for long-horizon tasks.
_DEFAULT_SOFT_TURNS = 54
_DEFAULT_HARD_TURNS = 72

_SOFT_ENV = "DEEPAGENTS_FINALIZE_SOFT_TURNS"
_HARD_ENV = "DEEPAGENTS_FINALIZE_HARD_TURNS"

_FINALIZE_NUDGE_TEXT = (
    "You are running low on step budget — stop exploring and secure a deliverable now. "
    "Write your best-effort solution to the exact path the task requires, then confirm "
    "it exists with `ls`/`cat`. Re-read your notes file if you have lost track of the "
    "contract. Run the task's own check (or `verify_implementation`) against it. A "
    "submitted best-effort artifact can score; an unfinished one cannot."
)

_HARD_TEXT = "Step budget exhausted — ending the run. Whatever is on disk is the final submission."


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
        soft = self._soft_override if self._soft_override is not None else _positive_int_env(_SOFT_ENV, _DEFAULT_SOFT_TURNS)
        hard = self._hard_override if self._hard_override is not None else _positive_int_env(_HARD_ENV, _DEFAULT_HARD_TURNS)
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
                "messages": [HumanMessage(content=_FINALIZE_NUDGE_TEXT)],
                "finalize_nudged": True,
                "finalize_turns": turns,
            }
        return {"finalize_turns": turns}

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: FinalizeState, runtime: Runtime[Any]) -> dict[str, Any]:
        """Async variant of ``before_model``.

        Returns:
            Same as ``before_model``.
        """
        return self.before_model(state, runtime)


# --- Anti-ramble --------------------------------------------------------------

# Output-token boundary above which a no-tool-call turn counts as rambling.
# Edit here or override via the env var below.
_DEFAULT_RAMBLE_OUTPUT_TOKENS = 8000
_RAMBLE_TOKENS_ENV = "DEEPAGENTS_RAMBLE_OUTPUT_TOKENS"

# Finish reasons that mean the model was cut off mid-generation.
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})

_RAMBLE_NUDGE_TEXT = (
    "STOP — you just produced a long response with NO tool call. You are talking, not "
    "acting, and that spends your output budget without putting anything on disk. Do "
    "NOT hand-write prose or file contents into your reply. Your next action MUST be a "
    "tool call: write and run a short script in the shell (or edit the file directly), "
    "then read the result back. The deliverable must be a file on disk, not text in "
    "this message — produce it now."
)


class RambleState(AgentState):
    """State schema for ``RambleMiddleware``."""

    # `PrivateStateAttr` keeps this flag out of the input/output schema while it
    # still persists internally across turns (default last-value reducer).
    ramble_nudged: NotRequired[Annotated[bool, PrivateStateAttr]]


class RambleMiddleware(AgentMiddleware):
    """Nudge the agent to act when it rambles instead of calling a tool.

    After each model turn, inspects the latest ``AIMessage``. If it has no tool calls
    and was either length-truncated or exceeds the output-token boundary, injects a
    ``HumanMessage`` and jumps back to the model. The boundary defaults to the module
    constant and is overridable via ``DEEPAGENTS_RAMBLE_OUTPUT_TOKENS``.

    The nudge re-arms per ramble *episode*: it fires once, and does not re-fire while
    the model keeps rambling back-to-back (loop-safe); the flag is cleared as soon as
    the model produces a non-ramble turn (i.e. acts), so a later ramble is nudged
    again rather than being ignored for the rest of the run.
    """

    state_schema = RambleState  # type: ignore[assignment]

    def __init__(self, *, output_tokens: int | None = None) -> None:
        """Initialize the middleware.

        Args:
            output_tokens: Output-token boundary. ``None`` reads env / default.
        """
        super().__init__()
        self._output_tokens_override = output_tokens

    def _boundary(self) -> int:
        """Return the output-token boundary (override, else env, else default)."""
        override = self._output_tokens_override
        if override is not None and override > 0:
            return override
        return _positive_int_env(_RAMBLE_TOKENS_ENV, _DEFAULT_RAMBLE_OUTPUT_TOKENS)

    def _is_ramble(self, message: AIMessage) -> bool:
        """Return True if ``message`` is a long/truncated turn with no tool call."""
        if message.tool_calls:
            return False
        finish = str(message.response_metadata.get("finish_reason") or message.response_metadata.get("stop_reason") or "").lower()
        if finish in _TRUNCATED_FINISH_REASONS:
            return True
        usage = message.usage_metadata or {}
        return int(usage.get("output_tokens") or 0) >= self._boundary()

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: RambleState,
        runtime: Runtime[Any],  # noqa: ARG002  (part of the hook signature)
    ) -> dict[str, Any] | None:
        """Nudge + loop back to the model when the latest turn rambled; re-arm on action.

        Returns:
            ``{"messages": [...], "jump_to": "model", "ramble_nudged": True}`` on a
            fresh ramble; ``{"ramble_nudged": False}`` to re-arm when the model acted
            after a nudge; otherwise ``None``.
        """
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None
        if not self._is_ramble(last):
            # The model acted (or produced a normal turn) — re-arm so the NEXT ramble
            # episode is nudged again, instead of the historical once-per-run cap.
            return {"ramble_nudged": False} if state.get("ramble_nudged") else None
        if state.get("ramble_nudged"):
            # Already nudged this episode; don't re-nudge a back-to-back ramble (the
            # model is ignoring it — re-arm only once it actually acts). Loop-safe.
            return None
        return {
            "messages": [HumanMessage(content=_RAMBLE_NUDGE_TEXT)],
            "jump_to": "model",
            "ramble_nudged": True,
        }

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: RambleState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        """Async variant of ``after_model``.

        Returns:
            Same as ``after_model``.
        """
        return self.after_model(state, runtime)


__all__ = ["FinalizeMiddleware", "RambleMiddleware"]
