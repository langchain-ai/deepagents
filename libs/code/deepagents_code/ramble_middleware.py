"""Anti-ramble middleware for the dcode agent.

When the model emits a long message with no tool call -- rambling prose instead of
writing and running a script -- nudge it once to act and loop back to the model so
the turn is not wasted. Detection is on the latest ``AIMessage``: no tool calls plus
either a length-truncated finish reason or an output-token count over a boundary.
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

# Output-token boundary above which a no-tool-call turn counts as rambling.
# Edit here or override via the env var below.
_DEFAULT_RAMBLE_OUTPUT_TOKENS = 8000
_RAMBLE_TOKENS_ENV = "DEEPAGENTS_RAMBLE_OUTPUT_TOKENS"

# Finish reasons that mean the model was cut off mid-generation.
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})

_NUDGE_TEXT = (
    "You produced a long response with no tool call. Writing prose or hand-authoring "
    "file contents in your reply burns your output budget before anything reaches "
    "disk. Stop and act: write a short script and run it in the shell to compute, "
    "generate, or verify the result, then read the output back. The graded deliverable "
    "must be a file on disk, not text in this message."
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


class RambleState(AgentState):
    """State schema for ``RambleMiddleware``."""

    # `PrivateStateAttr` keeps this flag out of the input/output schema while it
    # still persists internally across turns (default last-value reducer).
    ramble_nudged: NotRequired[Annotated[bool, PrivateStateAttr]]


class RambleMiddleware(AgentMiddleware):
    """Nudge the agent to act when it rambles instead of calling a tool.

    After each model turn, inspects the latest ``AIMessage``. If it has no tool calls
    and was either length-truncated or exceeds the output-token boundary, injects a
    one-time ``HumanMessage`` and jumps back to the model. The boundary defaults to the
    module constant and is overridable via ``DEEPAGENTS_RAMBLE_OUTPUT_TOKENS``. The
    one-time flag caps it at a single extra model re-invocation per run.
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
        finish = str(
            message.response_metadata.get("finish_reason")
            or message.response_metadata.get("stop_reason")
            or ""
        ).lower()
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
        """Nudge once + loop back to the model when the latest turn rambled.

        Returns:
            ``{"messages": [...], "jump_to": "model", "ramble_nudged": True}`` on a
            detected ramble not yet nudged; otherwise ``None``.
        """
        if state.get("ramble_nudged"):
            return None
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage) or not self._is_ramble(last):
            return None
        return {
            "messages": [HumanMessage(content=_NUDGE_TEXT)],
            "jump_to": "model",
            "ramble_nudged": True,
        }

    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self, state: RambleState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async variant of ``after_model``.

        Returns:
            Same as ``after_model``.
        """
        return self.after_model(state, runtime)
