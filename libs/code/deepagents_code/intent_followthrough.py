"""Keep a turn going when the model promises tool work but calls no tool.

Some turns end after the model emits a natural-language intent statement
("I'm checking the failed job's logs...") without any tool call, so the
promised fetch/read/run never happens and the developer gets a promise with no
deliverable. This middleware detects that specific shape — a final assistant
message that announces an action, has no tool calls, and answers a request that
references an external resource — and jumps back to the model so it acts on its
own plan. The heuristic is intentionally conservative so genuinely-complete
answers are not re-driven into a loop.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState, ContextT
    from langgraph.runtime import Runtime

# Prose openings that announce an action the assistant is about to take.
_INTENT_PATTERN = re.compile(
    r"\b("
    r"i'?m\s+(?:going\s+to|about\s+to|checking|reviewing|reading|fetching|"
    r"looking|pulling|running|inspecting|examining)"
    r"|i'?ll\s+(?:check|review|read|fetch|look|pull|run|inspect|examine|take\s+a\s+look)"
    r"|let\s+me\s+(?:check|review|read|fetch|look|pull|run|inspect|examine|take\s+a\s+look)"
    r")\b",
    re.IGNORECASE,
)

# Markers that the request needs external data the model cannot know without a
# tool call: a URL, a PR/issue reference, or a git action verb.
_EXTERNAL_RESOURCE_PATTERN = re.compile(
    r"(https?://\S+|#\d+|\b(?:pr|pull\s+request|issue|commit|push|merge|rebase|"
    r"clone|fetch|pull)\b)",
    re.IGNORECASE,
)

# One-line reminder appended so the model corrects course on the retry rather
# than re-emitting the same plan-only prose.
FOLLOW_THROUGH_REMINDER = (
    "You described an action but called no tool. Call the required tool now "
    "instead of only stating your plan."
)


def _message_text(message: AIMessage) -> str:
    """Return the plain-text content of an assistant message.

    Returns:
        The concatenated text of the message's content blocks.
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return " ".join(parts)


def _references_external_resource(state: AgentState) -> bool:
    """Whether the latest human request needs external data or a git action.

    Returns:
        `True` when the most recent genuine human message references a URL, a
        PR/issue number, or a git action verb.
    """
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage) and not _is_reminder(msg):
            return bool(_EXTERNAL_RESOURCE_PATTERN.search(msg.text))
    return False


def _already_nudged(state: AgentState) -> bool:
    """Whether a follow-through reminder was already injected this turn.

    Walking back from the latest assistant message, stop at the first genuine
    human turn (a `HumanMessage` that is not our reminder). If a reminder is seen
    before then, the model was already re-driven once and still declined, so we
    stop to avoid looping.

    Returns:
        `True` when a reminder was already injected after the last human turn.
    """
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return _is_reminder(msg)
    return False


def _is_reminder(message: HumanMessage) -> bool:
    """Whether a human message is a follow-through reminder we injected.

    Returns:
        `True` when the message content is exactly the reminder text.
    """
    content = message.content
    return isinstance(content, str) and content == FOLLOW_THROUGH_REMINDER


class IntentFollowThroughMiddleware(AgentMiddleware):
    """Continue the loop when the final turn is a plan-only intent statement."""

    @hook_config(can_jump_to=["model"])
    def after_model(  # noqa: PLR6301  # AgentMiddleware hook must be an instance method.
        self,
        state: AgentState,
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Jump back to the model when it promised tool work but called none.

        Returns:
            A jump-to-model state update, or `None` to end the turn normally.
        """
        return _decide(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(  # noqa: PLR6301  # AgentMiddleware hook must be an instance method.
        self,
        state: AgentState,
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of `after_model`.

        Returns:
            A jump-to-model state update, or `None` to end the turn normally.
        """
        return _decide(state)


def _decide(state: AgentState) -> dict[str, Any] | None:
    """Return a jump-to-model update when the turn is a bare intent statement.

    Returns:
        A state update that re-drives the model, or `None` to leave the turn
        as-is.
    """
    messages = state.get("messages") or []
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, AIMessage) or getattr(last, "tool_calls", None):
        return None
    if not _INTENT_PATTERN.search(_message_text(last)):
        return None
    if not _references_external_resource(state):
        return None
    if _already_nudged(state):
        return None
    return {
        "messages": [HumanMessage(content=FOLLOW_THROUGH_REMINDER)],
        "jump_to": "model",
    }
