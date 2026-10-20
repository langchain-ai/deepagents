"""Independent channel conversations scoped to the active invocation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from langchain_core.tools import tool

from deepagents_talon.background import _IN_SUBAGENT

BreakoutHandler = Callable[[str, str], Awaitable[str]]
BREAKOUT_HANDLER: ContextVar[BreakoutHandler | None] = ContextVar("talon_breakout", default=None)
MAX_TITLE_CHARS = 160
MAX_INSTRUCTIONS_CHARS = 12_000


@tool
async def breakout(title: str, instructions: str) -> str:
    """Start an independent task in a new Slack or Discord thread, keeping this conversation active.

    Use only when the user asks to break out separate work. Provide a self-contained
    task: the new conversation does not inherit this conversation's history.
    Only one breakout is allowed per turn; background runs and subagents cannot use it.

    Args:
        title: Short title for the new thread.
        instructions: Complete task and context for the independent agent.
    """
    if _IN_SUBAGENT.get():
        return "Breakout unavailable to subagents."
    handler = BREAKOUT_HANDLER.get()
    if handler is None:
        return "Breakout unavailable: this run has no originating channel."
    if not title.strip() or len(title) > MAX_TITLE_CHARS:
        return f"Title must contain 1-{MAX_TITLE_CHARS} characters."
    if not instructions.strip() or len(instructions) > MAX_INSTRUCTIONS_CHARS:
        return f"Instructions must contain 1-{MAX_INSTRUCTIONS_CHARS} characters."
    return await handler(title.strip(), instructions.strip())
