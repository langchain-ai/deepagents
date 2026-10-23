"""Turn a failing tool call into a message instead of a dead run.

LangGraph's `_default_handle_tool_errors` re-raises, so one exception out of a
tool ends the whole graph run. For a browsing agent almost every tool failure is
recoverable -- the page moved, a selector missed, the snippet threw -- and the
model's next turn is where that gets fixed.

This cost us 6 of the first 17 runs, and 8 of those deaths landed inside the
same 86 ms, which is the signature of a shared QuickJS interpreter being torn
down under concurrent runs rather than anything the agent did. Whatever the
upstream cause, a run should not die because one `eval` raised.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

_MAX_CHARS = 2_000


class ToolErrorsAreMessages(AgentMiddleware):
    """Report tool exceptions back to the model rather than raising them."""

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        try:
            return await handler(request)
        except Exception as err:  # noqa: BLE001 - the whole point is to contain it
            call = request.tool_call
            logger.debug("tool %s failed", call.get("name"), exc_info=True)
            detail = f"{type(err).__name__}: {err}"
            return ToolMessage(
                content=(
                    f"error: {detail[:_MAX_CHARS]}\n"
                    "This call failed but the run continues. Try a different "
                    "approach -- re-navigate if the page or session was lost."
                ),
                tool_call_id=call.get("id", ""),
                name=call.get("name"),
                status="error",
            )
