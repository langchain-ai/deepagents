"""Default tool-error handling for deep agents.

A tool that raises should cost the model one step, not the whole run. Without
this, only argument-binding failures come back as an error `ToolMessage`
(`ToolNode` converts those before the tool runs); anything the tool *body*
raises escapes the graph and ends `.invoke()`.

The handler here backs the `ToolErrorMiddleware` that `create_deep_agent`
places at the head of every stack it assembles, so the conversion applies to
built-in tools, caller-supplied tools, MCP tools, and tools running inside a
subagent alike.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import ToolErrorMiddleware
from langchain_core.tools import ToolException
from langgraph.errors import GraphRecursionError, NodeCancelledError, NodeTimeoutError

from deepagents.middleware.subagents import TASK_TOOL_NAME

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain.tools.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)


_NEVER_HANDLED: tuple[type[Exception], ...] = (
    # Recursion is a structural failure, not a tool failure. Reporting it to
    # the model invites the same runaway delegation that tripped the limit.
    GraphRecursionError,
    # Cancellation and node timeouts are the runtime deciding this work stops.
    # Answering the model as if the tool merely failed would undo that.
    NodeCancelledError,
    NodeTimeoutError,
)
"""Exceptions that keep propagating even though they reach `on_error`.

`GraphBubbleUp` (interrupts, parent commands) never reaches the handler at all
-- `ToolErrorMiddleware` re-raises it before calling us -- so human-in-the-loop
is unaffected. `KeyboardInterrupt`, `SystemExit`, and `asyncio.CancelledError`
derive from `BaseException` and are likewise out of reach.
"""


def _default_on_tool_error(exc: Exception, request: ToolCallRequest) -> str | None:
    """Convert a tool-execution exception into content for an error `ToolMessage`.

    Args:
        exc: Exception raised while executing the tool.
        request: The tool call that failed (name, args, call id).

    Returns:
        Content for an error `ToolMessage`, or `None` to let `exc` propagate.
    """
    if isinstance(exc, _NEVER_HANDLED):
        return None

    tool_name = request.tool.name if request.tool else request.tool_call["name"]

    if tool_name == TASK_TOOL_NAME:
        # `task` nests a whole agent run rather than doing leaf work. Tool
        # failures *inside* the subagent are already converted by that
        # subagent's own copy of this middleware, so anything still escaping
        # here is structural -- its model erroring out, a `CompiledSubAgent`
        # violating the state contract. Reporting those as a recoverable tool
        # error would hide a broken subagent behind a retry loop and strip the
        # `failed` status off the streamed subagent handle.
        return None

    if isinstance(exc, ToolException):
        # The one exception type whose message is written *for* the model:
        # "allows tools to signal errors without stopping the agent". This is
        # what `langchain_mcp_adapters` raises for an `isError` tool result, so
        # the text is the upstream server's own error report. Matches what
        # `handle_tool_error=True` produces in `langchain-core`.
        logger.debug("Tool %r raised ToolException; returning it to the model.", tool_name, exc_info=exc)
        return str(exc.args[0]) if exc.args else "Tool execution error"

    # Everything else is an unplanned failure whose message may carry internal
    # detail (paths, hosts, credentials in a URL). Name the type so the model
    # can react, and leave the detail in the logs for whoever owns the tool.
    logger.exception("Tool %r raised %s; returning an error to the model.", tool_name, type(exc).__name__, exc_info=exc)
    return f"Error: tool `{tool_name}` failed with {type(exc).__name__}. Check the arguments, or use a different approach if it keeps failing."


def _create_tool_error_middleware() -> AgentMiddleware[Any, Any, Any]:
    """Build the `ToolErrorMiddleware` that `create_deep_agent` installs by default.

    Placed first in each assembled stack, so it is the outermost
    `wrap_tool_call` and also catches failures raised by inner middleware.

    Returns:
        A `ToolErrorMiddleware` wired to [`_default_on_tool_error`][deepagents.middleware._tool_errors._default_on_tool_error].
    """
    return ToolErrorMiddleware(_default_on_tool_error)
