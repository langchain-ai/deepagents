"""Reference extension: middleware that gates and records shell commands.

Copy this file into `~/.deepagents/extensions/` to try it. It shows the shape of
a permission-gate middleware: the agent's `execute` calls are inspected, denied
commands are turned into a tool error the model can read, and everything else is
appended to an audit log.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain.agents.middleware.types import ToolCallRequest
    from langgraph.types import Command

    from deepagents_code.extensions import ExtensionAPI

DENIED = ("rm -rf /", "curl", "wget")
AUDIT_LOG = Path.home() / ".deepagents" / "shell-audit.log"


class ShellAudit(AgentMiddleware):
    """Records every shell command and blocks a small deny-list."""

    name = "ShellAudit"

    def wrap_tool_call(  # noqa: PLR6301  # AgentMiddleware hook signature
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Inspect an outgoing tool call before it executes.

        Args:
            request: The pending tool call.
            handler: The next handler in the middleware chain.

        Returns:
            An error `ToolMessage` for a denied command, otherwise whatever the
                downstream handler returns.
        """
        if request.tool_call["name"] != "execute":
            return handler(request)

        command = str(request.tool_call["args"].get("command", ""))
        if any(pattern in command for pattern in DENIED):
            return ToolMessage(
                content=f"Blocked by the ShellAudit extension: {command}",
                tool_call_id=request.tool_call["id"] or "",
                status="error",
            )

        with AUDIT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{command}\n")
        return handler(request)


def extension(d: ExtensionAPI) -> None:
    """Register the audit middleware.

    Args:
        d: The dcode extension API.
    """
    d.register_middleware(ShellAudit)
