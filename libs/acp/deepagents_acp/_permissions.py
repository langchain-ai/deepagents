"""Human-in-the-loop permissions: "always" answers and what a request shows the client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias
from uuid import uuid4

from acp.schema import PermissionOption, ToolCallLocation, ToolCallUpdate

from deepagents_acp._tool_calls import _TOOL_KINDS, _tool_path, _tool_title

if TYPE_CHECKING:
    from collections.abc import Mapping

    from deepagents_acp._session import AgentSessionContext
    from deepagents_acp._tool_calls import _ToolCall

_ExecuteKey: TypeAlias = "tuple[AgentSessionContext, str]"
"""An exact `execute` request: the options it runs under and its canonical arguments."""

_PLAN_REJECTED = (
    "The user rejected the plan. Please ask them for feedback "
    "on how the plan can be improved, then create a new "
    "and improved plan using this same write_todos tool."
)


@dataclass
class _Permissions:
    """The "always" answers the user gave during one opening of a session."""

    allowed_tools: set[str] = field(default_factory=set)
    rejected_tools: set[str] = field(default_factory=set)
    allowed_commands: set[_ExecuteKey] = field(default_factory=set)
    rejected_commands: set[_ExecuteKey] = field(default_factory=set)

    def remembered(self, tool_name: str, execute_key: _ExecuteKey | None) -> dict[str, Any] | None:
        """Return the decision an earlier "always" answer already made, if any."""
        if tool_name == "execute":
            if execute_key in self.allowed_commands:
                return {"type": "approve"}
            if execute_key in self.rejected_commands:
                return {"type": "reject", "message": "The user rejected this command."}
            return None
        if tool_name in self.allowed_tools:
            return {"type": "approve"}
        if tool_name in self.rejected_tools:
            return {"type": "reject", "message": f"The user rejected all {tool_name} calls."}
        return None

    def remember(self, tool_name: str, execute_key: _ExecuteKey | None, *, allow: bool) -> None:
        """Record an "always" answer: per exact command for `execute`, else per tool."""
        if tool_name == "execute":
            if execute_key is not None:
                (self.allowed_commands if allow else self.rejected_commands).add(execute_key)
        elif allow:
            self.allowed_tools.add(tool_name)
        else:
            self.rejected_tools.add(tool_name)


def _execute_arguments(tool_args: object) -> str | None:
    """Return an `execute` request's arguments in canonical form.

    Returns `None` when the request has no command or its arguments do not
    survive a JSON round trip, so it can never be matched by an "always" answer.
    """
    if not isinstance(tool_args, dict):
        return None
    command = tool_args.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        arguments = json.dumps(tool_args, sort_keys=True, allow_nan=False)
        if json.loads(arguments) != tool_args:
            return None
    except (TypeError, ValueError):
        return None
    return arguments


def _permission_options(tool_name: str, execute_key: _ExecuteKey | None) -> list[PermissionOption]:
    """Build the choices for one permission request."""
    options = [
        PermissionOption(option_id="approve", name="Approve", kind="allow_once"),
        PermissionOption(option_id="reject", name="Reject", kind="reject_once"),
    ]
    if tool_name == "execute" and execute_key is None:
        return options
    exact = tool_name == "execute"
    options += [
        PermissionOption(
            option_id="approve_always",
            name=(
                "Always allow this exact command in this session"
                if exact
                else f"Always allow {tool_name} commands"
            ),
            kind="allow_always",
        ),
        PermissionOption(
            option_id="reject_always",
            name=(
                "Always reject this exact command in this session"
                if exact
                else f"Always reject {tool_name} commands"
            ),
            kind="reject_always",
        ),
    ]
    return options


def _permission_tool_call_id(
    calls: Mapping[str, _ToolCall],
    action: Mapping[str, Any],
    scope: tuple[str, ...] | None = None,
) -> str:
    """Pick the `toolCallId` a permission request shows, marking that call as linked.

    Uses the action's own `tool_call_id` when the middleware provides one.
    Otherwise links to the one open streamed call from the same agent with
    the same name and arguments, and never guesses between identical calls: an ambiguous
    request gets a fresh id so it can't be shown against the wrong call.
    """
    explicit = action.get("tool_call_id")
    if isinstance(explicit, str) and explicit:
        if explicit in calls:
            calls[explicit].linked = True
        return explicit
    matches = [
        tool_id
        for tool_id, call in calls.items()
        if not call.finished
        and not call.linked
        and (scope is None or call.scope == scope)
        and call.name == action.get("name")
        and call.args == action.get("args")
    ]
    if len(matches) == 1:
        calls[matches[0]].linked = True
        return matches[0]
    return uuid4().hex


def _permission_request(
    tool_call_id: str,
    tool_name: str,
    tool_args: Any,  # whatever the middleware put in the action
) -> ToolCallUpdate:
    """Describe one action for the client to approve."""
    args = tool_args if isinstance(tool_args, dict) else {}
    if tool_name == "write_todos":
        title = "Review Plan"
    elif tool_name == "execute":
        command = args.get("command", "")
        title = f"Execute: `{command}`" if command else "Execute command"
    else:
        title = _tool_title(tool_name, args)
    path = _tool_path(tool_name, args)
    return ToolCallUpdate(
        tool_call_id=tool_call_id,
        title=title,
        kind=_TOOL_KINDS.get(tool_name, "other"),
        # Clients approve edits by location, so the request names the real path.
        locations=[ToolCallLocation(path=path)] if path else None,
        raw_input=tool_args,
    )


def _plan_text(todos: list[dict[str, Any]]) -> str:
    """Render a proposed plan as the message shown above its approval request."""
    return "## Plan\n\n" + "".join(
        f"{i}. {todo.get('content', '')}\n" for i, todo in enumerate(todos, 1)
    )


def _all_tasks_completed(plan: list[dict[str, Any]]) -> bool:
    """Check if all tasks in a plan are completed.

    Args:
        plan: List of todo dictionaries

    Returns:
        True if all tasks have status 'completed', False otherwise
    """
    if not plan:
        return True

    return all(todo.get("status") == "completed" for todo in plan)
