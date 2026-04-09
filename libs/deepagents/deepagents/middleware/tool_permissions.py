"""Middleware for enforcing FilesystemPermission and ToolPermission rules."""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

import wcmatch.fnmatch as wcfnmatch
import wcmatch.glob as wcglob
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.backends.utils import (
    format_grep_matches,
    truncate_if_too_long,
    validate_path,
)
from deepagents.permissions import FilesystemOperation, FilesystemPermission, ToolPermission

_TOOL_WCMATCH_FLAGS = wcfnmatch.BRACE
_FS_WCMATCH_FLAGS = wcglob.BRACE | wcglob.GLOBSTAR

# Maps filesystem tool names to (operation, path_arg_key).
_FS_TOOL_MAP: dict[str, tuple[FilesystemOperation, str]] = {
    "read_file": ("read", "file_path"),
    "write_file": ("write", "file_path"),
    "edit_file": ("write", "file_path"),
    "ls": ("read", "path"),
    "glob": ("read", "path"),
    "grep": ("read", "path"),
}

# Tools whose results contain multiple paths and need post-filtering.
_POST_FILTER_TOOLS = frozenset({"ls", "glob", "grep"})


def _check_tool_permission(
    rules: list[ToolPermission],
    tool_name: str,
    args: dict,
) -> Literal["allow", "deny"]:
    """Evaluate tool permission rules for a tool invocation.

    Iterates rules in declaration order. The first matching rule's mode
    is applied. If no rule matches, returns `"allow"` (permissive default).

    A rule matches when its `name` equals `tool_name` and every entry in
    `args` (if any) glob-matches the corresponding arg value.

    Args:
        rules: Ordered list of `ToolPermission` rules to evaluate.
        tool_name: The name of the tool being invoked.
        args: The arguments passed to the tool call.

    Returns:
        `"allow"` if the call should proceed, `"deny"` if it should be blocked.
    """
    for rule in rules:
        if rule.name != tool_name:
            continue
        if rule.args is not None and not all(
            wcfnmatch.fnmatch(str(args.get(arg_name, "")), pattern, flags=_TOOL_WCMATCH_FLAGS) for arg_name, pattern in rule.args.items()
        ):
            continue
        return rule.mode
    return "allow"


def _check_fs_permission(
    rules: list[FilesystemPermission],
    operation: FilesystemOperation,
    path: str,
) -> Literal["allow", "deny"]:
    """Evaluate filesystem permission rules for an operation on a path.

    Iterates rules in declaration order. The first matching rule's mode
    is returned. If no rule matches, returns `"allow"` (permissive default).

    Args:
        rules: Ordered list of `FilesystemPermission` rules to evaluate.
        operation: The operation being performed (`"read"` or `"write"`).
        path: The canonicalized absolute path being accessed.

    Returns:
        `"allow"` if the call should proceed, `"deny"` if it should be blocked.
    """
    for rule in rules:
        if operation not in rule.operations:
            continue
        if any(wcglob.globmatch(path, pattern, flags=_FS_WCMATCH_FLAGS) for pattern in rule.paths):
            return rule.mode
    return "allow"


def _filter_paths_by_permission(
    rules: list[FilesystemPermission],
    operation: FilesystemOperation,
    paths: list[str],
) -> list[str]:
    """Filter a list of paths to only those allowed by the permission rules."""
    if not rules:
        return paths
    return [p for p in paths if _check_fs_permission(rules, operation, p) == "allow"]


class PermissionMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Unified middleware that enforces both `FilesystemPermission` and `ToolPermission` rules.

    Intercepts each tool call via `wrap_tool_call` / `awrap_tool_call`:

    1. **Tool permission pre-check** -- denies calls matching a `ToolPermission` deny rule.
    2. **Filesystem permission pre-check** -- denies filesystem tool calls whose path
       argument is denied by a `FilesystemPermission` rule.
    3. **Post-filtering** -- for ``ls``, ``glob``, and ``grep`` results, filters out
       paths denied by `FilesystemPermission` rules using the tool result's artifact.

    This middleware should be placed last in the stack so it sees the final set
    of tools (including those injected by other middleware).

    Args:
        rules: Mixed list of `FilesystemPermission` and `ToolPermission` rules.
            Rules are evaluated in declaration order per type; the first match
            wins. If no rule matches, the call is allowed (permissive default).

    Example:
        ```python
        from deepagents.permissions import FilesystemPermission, ToolPermission
        from deepagents.middleware.tool_permissions import PermissionMiddleware

        middleware = PermissionMiddleware(
            rules=[
                FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
                ToolPermission(name="execute", args={"command": "pytest *"}),
                ToolPermission(name="execute", mode="deny"),
            ]
        )
        ```
    """

    def __init__(self, *, rules: list[FilesystemPermission | ToolPermission]) -> None:
        """Initialize with a mixed list of permission rules, split by type internally."""
        self._tool_rules: list[ToolPermission] = []
        self._fs_rules: list[FilesystemPermission] = []
        for r in rules:
            if isinstance(r, ToolPermission):
                self._tool_rules.append(r)
            elif isinstance(r, FilesystemPermission):
                self._fs_rules.append(r)
            else:
                msg = f"Unknown permission type: {type(r).__name__}"
                raise TypeError(msg)

    # -- model call: pass through ------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Pass through -- permission checks happen in wrap_tool_call."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Pass through -- permission checks happen in awrap_tool_call."""
        return await handler(request)

    # -- tool call: enforce permissions -------------------------------------------

    def _fs_pre_check(self, tool_name: str, args: dict, tool_call_id: str) -> ToolMessage | None:
        """Return a denial ToolMessage if the filesystem path arg is denied, else None."""
        fs_info = _FS_TOOL_MAP.get(tool_name)
        if not fs_info or not self._fs_rules:
            return None
        operation, path_key = fs_info
        path_val = args.get(path_key)
        if path_val is None:
            return None
        try:
            validated = validate_path(path_val)
        except ValueError:
            return None  # let the tool handle invalid paths
        if _check_fs_permission(self._fs_rules, operation, validated) == "deny":
            return ToolMessage(
                content=f"Error: permission denied for {operation} on {validated}",
                name=tool_name,
                tool_call_id=tool_call_id,
            )
        return None

    def _post_filter(self, tool_name: str, args: dict, msg: ToolMessage) -> ToolMessage:
        """Filter paths from ls/glob/grep artifacts using filesystem permission rules."""
        if not self._fs_rules or tool_name not in _POST_FILTER_TOOLS:
            return msg
        artifact = getattr(msg, "artifact", None)
        if artifact is None:
            return msg

        if tool_name in ("ls", "glob"):
            # artifact is list[str] of paths
            filtered = _filter_paths_by_permission(self._fs_rules, "read", artifact)
            content = str(truncate_if_too_long(filtered))
            return ToolMessage(
                content=content,
                artifact=filtered,
                name=msg.name,
                tool_call_id=msg.tool_call_id,
            )

        if tool_name == "grep":
            # artifact is list[GrepMatch] (dicts with "path", "line", "text")
            filtered = [m for m in artifact if _check_fs_permission(self._fs_rules, "read", m.get("path", "")) == "allow"]
            output_mode = args.get("output_mode", "files_with_matches")
            formatted = format_grep_matches(filtered, output_mode)
            content = truncate_if_too_long(formatted)
            return ToolMessage(
                content=content,
                artifact=filtered,
                name=msg.name,
                tool_call_id=msg.tool_call_id,
            )

        return msg

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Enforce permission rules: pre-check, execute, post-filter."""
        tool_name = request.tool_call["name"]
        args = request.tool_call.get("args", {}) or {}

        # 1. Tool permission pre-check
        if _check_tool_permission(self._tool_rules, tool_name, args) == "deny":
            return ToolMessage(
                content="Error: tool use denied by permission",
                name=tool_name,
                tool_call_id=request.tool_call["id"],
            )

        # 2. Filesystem permission pre-check
        tool_call_id: str = request.tool_call.get("id") or ""
        denial = self._fs_pre_check(tool_name, args, tool_call_id)
        if denial is not None:
            return denial

        # 3. Execute tool
        result = handler(request)

        # 4. Post-filter for ls/glob/grep
        if isinstance(result, ToolMessage):
            result = self._post_filter(tool_name, args, result)

        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Enforce permission rules (async): pre-check, execute, post-filter."""
        tool_name = request.tool_call["name"]
        args = request.tool_call.get("args", {}) or {}

        # 1. Tool permission pre-check
        if _check_tool_permission(self._tool_rules, tool_name, args) == "deny":
            return ToolMessage(
                content="Error: tool use denied by permission",
                name=tool_name,
                tool_call_id=request.tool_call["id"],
            )

        # 2. Filesystem permission pre-check
        tool_call_id: str = request.tool_call.get("id") or ""
        denial = self._fs_pre_check(tool_name, args, tool_call_id)
        if denial is not None:
            return denial

        # 3. Execute tool
        result = await handler(request)

        # 4. Post-filter for ls/glob/grep
        if isinstance(result, ToolMessage):
            result = self._post_filter(tool_name, args, result)

        return result
