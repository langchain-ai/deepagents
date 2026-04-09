"""Unified permission middleware for filesystem and tool access control.

Consolidates both ``FilesystemPermission`` and ``ToolPermission`` enforcement
into a single ``wrap_tool_call`` / ``awrap_tool_call`` boundary.
"""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

import wcmatch.fnmatch as wcfnmatch
import wcmatch.glob as wcglob
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ResponseT,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.backends.protocol import GlobResult, GrepResult, LsResult
from deepagents.backends.utils import (
    format_grep_matches,
    truncate_if_too_long,
    validate_path,
)
from deepagents.permissions import (
    FilesystemOperation,
    FilesystemPermission,
    ToolPermission,
)

# ---------------------------------------------------------------------------
# Glob flags
# ---------------------------------------------------------------------------

_FS_WCMATCH_FLAGS = wcglob.BRACE | wcglob.GLOBSTAR
_TOOL_WCMATCH_FLAGS = wcfnmatch.BRACE

# ---------------------------------------------------------------------------
# Default mapping: filesystem tool name → operation type
# ---------------------------------------------------------------------------

_DEFAULT_FS_TOOL_OPS: dict[str, FilesystemOperation] = {
    "ls": "read",
    "read_file": "read",
    "glob": "read",
    "grep": "read",
    "write_file": "write",
    "edit_file": "write",
}

# ---------------------------------------------------------------------------
# Pure check functions (stateless, reusable)
# ---------------------------------------------------------------------------


def _check_fs_permission(
    rules: list[FilesystemPermission],
    operation: FilesystemOperation,
    path: str,
) -> Literal["allow", "deny"]:
    """Evaluate filesystem permission rules for an operation on a path.

    Iterates rules in declaration order. The first matching rule's mode
    is returned. If no rule matches, returns ``"allow"`` (permissive default).

    Args:
        rules: Ordered list of ``FilesystemPermission`` rules to evaluate.
        operation: The operation being performed (``"read"`` or ``"write"``).
        path: The canonicalized absolute path being accessed.

    Returns:
        ``"allow"`` if the call should proceed, ``"deny"`` if it should be blocked.
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
    """Filter a list of paths to only those allowed by the permission rules.

    Args:
        rules: Ordered list of ``FilesystemPermission`` rules to evaluate.
        operation: The operation being performed (typically ``"read"``).
        paths: The raw list of paths to filter.

    Returns:
        The filtered list of allowed paths.
    """
    if not rules:
        return paths
    return [p for p in paths if _check_fs_permission(rules, operation, p) == "allow"]


def _check_tool_permission(
    rules: list[ToolPermission],
    tool_name: str,
    args: dict,
) -> Literal["allow", "deny"]:
    """Evaluate tool permission rules for a tool invocation.

    Iterates rules in declaration order. The first matching rule's mode
    is applied. If no rule matches, returns ``"allow"`` (permissive default).

    A rule matches when its ``name`` equals ``tool_name`` and every entry in
    ``args`` (if any) glob-matches the corresponding arg value.

    Args:
        rules: Ordered list of ``ToolPermission`` rules to evaluate.
        tool_name: The name of the tool being invoked.
        args: The arguments passed to the tool call.

    Returns:
        ``"allow"`` if the call should proceed, ``"deny"`` if it should be blocked.
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


# ---------------------------------------------------------------------------
# Unified middleware
# ---------------------------------------------------------------------------


class PermissionMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Unified middleware enforcing both filesystem and tool permission rules.

    Intercepts each tool call via ``wrap_tool_call`` / ``awrap_tool_call``.

    **Pre-check** (before tool execution):

    1. ``ToolPermission`` rules are evaluated against the tool name and args.
    2. For known filesystem tools, ``FilesystemPermission`` rules are evaluated
       against the path extracted from the tool args.

    **Post-filter** (after tool execution):

    For tools whose ``ToolMessage.artifact`` carries structured path data
    (``ls``, ``glob``, ``grep``), denied paths are filtered from the result
    and the content is rebuilt.

    This middleware must be placed **last** in the stack so it sees the final
    set of tools (including those injected by other middleware).

    Args:
        rules: Flat list of ``FilesystemPermission`` and ``ToolPermission``
            rules.  Rules are evaluated in declaration order; the first match
            wins.  If no rule matches, the call is allowed (permissive default).

    Example:
        ```python
        from deepagents.permissions import FilesystemPermission, ToolPermission
        from deepagents.middleware.permissions import PermissionMiddleware

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
        """Initialize the permission middleware.

        Args:
            rules: Flat list of permission rules. ``FilesystemPermission`` and
                ``ToolPermission`` instances may be mixed freely. Rules are
                evaluated in declaration order; the first match wins.
        """
        fs_rules: list[FilesystemPermission] = []
        tool_rules: list[ToolPermission] = []
        for r in rules:
            if isinstance(r, FilesystemPermission):
                fs_rules.append(r)
            elif isinstance(r, ToolPermission):
                tool_rules.append(r)
            else:
                msg = f"Unknown permission type: {type(r).__name__}"
                raise TypeError(msg)
        self._fs_rules = fs_rules
        self._tool_rules = tool_rules
        self._fs_tool_ops: dict[str, FilesystemOperation] = dict(_DEFAULT_FS_TOOL_OPS)

    # ------------------------------------------------------------------
    # Tool call: unified enforcement
    # ------------------------------------------------------------------

    def _pre_check(self, tool_name: str, tool_call_id: str | None, args: dict) -> ToolMessage | None:
        """Run tool and filesystem pre-checks.  Returns an error ToolMessage on deny, else None."""
        # 1. Tool permission rules
        if self._tool_rules and _check_tool_permission(self._tool_rules, tool_name, args) == "deny":
            return ToolMessage(
                content="Error: tool use denied by permission",
                name=tool_name,
                tool_call_id=tool_call_id,
            )

        # 2. Filesystem permission rules (path-based)
        if self._fs_rules and tool_name in self._fs_tool_ops:
            operation = self._fs_tool_ops[tool_name]
            path = args.get("file_path") or args.get("path")
            if path is not None:
                try:
                    canonical = validate_path(path)
                except ValueError:
                    # Let the tool handle the invalid path error itself
                    return None
                if _check_fs_permission(self._fs_rules, operation, canonical) == "deny":
                    return ToolMessage(
                        content=f"Error: permission denied for {operation} on {canonical}",
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
        return None

    def _post_filter(self, result: ToolMessage) -> ToolMessage:  # noqa: PLR0911
        """Filter denied paths from artifact-bearing ToolMessages.

        Artifacts are the backend result objects (``LsResult``, ``GlobResult``,
        ``GrepResult``) which allow faithful reconstruction of the content
        string after filtering.
        """
        artifact = result.artifact

        # Handle ls results
        if isinstance(artifact, LsResult):
            entries = artifact.entries or []
            paths = [fi.get("path", "") for fi in entries]
            filtered = _filter_paths_by_permission(self._fs_rules, "read", paths)
            if len(filtered) == len(paths):
                return result
            return ToolMessage(
                content=str(truncate_if_too_long(filtered)),
                tool_call_id=result.tool_call_id,
                name=result.name,
                id=result.id,
                status=result.status,
                additional_kwargs=dict(result.additional_kwargs),
                response_metadata=dict(result.response_metadata),
            )

        # Handle glob results
        if isinstance(artifact, GlobResult):
            matches = artifact.matches or []
            paths = [fi.get("path", "") for fi in matches]
            filtered = _filter_paths_by_permission(self._fs_rules, "read", paths)
            if len(filtered) == len(paths):
                return result
            return ToolMessage(
                content=str(truncate_if_too_long(filtered)),
                tool_call_id=result.tool_call_id,
                name=result.name,
                id=result.id,
                status=result.status,
                additional_kwargs=dict(result.additional_kwargs),
                response_metadata=dict(result.response_metadata),
            )

        # Handle grep results (dict wrapping GrepResult + output_mode)
        if isinstance(artifact, dict) and isinstance(artifact.get("result"), GrepResult):
            grep_result: GrepResult = artifact["result"]
            output_mode = artifact.get("output_mode", "files_with_matches")
            matches = grep_result.matches or []
            filtered = [m for m in matches if _check_fs_permission(self._fs_rules, "read", m.get("path", "")) == "allow"]
            if len(filtered) == len(matches):
                return result
            return ToolMessage(
                content=truncate_if_too_long(format_grep_matches(filtered, output_mode)),
                tool_call_id=result.tool_call_id,
                name=result.name,
                id=result.id,
                status=result.status,
                additional_kwargs=dict(result.additional_kwargs),
                response_metadata=dict(result.response_metadata),
            )

        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Enforce permission rules before and after tool execution.

        Args:
            request: The tool call request being processed.
            handler: The next handler in the chain.

        Returns:
            An error ``ToolMessage`` on deny, otherwise the (possibly filtered) handler result.
        """
        tool_name = request.tool_call["name"]
        args = request.tool_call.get("args", {}) or {}

        denial = self._pre_check(tool_name, request.tool_call["id"], args)
        if denial is not None:
            return denial

        result = handler(request)

        if self._fs_rules and isinstance(result, ToolMessage) and result.artifact:
            result = self._post_filter(result)

        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async) Enforce permission rules before and after tool execution.

        Args:
            request: The tool call request being processed.
            handler: The next handler in the chain.

        Returns:
            An error ``ToolMessage`` on deny, otherwise the (possibly filtered) handler result.
        """
        tool_name = request.tool_call["name"]
        args = request.tool_call.get("args", {}) or {}

        denial = self._pre_check(tool_name, request.tool_call["id"], args)
        if denial is not None:
            return denial

        result = await handler(request)

        if self._fs_rules and isinstance(result, ToolMessage) and result.artifact:
            result = self._post_filter(result)

        return result
