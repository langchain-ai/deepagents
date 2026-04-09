"""Permission middleware for filesystem access control.

Enforces ``FilesystemPermission`` rules via ``wrap_tool_call`` /
``awrap_tool_call``.
"""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

import wcmatch.glob as wcglob
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ResponseT,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol, GlobResult, GrepResult, LsResult
from deepagents.backends.utils import (
    format_grep_matches,
    truncate_if_too_long,
    validate_path,
)
from deepagents.middleware.filesystem import supports_execution
from deepagents.permissions import (
    FilesystemOperation,
    FilesystemPermission,
)

# ---------------------------------------------------------------------------
# Glob flags
# ---------------------------------------------------------------------------

_FS_WCMATCH_FLAGS = wcglob.BRACE | wcglob.GLOBSTAR

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


class PermissionMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware enforcing filesystem permission rules.

    Intercepts each tool call via ``wrap_tool_call`` / ``awrap_tool_call``.

    **Pre-check** (before tool execution):

    For known filesystem tools, ``FilesystemPermission`` rules are evaluated
    against the path extracted from the tool args.

    **Post-filter** (after tool execution):

    For tools whose ``ToolMessage.artifact`` carries structured path data
    (``ls``, ``glob``, ``grep``), denied paths are filtered from the result
    and the content is rebuilt.

    This middleware must be placed **last** in the stack so it sees the final
    set of tools (including those injected by other middleware).

    Args:
        rules: List of ``FilesystemPermission`` rules. Rules are evaluated in
            declaration order; the first match wins. If no rule matches, the
            call is allowed (permissive default).

    Example:
        ```python
        from deepagents.permissions import FilesystemPermission
        from deepagents.middleware.permissions import PermissionMiddleware

        middleware = PermissionMiddleware(
            rules=[
                FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
            ]
        )
        ```
    """

    def __init__(self, *, rules: list[FilesystemPermission], backend: BACKEND_TYPES) -> None:
        """Initialize the permission middleware.

        Args:
            rules: List of ``FilesystemPermission`` rules. Rules are evaluated
                in declaration order; the first match wins.
            backend: The backend instance. If it supports execution
                (``SandboxBackendProtocol``), a ``NotImplementedError`` is
                raised because tool-level permissions for the ``execute``
                tool are not yet implemented.

        Raises:
            NotImplementedError: If the backend supports command execution.
        """
        if isinstance(backend, BackendProtocol) and supports_execution(backend):
            msg = (
                "PermissionMiddleware does not yet support backends with command "
                "execution (SandboxBackendProtocol). Tool-level permissions for "
                "the execute tool are not implemented. Either remove permissions "
                "or use a backend without execution support."
            )
            raise NotImplementedError(msg)
        self._fs_rules = list(rules)
        self._fs_tool_ops: dict[str, FilesystemOperation] = dict(_DEFAULT_FS_TOOL_OPS)

    # ------------------------------------------------------------------
    # Tool call: enforcement
    # ------------------------------------------------------------------

    def _pre_check(self, tool_name: str, tool_call_id: str | None, args: dict) -> ToolMessage | None:
        """Run filesystem pre-checks.  Returns an error ToolMessage on deny, else None."""
        if self._fs_rules and tool_name in self._fs_tool_ops:
            operation = self._fs_tool_ops[tool_name]
            path = args.get("file_path") if "file_path" in args else args.get("path")
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
