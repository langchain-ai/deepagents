"""Glue between `FilesystemPermission` rules and `HumanInTheLoopMiddleware`.

`FilesystemMiddleware` itself doesn't know about HITL — it only enforces deny
rules and filters denied results. The graph-assembly code in
`deepagents.graph` calls `_build_interrupt_on_from_permissions` to turn the
filesystem permissions into an `interrupt_on` mapping for
`HumanInTheLoopMiddleware`, using a `when` predicate that decides per call
whether the access intersects an interrupt-mode rule.
"""

from collections.abc import Callable
from typing import Literal

from langchain.agents.middleware import InterruptOnConfig
from langchain.tools.tool_node import ToolCallRequest

from deepagents.backends.utils import _glob_anchor, _paths_overlap, validate_path
from deepagents.middleware.filesystem import FilesystemOperation, FilesystemPermission, _check_fs_permission

# Scope of a filesystem tool's path argument:
#   - "exact": the call operates on exactly the named path (read_file,
#     write_file, edit_file). Interrupt fires iff that path matches an
#     interrupt-mode rule.
#   - "bulk":  the call's path argument names a search root and the call may
#     surface any descendant (ls, glob, grep). Interrupt fires whenever the
#     search subtree intersects an interrupt-mode rule's pattern, and — when
#     the path argument is omitted (`grep(path=None)`) — fires unconditionally
#     for any interrupt-mode rule, because a pathless bulk call can touch
#     anything.
ToolScope = Literal["exact", "bulk"]

# Map filesystem tool name → (operation, path-arg name, scope). Drives
# `_build_interrupt_on_from_permissions` when synthesizing `when` predicates
# per tool.
_FS_TOOL_PATH_ARGS: dict[str, tuple[FilesystemOperation, str, ToolScope]] = {
    "ls": ("read", "path", "bulk"),
    "read_file": ("read", "file_path", "exact"),
    "write_file": ("write", "file_path", "exact"),
    "edit_file": ("write", "file_path", "exact"),
    "glob": ("read", "path", "bulk"),
    "grep": ("read", "path", "bulk"),
}


def _make_fs_when_predicate(
    rules: list[FilesystemPermission],
    operation: FilesystemOperation,
    path_arg_name: str,
    scope: ToolScope,
) -> Callable[[ToolCallRequest], bool]:
    """Build a `when` predicate that fires on interrupt-mode rule matches.

    The predicate's behavior depends on the tool's `ToolScope`:

    - `"exact"`: fire iff the call's path matches an interrupt-mode rule
      with normal first-match precedence. A preceding `deny` rule wins and
      the interrupt does not fire — the tool returns a permission-denied
      error instead.
    - `"bulk"`: fire iff the call's search subtree could intersect an
      interrupt-mode rule. With no path argument (e.g. `grep(path=None)`)
      we cannot localize the call, so we fire unconditionally for any
      interrupt-mode rule on the operation.
    """
    if scope == "exact":
        return _make_exact_when_predicate(rules, operation, path_arg_name)
    return _make_bulk_when_predicate(rules, operation, path_arg_name)


def _make_exact_when_predicate(
    rules: list[FilesystemPermission],
    operation: FilesystemOperation,
    path_arg_name: str,
) -> Callable[[ToolCallRequest], bool]:
    def when(req: ToolCallRequest) -> bool:
        raw_path = req.tool_call.get("args", {}).get(path_arg_name)
        if not isinstance(raw_path, str):
            return False
        try:
            normalized = validate_path(raw_path)
        except ValueError:
            return False
        return _check_fs_permission(rules, operation, normalized) == "interrupt"

    return when


def _make_bulk_when_predicate(
    rules: list[FilesystemPermission],
    operation: FilesystemOperation,
    path_arg_name: str,
) -> Callable[[ToolCallRequest], bool]:
    # Precompute interrupt-mode rule anchors for this op so the predicate is
    # a single pass per call.
    interrupt_anchors: list[str] = [
        _glob_anchor(pattern) for rule in rules if rule.mode == "interrupt" and operation in rule.operations for pattern in rule.paths
    ]

    def when(req: ToolCallRequest) -> bool:
        if not interrupt_anchors:
            return False
        raw_path = req.tool_call.get("args", {}).get(path_arg_name)
        if raw_path is None:
            # Pathless bulk call — fire because we cannot localize the access.
            return True
        if not isinstance(raw_path, str):
            return False
        try:
            normalized = validate_path(raw_path)
        except ValueError:
            return False
        # `validate_path` returns `/.` for current-directory aliases like
        # `"."`, `""`, and `"./"`. Those refer to the whole accessible tree
        # just like a missing path arg, so collapse to `/` so the
        # root-overlaps-everything branch in `_paths_overlap` fires. Without
        # this, an agent could pass `path="."` to bypass HITL.
        if normalized == "/.":
            normalized = "/"
        return any(_paths_overlap(normalized, anchor) for anchor in interrupt_anchors)

    return when


def _build_interrupt_on_from_permissions(
    rules: list[FilesystemPermission],
) -> dict[str, InterruptOnConfig]:
    """Generate `interrupt_on` configs from interrupt-mode permissions.

    Returns an entry for each filesystem tool whose operation could be triggered
    by at least one interrupt-mode rule. Each entry uses a `when` predicate so
    the interrupt only fires when the tool call's path argument matches an
    interrupt-mode rule.
    """
    if not any(r.mode == "interrupt" for r in rules):
        return {}

    # Annotated so ty narrows to `list[DecisionType]` instead of `list[str]`.
    allowed: list[Literal["approve", "edit", "reject", "respond"]] = ["approve", "reject"]
    result: dict[str, InterruptOnConfig] = {}
    for tool_name, (op, arg, scope) in _FS_TOOL_PATH_ARGS.items():
        if not any(r.mode == "interrupt" and op in r.operations for r in rules):
            continue
        result[tool_name] = InterruptOnConfig(
            allowed_decisions=allowed,
            when=_make_fs_when_predicate(rules, op, arg, scope),
        )
    return result
