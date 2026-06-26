"""Glue between `FilesystemPermission` rules and `HumanInTheLoopMiddleware`.

`FilesystemMiddleware` itself doesn't know about HITL — it only enforces deny
rules and filters denied results. The graph-assembly code in
`deepagents.graph` calls `_build_interrupt_on_from_permissions` to turn the
filesystem permissions into an `interrupt_on` mapping for
`HumanInTheLoopMiddleware`, using a `when` predicate that decides per call
whether the access intersects an interrupt-mode rule.
"""

from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Literal

from langchain.agents.middleware import InterruptOnConfig
from langchain.tools.tool_node import ToolCallRequest

from deepagents.backends.utils import _glob_anchor, _paths_overlap, to_posix_path, validate_path
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

# Map filesystem tool name → (operation, path-arg name, scope, pattern-arg name).
# Drives `_build_interrupt_on_from_permissions` when synthesizing `when`
# predicates per tool. The optional pattern-arg name is set only for `glob`,
# whose `pattern` argument can itself redirect the search root (an absolute
# pattern ignores the call's `path`); see `_make_bulk_when_predicate`.
_FS_TOOL_PATH_ARGS: dict[str, tuple[FilesystemOperation, str, ToolScope, str | None]] = {
    "ls": ("read", "path", "bulk", None),
    "read_file": ("read", "file_path", "exact", None),
    "write_file": ("write", "file_path", "exact", None),
    "edit_file": ("write", "file_path", "exact", None),
    "glob": ("read", "path", "bulk", "pattern"),
    "grep": ("read", "path", "bulk", None),
}


def _make_fs_when_predicate(
    rules: list[FilesystemPermission],
    operation: FilesystemOperation,
    path_arg_name: str,
    scope: ToolScope,
    pattern_arg_name: str | None = None,
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
        interrupt-mode rule on the operation. `pattern_arg_name` (set for
        `glob`) additionally gates the call's `pattern`, which can redirect
        the search root independently of `path`.
    """
    if scope == "exact":
        return _make_exact_when_predicate(rules, operation, path_arg_name)
    return _make_bulk_when_predicate(rules, operation, path_arg_name, pattern_arg_name)


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
    pattern_arg_name: str | None = None,
) -> Callable[[ToolCallRequest], bool]:
    # Precompute interrupt-mode rule anchors for this op so the predicate is
    # a single pass per call.
    interrupt_anchors: list[str] = [
        _glob_anchor(pattern) for rule in rules if rule.mode == "interrupt" and operation in rule.operations for pattern in rule.paths
    ]

    def when(req: ToolCallRequest) -> bool:
        if not interrupt_anchors:
            return False
        args = req.tool_call.get("args", {})
        raw_path = args.get(path_arg_name)
        if not isinstance(raw_path, str):
            # A missing path (pathless bulk call) can't be localized, so fire;
            # any other non-string is malformed, so don't.
            return raw_path is None
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
        if any(_paths_overlap(normalized, anchor) for anchor in interrupt_anchors):
            return True
        # `glob`'s `pattern` can redirect the search root away from `path`, so
        # gating on `path` alone would let `glob(pattern="/secrets/**",
        # path="/workspace")` bypass an interrupt rule on `/secrets/**`.
        if pattern_arg_name is not None:
            raw_pattern = args.get(pattern_arg_name)
            if isinstance(raw_pattern, str) and _bulk_pattern_fires(raw_pattern, interrupt_anchors):
                return True
        return False

    return when


def _bulk_pattern_fires(raw_pattern: str, interrupt_anchors: list[str]) -> bool:
    """Whether a glob `pattern` reaches an interrupt-mode subtree regardless of `path`.

    An absolute pattern is matched from its own root — Python's `glob` ignores
    the backend's `os.chdir(path)` — so gate on the pattern's anchor. A relative
    pattern containing `..` can climb out of `path`; we cannot localize where it
    lands, so treat it as firing. Absoluteness comes from the raw pattern, not
    `_glob_anchor`: the anchor of a leading-wildcard relative pattern (`*.txt`)
    collapses to `/`, which would otherwise look absolute.
    """
    posix_pattern = to_posix_path(raw_pattern)
    if posix_pattern.startswith("/"):
        return any(_paths_overlap(_glob_anchor(raw_pattern), anchor) for anchor in interrupt_anchors)
    return ".." in PurePosixPath(posix_pattern).parts


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

    # Offer the approver the full decision set, matching the default for
    # user-supplied `interrupt_on` tools. All four are human-controlled, so the
    # human stays the authorization gate: `edit`ed calls still re-enter the tool
    # and hit its pre-execution deny check, and `respond` skips execution.
    # Annotated so ty narrows to `list[DecisionType]` instead of `list[str]`.
    allowed: list[Literal["approve", "edit", "reject", "respond"]] = ["approve", "edit", "reject", "respond"]
    result: dict[str, InterruptOnConfig] = {}
    for tool_name, (op, arg, scope, pattern_arg) in _FS_TOOL_PATH_ARGS.items():
        if not any(r.mode == "interrupt" and op in r.operations for r in rules):
            continue
        result[tool_name] = InterruptOnConfig(
            allowed_decisions=allowed,
            when=_make_fs_when_predicate(rules, op, arg, scope, pattern_arg),
        )
    return result
