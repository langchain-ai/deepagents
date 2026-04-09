"""Permission rule types for filesystem and tool access control."""

from dataclasses import dataclass, field
from typing import Literal

FilesystemOperation = Literal["read", "write"]
"""Operation type for filesystem permission rules.

- `read`: covers `ls`, `read_file`, `glob`, `grep`
- `write`: covers `write_file`, `edit_file`
"""


@dataclass
class FilesystemPermission:
    """A single access rule for filesystem operations.

    Rules are evaluated in declaration order. The first matching rule's
    `mode` is applied. If no rule matches, the call is allowed (permissive
    default).

    Args:
        operations: Operations this rule applies to. `"read"` covers
            `ls`, `read_file`, `glob`, `grep`. `"write"` covers
            `write_file`, `edit_file`.
        paths: Glob patterns for matching file paths
            (e.g. `["/workspace/**", "/tmp/*.log"]`). Uses
            `wcmatch` with `BRACE | GLOBSTAR` flags. Paths are
            canonicalized before matching to prevent traversal bypasses.
        mode: Whether to allow or deny matching calls.

    Example:
        ```python
        from deepagents.permissions import FilesystemPermission

        # Deny all writes anywhere
        FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")

        # Allow reads only under /workspace
        FilesystemPermission(operations=["read"], paths=["/workspace/**"])
        ```
    """

    operations: list[FilesystemOperation]
    paths: list[str]
    mode: Literal["allow", "deny"] = "allow"


@dataclass
class ToolPermission:
    """A single rule governing invocations of one tool.

    Rules are evaluated in declaration order. The first matching rule's
    `mode` is applied. If no rule matches, the call is allowed (permissive
    default).

    Args:
        name: Exact tool name this rule applies to (e.g. `"execute"`, `"grep"`).
        args: Optional argument-level constraints. Maps argument name to
            a glob pattern matched against the actual argument value at call
            time. All entries must match for the rule to apply. If `None`,
            the rule matches all invocations of the tool.
        mode: Whether to allow or deny matching calls.

    Example:
        ```python
        from deepagents.permissions import ToolPermission

        # Allow only pytest invocations, deny everything else
        ToolPermission(name="execute", args={"command": "pytest *"})
        ToolPermission(name="execute", mode="deny")

        # Disable a tool entirely
        ToolPermission(name="execute", mode="deny")
        ```
    """

    name: str
    args: dict[str, str] | None = field(default=None)
    mode: Literal["allow", "deny"] = "allow"
