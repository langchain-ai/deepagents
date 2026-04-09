"""Permission rule types for filesystem access control."""

from dataclasses import dataclass
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
