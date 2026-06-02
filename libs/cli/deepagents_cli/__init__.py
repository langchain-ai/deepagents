"""Deep Agents CLI - deployment tooling (`init`, `dev`, `deploy`).

For the interactive coding agent, install the `deepagents-code` package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_cli._version import __version__

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "__version__",
    "cli_main",  # noqa: F822  # resolved lazily by __getattr__
]


def __getattr__(name: str) -> Callable[[], None]:
    """Lazy import for `cli_main` to keep `import deepagents_cli` cheap.

    Args:
        name: Attribute being looked up on the package.

    Returns:
        The requested attribute (only `cli_main` is provided lazily).

    Raises:
        AttributeError: If `name` is not a lazily-provided attribute.
    """
    if name == "cli_main":
        from deepagents_cli.main import cli_main

        return cli_main
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
