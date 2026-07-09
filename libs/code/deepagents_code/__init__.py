"""Deep Agents Code - Interactive AI coding assistant."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents_code._debug import configure_debug_logging
from deepagents_code._debug_buffer import install_log_buffer
from deepagents_code._version import __version__

if TYPE_CHECKING:
    from collections.abc import Callable

configure_debug_logging(logging.getLogger(__name__))  # noqa: RUF067  # package logger must be configured before child modules emit logs
install_log_buffer(logging.getLogger(__name__))  # noqa: RUF067  # always-on tail must attach at import so early logs are captured

__all__ = [
    "__version__",
    "cli_main",  # noqa: F822  # resolved lazily by __getattr__
]


def __getattr__(name: str) -> Callable[[], None]:
    """Lazy import for `cli_main` to avoid loading `main.py` at package import.

    `main.py` pulls in `argparse`, signal handling, and other startup machinery
    that isn't needed when submodules like `config` or `widgets` are
    imported directly.

    Returns:
        The requested callable.

    Raises:
        AttributeError: If *name* is not a lazily-provided attribute.
    """
    if name == "cli_main":
        from deepagents_code.main import cli_main

        return cli_main
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
