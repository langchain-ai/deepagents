"""Exception type for tool arguments the model authored incorrectly.

Lives in its own module so both the tools that raise it and the agent wiring
that recovers from it can import it without a circular import.
"""

from __future__ import annotations


class ToolArgumentError(ValueError):
    """A tool rejected arguments the model authored.

    Raise this only when the model can fix the call by rewriting its arguments.
    `create_cli_agent` wires a `ToolErrorMiddleware` that turns this into an
    error `ToolMessage` the model can retry from. Every other exception, plain
    `ValueError` included, stays fatal.

    Subclasses `ValueError` so existing callers that catch `ValueError` around
    tool argument validation keep working.
    """
