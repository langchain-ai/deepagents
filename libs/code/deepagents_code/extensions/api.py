"""The public contract extensions are written against.

An extension is a Python module exposing a single factory callable::

    def extension(d: ExtensionAPI) -> None:
        d.register_tool(my_tool)

The factory may be `async def`; dcode awaits it before the session starts. Every
unit kind is registered through one explicit verb on `ExtensionAPI`, and the
factory signature never changes as new unit kinds are added.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from deepagents_code.extensions.models import ExtensionError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.tools import BaseTool

    from deepagents_code.extensions.models import CommandHandler, SourceInfo
    from deepagents_code.extensions.registry import ExtensionRegistry

logger = logging.getLogger(__name__)

_COMMAND_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
"""Command names are restricted so a registration cannot forge slash syntax."""


class ExtensionAPI:
    """Registrar and session context handed to an extension factory.

    One instance is created per extension file, so every registration is
    attributed to the file it came from without the author passing provenance.
    """

    def __init__(
        self,
        registry: ExtensionRegistry,
        source: SourceInfo,
        *,
        cwd: Path,
        mode: str,
    ) -> None:
        """Bind a registrar to one extension's provenance.

        Args:
            registry: Shared registry receiving the registrations.
            source: Provenance recorded on every unit this instance registers.
            cwd: Working directory of the session.
            mode: Runtime mode: `interactive` or `headless`.
        """
        self._registry = registry
        self._source = source
        self._cwd = cwd
        self._mode = mode

    @property
    def cwd(self) -> Path:
        """Working directory of the session."""
        return self._cwd

    @property
    def mode(self) -> str:
        """Runtime mode: `interactive` or `headless`."""
        return self._mode

    @property
    def path(self) -> Path | None:
        """File this extension was loaded from."""
        return self._source.path

    def register_middleware(
        self,
        middleware: AgentMiddleware[Any, Any] | type[AgentMiddleware[Any, Any]],
    ) -> None:
        """Install a LangChain `AgentMiddleware` on the agent.

        Interception semantics (wrap, chain, short-circuit) are LangChain's;
        dcode adds no parallel event taxonomy. A class is instantiated with no
        arguments so dcode owns its lifecycle; pass an instance when the author
        needs to control construction.

        Args:
            middleware: Middleware class or instance.

        Raises:
            ExtensionError: If a class cannot be instantiated without arguments.
        """
        instance: AgentMiddleware[Any, Any]
        if isinstance(middleware, type):
            try:
                instance = middleware()
            except TypeError as exc:
                msg = (
                    f"Middleware {middleware.__name__} needs constructor "
                    "arguments; register an instance instead"
                )
                raise ExtensionError(msg) from exc
        else:
            instance = middleware
        self._registry.add_middleware(instance, self._source)

    def register_tool(self, tool: BaseTool | Callable[..., Any]) -> None:
        """Expose an LLM-callable tool.

        A plain function is converted with LangChain's `tool` decorator, which
        derives the schema from the signature and docstring. Pass a `BaseTool`
        when the schema needs to be declared explicitly.

        Args:
            tool: Tool instance or plain callable.

        Raises:
            ExtensionError: If a callable cannot be converted into a tool.
        """
        # Deferred: LangChain is heavy on the startup path.
        from langchain_core.tools import (
            BaseTool,
            tool as as_tool,
        )

        if isinstance(tool, BaseTool):
            self._registry.add_tool(tool, self._source)
            return

        try:
            converted = as_tool(tool)
        except Exception as exc:
            name = getattr(tool, "__name__", repr(tool))
            msg = f"Could not convert {name} into a tool: {exc}"
            raise ExtensionError(msg) from exc
        self._registry.add_tool(converted, self._source)  # type: ignore[arg-type]  # `tool()` returns a `BaseTool` for a plain callable

    def on_shutdown(self, hook: Callable[[], Any]) -> None:
        """Register a teardown callback for deterministic session cleanup.

        Called once when the session ends, in load order. Not a unit kind — it is
        the lifecycle counterpart of the async factory, so session-scoped
        resources (connections, temp files) are released even when the session
        ends abruptly.

        Args:
            hook: Callable invoked at shutdown; may be async.

        Raises:
            ExtensionError: If the hook is not callable.
        """
        if not callable(hook):
            msg = "Shutdown hook is not callable"
            raise ExtensionError(msg)
        self._registry.add_shutdown_hook(hook, self._source)

    def register_command(
        self,
        name: str,
        handler: CommandHandler,
        *,
        description: str = "",
    ) -> str:
        """Register an executable `/name` handler.

        Args:
            name: Command name without the leading slash.
            handler: Callable receiving a `CommandContext`; may be async.
            description: Short user-facing description for autocomplete.

        Returns:
            The name the command was registered under; a colliding name is
                suffixed rather than replacing the existing command.

        Raises:
            ExtensionError: If the name is not a lowercase slug or the handler
                is not callable.
        """
        normalized = name.lstrip("/").strip().lower()
        if not _COMMAND_NAME.match(normalized):
            msg = (
                f"Invalid command name {name!r}: use lowercase letters, digits, "
                "hyphens, and underscores"
            )
            raise ExtensionError(msg)
        if not callable(handler):
            msg = f"Handler for command {normalized!r} is not callable"
            raise ExtensionError(msg)
        return self._registry.add_command(
            normalized,
            handler,
            self._source,
            description=description,
        )
