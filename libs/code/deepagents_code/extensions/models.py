"""Data types shared by the extension discovery, loading, and registry layers.

`SourceInfo` is the provenance record attached to every registered unit. It
exists for display, filtering, and enable/disable decisions — never for
collision precedence, which is decided by load order alone (see
`registry.ExtensionRegistry`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from pathlib import Path

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.tools import BaseTool


class UnitSource(StrEnum):
    """Where a registered unit came from."""

    BUILTIN = "builtin"
    """Shipped with dcode."""

    EXTENSION = "extension"
    """Registered by a loaded extension."""

    SDK = "sdk"
    """Registered by an embedding application through the public API."""


class UnitScope(StrEnum):
    """Configuration scope an extension was discovered in."""

    USER = "user"
    """Global/user-level source (e.g. `~/.deepagents/extensions/`)."""

    PROJECT = "project"
    """Project-level source (e.g. `.deepagents/extensions/`); trust-gated."""

    TEMPORARY = "temporary"
    """Supplied for a single run (e.g. an explicit path)."""


class UnitOrigin(StrEnum):
    """Whether a unit came from a packaged extension or a top-level file."""

    PACKAGE = "package"
    TOP_LEVEL = "top-level"


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Provenance for one registered unit."""

    path: Path | None
    """File the unit was loaded from; `None` for built-in units."""

    source: UnitSource
    """Broad category of the registering party."""

    scope: UnitScope
    """Configuration scope the extension was discovered in."""

    origin: UnitOrigin
    """Whether the extension is a package directory or a single file."""

    @property
    def label(self) -> str:
        """Short human-readable provenance label.

        Returns:
            The extension's file or directory name, or the source name when the
                unit has no path.
        """
        if self.path is None:
            return str(self.source)
        if self.origin is UnitOrigin.PACKAGE:
            return self.path.parent.name
        return self.path.stem


@dataclass(frozen=True, slots=True)
class ExtensionFile:
    """One discovered extension entry file, with the provenance it will carry."""

    path: Path
    source: SourceInfo


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RegisteredUnit(Generic[T]):
    """A registered unit paired with its provenance."""

    name: str
    unit: T
    source: SourceInfo


class CommandHandler(Protocol):
    """Signature an extension command handler must satisfy."""

    def __call__(self, ctx: CommandContext) -> Awaitable[str | None] | str | None:
        """Run the command.

        Args:
            ctx: Invocation context (arguments, working directory, mode).

        Returns:
            Optional text to display to the user.
        """
        ...


@dataclass(frozen=True, slots=True)
class CommandContext:
    """Context handed to an extension command handler when it runs."""

    args: str
    """Raw argument string typed after the command name."""

    cwd: Path
    """Working directory of the session."""

    mode: str
    """Runtime mode: `interactive` or `headless`."""

    data: dict[str, Any] = field(default_factory=dict)
    """Free-form data the host may attach; reserved for future host services."""


@dataclass(frozen=True, slots=True)
class LoadedExtension:
    """Result of loading a single extension file."""

    path: Path
    source: SourceInfo
    middleware: tuple[AgentMiddleware[Any, Any], ...] = ()
    tools: tuple[BaseTool, ...] = ()
    commands: tuple[str, ...] = ()
    backend_routes: tuple[str, ...] = ()
    """Path prefixes this extension mounted a backend route at."""


class ExtensionError(Exception):
    """Raised when an extension file cannot be imported or initialized."""
