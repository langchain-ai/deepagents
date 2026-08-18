"""Shared data types for Python extension discovery and registration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.tools import BaseTool


class UnitScope(StrEnum):
    """Configuration scope where an extension was discovered."""

    USER = "user"
    PROJECT = "project"


class UnitOrigin(StrEnum):
    """Shape of an extension on disk."""

    PACKAGE = "package"
    TOP_LEVEL = "top-level"


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Provenance attached to every extension registration."""

    path: Path
    """Entry file that registered the unit."""

    scope: UnitScope
    """Configuration scope that supplied the extension."""

    origin: UnitOrigin
    """Whether the extension is a package or top-level file."""

    @property
    def label(self) -> str:
        """Short human-readable extension label."""
        if self.origin is UnitOrigin.PACKAGE:
            return self.path.parent.name
        return self.path.stem


@dataclass(frozen=True, slots=True)
class ExtensionFile:
    """Discovered extension entry file and its provenance."""

    path: Path
    source: SourceInfo


@dataclass(frozen=True, slots=True)
class RegisteredUnit[T]:
    """Registered extension unit paired with its provenance."""

    name: str
    unit: T
    source: SourceInfo


@dataclass(frozen=True, slots=True)
class LoadedExtension:
    """Registrations produced by one successfully loaded extension."""

    path: Path
    source: SourceInfo
    middleware: tuple[AgentMiddleware[Any, Any], ...] = ()
    tools: tuple[BaseTool, ...] = ()
    backend_routes: tuple[str, ...] = ()


class ExtensionError(Exception):
    """Raised when an extension cannot be imported or initialized."""
