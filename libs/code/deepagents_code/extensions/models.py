"""Internal data types for Python extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path


ExtensionScope = Literal["plugin", "project", "local"]


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """An extension entry file, ownership, and import shape."""

    path: Path
    is_package: bool = False
    scope: ExtensionScope = "local"
    plugin_id: str | None = None
    plugin_version: str | None = None
    package_root: Path | None = None
    data_dir: Path | None = None

    @property
    def label(self) -> str:
        """Short extension label."""
        if self.plugin_id is not None:
            return self.plugin_id
        return self.path.parent.name if self.is_package else self.path.stem


@dataclass(frozen=True, slots=True)
class RegisteredUnit[T]:
    """A registered unit paired with its source."""

    name: str
    unit: T
    source: SourceInfo


class ExtensionError(Exception):
    """Raised when an extension cannot be loaded."""
