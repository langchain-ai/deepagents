"""Internal data types for Python extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """An extension entry file and its import shape."""

    path: Path
    is_package: bool = False

    @property
    def label(self) -> str:
        """Short extension label."""
        return self.path.parent.name if self.is_package else self.path.stem


@dataclass(frozen=True, slots=True)
class RegisteredUnit[T]:
    """A registered unit paired with its source."""

    name: str
    unit: T
    source: SourceInfo


class ExtensionError(Exception):
    """Raised when an extension cannot be loaded."""
