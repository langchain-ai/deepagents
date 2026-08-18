"""Synchronous providers for local configuration sources."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_code.configuration.types import (
    Found,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    TomlSnapshot,
    Unset,
)


class ConfigProvider[T](Protocol):
    """Provider of values keyed by canonical config identifier."""

    name: str
    rank: int

    def get(self, key: str) -> ProviderResult[T]:
        """Return this provider's result for `key`."""
        ...


@dataclass(frozen=True, slots=True)
class MappingProvider[T]:
    """Immutable snapshot provider backed by a mapping."""

    name: str
    rank: int
    values: dict[str, T]

    def get(self, key: str) -> ProviderResult[T]:
        """Return the mapped value or `Unset`."""
        if key not in self.values:
            return Unset()
        return Found(self.values[key], self.name)


class EnvProvider[T](MappingProvider[T]):
    """Snapshot of explicitly resolved environment values."""


class CliProvider[T](MappingProvider[T]):
    """Snapshot of explicitly supplied CLI values."""


@dataclass(frozen=True, slots=True)
class TomlFileProvider:
    """Provider that parses one local TOML file per `load` call."""

    name: str
    path: Path
    rank: int

    def load(self) -> TomlSnapshot:
        """Parse the file and classify missing, unreadable, or corrupt states.

        Returns:
            Parsed data and provider health.
        """
        try:
            with self.path.open("rb") as handle:
                data = tomllib.load(handle)
        except FileNotFoundError:
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.MISSING,
                ),
            )
        except OSError as exc:
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.UNREADABLE,
                    type(exc).__name__,
                ),
            )
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            detail = (
                "not UTF-8 encoded" if isinstance(exc, UnicodeDecodeError) else str(exc)
            )
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.CORRUPT,
                    detail,
                ),
            )
        if not isinstance(data, dict):
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.CORRUPT,
                    "top-level TOML value is not a table",
                ),
            )
        return TomlSnapshot(
            data,
            ProviderStatus(self.name, self.path, ProviderHealth.OK),
        )

    def get_path(
        self, data: dict[str, Any], keys: tuple[str, ...]
    ) -> ProviderResult[Any]:
        """Return a nested TOML value without conflating empty and missing values."""
        node: Any = data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return Unset()
            node = node[key]
        return Found(node, self.name)
