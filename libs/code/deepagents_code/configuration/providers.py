"""Synchronous providers for local configuration sources."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_code.configuration.types import (
    ProviderHealth,
    ProviderStatus,
    TomlSnapshot,
)


@dataclass(frozen=True, slots=True)
class TomlFileProvider:
    """Provider that parses one local TOML file per `load` call."""

    name: str
    path: Path

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
